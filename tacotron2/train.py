import os
import time
import argparse
import math
from numpy import finfo
import os
import pandas as pd
from pydub import AudioSegment
import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from text import symbols
import numpy as np
import librosa
import scipy.io.wavfile as wav

            # mel_spec = torch.detach(mel_spec)
            # mel_spec = torch.Tensor.cpu(mel_spec)

from pydub.silence import detect_nonsilent

def play(mel_spec, sample_rate=16000, min_silence_len=100, silence_thresh=-50):
    temp_path = "/media/fathom/agi/gpt-Rogan/audio/audiofiles/temp/"
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    for i, mel_spec in enumerate(mel_spec):
        if torch.sum(mel_spec) > 0.1:
            waveform = librosa.feature.inverse.mel_to_audio(mel_spec.numpy().T, sr=sample_rate, n_mels=80)
            waveform = (waveform * 32767).astype(np.int16)
            wav.write(temp_path + f"y_{i}.wav", sample_rate, waveform)
            audio = AudioSegment.from_wav(temp_path + f"y_{i}.wav")
            nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
            for j, nonsilent_range in enumerate(nonsilent_ranges):
                start_time = nonsilent_range[0]
                end_time = nonsilent_range[1]
                audio_segment = audio[start_time:end_time]
                audio_segment = audio_segment.set_channels(1)
                audio_segment.export(temp_path + f"y_mono_{i}_{j}.wav", format="wav")
                os.system("aplay " + temp_path + f"y_mono_{i}_{j}.wav")

    

class Hparams:
    def __init__(self, **entries):
        self.__dict__.update(entries)
hparams = Hparams(epochs=10,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=True,
        distributed_run=True,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],
        load_mel_from_disk=False,
        training_files="/media/fathom/agi/gpt-Rogan/audio/audiofiles/segments/train.txt",
        validation_files="/media/fathom/agi/gpt-Rogan/audio/audiofiles/segments/val.txt",
        text_cleaners=['english_cleaners'],
        max_wav_value=32768.0,
        sampling_rate=16000,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        n_frames_per_step=1,
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        attention_rnn_dim=1024,
        attention_dim=128,
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=10,
        mask_padding=True)
    

def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
            val_loss = val_loss / (i + 1)
            model.train()
            if rank == 0:
                print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
                logger.log_validation(val_loss, model, y, y_pred, iteration)


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            if i % 10 == 0:
                play(y_pred)
            loss = criterion(y_pred, y)
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1








torch.backends.cudnn.enabled = hparams.cudnn_enabled
torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
print("FP16 Run:", hparams.fp16_run)
print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
print("Distributed Run:", hparams.distributed_run)
print("cuDNN Enabled:", hparams.cudnn_enabled)
print("cuDNN Benchmark:", hparams.cudnn_benchmark)







picklesfolder = "/media/fathom/agi/gpt-Rogan/audio/audiofiles/parsed"
wav_folder = "/media/fathom/agi/gpt-Rogan/audio/audiofiles/wav"
segments = "/media/fathom/agi/gpt-Rogan/audio/audiofiles/segments"
trainpath=str(segments + "/" + "train.txt")
valpath=str(segments + "/" + "val.txt")
for picklefile in os.listdir(picklesfolder):
    sound = wav_folder + "/" + picklefile[:-4]
    sound = AudioSegment.from_wav(sound)
    sound = sound.set_frame_rate(16000)
    file_path = os.path.join(picklesfolder, picklefile)
    with open(file_path, 'rb') as handle:
        df = pd.read_pickle(handle)
        num_speakers = len(df["speaker"].unique())
        speaker_list = []
        current_speaker = None
        texts = ""
        for i in range(20,(len(df))):
            start = df.iloc[i]["start"]
            end = df.iloc[i]["end"]
            speaker = df.iloc[i]["speaker"]
            if current_speaker != speaker:
                if current_speaker is not None:
                    end_time = df.iloc[i-1]["end"]
                    speaker_list.append([current_speaker, texts, start_time, end_time])
                    texts = ""
                start_time = df.iloc[i]["start"]
                current_speaker = speaker
            texts += " " +df.iloc[i]["text"]
        end_time = df.iloc[len(df)-1]["end"]
        speaker_list.append([current_speaker, texts, start_time, end_time])
        speaker_list = [x for x in speaker_list if x[0] == "SPEAKER_00"]
        speaker_list = [x for x in speaker_list if x[3] - x[2] > 2 and x[3] - x[2] < 15]
        for count, p in enumerate(speaker_list):
                sound1 = sound[p[2]*1000:p[3]*1000]
                audio_path = segments + "/" + str(count) + ".wav"
                sound1.export(audio_path, format="wav")
                audio_path = os.path.abspath(audio_path)
                if count < 40:
                    with open(trainpath, "a") as f:
                        f.write(audio_path + "|" + p[1] + "\n")
                else:
                    with open(valpath, "a") as f:
                        f.write(audio_path + "|" + p[1] + "\n")
    train(output_directory="/media/fathom/agi/gpt-Rogan/outdir",log_directory="/media/fathom/agi/gpt-Rogan/logs",checkpoint_path="/media/fathom/agi/gpt-Rogan/outdir/checkpoint_0",warm_start=False,n_gpus=1,rank=0,group_name="joe",hparams=hparams)
    with open (segments + "/train.txt", "w") as f:
        f.write("")
    with open (segments + "/val.txt", "w") as f:
        f.write("")
    for file in os.listdir(segments):
        if file.endswith(".wav"):
            os.remove(segments + "/" + file)
    