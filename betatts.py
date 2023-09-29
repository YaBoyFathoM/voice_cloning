import os
import pandas as pd
from pydub import AudioSegment
import torch
import librosa
import sys
import matplotlib
from tacotron2.model import Tacotron2
from tacotron2.loss_function import Tacotron2Loss
from waveglow.glow import WaveGlow,WaveGlowLoss
from text import symbols
import pandas as pd
from tqdm import tqdm
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import subprocess
import torchaudio
import glob
from tacotron2.data_utils import TextMelLoader, TextMelCollate
import torch
import torch.optim as optim
import torchaudio
from text import sequence_to_text
import numpy as np
import torch.cuda.amp as amp


if torch.cuda.is_available():
    print("GPU available")
else:
    raise ValueError("GPU not available")
torch.cuda.empty_cache()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Hparams:
    def __init__(self, **entries):
        self.__dict__.update(entries)
hparams = Hparams(epochs=1000,
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
        training_files="/media/cam/agi/gpt-Rogan/audio/audiofiles/segments/train.txt",
        validation_files="/media/cam/agi/gpt-Rogan/audio/audiofiles/segments/val.txt",
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
        max_decoder_steps=6000,
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
        batch_size=1,
        mask_padding=True)

def plot_and_play(spec, wav_file):
    figsize=(16, 4)
    fig, axes = plt.subplots(figsize=figsize)
    axes.imshow(spec[0,:,:], aspect='auto', origin='lower', 
                    interpolation='none')
    subprocess.Popen(["ffplay", "-nodisp", "-autoexit", wav_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    duration = librosa.get_duration(filename=wav_file)
    print("duration: ", duration)
    plt.pause(duration)
    print(duration)
    plt.cla()
    plt.close("all")


class JoeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fresh = True
        self.metadata_file = "/media/cam/agi/gpt-Rogan/audio/audiofiles/segments/train.txt"
        self.checkpoint_path = "/media/cam/agi/gpt-Rogan/outdir"
        self.wav_folder = "/media/cam/agi/gpt-Rogan/audio/audiofiles/wav"
        self.segments = "/media/cam/agi/gpt-Rogan/audio/audiofiles/segments"
        self.trainpath=str(self.segments + "/" + "train.txt")
        self.valpath=str(self.segments + "/" + "val.txt")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2*1)
        self.conv4 = torch.nn.Conv2d(64, 32, kernel_size=3)
        self.conv5 = torch.nn.Conv2d(32, 16, kernel_size=3)
        self.conv6 = torch.nn.Conv2d(16, 1, kernel_size=3)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2*1)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(5175, 400)
        self.fc2 = torch.nn.Linear(400, 200)
        self.fc3 = torch.nn.Linear(200, 100)
        self.fc4 = torch.nn.Linear(100, 1) 
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_func = torch.nn.BCELoss()
        for param in self.parameters():
            param.requires_grad = True
    
        
        
    def forward(self, x):
        # Apply the layers in the defined order
        x=x.view(1,1,80,1400)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = x.view(1,5175)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

class Tacotron2WaveglowTrainer:
    def __init__(self, hparams,fresh=True):
        self.hp = hparams
        self.fresh = fresh
        self.metadata_file = "/media/cam/agi/gpt-Rogan/audio/audiofiles/segments/train.txt"
        self.checkpoint_path = "/media/cam/agi/gpt-Rogan/models"
        self.picklesfolder = "/media/cam/agi/gpt-Rogan/audio/audiofiles/parsed"
        self.wav_folder = "/media/cam/agi/gpt-Rogan/audio/audiofiles/wav"
        self.segments = "/media/cam/agi/gpt-Rogan/audio/audiofiles/segments"
        self.trainpath=str(self.segments + "/" + "train.txt")
        self.valpath=str(self.segments + "/" + "val.txt")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    
    def detect_joe(self,audio_input_path,detector):
        audio_input = AudioSegment.from_wav(audio_input_path)
        audio_input = audio_input.set_frame_rate(16000)
        audio_input.set_channels(1)
        audio_input = torch.tensor(audio_input.get_array_of_samples()).view(1,-1).float()
        audio_input = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024, hop_length=256, n_mels=80)(audio_input)
        pad=torch.zeros((1,80,1400-audio_input.shape[2]))
        audio_input=torch.cat((audio_input,pad),2)
        audio_input = audio_input.view(1,1,80,1400)
        with torch.no_grad():
            output = detector.forward(audio_input)
        if torch.sum(output) > 0.5:
            return True
        else:
            return False
    def generate_data(self,picklefolder):
        detector=JoeDetector().to("cpu")
        detector.load_state_dict(torch.load("/media/cam/agi/gpt-Rogan/models/joedet_2590.pt")['state_dict'])
        if self.fresh==True:
            for file in os.listdir("/media/cam/agi/gpt-Rogan/outdir"):
                if file.endswith(".pt"):
                    os.remove("/media/cam/agi/gpt-Rogan/outdir" + "/" + file)
        #random file order
        for picklefile in os.listdir(picklefolder)[2:]:
            self.clear()
            samples = 0
            sound = self.wav_folder + "/" + picklefile[:-4]
            sound = AudioSegment.from_wav(sound)
            sound = sound.set_frame_rate(16000)
            file_path = os.path.join(picklefolder, picklefile)
            with open(file_path, 'rb') as handle:
                df = pd.read_pickle(handle)
                speaker_list = []
                current_speaker = None
                texts = ""
                for i in range((len(df))):
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
                for count, p in enumerate(speaker_list):
                    duration=int(p[3]-p[2])
                    if 2<duration<10:
                        sound1 = sound[p[2]*1000:p[3]*1000]
                        audio_path = self.segments + "/" + str(count) + ".wav"
                        sound1.export(audio_path, format="wav")
                        audio_path = os.path.abspath(audio_path)
                        if self.detect_joe(audio_path,detector)==True:
                            with open(self.trainpath, "a",encoding="utf-8") as f:
                                f.write(audio_path + "|" + p[1] + "\n")
                                samples+=1
                                print("samples generated",samples)
                del df, speaker_list, sound, sound1
                self.train()
                self.fresh=False
                self.clear()

    def train(self):
            print("TRAINING")
            WN_config = dict(n_layers=8, kernel_size=5, n_channels=80)
            textloader = TextMelLoader(self.metadata_file, self.hp)
            collate_fn = TextMelCollate(1)
            tacotron2 = Tacotron2(self.hp).to(self.device)
            waveglow = WaveGlow(n_mel_channels=80, WN_config=WN_config, n_flows=12, n_group=8, n_early_every=4, n_early_size=2).to(self.device)
            tloss_fn = Tacotron2Loss().to(self.device)
            wloss_fn = WaveGlowLoss().to(self.device)
            topt = optim.Adam(tacotron2.parameters())
            wopt = optim.Adam(waveglow.parameters())
            if os.listdir(self.checkpoint_path) != []:
                latest_checkpoint_tacotron = max(glob.glob(self.checkpoint_path + "/tacotron2_*.pt"), key=os.path.getctime)
                latest_checkpoint_waveglow = max(glob.glob(self.checkpoint_path + "/waveglow_*.pt"), key=os.path.getctime)
                tacotron2.load_state_dict(torch.load(latest_checkpoint_tacotron)['state_dict'])
                waveglow.load_state_dict(torch.load(latest_checkpoint_waveglow)['state_dict'])
            tacotron2.train()
            waveglow.train()
            total_loss = 0
            scaler = amp.GradScaler()
            for epoch in range(1000):
                for i, _ in enumerate(textloader):
                    linedata, audio_path = textloader.__getitem__(i)
                    inputs, targets = tacotron2.parse_batch(collate_fn([linedata]))
                    model_outputs = tacotron2.forward(inputs)
                    audio, _ = torchaudio.load(audio_path)
                    audio = audio.to(self.device)
                    with amp.autocast():
                        tloss = tloss_fn(model_outputs, targets)
                        wav_outputs = waveglow.forward((model_outputs[0], audio))
                        wloss = wloss_fn(wav_outputs)
                        combined_loss = tloss + wloss
                    scaler.scale(combined_loss).backward()
                    scaler.step(topt)
                    scaler.step(wopt)
                    scaler.update()
                    topt.zero_grad()
                    wopt.zero_grad()
                    total_loss += combined_loss.item()
                avg_loss = total_loss / len(textloader)
                print("Epoch: {} Average Loss: {:.4f}".format(epoch, avg_loss))
                # with torch.no_grad():
                #     print("Saving checkpoint...")
                #     torch.save({'state_dict': tacotron2.state_dict()}, self.checkpoint_path + "/tacotron2_{}.pt".format(epoch))
                #     torch.save({'state_dict': waveglow.state_dict()}, self.checkpoint_path + "/waveglow_{}.pt".format(epoch))
                #     print("Saved checkpoint")
    def clear(self):
        with open (self.trainpath, "w") as f:
            f.write("")
        with open (self.valpath, "w") as f:
            f.write("")
        for file in os.listdir(self.segments):
            if file.endswith(".wav"):
                os.remove(self.segments + "/" + file)
            
trainer=Tacotron2WaveglowTrainer(hparams,fresh=False)
trainer.generate_data(picklefolder = "/media/cam/agi/gpt-Rogan/audio/audiofiles/parsed")
    


# subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "/media/cam/agi/gpt-Rogan/audio/audiofiles/segments/6.wav"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
# proceed = input("Proceed? (y/n)")
# if proceed == "n":
#     shutil.move(picklesfolder + "/" + picklefile, filterfolder + "/" + picklefile)
#     continue