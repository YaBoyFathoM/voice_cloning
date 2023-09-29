import whisperx
import datetime
import subprocess
import pandas as pd
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from torch.utils.cpp_extension import CUDA_HOME
import os
import wave
#check if torch is using GPU
if torch.cuda.is_available()==False:
  print("GPU not available")
  np.random.seed(-5)
#set unlimited pytorch memory usage


model = whisperx.load_model("base")
embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

def transcribe(audio, num_speakers,file_name):
  path, error = convert_to_wav(audio)
  if error is not None:
    return error

  duration = get_duration(path)
  if duration > 4 * 60 * 60:
    return "Audio duration too long"

  result = model.transcribe(path,verbose=True)
  segments = result["segments"]

  num_speakers = min(max(round(num_speakers), 1), len(segments))
  if len(segments) == 1:
    segments[0]['speaker'] = '0'
  else:
    embeddings = make_embeddings(path, segments, duration)
    segments= add_speaker_labels(segments, embeddings, num_speakers)
  df=pd.DataFrame(columns=["speaker","tokens","audio"])
  for seg in segments:
    speaker=seg["speaker"]
    start=seg["start"]
    end=seg["end"]
    #get audio clip
    clip=wave.open(path)
    clip=clip.readframes(int((end-start)*clip.getframerate()))
    tokens=seg["tokens"]
    df=df.append({"speaker":int(speaker),"tokens":list(tokens),"audio":clip},ignore_index=True)
  df.to_csv(dest_folder+"/"+file_name,index=True)
  torch.cuda.empty_cache()
def convert_to_wav(path):
  if path[-3:] != 'wav':
    new_path = '.'.join(path.split('.')[:-1]) + '.wav'
    try:
      subprocess.call(['ffmpeg', '-i', path, new_path, '-y'])
    except:
      return path, 'Error: Could not convert file to .wav'
    os.remove(path)
    path = new_path
  return path, None

def get_duration(path):
  with contextlib.closing(wave.open(path,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    return frames / float(rate)

def make_embeddings(path, segments, duration):
  embeddings = np.zeros(shape=(len(segments), 192))
  for i, segment in enumerate(segments):
    embeddings[i] = segment_embedding(path, segment, duration)
  return np.nan_to_num(embeddings)

audio = Audio()

def segment_embedding(path, segment, duration):
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)
  return embedding_model(waveform[None])

def add_speaker_labels(segments, embeddings, num_speakers):
  clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
  labels = clustering.labels_
  for i in range(len(segments)):
    segments[i]["speaker"] = str(labels[i])
  return segments

def time(secs):
  return datetime.timedelta(seconds=round(secs))




  

audi_folder="audio/audiofiles/mp3"
dest_folder="audio/audiofiles/parsed"
for audi in os.listdir(audi_folder):
    if audi.endswith(".mp3"):
      wav=convert_to_wav(audi_folder+"/"+audi)
    #     file_name = audi.split("#")[1].split(".")[0]+".csv"
    # if "&" in file_name:
    #     num_speakers = 3
    # else:
    #     num_speakers = 2
    # transcribe(audi_folder+"/"+audi, num_speakers,file_name)