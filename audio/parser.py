import numpy as np
import os
import wave
import ffmpeg
import torch
import subprocess
#add to path

#connect torch to gpu

if torch.cuda.is_available() != True:
    raise ValueError("GPU not available")
# usage: whisperx [-h] [--model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large}]
#                 [--model_dir MODEL_DIR] [--device DEVICE] [--align_model ALIGN_MODEL] [--align_extend ALIGN_EXTEND]
#                 [--align_from_prev ALIGN_FROM_PREV] [--interpolate_method {nearest,linear,ignore}] [--vad_filter]
#                 [--parallel_bs PARALLEL_BS] [--diarize] [--min_speakers MIN_SPEAKERS] [--max_speakers MAX_SPEAKERS]
#                 [--output_dir OUTPUT_DIR] [--output_type {all,srt,srt-word,vtt,txt,tsv,ass,ass-char,pickle,vad}]
#                 [--verbose VERBOSE] [--task {transcribe,translate}]
folder = "audio/audiofiles/wav"
dfolder = "audio/audiofiles/parsed"
for f in os.listdir(folder):
    file_path = os.path.join(folder, f)
    os.system(f'whisperx "{file_path}" --model base.en --device cuda --align_model WAV2VEC2_ASR_BASE_960H --hf_token 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 --diarize --max_speakers 4 --vad_filter --output_dir "{dfolder}" --output_type tsv --verbose True --task transcribe')
    break



#df=transcribe(file_sound,num_speakers,file_name)
#df.to_csv(dfolder+"/"+file_name,index=True)
"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"

