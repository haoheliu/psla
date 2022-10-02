import os
import wave
from glob import glob
from tqdm import tqdm
import numpy as np

def get_duration(fname):
    with wave.open(fname) as f:
        params = f.getparams()
    return params[3]

def recursive_glob(path, suffix):
    return glob(os.path.join(path,"*" + suffix)) + \
                glob(os.path.join(path,"*/*" + suffix)) + \
                    glob(os.path.join(path,"*/*/*" + suffix)) + \
                        glob(os.path.join(path,"*/*/*/*" + suffix)) + \
                            glob(os.path.join(path,"*/*/*/*/*" + suffix)) + \
                                glob(os.path.join(path,"*/*/*/*/*/*" + suffix))
                                
                                
# ROOT = "/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/FSD50K"                            
# ROOT = "/mnt/fast/datasets/audio/audioset/2million_audioset_wav"                            
# ROOT = "/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/speechcommands"
# ROOT = "/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/nsynth"
ROOT = "/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla/egs/esc50/data/ESC-50-master/audio_16k"
files = recursive_glob(ROOT, ".wav")
duration_list = []
for file in tqdm(files):
    path = os.path.join(ROOT, file)
    duration = get_duration(path)
    duration_list.append(duration/16000)
    
print(np.mean(duration_list), np.std(duration_list))
import ipdb; ipdb.set_trace()
    