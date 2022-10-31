import os
import numpy as np
import tqdm
import pandas as pd

ensemble_label_path = "/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/audioset_segment_labels_ensemble/averaged"
audio_file_path = "audio"


def build_id_to_label(
    path="/mnt/fast/nobackup/users/hl01486/metadata/audioset/class_labels_indices.csv",
):
    ret = {}
    id2num = {}
    num2label = {}
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        index, mid, display_name = row["index"], row["mid"], row["display_name"]
        ret[mid] = display_name
        id2num[mid] = index
        num2label[index] = display_name
    return ret, id2num, num2label

_,_,num2label = build_id_to_label()


audio_files_eval = [
    "Y---2_BBVHAA.wav",
    "Y---B_v8ZoBY.wav",
    "Y---N4cFAE1A.wav",
    "Y---fcVQUf3E.wav",
    "Y---g9OGAhwc.wav",
    "Y---lTs1dxhU.wav",
    "Y---qub7uxgs.wav",
]

def visualize_label(label, file):
    import matplotlib.pyplot as plt
    output_pred = np.mean(label, axis=0)
    index = np.argsort(output_pred)[::-1][:5]
    output = label[:, index]
    plt.figure(figsize=(8,8))
    plt.imshow(output.T, aspect="auto")
    plt.title([num2label[x] for x in index])
    plt.savefig(file.replace('.wav',".png"))

for file in audio_files_eval:
    audio_path = os.path.join(audio_file_path, file)
    label_path = os.path.join(ensemble_label_path, file.replace(".wav",".npy"))    
    if(not os.path.exists(label_path)):
        print(label_path,"not found")
        continue
    label = np.load(label_path)
    visualize_label(label, file)
    os.system("cp %s ." % audio_path)