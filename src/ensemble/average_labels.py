import os
from tkinter import ALL
from tqdm import tqdm 
import numpy as np

audioset_labels_part = "/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/audioset_segment_labels_ensemble"
averaged_dir_path = "/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/audioset_segment_labels_ensemble/averaged"
os.makedirs(averaged_dir_path, exist_ok=True)

npy_files = os.listdir(os.path.join(audioset_labels_part, "as_0"))

for file in tqdm(npy_files):
    outputs = []
    if(os.path.exists(os.path.join(averaged_dir_path, file))):
        continue
    
    ALL_FINISHED = True
    for i in range(0,10):
        if(not os.path.exists(os.path.join(audioset_labels_part, "as_%s" % i, file))):
            ALL_FINISHED=False
            break
        
    if(not ALL_FINISHED):
        continue
    
    for i in range(0,10):
        model_output = os.path.join(audioset_labels_part, "as_%s" % i, file)
        outputs.append(np.load(model_output))
    outputs = np.concatenate([x[None,...] for x in outputs], axis=0)
    avg_output = np.mean(outputs, axis=0) # [53, 527]
    
    if(not os.path.exists(os.path.join(averaged_dir_path, file))):
        np.save(os.path.join(averaged_dir_path, file), avg_output)
        
        

    
        
    