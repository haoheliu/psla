import pickle

def save_pickle(obj,fname):
    print("Save pickle at "+fname)
    with open(fname,'wb') as f:
        pickle.dump(obj,f)

def load_pickle(fname):
    print("Load pickle at "+fname)
    with open(fname,'rb') as f:
        res = pickle.load(f)
    return res

import os
import numpy as np

PATH = "/media/Disk_HDD/haoheliu/projects/psla/egs/fsd50k/exp/nerual_sampler-0.25-weightnorm-feature-efficientnet-2-5e-4-fsd50k-impretrain-True-fm48-tm192-mix0.5-bal-True-b48-le-2"
for file in os.listdir(PATH):
    if("pickle" in file):
        stats = load_pickle(os.path.join(PATH, file))
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        print(file, mAP, mAUC)