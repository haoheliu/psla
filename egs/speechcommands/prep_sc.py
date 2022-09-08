# -*- coding: utf-8 -*-
# @Time    : 6/23/21 3:19 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : prep_sc.py

import numpy as np
import json
import os
import wget
from torchaudio.datasets import SPEECHCOMMANDS

# prepare the data of the speechcommands dataset.
print('Now download and process speechcommands dataset, it will take a few moments...')

PATH = "/mnt/fast/nobackup/scratch4weeks/hl01486/datasets/speechcommands/"
# download the speechcommands dataset

# we use the 35 class v2 dataset, which is used in torchaudio https://pytorch.org/audio/stable/_modules/torchaudio/datasets/speechcommands.html
# if(not os.path.exists("speech_commands_v0.02.tar.gz")):
#     sc_url = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
#     wget.download(sc_url, out='.')

# if(len(os.listdir(PATH)) < 5):
#     os.makedirs(PATH, exist_ok=True)
#     os.system('tar -xzvf speech_commands_v0.02.tar.gz -C %s' % PATH)
#     os.remove('speech_commands_v0.02.tar.gz')

# generate training list = all samples - validation_list - testing_list
if os.path.exists(os.path.join(PATH, 'train_list.txt'))==False:
    with open(os.path.join(PATH, 'validation_list.txt'), 'r') as f:
        val_list = f.readlines()

    with open(os.path.join(PATH, 'testing_list.txt'), 'r') as f:
        test_list = f.readlines()

    val_test_list = list(set(test_list+val_list))

    def get_immediate_subdirectories(a_dir):
        return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
    def get_immediate_files(a_dir):
        return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

    base_path = PATH
    all_cmds = get_immediate_subdirectories(base_path)
    all_list = []
    for cmd in all_cmds:
        if cmd != '_background_noise_':
            cmd_samples = get_immediate_files(base_path+'/'+cmd)
            for sample in cmd_samples:
                all_list.append(cmd + '/' + sample+'\n')

    training_list = [x for x in all_list if x not in val_test_list]

    with open(os.path.join(PATH, 'train_list.txt'), 'w') as f:
        f.writelines(training_list)

label_set = np.loadtxt('./data/speechcommands_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]

os.makedirs('./data/datafiles', exist_ok=True)
base_path = PATH
for split in ['testing', 'validation', 'train']:
    wav_list = []
    with open(base_path+split+'_list.txt', 'r') as f:
        filelist = f.readlines()
    for file in filelist:
        cur_label = label_map[file.split('/')[0]]
        cur_path = os.path.join(PATH, file.strip()) 
        cur_dict = {"wav": cur_path, "labels": '/m/spcmd'+cur_label.zfill(2)}
        wav_list.append(cur_dict)
    np.random.shuffle(wav_list)
    if split == 'train':
        with open('./data/datafiles/speechcommand_train_data.json', 'w') as f:
            json.dump({'data': wav_list}, f, indent=1)
    if split == 'testing':
        with open('./data/datafiles/speechcommand_eval_data.json', 'w') as f:
            json.dump({'data': wav_list}, f, indent=1)
    if split == 'validation':
        with open('./data/datafiles/speechcommand_valid_data.json', 'w') as f:
            json.dump({'data': wav_list}, f, indent=1)
    print(split + ' data processing finished, total {:d} samples'.format(len(wav_list)))

print('Speechcommands dataset processing finished.')



