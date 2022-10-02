# @Time    : 3/8/22 
# @Author : Juncheng B Li

import numpy as np
import json
import os
import zipfile
import wget
import csv
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--slurm-id", type=str, default='5689254', help="the root path of audio data")

if __name__ == '__main__':
    args = parser.parse_args()
    if(os.path.exists("/media/Disk_HDD")):
        base_path = '/media/Disk_HDD/haoheliu/projects/psla/egs/audioset'
        list_path = "/media/Disk_HDD/haoheliu/projects/psla/egs/audioset/data/"
        data_path = "/media/Disk_HDD/haoheliu/datasets/AudioSet"
    if(os.path.exists("/mnt/fast")):
        base_path = '/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla/egs/audioset'
        list_path = "/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla/egs/audioset/data/"
        data_path = "/mnt/fast/datasets/audio/audioset/2million_audioset_wav"
    else:
        base_path = '/vol/research/NOBACKUP/CVSSP/scratch_4weeks/hl01486/projects/psla/egs/audioset_condor'
        list_path = "/vol/research/NOBACKUP/CVSSP/scratch_4weeks/hl01486/projects/psla/egs/audioset/data/"
        data_path = "/vol/research/datasets/audio/audioset/2million_audioset_wav"

    # fix bug: generate an empty directory to save json files
    if os.path.exists(base_path + '/data/datafiles') == False:
        os.mkdir(base_path + '/data/datafiles')

    bal_train_wav_list = []
    unbal_train_wav_list = []
    eval_wav_list = []
    bal_unbal_train_wav_list = []
    with open(os.path.join(list_path, 'unbalanced_train_segments.csv')) as unbal_csv_file:
        unbal_csv_reader = csv.reader(unbal_csv_file, delimiter=',')

        unbal_count = 0
        missing_count = 0
        for row in tqdm(unbal_csv_reader):
            if unbal_count <3:
                print(f'Column names are {", ".join(row)}')
                unbal_count += 1
            else:
                for i in range(0,42):
                    wav_path = os.path.join(data_path, 'unbalanced_train_segments_part%d' % i,"Y"+row[0] +'.wav')
                    if os.path.exists(wav_path):
                        cur_unbal_dict = {"wav": wav_path, "labels": eval(','.join(row[3:]))}
                        unbal_train_wav_list.append(cur_unbal_dict)
                        bal_unbal_train_wav_list.append(cur_unbal_dict)
                        unbal_count += 1
                        break
                if(i == 41):
                    missing_count+=1
    #                 print("missing training: " + row[0])
    print(f'unbalanced count: {unbal_count}, missing count: {missing_count}')
    with open(base_path + '/data/datafiles/audioset_unbal_train_data' +'.json', 'w') as f:
        json.dump({'data': unbal_train_wav_list}, f, indent=1)

    with open(os.path.join(list_path, 'balanced_train_segments.csv')) as bal_csv_file:
        bal_csv_reader = csv.reader(bal_csv_file, delimiter=',')

        bal_count = 0
        missing_count = 0
        for row in tqdm(bal_csv_reader):
            if bal_count <3:
                print(f'Column names are {", ".join(row)}')
                bal_count += 1
            else:
                wav_path = os.path.join(data_path, 'balanced_train_segments',"Y"+row[0] +'.wav')
                if os.path.exists(wav_path):
                    cur_bal_dict = {"wav": wav_path, "labels": eval(','.join(row[3:]))}
    #             print(cur_bal_dict)
                    bal_train_wav_list.append(cur_bal_dict)
                    bal_unbal_train_wav_list.append(cur_bal_dict)
                    bal_count += 1

                else:
                    missing_count+=1
    #                 print("missing training: " + row[0])
    print(f'balanced count: {bal_count}, missing count: {missing_count}')
    with open(base_path + '/data/datafiles/audioset_bal_train_data' +'.json', 'w') as f:
        json.dump({'data': bal_train_wav_list}, f, indent=1)

    with open(base_path + '/data/datafiles/audioset_bal_unbal_train_data' +'.json', 'w') as f:
        json.dump({'data': bal_unbal_train_wav_list}, f, indent=1)

    with open(os.path.join(list_path, 'eval_segments.csv')) as eval_csv_file:
        eval_csv_reader = csv.reader(eval_csv_file, delimiter=',')

        eval_count = 0
        missing_count = 0
        for row in eval_csv_reader:
            if eval_count <3:
                print(f'Column names are {", ".join(row)}')
                eval_count += 1
            else:
                wav_path = os.path.join(data_path, 'eval_segments',"Y"+row[0] +'.wav')
                if os.path.exists(wav_path):
                    cur_eval_dict = {"wav": wav_path, "labels": eval(','.join(row[3:]))}
    #             print(cur_bal_dict)
                    eval_wav_list.append(cur_eval_dict)
                    eval_count += 1

                else:
                    missing_count+=1
    #                 print("missing training: " + row[0])
    print(f'eval count: {eval_count}, missing count: {missing_count}')
    with open(base_path + '/data/datafiles/audioset_eval_data' +'.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)


    print('Finished AudioSet Preparation')