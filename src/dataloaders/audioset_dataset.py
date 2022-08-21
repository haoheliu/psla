# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import logging
# from torchsubband import SubbandDSP
import os

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None, hop_ms=10):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.datapath = dataset_json_file
        self.hop_ms = hop_ms
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.audio_conf = audio_conf
        logging.info('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        logging.info('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        logging.info('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        logging.info('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.mode = self.audio_conf.get('mode')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            logging.info('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            logging.info('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            logging.info('now use noise augmentation')
        # self.dsp = SubbandDSP()
        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        logging.info('number of classes is {:d}'.format(self.label_num))

    def resample_16k(self, data, sr):
        if(sr == 16000): 
            return data, 16000
        elif(sr == 32000):
            return data[:,::2], 16000
        else:
            raise RuntimeError("Unexpected sampling rate %s" % (sr))

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform, sr = self.resample_16k(waveform, sr)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform1, sr = self.resample_16k(waveform1, sr)
            waveform2, sr = torchaudio.load(filename2)
            waveform2, sr = self.resample_16k(waveform2, sr)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, :waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, :waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        # torch.Size([1, 160000]) torch.Size([998, 128])

        target_length = self.audio_conf.get('waveform_length')
        n_samples = waveform.shape[1]

        p = target_length - n_samples

        # cut and pad
        if p > 0:
            waveform = torch.nn.functional.pad(waveform, (0, p), mode='constant', value=0) 
        elif p < 0:
            waveform = waveform[:, :target_length]
        
        assert waveform.size()[1] == target_length, "%s %s %s" % (n_samples, p, waveform.size())
        
        if filename2 == None:
            return None, 0, waveform
        else:
            return None, mix_lambda, waveform
    

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # print(self.index_dict.keys())
        # print(len(self.index_dict.keys()))
        # do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            while(True):
                try:
                    datum = self.data[index]
                    # find another sample to mix, also do balance sampling
                    # sample the other sample from the multinomial distribution, will make the performance worse
                    # mix_sample_idx = np.random.choice(len(self.data), p=self.sample_weight_file)
                    # sample the other sample from the uniform distribution
                    mix_sample_idx = random.randint(0, len(self.data)-1)
                    mix_datum = self.data[mix_sample_idx]
                    # get the mixed fbank
                    fbank, mix_lambda, waveform = self._wav2fbank(datum['wav'], mix_datum['wav'])
                    break
                except:
                    print("error reading file during mixup", datum['wav'], mix_datum['wav'])
                    logging.warning("Error reading file: %s, %s" % (datum['wav'], mix_datum['wav']))
                    index += 1
                    index = index % len(self.data)

            # initialize the label
            label_indices = np.zeros(self.label_num)
            # add sample 1 labels
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda
            # add sample 2 labels
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0-mix_lambda)
            label_indices = torch.FloatTensor(label_indices)
        # if not do mixup
        else:
            while(True):
                try:
                    datum = self.data[index]
                    label_indices = np.zeros(self.label_num)
                    fbank, mix_lambda, waveform = self._wav2fbank(datum['wav'])
                    break
                except Exception as e:
                    print("error reading file", datum['wav'])
                    logging.warning("Error reading file: %s" % datum['wav'])
                    index += 1
                    index = index % len(self.data)

            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0

            label_indices = torch.FloatTensor(label_indices)
        # if self.noise == True:
        #     fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
        #     fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
        return waveform, label_indices, os.path.basename(datum['wav'])

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end-start) * val

    def frequency_masking(self, fbank, freqm):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(freqm // 8, freqm))
        mask_start = int(self.random_uniform(start=0, end=freq-mask_len))
        fbank[:,mask_start:mask_start+mask_len,:] *= 0.0
        # value = self.random_uniform(0.0, 1.0)
        # fbank[:,mask_start:mask_start+mask_len,:] += value
        return fbank

    def time_masking(self, fbank, timem):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(timem // 8, timem))
        mask_start = int(self.random_uniform(start=0, end=tsteps-mask_len))
        fbank[:,:,mask_start:mask_start+mask_len] *= 0.0
        # value = self.random_uniform(0.0, 1.0)
        # fbank[:,:,mask_start:mask_start+mask_len] += value
        return fbank

    def frequency_fading(self, fbank, freqm):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(freqm // 8, freqm))
        if(mask_len % 2 == 1): mask_len += 1
        mask_start = int(self.random_uniform(start=0, end=freq-mask_len-1))
        
        weight = torch.cat([torch.linspace(1,0,mask_len//2), torch.linspace(0,1,mask_len//2)])
        weight = weight[None, : ,None].expand(fbank.size(0), mask_len, fbank.size(2))

        fbank[:,mask_start:mask_start+mask_len,:] *= weight

        return fbank

    def time_fading(self, fbank, timem):
        bs, freq, tsteps = fbank.size()
        mask_len = int(self.random_uniform(timem // 8, timem))
        if(mask_len % 2 == 1): mask_len += 1
        mask_start = int(self.random_uniform(start=0, end=tsteps-mask_len-1))
        
        weight = torch.cat([torch.linspace(1,0,mask_len//2), torch.linspace(0,1,mask_len//2)])
        weight = weight[None, None, : ].expand(fbank.size(0), fbank.size(1), mask_len)

        fbank[:,:,mask_start:mask_start+mask_len] *= weight
        
        return fbank


    def __len__(self):
        return len(self.data)