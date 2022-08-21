from turtle import forward
from unicodedata import bidirectional
import torch.nn as nn
import torch
import torchlibrosa as tl

if __name__ == '__main__':
    from HigherModels import *
    from neural_sampler import *
else:
    from .HigherModels import *
    from .neural_sampler import *

from efficientnet_pytorch import EfficientNet
import torchvision

class ResNetAttention(nn.Module):
    def __init__(self, label_dim=527, pretrain=True):
        super(ResNetAttention, self).__init__()

        self.model = torchvision.models.resnet50(pretrained=False)

        if pretrain == False:
            print('ResNet50 Model Trained from Scratch (ImageNet Pretraining NOT Used).')
        else:
            print('Now Use ImageNet Pretrained ResNet50 Model.')

        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # remove the original ImageNet classification layers to save space.
        self.model.fc = torch.nn.Identity()
        self.model.avgpool = torch.nn.Identity()

        # attention pooling module
        self.attention = Attention(
            2048,
            label_dim,
            att_activation='sigmoid',
            cla_activation='sigmoid')
        self.avgpool = nn.AvgPool2d((4, 1))

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        batch_size = x.shape[0]
        x = self.model(x)
        x = x.reshape([batch_size, 2048, 4, 33])
        x = self.avgpool(x)
        x = x.transpose(2,3)
        out, norm_att = self.attention(x)
        return out

class MBNet(nn.Module):
    def __init__(self, label_dim=527, pretrain=True):
        super(MBNet, self).__init__()

        self.model = torchvision.models.mobilenet_v2(pretrained=pretrain)

        self.model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = torch.nn.Linear(in_features=1280, out_features=label_dim, bias=True)

    def forward(self, x, nframes=1056):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        out = torch.sigmoid(self.model(x))
        return out

class EffNetAttention(nn.Module):
    def __init__(self, label_dim=527, b=0, pretrain=True, head_num=4, input_seq_length=3000, sampler=None, preserve_ratio=0.1, alpha=1.0, learn_pos_emb=False, hop_size=160, mean=None, std=None):
        super(EffNetAttention, self).__init__()
        
        sample_rate = 16000
        win_length = 512
        hop_length = hop_size
        n_mels = 128
        
        self.mean = mean
        self.std = std
        
        from torchlibrosa.augmentation import SpecAugmentation
        self.specaug=SpecAugmentation(time_drop_width=192, time_stripes_num=1, freq_drop_width=48, freq_stripes_num=1)
        self.feature_extractor = torch.nn.Sequential(
                        tl.Spectrogram(
                            hop_length=hop_length,
                            win_length=win_length,
                        ), tl.LogmelFilterBank(
                            sr=sample_rate,
                            n_mels=n_mels,
                            is_log=False, # Default is true
                        ))
        
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        self.input_seq_length = input_seq_length
        print("Use %s with preserve ratio of %s" % (str(sampler), str(preserve_ratio)))
        self.learn_pos_emb = learn_pos_emb
        self.alpha = alpha

        self.neural_sampler = sampler(input_seq_length, preserve_ratio, self.alpha, self.learn_pos_emb)
        
        if pretrain == False:
            print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=self.neural_sampler.feature_channels)
        else:
            print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=self.neural_sampler.feature_channels)
        # multi-head attention pooling
        if head_num > 1:
            print('Model with {:d} attention heads'.format(head_num))
            self.attention = MHeadAttention(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        # single-head attention pooling
        elif head_num == 1:
            print('Model with single attention heads')
            self.attention = Attention(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        # mean pooling (no attention)
        elif head_num == 0:
            print('Model with mean pooling (NO Attention Heads)')
            self.attention = MeanPooling(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        else:
            raise ValueError('Attention head must be integer >= 0, 0=mean pooling, 1=single-head attention, >1=multi-head attention.')
        
        self.avgpool = nn.AvgPool2d((4, 1))
        self.effnet._fc = nn.Identity()
        self.batch_idx=0
        self.rank = None
        self.data = None

    def feature_extraction(self, x):
        mel = self.feature_extractor(x.squeeze(1))
        mel = self.specaug(mel)
        mel = mel.squeeze(1)
        return (torch.log(mel + 1e-7) - self.mean) / self.std

    def forward(self, x, nframes=1056):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        logmel = self.feature_extraction(x)
        
        # if(self.data is None):
        #     self.data = logmel.flatten()
        # else:
        #     self.data = torch.cat([self.data, logmel.flatten()])
        # print(torch.mean(self.data), torch.std(self.data))
        
        ret = self.neural_sampler(logmel)
        x, score, energy = ret['feature'], ret['score'], ret['energy']
        
        if(self.rank == 0 and self.batch_idx % 2000 == 0 and self.training):
            self.neural_sampler.visualize(ret)

        x = x.transpose(2, 3)
        x = self.effnet.extract_features(x) # torch.Size([10, 1280, 4, 4])
        x = self.avgpool(x) # torch.Size([10, 1280, 1, 4])
        x = x.transpose(2,3)
        out, norm_att = self.attention(x)
        if(self.training): self.batch_idx += 1
        return out, score, energy, logmel
    
def test_model():
    input_tdim = 3000
    from thop import clever_format
    from thop import profile

    # model = MBNet(pretrain=False)
    model = EffNetAttention(pretrain=False, b=0, head_num=0, sampler=NeuralSamplerLargeEnergy, preserve_ratio=0.25) # 2.688G 717.103K
    
    test_input = torch.rand([10, input_tdim, 128])
    flops, params = profile(model, inputs=(test_input, ))
    flops, params = clever_format([flops, params], "%.3f")

    print(flops, params)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    test_model()
    