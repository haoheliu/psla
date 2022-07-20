from turtle import forward
from unicodedata import bidirectional
import torch.nn as nn
import torch
from .HigherModels import *
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

    def forward(self, x, nframes):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        out = torch.sigmoid(self.model(x))
        return out


class EffNetAttention(nn.Module):
    def __init__(self, label_dim=527, b=0, pretrain=True, head_num=4):
        super(EffNetAttention, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        if pretrain == False:
            print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
        else:
            print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)
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
        #remove the original ImageNet classification layers to save space.
        self.effnet._fc = nn.Identity()
        print("Use Neural Sampler with drop ratio of 0.1")
        self.neural_sampler = NeuralSampler(input_dim=128, latent_dim=128, num_layers=2, drop_radio=0.1)
        # self.pooling = nn.AvgPool2d((10, 1), stride=(10,1))

    def forward(self, x, nframes=1056):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x, score_loss = self.neural_sampler(x)

        # x = self.pooling(x); score_loss=torch.tensor([0.0]).cuda()

        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        
        x = self.effnet.extract_features(x) # torch.Size([10, 1280, 4, 4])
        x = self.avgpool(x) # torch.Size([10, 1280, 1, 4])
        x = x.transpose(2,3)
        out, norm_att = self.attention(x)
        return out, score_loss

class NeuralSampler(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers=2, drop_radio=0.75):
        super(NeuralSampler, self).__init__()
        # We can also predict the weight by NN
        self.feature_lstm = nn.LSTM(input_dim, latent_dim, num_layers, batch_first=True, bidirectional=False)
        self.score_lstm = nn.LSTM(latent_dim, latent_dim, num_layers, batch_first=True, bidirectional=False)
        self.score_linear = nn.Linear(latent_dim, 1)
        self.drop_radio = drop_radio
    
    def forward(self, x):
        feature, (hn, cn) = self.feature_lstm(x)
        score, (hn, cn) = self.score_lstm(feature, (hn, cn))
        score = torch.sigmoid(self.score_linear(score))

        # Range norm
        # max_score, min_score = torch.max(score, dim=1, keepdim=True)[0], torch.min(score, dim=1, keepdim=True)[0]
        # score = (score - min_score)/(max_score-min_score)

        feature, score_loss = self.select_feature_fast(feature, score, total_length=int(x.size(1)*self.drop_radio))
        return feature, score_loss

    def select_feature_fast(self, feature, score, total_length):
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        max_val = torch.max(score, dim=1)[0]
        max_val = max_val[..., 0]
        dims_need_norm = max_val >= 1
        if(torch.sum(dims_need_norm) > 0):
            score[dims_need_norm] = score[dims_need_norm] / max_val[dims_need_norm][..., None, None]
        # print(score)
        # print(torch.mean(torch.std(score, dim=1)))
        # feature.size(): torch.Size([10, 100, 128])
        # weight: torch.Size([10, 75, 100])
        # score: torch.Size([10, 100, 1])
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        weight = score.expand(feature.size(0), feature.size(1), total_length)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        cumsum_weight = cumsum_weight * mask
        weight = weight * mask
        weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        # weight = weight.permute(0, 2, 1)
        return tensor_list, torch.mean(torch.std(score, dim=1))

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

    def update_element_weight(self, weight):
        # weight: [T, newT]
        i,j=0,0
        _sum = 0
        weight = weight.T
        height, width = weight.size(0), weight.size(1)
        while(i < height - 1 and j < width):
            if(weight[i,j] == 0.0):
                weight[i, j] = 1 - _sum
                weight[i+1, j] -= weight[i, j]
                _sum=0
                i += 1
            else:
                _sum += weight[i,j]
                j += 1
        return weight.T

    def select_feature(self, feature, score, total_length):
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        max_val = torch.max(score, dim=1)[0]
        max_val = max_val[..., 0]
        dims_need_norm = max_val >= 1
        if(torch.sum(dims_need_norm) > 0 ):
            score[dims_need_norm] = score[dims_need_norm] / max_val[dims_need_norm]
        # print(score)
        # print(torch.std(score, dim=1))
        # feature.size(): torch.Size([10, 100, 128])
        # weight: torch.Size([10, 75, 100])
        tensor_list = []
        for i in range(score.size(0)):
            sum=0    
            current=0
            total_score = 0
            tensor = feature[i].clone()
            tensor *= 0.0

            while current < score.size(1):
                val_score = score[i, current, 0]
                if(val_score + total_score >= 1):
                    # Add a tensor element
                    score[i, current, 0] = score[i, current, 0]-(1-total_score)
                    val_score = (1-total_score)
                    total_score = 0

                    sum += val_score
                    tensor[int(sum)] += val_score * feature[i][current]
                else:
                    sum += val_score
                    tensor[int(sum)] += val_score * feature[i][current]

                    current += 1
            tensor_list.append(tensor.unsqueeze(0))
        tensor_list = torch.cat(tensor_list, dim=0)
        return tensor_list[:,:total_length,:] # TODO need scrutinize

def test_model():
    input_tdim = 100
    #ast_mdl = ResNetNewFullAttention(pretrain=False)
    psla_mdl = EffNetAttention(pretrain=False, b=0, head_num=0)
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = psla_mdl(test_input)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(test_output.shape)
    import ipdb; ipdb.set_trace()


def test_sampler():
    input_tdim = 100
    sampler = NeuralSampler(input_dim=128, latent_dim=128)
    test_input = torch.rand([10, input_tdim, 128])
    output =sampler(test_input)
    import ipdb; ipdb.set_trace()

def test_select_feature():
    # score.shape: torch.Size([10, 100, 1])
    # feature.shape: torch.Size([10, 100, 256])
    input_tdim = 100
    sampler = NeuralSampler(input_dim=128, latent_dim=128)
    feature = torch.tensor([[[ 0.1, 0.2, 0.3],
         [0.4, 0.5, 0.6],
         [-0.1, -0.2, -0.3],
         [-0.4, -0.5, -0.6],
         [ 1.1, 1.2, 1.3],
         [-1.1, -1.2, -1.3]]])
    score = torch.tensor([[[ 0.2],
         [0.7],
         [ 0.4],
         [ 0.3],
         [ 0.9],
         [0.5]]])
    res = sampler.select_feature_fast(feature, score, total_length=3)
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    test_sampler()