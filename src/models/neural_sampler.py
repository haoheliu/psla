from turtle import forward
from unicodedata import bidirectional
import torch.nn as nn
import torch
import math
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer

import logging
import os

# from HigherModels import *
# from neural_sampler import *
# from pooling import Pooling_layer
# from transformerencoder.model import TextEncoder

from .HigherModels import *
from .neural_sampler import *
from .pooling import Pooling_layer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).permute(1,0,2)

# 07/23/2022 12:48:27 PM - INFO: mAP: 0.206710
# 07/23/2022 12:48:27 PM - INFO: AUC: 0.849850
# 07/23/2022 12:48:27 PM - INFO: Avg Precision: 0.010241
# 07/23/2022 12:48:27 PM - INFO: Avg Recall: 0.857787
# 07/23/2022 12:48:27 PM - INFO: d_prime: 1.464827
# 07/23/2022 12:48:27 PM - INFO: train_loss: 0.014192
# 07/23/2022 12:48:27 PM - INFO: valid_loss: 0.023854
class NeuralSamplerPosEmbLearnableNoFakeSoftmax(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSamplerPosEmbLearnableNoFakeSoftmax, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True
        self.feature_lstm = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Linear(self.latent_dim*2, 1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        score, (hn, cn) = self.feature_lstm(x)
        score = torch.sigmoid(self.score_linear(score))
        ret = self.select_feature_fast(x, score, total_length=int(x.size(1)*self.preserv_ratio))
        ret['score']=score
        ret['x']=x
        return ret

    def visualize(self, ret):
        x, y, emb, score = ret['x'], ret['feature'], ret['emb'], ret['score']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(411)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(412)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)

        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        return ret

# TODO concatenate feature on the channel dimension
class NeuralSamplerPosEmbLearnableLargeWithMAFeat(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSamplerPosEmbLearnableLargeWithMAFeat, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=256
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True

        # data = torch.randn((3, 100, 128))
        # a,b = model(data, data, data)
        self.ma = torch.nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=4, batch_first=True)
        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim*2),
            nn.ReLU(inplace=True),
        )
        self.pre_linear_2 = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.score_lstm = nn.LSTM(self.input_dim*2, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.feature_lstm = nn.LSTM(self.input_dim, self.input_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.feature_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.score_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, 1),
        )

        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        pre_x = self.pre_linear(x)
        score, (hn, cn) = self.score_lstm(pre_x)
        score = torch.sigmoid(self.score_linear(score))
        
        pre_x_2 = self.pre_linear_2(x)
        pre_x_2 = pre_x_2 + self.pos_emb
        ma_x,_ = self.ma(pre_x_2, pre_x_2, pre_x_2)
        ma_x = ma_x[:,:int(self.input_seq_length*self.preserv_ratio),:]
        feature_x, (hn, cn) = self.feature_lstm(ma_x) 
        feat = self.feature_linear(feature_x)
        feat = feat + ma_x

        ret = self.select_feature_fast(x, score, feat, total_length=int(x.size(1)*self.preserv_ratio))
        ret['x']=x
        return ret

    def visualize(self, ret):
        try:
            x, y, emb, score = ret['x'], ret['feature'], ret['emb'], ret['score']
            y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
            import matplotlib.pyplot as plt
            for i in range(10):
                plt.figure(figsize=(6, 8))
                plt.subplot(511)
                plt.plot(score[i,:,0].detach().cpu().numpy())
                plt.subplot(512)
                plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
                plt.subplot(513)
                plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
                plt.subplot(514)
                plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
                plt.subplot(515)
                plt.imshow(ret['feat'][i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
                path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
                plt.savefig(os.path.join(path, "%s.png" % i))
                plt.close()
        except Exception as e:
            logging.exception("message")
            print(e)

    def select_feature_fast(self, feature, score, feat, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # cumsum_weight = cumsum_weight - (score/2)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        ret['emb'] = pos_emb
        ret['feat'] = feat
        tensor_list = tensor_list + feat
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

# High both ends
class NeuralSamplerPosEmbLearnableLargeWithMAFeatCh(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio,alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeWithMAFeatCh, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=256
        self.num_layers=2
        self.feature_channels=3
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True

        # data = torch.randn((3, 100, 128))
        # a,b = model(data, data, data)
        self.ma = torch.nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=4, batch_first=True)
        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim*2),
            nn.ReLU(inplace=True),
        )
        self.pre_linear_2 = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.score_lstm = nn.LSTM(self.input_dim*2, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.feature_lstm = nn.LSTM(self.input_dim, self.input_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.feature_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.score_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, 1),
        )

        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        pre_x = self.pre_linear(x)
        score, (hn, cn) = self.score_lstm(pre_x)
        score = torch.sigmoid(self.score_linear(score))
        
        pre_x_2 = self.pre_linear_2(x)
        pre_x_2 = pre_x_2 + self.pos_emb
        ma_x,_ = self.ma(pre_x_2, pre_x_2, pre_x_2)
        ma_x = ma_x[:,:int(self.input_seq_length*self.preserv_ratio),:]
        feature_x, (hn, cn) = self.feature_lstm(ma_x) 
        feat = self.feature_linear(feature_x)
        feat = feat + ma_x

        ret = self.select_feature_fast(x, score, feat, total_length=int(x.size(1)*self.preserv_ratio))
        ret['x']=x
        return ret

    def visualize(self, ret):
        try:
            x, y, emb, score = ret['x'], ret['feature'], ret['emb'], ret['score']
            y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
            import matplotlib.pyplot as plt
            for i in range(10):
                plt.figure(figsize=(6, 8))
                plt.subplot(511)
                plt.plot(score[i,:,0].detach().cpu().numpy())
                plt.subplot(512)
                plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
                plt.subplot(513)
                plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
                plt.subplot(514)
                plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
                plt.subplot(515)
                plt.imshow(ret['feat'][i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
                path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
                plt.savefig(os.path.join(path, "%s.png" % i))
                plt.close()
        except Exception as e:
            logging.exception("message")
            print(e)

    def select_feature_fast(self, feature, score, feat, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # cumsum_weight = cumsum_weight - (score/2)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        ret['emb'] = pos_emb
        ret['feat'] = feat
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1), feat.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class NeuralSampler_NNMA(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSampler_NNMA, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=256
        self.num_layers=2
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True

        # data = torch.randn((3, 100, 128))
        # a,b = model(data, data, data)
        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.ma = torch.nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=4, batch_first=True)

        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        ret = {}
        pre_x = self.pre_linear(x)
        pre_x = pre_x + self.pos_emb
        ma_x,_ = self.ma(pre_x, pre_x, pre_x)
        ret['feature']=ma_x[:,:int(self.preserv_ratio * self.input_seq_length),:].unsqueeze(1)
        ret['x']=x
        ret['score_loss']=torch.tensor([0.0]).cuda()
        return ret

    def visualize(self, ret):
        x, y = ret['x'], ret['feature']
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(211)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

class NeuralSamplerPosEmbLearnableLargeWithMA(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeWithMA, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=256
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True

        # data = torch.randn((3, 100, 128))
        # a,b = model(data, data, data)
        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.ma = torch.nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=4, batch_first=True)

        self.score_lstm = nn.LSTM(self.input_dim, self.input_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.score_lstm_linear = nn.Sequential(
            nn.Linear(self.input_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, self.input_dim),
        )
        self.score_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, 1),
        )

        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        pre_x = self.pre_linear(x)
        pre_x = pre_x + self.pos_emb
        ma_x,_ = self.ma(pre_x, pre_x, pre_x)

        score, (hn, cn) = self.score_lstm(ma_x)
        score = self.score_lstm_linear(score)

        score = score + ma_x
        score = torch.sigmoid(self.score_linear(score))

        ret = self.select_feature_fast(x, score, total_length=int(x.size(1)*self.preserv_ratio))
        ret['x']=x
        return ret

    def visualize(self, ret):
        try:
            x, y, emb, score = ret['x'], ret['feature'], ret['emb'], ret['score']
            y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
            import matplotlib.pyplot as plt
            for i in range(10):
                plt.figure(figsize=(6, 8))
                plt.subplot(411)
                plt.plot(score[i,:,0].detach().cpu().numpy())
                plt.subplot(412)
                plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
                plt.subplot(413)
                plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
                plt.subplot(414)
                plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
                path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
                plt.savefig(os.path.join(path, "%s.png" % i))
                plt.close()
        except Exception as e:
            logging.exception("message")
            print(e)

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # cumsum_weight = cumsum_weight - (score/2)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)


class NeuralSamplerPosEmbLearnableLargeEnergy(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeEnergy, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.alpha = alpha
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True
        print("ALPHA %s" % self.alpha)
        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim*2),
            nn.ReLU(inplace=True),
        )
        self.feature_lstm = nn.LSTM(self.input_dim*2, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, 1),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp()**self.alpha, dim=2, keepdim=True)
        score = magnitude/torch.max(magnitude)
        ret = self.select_feature_fast(x, score, total_length=int(x.size(1)*self.preserv_ratio))
        ret['x']=x
        return ret

    def visualize(self, ret, name=None):
        x, y, emb, score = ret['x'], ret['feature'], ret['emb'], ret['score']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(411)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(412)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            if(name is None):
                plt.savefig(os.path.join(path, "%s.png" % i))
            else:
                plt.savefig(os.path.join(path, "%s_%s.png" % (name, i)))
            plt.close()

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        ####################################################################
        # Trying to rescale the total score 
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        ####################################################################

        ####################################################################
        # If the weight for one frame is greater than one, rescale the batch
        max_val = torch.max(score, dim=1)[0]
        max_val = max_val[..., 0]
        dims_need_norm = max_val >= 1
        if(torch.sum(dims_need_norm) > 0):
            score[dims_need_norm] = score[dims_need_norm] / max_val[dims_need_norm][..., None, None]
        ####################################################################

        ####################################################################
        # Remove the zero pad at the end, using the rescaling of the weight in between 
        # torch.Size([32, 1056, 1])
        if(torch.sum(dims_need_norm) > 0):
            sum_score = torch.sum(score, dim=(1,2), keepdim=True)
            distance_with_target_length = (total_length-sum_score)[:,0,0]
            axis = torch.logical_and(score < 0.99**self.alpha, score > 0.01**self.alpha) # TODO here 0.1 or 0.01
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################

        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # cumsum_weight = cumsum_weight - (score/2)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        # import ipdb; ipdb.set_trace()
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class NeuralSamplerPosEmbLearnableLargeEnergyMaxPooling(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeEnergyMaxPooling, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.alpha = alpha
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True
        print("ALPHA %s" % self.alpha)
        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim*2),
            nn.ReLU(inplace=True),
        )
        self.feature_lstm = nn.LSTM(self.input_dim*2, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, 1),
        )
        self.score_pooling = nn.MaxPool1d(11, stride=1, padding=5)
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp()**self.alpha, dim=2, keepdim=True)
        score = magnitude/torch.max(magnitude)
        ret = self.select_feature_fast(x, score, total_length=int(x.size(1)*self.preserv_ratio))
        ret['x']=x
        return ret

    def visualize(self, ret, name=None):
        x, y, emb, score = ret['x'], ret['feature'], ret['emb'], ret['score']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(411)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(412)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            if(name is None):
                plt.savefig(os.path.join(path, "%s.png" % i))
            else:
                plt.savefig(os.path.join(path, "%s_%s.png" % (name, i)))
            plt.close()

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        
        score = self.score_pooling(score.permute(0,2,1)).permute(0,2,1)

        ####################################################################
        # Trying to rescale the total score 
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        ####################################################################

        ####################################################################
        # If the weight for one frame is greater than one, rescale the batch
        max_val = torch.max(score, dim=1)[0]
        max_val = max_val[..., 0]
        dims_need_norm = max_val >= 1
        if(torch.sum(dims_need_norm) > 0):
            score[dims_need_norm] = score[dims_need_norm] / max_val[dims_need_norm][..., None, None]
        ####################################################################

        ####################################################################
        # Remove the zero pad at the end, using the rescaling of the weight in between 
        # torch.Size([32, 1056, 1])
        if(torch.sum(dims_need_norm) > 0):
            sum_score = torch.sum(score, dim=(1,2), keepdim=True)
            distance_with_target_length = (total_length-sum_score)[:,0,0]
            axis = torch.logical_and(score < 0.99**self.alpha, score > 0.01**self.alpha) # TODO here 0.1 or 0.01
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################

        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # cumsum_weight = cumsum_weight - (score/2)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        # import ipdb; ipdb.set_trace()
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class NeuralSamplerPosEmbLearnableLargeEnergyRandAlpha(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeEnergyRandAlpha, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.alpha = alpha
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True
        print("ALPHA %s" % self.alpha)
        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim*2),
            nn.ReLU(inplace=True),
        )
        self.feature_lstm = nn.LSTM(self.input_dim*2, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, 1),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        # torch.Size([96, 1056, 128])
        if(self.training):
            self.alpha = torch.rand(1).item()*1.5+0.5
        else:
            self.alpha=1.0

        magnitude = torch.sum(x.exp()**self.alpha, dim=2, keepdim=True)
        score = magnitude/torch.max(magnitude)
        ret = self.select_feature_fast(x, score, total_length=int(x.size(1)*self.preserv_ratio))
        ret['x']=x
        return ret

    def visualize(self, ret, name=None):
        x, y, emb, score = ret['x'], ret['feature'], ret['emb'], ret['score']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(411)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(412)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            if(name is None):
                plt.savefig(os.path.join(path, "%s.png" % i))
            else:
                plt.savefig(os.path.join(path, "%s_%s.png" % (name, i)))
            plt.close()

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        ####################################################################
        # Trying to rescale the total score 
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        ####################################################################

        ####################################################################
        # If the weight for one frame is greater than one, rescale the batch
        max_val = torch.max(score, dim=1)[0]
        max_val = max_val[..., 0]
        dims_need_norm = max_val >= 1
        if(torch.sum(dims_need_norm) > 0):
            score[dims_need_norm] = score[dims_need_norm] / max_val[dims_need_norm][..., None, None]
        ####################################################################

        ####################################################################
        # Remove the zero pad at the end, using the rescaling of the weight in between 
        # torch.Size([32, 1056, 1])
        if(torch.sum(dims_need_norm) > 0):
            sum_score = torch.sum(score, dim=(1,2), keepdim=True)
            distance_with_target_length = (total_length-sum_score)[:,0,0]
            axis = torch.logical_and(score < 0.99**self.alpha, score > 0.01**self.alpha) # TODO here 0.1 or 0.01
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################

        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # cumsum_weight = cumsum_weight - (score/2)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        # import ipdb; ipdb.set_trace()
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class NeuralSamplerPosEmbLearnableLargeEnergyNN(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeEnergyNN, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim*2),
            nn.ReLU(inplace=True),
        )
        self.feature_lstm = nn.LSTM(self.input_dim*2, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, 1),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        pre_x = self.pre_linear(x)
        score, (hn, cn) = self.feature_lstm(pre_x)
        score = torch.sigmoid(self.score_linear(score))
        ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        ret['x']=x
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        return ret

    def visualize(self, ret):
        x, y, emb, score, energy = ret['x'], ret['feature'], ret['emb'], ret['score'], ret['energy']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(511)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(512)
            plt.plot(energy[i,:,0].detach().cpu().numpy())
            plt.subplot(513)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(514)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(515)
            plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

    def score_norm(self, score, total_length):
        ####################################################################
        # Trying to rescale the total score 
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        ####################################################################

        ####################################################################
        # If the weight for one frame is greater than one, rescale the batch
        max_val = torch.max(score, dim=1)[0]
        max_val = max_val[..., 0]
        dims_need_norm = max_val >= 1
        if(torch.sum(dims_need_norm) > 0):
            score[dims_need_norm] = score[dims_need_norm] / max_val[dims_need_norm][..., None, None]
        ####################################################################

        ####################################################################
        # Remove the zero pad at the end, using the rescaling of the weight in between 
        # torch.Size([32, 1056, 1])
        if(torch.sum(dims_need_norm) > 0):
            sum_score = torch.sum(score, dim=(1,2), keepdim=True)
            distance_with_target_length = (total_length-sum_score)[:,0,0]
            axis = torch.logical_and(score < 0.99, score > 0.01) # TODO here 0.1 or 0.01
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        score, total_length = self.score_norm(score, total_length)

        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # cumsum_weight = cumsum_weight - (score/2)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)

        sum_score = torch.sum(score, dim=(1,2), keepdim=True)

        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class NeuralSamplerPosEmbLearnableLargeEnergyNNPosFreeze(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeEnergyNNPosFreeze, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim*2),
            nn.ReLU(inplace=True),
        )
        self.feature_lstm = nn.LSTM(self.input_dim*2, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, 1),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        pre_x = self.pre_linear(x)
        score, (hn, cn) = self.feature_lstm(pre_x)
        score = torch.sigmoid(self.score_linear(score))
        ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        ret['x']=x
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        return ret

    def visualize(self, ret):
        x, y, emb, score, energy = ret['x'], ret['feature'], ret['emb'], ret['score'], ret['energy']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(511)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(512)
            plt.plot(energy[i,:,0].detach().cpu().numpy())
            plt.subplot(513)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(514)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(515)
            plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

    def score_norm(self, score, total_length):
        ####################################################################
        # Trying to rescale the total score 
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        ####################################################################

        ####################################################################
        # If the weight for one frame is greater than one, rescale the batch
        max_val = torch.max(score, dim=1)[0]
        max_val = max_val[..., 0]
        dims_need_norm = max_val >= 1
        if(torch.sum(dims_need_norm) > 0):
            score[dims_need_norm] = score[dims_need_norm] / max_val[dims_need_norm][..., None, None]
        ####################################################################

        ####################################################################
        # Remove the zero pad at the end, using the rescaling of the weight in between 
        # torch.Size([32, 1056, 1])
        if(torch.sum(dims_need_norm) > 0):
            sum_score = torch.sum(score, dim=(1,2), keepdim=True)
            distance_with_target_length = (total_length-sum_score)[:,0,0]
            axis = torch.logical_and(score < 0.99, score > 0.01) # TODO here 0.1 or 0.01
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        score, total_length = self.score_norm(score, total_length)

        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # cumsum_weight = cumsum_weight - (score/2)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)

        sum_score = torch.sum(score, dim=(1,2), keepdim=True)

        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)


class NeuralSamplerPosEmbLearnableLargeEnergyNNMaxPooling(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeEnergyNNMaxPooling, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim*2),
            nn.ReLU(inplace=True),
        )
        self.feature_lstm = nn.LSTM(self.input_dim*2, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, 1),
        )

        self.score_pooling = nn.MaxPool1d(11, stride=1, padding=5)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        pre_x = self.pre_linear(x)
        score, (hn, cn) = self.feature_lstm(pre_x)
        score = torch.sigmoid(self.score_linear(score))
        ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        ret['x']=x
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        return ret

    def visualize(self, ret):
        x, y, emb, score, energy = ret['x'], ret['feature'], ret['emb'], ret['score'], ret['energy']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(511)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(512)
            plt.plot(energy[i,:,0].detach().cpu().numpy())
            plt.subplot(513)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(514)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(515)
            plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

    def score_norm(self, score, total_length):
        score = self.score_pooling(score.permute(0,2,1)).permute(0,2,1)

        ####################################################################
        # Trying to rescale the total score 
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        ####################################################################

        ####################################################################
        # If the weight for one frame is greater than one, rescale the batch
        max_val = torch.max(score, dim=1)[0]
        max_val = max_val[..., 0]
        dims_need_norm = max_val >= 1
        if(torch.sum(dims_need_norm) > 0):
            score[dims_need_norm] = score[dims_need_norm] / max_val[dims_need_norm][..., None, None]
        ####################################################################

        ####################################################################
        # Remove the zero pad at the end, using the rescaling of the weight in between 
        # torch.Size([32, 1056, 1])
        if(torch.sum(dims_need_norm) > 0):
            sum_score = torch.sum(score, dim=(1,2), keepdim=True)
            distance_with_target_length = (total_length-sum_score)[:,0,0]
            axis = torch.logical_and(score < 0.99, score > 0.01) # TODO here 0.1 or 0.01
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        
        return score, total_length

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        score, total_length = self.score_norm(score, total_length)

        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # cumsum_weight = cumsum_weight - (score/2)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)

        sum_score = torch.sum(score, dim=(1,2), keepdim=True)

        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

# 07/23/2022 08:41:02 AM - INFO: mAP: 0.220362
# 07/23/2022 08:41:02 AM - INFO: AUC: 0.918260
# 07/23/2022 08:41:02 AM - INFO: Avg Precision: 0.014869
# 07/23/2022 08:41:02 AM - INFO: Avg Recall: 0.926952
# 07/23/2022 08:41:02 AM - INFO: d_prime: 1.970651
# 07/23/2022 08:41:02 AM - INFO: train_loss: 0.013516
# 07/23/2022 08:41:02 AM - INFO: valid_loss: 0.018110
class NeuralSamplerPosEmbLearnable(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnable, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True
        self.feature_lstm = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Linear(self.latent_dim*2, 1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        score, (hn, cn) = self.feature_lstm(x)
        score = torch.sigmoid(self.score_linear(score))
        ret = self.select_feature_fast(x, score, total_length=int(x.size(1)*self.preserv_ratio))
        ret['x']=x
        return ret

    def visualize(self, ret):
        x, y, emb, score = ret['x'], ret['feature'], ret['emb'], ret['score']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(411)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(412)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # cumsum_weight = cumsum_weight - (score/2)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)

        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

# Feat do not learn anything
class NeuralSamplerPosEmbLearnableLargeWithFeatv3(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeWithFeatv3, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=256
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True

        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim*2),
            nn.ReLU(inplace=True),
        )
        self.feature_lstm = nn.LSTM(self.input_dim*2, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.feature_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.score_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, 1),
        )

        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        pre_x = self.pre_linear(x)
        score, (hn, cn) = self.feature_lstm(pre_x)
        feat = self.feature_linear(score)
        score = torch.sigmoid(self.score_linear(score))
        ret = self.select_feature_fast(x, score, feat, total_length=int(x.size(1)*self.preserv_ratio))
        ret['x']=x
        return ret

    def visualize(self, ret):
        x, y, emb, score = ret['x'], ret['feature'], ret['emb'], ret['score']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(511)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(512)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(513)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(514)
            plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(515)
            plt.imshow(ret['feat'][i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

    def select_feature_fast(self, feature, score, feat, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # cumsum_weight = cumsum_weight - (score/2)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)

        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        feat = torch.matmul(weight.permute(0,2,1), feat)
        tensor_list = tensor_list + feat
        ret['emb'] = pos_emb
        ret['feat'] = feat
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

# 07/23/2022 06:42:49 AM - INFO: mAP: 0.221230
# 07/23/2022 06:42:49 AM - INFO: AUC: 0.916620
# 07/23/2022 06:42:49 AM - INFO: Avg Precision: 0.014316
# 07/23/2022 06:42:49 AM - INFO: Avg Recall: 0.926914
# 07/23/2022 06:42:49 AM - INFO: d_prime: 1.955419
# 07/23/2022 06:42:49 AM - INFO: train_loss: 0.013721
# 07/23/2022 06:42:49 AM - INFO: valid_loss: 0.017955
class NeuralSamplerNoFakeSoftmax(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerNoFakeSoftmax, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.num_layers=2
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = False
        self.feature_lstm = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Linear(self.latent_dim*2, 1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
    def forward(self, x):
        score, (hn, cn) = self.feature_lstm(x)
        score = torch.sigmoid(self.score_linear(score))
        ret = self.select_feature_fast(x, score, total_length=int(x.size(1)*self.preserv_ratio))
        ret['score']=score
        ret['x']=x
        return ret

    def visualize(self, ret):
        x, y, score = ret['x'], ret['feature'], ret['score']
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(311)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(312)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(313)
            plt.imshow(y[i,0,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        sum_score = torch.sum(score, dim=(1,2), keepdim=True)
        # Normalize the sum of score to the total length
        score = (score / sum_score) * total_length
        # If the original total legnth is smaller, we need to normalize the value greater than 1.  
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # cumsum_weight = cumsum_weight - (score/2)

        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # cumsum_weight = cumsum_weight * mask
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask
        # weight = self.weight_fake_softmax(weight, mask)
        # for i in range(weight.size(0)):
        #     weight[i] = self.update_element_weight(weight[i])
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)

        ret['feature'] = tensor_list.unsqueeze(1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        return ret

class NeuralSamplerMaxPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerMaxPool, self).__init__()
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = False
        self.pooling = Pooling_layer(pooling_type="max", factor=preserve_ratio)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
    def forward(self, x):
        ret={}
        feature = self.pooling(x.unsqueeze(1))
        ret['score_loss']=torch.tensor([0.0]).cuda()
        ret['feature']=feature
        ret['x']=x
        return ret

    def visualize(self, ret):
        x, y = ret['x'], ret['feature']
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(211)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

# 07/23/2022 05:36:10 AM - INFO: mAP: 0.216001
# 07/23/2022 05:36:10 AM - INFO: AUC: 0.919990
# 07/23/2022 05:36:10 AM - INFO: Avg Precision: 0.013685
# 07/23/2022 05:36:10 AM - INFO: Avg Recall: 0.930226
# 07/23/2022 05:36:10 AM - INFO: d_prime: 1.986981
# 07/23/2022 05:36:10 AM - INFO: train_loss: 0.014289
# 07/23/2022 05:36:10 AM - INFO: valid_loss: 0.017917
class NeuralSamplerAvgMaxPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerAvgMaxPool, self).__init__()
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = False
        self.pooling = Pooling_layer(pooling_type="avg-max", factor=preserve_ratio)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
    def forward(self, x):
        ret={}
        feature = self.pooling(x.unsqueeze(1))
        ret['score_loss']=torch.tensor([0.0]).cuda()
        ret['feature']=feature
        ret['x']=x
        return ret

    def visualize(self, ret):
        x, y = ret['x'], ret['feature']
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(211)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

class NeuralSamplerSpecPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerSpecPool, self).__init__()
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = False
        self.pooling = Pooling_layer(pooling_type="spec", factor=preserve_ratio)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
    def forward(self, x):
        ret={}
        feature = self.pooling(x.unsqueeze(1))
        ret['score_loss']=torch.tensor([0.0]).cuda()
        ret['feature']=feature
        ret['x']=x
        return ret

    def visualize(self, ret):
        x, y = ret['x'], ret['feature']
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(211)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

class NeuralSamplerAvgPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerAvgPool, self).__init__()
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = False
        self.pooling = Pooling_layer(pooling_type="avg", factor=preserve_ratio)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
    def forward(self, x):
        ret={}
        feature = self.pooling(x.unsqueeze(1))
        ret['score_loss']=torch.tensor([0.0]).cuda()
        ret['feature']=feature
        ret['x']=x
        return ret

    def visualize(self, ret):
        x, y = ret['x'], ret['feature']
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(211)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

class NeuralSampler_NNLSTM(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSampler_NNLSTM, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.num_layers=2
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.preserv_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = False
        self.feature_lstm = nn.LSTM(self.input_dim, self.input_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Linear(self.input_dim*2, self.input_dim)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
    def forward(self, x):
        ret = {}
        score, (hn, cn) = self.feature_lstm(x)
        ret['score']=None
        ret['x']=x
        ret['feature']=score[:,:self.preserv_length, :].unsqueeze(1)
        ret['score_loss']=torch.tensor([0.0]).cuda()
        return ret

    def visualize(self, ret):
        x, y = ret['x'], ret['feature']
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(211)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

# 07/23/2022 04:02:14 AM - INFO: mAP: 0.307234
# 07/23/2022 04:02:14 AM - INFO: AUC: 0.951799
# 07/23/2022 04:02:14 AM - INFO: Avg Precision: 0.028310
# 07/23/2022 04:02:14 AM - INFO: Avg Recall: 0.945901
# 07/23/2022 04:02:14 AM - INFO: d_prime: 2.351210
# 07/23/2022 04:02:14 AM - INFO: train_loss: 0.013613
# 07/23/2022 04:02:14 AM - INFO: valid_loss: 0.015424
class NoAction(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha):
        super(NoAction, self).__init__()
        self.feature_channels=1

    def forward(self, x):
        ret = {}
        ret['feature']=x.unsqueeze(1)
        ret['score_loss']=torch.tensor([0.0]).cuda()
        return ret

    def visualize(self, ret):
        return


def test_sampler(sampler):
    input_tdim = 1000
    sampler = sampler(input_seq_length=input_tdim, preserve_ratio=0.1)
    test_input = torch.rand([10, input_tdim, 128])
    ret =sampler(test_input)
    sampler.visualize(ret)
    print("Perfect!", sampler, ret["feature"].size(), ret["score_loss"].size(), ret["score_loss"])

def test_select_feature():
    # score.shape: torch.Size([10, 100, 1])
    # feature.shape: torch.Size([10, 100, 256])
    input_tdim = 100
    sampler = NeuralSamplerTopK(6, 0.5)
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

def test_feature():
    import librosa
    import numpy as np
    PATH = "/media/Disk_HDD/haoheliu/datasets/AudioSet/eval_segments"
    for i, file in enumerate(os.listdir(PATH)):
        if(i > 20): break
        x,_ = librosa.load(os.path.join(PATH, file), sr=None)
        spec = torch.tensor(np.abs(librosa.feature.melspectrogram(x))).log()
        spec = spec[None,...].permute(0,2,1)
        spec = torch.nn.functional.pad(spec, (0, 0, 0,374), value=-15.7)
        sampler = NeuralSamplerPosEmbLearnableLargeEnergy(1000, 0.2, alpha=0.5)
        print(spec.size())
        ret = sampler(spec)
        sampler.visualize(ret, name=file.split('.')[0])

def test_feature_single():
    import librosa
    import numpy as np
    x,_ = librosa.load(os.path.join("/media/Disk_HDD/haoheliu/datasets/AudioSet/eval_segments", "YZxq2_xOLT8o.wav"), sr=None)
    spec = torch.tensor(np.abs(librosa.feature.melspectrogram(x)) + 1e-8).log()
    spec = spec[None,...].permute(0,2,1)
    spec = torch.nn.functional.pad(spec, (0, 0, 0,374), value=-15.7)
    sampler = NeuralSamplerPosEmbLearnableLargeEnergyv2(1000, 0.25, alpha=1.0)
    print(spec.size())
    ret = sampler(spec)
    sampler.visualize(ret, name="test")

# YZxq2_xOLT8o_0
if __name__ == "__main__":
    from HigherModels import *
    from neural_sampler import *
    from pooling import Pooling_layer
    from transformerencoder.model import TextEncoder
    import logging
    
    import numpy as np

    logging.basicConfig(
    filename="log.txt",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    # test_feature_single()

    # test_select_feature()
    test_sampler(NeuralSamplerPosEmbLearnableLargeEnergyNNMaxPooling)
    # test_sampler(NeuralSamplerPosEmbLearnableLargeEnergyv4)
    # test_sampler(NeuralSamplerPosEmbLearnableLargeEnergyv5)
    # test_sampler(NeuralSamplerPosEmbLearnableLarge)
    # test_sampler(NeuralSamplerPosEmbLearnableLargeWithFeat)
    # test_sampler(NeuralSamplerPosEmbLearnableLargeWithFeatv2)
    # test_sampler(NeuralSamplerTopK)
    # test_sampler(NeuralSamplerMaxPool)
    # test_sampler(NeuralSamplerSpecPool)
    # test_sampler(NeuralSamplerAvgMaxPool)
    # test_sampler(NeuralSamplerAvgPool)

    # test_sampler(NeuralSamplerNoFakeSoftmax)                                                # Better than NeuralSampler
    # test_sampler(NeuralSamplerPosEmbLearnable)                                              # Better than NeuralSampler

    # test_sampler(NoAction)
    