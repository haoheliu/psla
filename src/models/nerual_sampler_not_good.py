from turtle import forward
from unicodedata import bidirectional
import torch.nn as nn
import torch
import math
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer
from .pooling import Pooling_layer
import logging
import os
from .transformerencoder.model import TextEncoder

TRANSFORMER_ENCODER_LAYERS=3
TRANSFORMER_ENCODER_NHEAD=2

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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
    
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
    
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
    
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
    
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
    
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


# Use Maxpooling to smoothing the importance score
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
    
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

# Use random alpha value to perform data augmentations
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
    
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
# Use max pooling to smooth the importance score
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
    

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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
    
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
    
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
    
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
    
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

# Use ResUNet as feature extractor
class NeuralSamplerPosEmbLearnableLargeEnergyNNResUNet6_0_25(nn.Module):

class NeuralSamplerPosEmbLearnableLargeEnergyNNv3(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeEnergyNNv3, self).__init__()
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
            axis = torch.logical_and(score < 1, score > 1e-4) # TODO here 0.1 or 0.01
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
class NeuralSamplerPosEmbLearnableLargeEnergyNNv2(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeEnergyNNv2, self).__init__()
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
            axis = torch.logical_and(score < 1, score > torch.min(score, dim=1, keepdim=True)[0] + 1e-4) # TODO here 0.1 or 0.01
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

# Old alpha, fix 1e-4
class NeuralSamplerPosEmbLearnableLargeEnergyv4(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeEnergyv4, self).__init__()
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
            axis = torch.logical_and(score < 1, score > 1e-4) # TODO here 0.1 or 0.01
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

# Old alpha, adaptive min
class NeuralSamplerPosEmbLearnableLargeEnergyv5(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeEnergyv5, self).__init__()
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
            axis = torch.logical_and(score < 1, score > torch.min(score, dim=1, keepdim=True)[0] + 1e-4) # TODO here 0.1 or 0.01
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
# New alpha, fix 1e-4
class NeuralSamplerPosEmbLearnableLargeEnergyv3(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeEnergyv3, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.alpha = alpha - 1.0
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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        score = magnitude/torch.max(magnitude, dim=1, keepdim=True)[0] # TODO
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
        #####################################################################
        # Alpha
        assert torch.max(score) <= 1.0
        intervel = 1.0-score
        delta_intervel = intervel * self.alpha
        score -= delta_intervel
        score = (score - torch.min(score, dim=1, keepdim=True)[0]) 
        score = score/torch.max(score, dim=1, keepdim=True)[0] 
        #####################################################################
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
            axis = torch.logical_and(score < 1, score > 1e-4) # TODO here 0.1 or 0.01
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1): # Have empty frames
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

# New alpha, adaptive min
class NeuralSamplerPosEmbLearnableLargeEnergyv2(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(NeuralSamplerPosEmbLearnableLargeEnergyv2, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.alpha = alpha - 1.0
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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        score = magnitude/torch.max(magnitude, dim=1, keepdim=True)[0] # TODO
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
        #####################################################################
        # Alpha
        assert torch.max(score) <= 1.0
        intervel = 1.0-score
        delta_intervel = intervel * self.alpha
        score -= delta_intervel
        score = (score - torch.min(score, dim=1, keepdim=True)[0]) 
        score = score/torch.max(score, dim=1, keepdim=True)[0] 
        #####################################################################
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
            axis = torch.logical_and(score < 1, score > torch.min(score, dim=1, keepdim=True)[0] + 1e-4) # TODO here 0.1 or 0.01
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1): # Have empty frames
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

# Feat do not learn anything
class NeuralSamplerPosEmbLearnableLargeWithFeatv2(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSamplerPosEmbLearnableLargeWithFeatv2, self).__init__()
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


# No significant improvement using channel pooling feature
class NeuralSamplerPosEmbLearnableLargeLSTMChPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSamplerPosEmbLearnableLargeLSTMChPool, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=3
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
        self.ch_lstm = nn.LSTM(self.input_dim*2, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.ch_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, self.input_dim),
        )
        self.score_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, 1),
        )
        self.pooling = Pooling_layer(pooling_type="avg-max", factor=preserve_ratio)
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        pre_x = self.pre_linear(x)
        score, (hn, cn) = self.feature_lstm(pre_x)
        score = torch.sigmoid(self.score_linear(score))

        ch_feature, (hn, cn) = self.ch_lstm(pre_x)
        ch_feature = self.ch_linear(ch_feature)
        ch_feature = self.pooling(ch_feature.unsqueeze(1))

        ret = self.select_feature_fast(x, score, ch_feature, total_length=int(x.size(1)*self.preserv_ratio))
        ret['feat']=ch_feature
        ret['x']=x
        return ret

    def visualize(self, ret):
        x, y, emb, score, feat, weight = ret['x'], ret['feature'], ret['emb'], ret['score'], ret['feat'], ret['weight']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(8, 16))
            plt.subplot(611)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(612)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(613)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(614)
            plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(615)
            plt.imshow(feat[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(616)
            plt.imshow(weight[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
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
        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1), feat], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        ret['weight'] = weight
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)
        
# Feat do not learn anything
class NeuralSamplerPosEmbLearnableLargeWithFeat(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSamplerPosEmbLearnableLargeWithFeat, self).__init__()
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
        self.feature_lstm_2 = nn.LSTM(self.input_dim*2, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
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
        
        feature_x, (hn, cn) = self.feature_lstm_2(pre_x)
        feat = self.feature_linear(feature_x)

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
        ret['emb'] = pos_emb
        tensor_list = tensor_list + feat[:,:int(self.input_seq_length * self.preserv_ratio),:]
        ret['feat'] = feat[:,:int(self.input_seq_length * self.preserv_ratio),:]
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)


# 07/23/2022 02:07:54 PM - INFO: mAP: 0.094352
# 07/23/2022 02:07:54 PM - INFO: AUC: 0.860832
# 07/23/2022 02:07:54 PM - INFO: Avg Precision: 0.010910
# 07/23/2022 02:07:54 PM - INFO: Avg Recall: 0.886058
# 07/23/2022 02:07:54 PM - INFO: d_prime: 1.533099
# 07/23/2022 02:07:54 PM - INFO: train_loss: 0.020625
# 07/23/2022 02:07:54 PM - INFO: valid_loss: 0.021104
class NeuralSamplerTopK(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSamplerTopK, self).__init__()
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        # score.shape: torch.Size([10, 100, 1])
        # feature.shape: torch.Size([10, 100, 256])
        indices=torch.topk(score, k=total_length, dim=1).indices # torch.Size([10, 300, 1])
        indices = indices.expand(feature.size(0), total_length, self.input_dim)
        tensor_list = torch.gather(feature, 1, indices.long())
        pos_emb = torch.gather(self.pos_emb.expand(feature.size(0), self.input_seq_length, self.input_dim), 1, indices.long())
        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        return ret

############################Transformer based model###########################
class NeuralSampler_v2_TransLSTMRes(nn.Module):
    # Transformer encoder (with pos embedding) + Residual connection + LSTM
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSampler_v2_TransLSTMRes, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True
        self.encoder = TextEncoder(hidden_channels=128, filter_channels=128, n_heads=TRANSFORMER_ENCODER_NHEAD, n_layers=TRANSFORMER_ENCODER_LAYERS, kernel_size=3, p_dropout=0.1)
        self.feature_lstm = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Linear(self.latent_dim*2, 1)
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        y = self.encoder(x+self.pos_emb)
        y = y + x
        score, (hn, cn) = self.feature_lstm(y)
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
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
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class NeuralSampler_v2_NNTrans(nn.Module):
    # Transformer encoder (with pos embedding) -> select first few frames
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSampler_v2_NNTrans, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.num_layers=2
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True
        self.encoder = TextEncoder(hidden_channels=128, filter_channels=128, n_heads=TRANSFORMER_ENCODER_NHEAD, n_layers=TRANSFORMER_ENCODER_LAYERS, kernel_size=3, p_dropout=0.1)
        self.feature_lstm = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Linear(self.latent_dim*2, 1)
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        ret = {}
        y = self.encoder(x+self.pos_emb)
        ret['feature']=y[:,:int(self.input_seq_length * self.preserv_ratio),:].unsqueeze(1)
        ret['emb'] = self.pos_emb
        ret['x']=x
        ret['score_loss']=torch.tensor([0.0]).cuda()
        return ret

    def visualize(self, ret):
        x, y, emb = ret['x'], ret['feature'], ret['emb']
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(311)
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(312)
            plt.imshow(y[i,0,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(313)
            plt.imshow(emb[0,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class NeuralSampler_v2_TransLSTM(nn.Module):
    # Transformer encoder (with pos embedding) + LSTM
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSampler_v2_TransLSTM, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True
        self.encoder = TextEncoder(hidden_channels=128, filter_channels=128, n_heads=TRANSFORMER_ENCODER_NHEAD, n_layers=TRANSFORMER_ENCODER_LAYERS, kernel_size=3, p_dropout=0.1)
        self.feature_lstm = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Linear(self.latent_dim*2, 1)
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        y = self.encoder(x+self.pos_emb)
        score, (hn, cn) = self.feature_lstm(y)
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
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
        # ret['score_loss'] = torch.mean(torch.abs(score - self.preserv_ratio))
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class NeuralSampler_v2_TransNoPlusPosEmb(nn.Module):
    # Transformer encoder (without pos embedding)
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSampler_v2_TransNoPlusPosEmb, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True
        self.encoder = TextEncoder(hidden_channels=128, filter_channels=128, n_heads=TRANSFORMER_ENCODER_NHEAD, n_layers=TRANSFORMER_ENCODER_LAYERS, kernel_size=3, p_dropout=0.1)
        self.score_linear = nn.Linear(self.latent_dim*2, 1)
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        y = self.encoder(x)
        score = torch.sigmoid(self.score_linear(y))
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
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
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class NeuralSampler_v2_Trans(nn.Module):
    # Transformer encoder (with pos embedding)
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSampler_v2_Trans, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True
        self.encoder = TextEncoder(hidden_channels=128, filter_channels=128, n_heads=TRANSFORMER_ENCODER_NHEAD, n_layers=TRANSFORMER_ENCODER_LAYERS, kernel_size=3, p_dropout=0.1)
        self.score_linear = nn.Linear(self.latent_dim*2, 1)
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=True)
    
    def forward(self, x):
        y = self.encoder(x+self.pos_emb)
        score = torch.sigmoid(self.score_linear(y))
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
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

##############################################################################


# 07/23/2022 09:52:42 AM - INFO: mAP: 0.191025
# 07/23/2022 09:52:42 AM - INFO: AUC: 0.832402
# 07/23/2022 09:52:42 AM - INFO: Avg Precision: 0.009599
# 07/23/2022 09:52:42 AM - INFO: Avg Recall: 0.841954
# 07/23/2022 09:52:42 AM - INFO: d_prime: 1.362878
# 07/23/2022 09:52:42 AM - INFO: train_loss: 0.014404
# 07/23/2022 09:52:42 AM - INFO: valid_loss: 0.028033
class NeuralSamplerPosEmbSum(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSamplerPosEmbSum, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.num_layers=2
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True
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
        x, y, emb, score = ret['x'], ret['feature'], ret['emb'], ret['score']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(411)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(412)
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
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
        ret['feature'] = tensor_list.unsqueeze(1) + pos_emb.unsqueeze(1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class NeuralSamplerPosEmbSumLearnable(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSamplerPosEmbSumLearnable, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.num_layers=2
        self.feature_channels=1
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
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
        ret['feature'] = tensor_list.unsqueeze(1) + pos_emb.unsqueeze(1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class NeuralSamplerPosEmbLearnable(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
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
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

# 07/23/2022 07:41:49 AM - INFO: mAP: 0.218144
# 07/23/2022 07:41:49 AM - INFO: AUC: 0.899066
# 07/23/2022 07:41:49 AM - INFO: Avg Precision: 0.012408
# 07/23/2022 07:41:49 AM - INFO: Avg Recall: 0.909120
# 07/23/2022 07:41:49 AM - INFO: d_prime: 1.804883
# 07/23/2022 07:41:49 AM - INFO: train_loss: 0.013924
# 07/23/2022 07:41:49 AM - INFO: valid_loss: 0.018248
class NeuralSamplerPosEmb(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSamplerPosEmb, self).__init__()
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(413)
            plt.imshow(y[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(emb[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
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
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class NeuralSamplerNoFakeSoftmax(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(313)
            plt.imshow(y[i,0,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
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
    def __init__(self, input_seq_length, preserve_ratio):
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

class NeuralSamplerAvgMaxPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

class NeuralSamplerSpecPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

class NeuralSamplerAvgPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

class NeuralSampler_NNLSTM(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSampler, self).__init__()
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
        ret['feature']=score[:,:self.preserv_length, :]
        ret['score_loss']=torch.tensor([0.0]).cuda()
        return ret

    def visualize(self, ret):
        x, y, score = ret['x'], ret['feature']
        import matplotlib.pyplot as plt
        for i in range(10):
            plt.figure(figsize=(6, 8))
            plt.subplot(211)
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

class NoAction(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NoAction, self).__init__()
        self.feature_channels=1

    def forward(self, x):
        ret = {}
        ret['feature']=x.unsqueeze(1)
        ret['score_loss']=torch.tensor([0.0]).cuda()
        return ret

    def visualize(self, ret):
        return

# 07/23/2022 10:50:58 AM - INFO: mAP: 0.213149
# 07/23/2022 10:50:58 AM - INFO: AUC: 0.852090
# 07/23/2022 10:50:58 AM - INFO: Avg Precision: 0.009899
# 07/23/2022 10:50:58 AM - INFO: Avg Recall: 0.863098
# 07/23/2022 10:50:58 AM - INFO: d_prime: 1.478474
# 07/23/2022 10:50:58 AM - INFO: train_loss: 0.014461
# 07/23/2022 10:50:58 AM - INFO: valid_loss: 0.024842
class NeuralSamplerMiddle(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSamplerMiddle, self).__init__()
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
        out, (hn, cn) = self.feature_lstm(x)
        score = torch.sigmoid(self.score_linear(out))
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(313)
            plt.imshow(y[i,0,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
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
        cumsum_weight = cumsum_weight - (score/2)

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

        ret['feature'] = tensor_list.unsqueeze(1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

# 07/22/2022 11:47:59 PM - INFO: mAP: 0.220006
# 07/22/2022 11:47:59 PM - INFO: AUC: 0.900948
# 07/22/2022 11:47:59 PM - INFO: Avg Precision: 0.012848
# 07/22/2022 11:47:59 PM - INFO: Avg Recall: 0.911499
# 07/22/2022 11:47:59 PM - INFO: d_prime: 1.820051
# 07/22/2022 11:47:59 PM - INFO: train_loss: 0.014196
# 07/22/2022 11:47:59 PM - INFO: valid_loss: 0.018107
class NeuralSampler(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio):
        super(NeuralSampler, self).__init__()
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
            plt.imshow(x[i,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
            plt.subplot(313)
            plt.imshow(y[i,0,...].detach().cpu().numpy(), aspect="auto", interpolation='none')
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

        ret['feature'] = tensor_list.unsqueeze(1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class ProposedDilatedConv_2_18_3FreezePos(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(ProposedDilatedConv_2_18_3FreezePos, self).__init__()
        from models.dilated_convolutions.conv import DilatedConv
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim // 2, self.input_dim // 2),
        )
        self.dilated_conv = DilatedConv(1, 1, layers=4, dilation_rate=2, channel_increase_rate=1.5, kernel_size=3, input_size=(input_seq_length, self.input_dim // 2), stride=1)
        self.feature_lstm = nn.LSTM(self.input_dim // 2, self.input_dim // 4, self.num_layers, batch_first=True, bidirectional=True)

        self.score_linear = nn.Sequential(
            nn.Linear((self.input_dim // 4)*2, self.input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim // 4, 1),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=POS_EMB_REQUIRES_GRAD)
        
        init_gru(self.feature_lstm)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        pre_x = self.pre_linear(x)
        dilated_x = self.dilated_conv(pre_x.unsqueeze(1)).squeeze(1)
        score, (hn, cn) = self.feature_lstm(dilated_x)
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

        ret['emb'] = pos_emb
        ret['feature'] = tensor_list.unsqueeze(1) + pos_emb.unsqueeze(1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

class ProposedFreezePosDilatedConv_2_18_3(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0):
        super(ProposedFreezePosDilatedConv_2_18_3, self).__init__()
        from models.dilated_convolutions.conv import DilatedConv
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=1
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
        self.dilated_conv = DilatedConv(1, 1, layers=4, dilation_rate=2, channel_increase_rate=1.5, kernel_size=3, input_size=(input_seq_length, self.input_dim*2), stride=1)
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

        init_gru(self.feature_lstm)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        pre_x = self.pre_linear(x)
        dilated_x = self.dilated_conv(pre_x.unsqueeze(1)).squeeze(1)
        score, (hn, cn) = self.feature_lstm(dilated_x)
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

        ret['emb'] = pos_emb
        ret['feature'] = tensor_list.unsqueeze(1) + pos_emb.unsqueeze(1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)



def test_sampler(sampler):
    input_tdim = 3000
    sampler = sampler(input_seq_length=3000, preserve_ratio=0.1)
    test_input = torch.rand([10, input_tdim, 128])
    ret =sampler(test_input)
    sampler.visualize(ret)
    print("Perfect!", sampler, ret["feature"].size(), ret["score_loss"].size(), ret["score_loss"])

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

if __name__ == "__main__":
    import logging

    logging.basicConfig(
    filename="log.txt",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    
    test_sampler(NeuralSamplerMaxPool)
    test_sampler(NeuralSamplerSpecPool)
    # 07/23/2022 05:36:10 AM - INFO: mAP: 0.216001
    # 07/23/2022 05:36:10 AM - INFO: AUC: 0.919990
    # 07/23/2022 05:36:10 AM - INFO: Avg Precision: 0.013685
    # 07/23/2022 05:36:10 AM - INFO: Avg Recall: 0.930226
    # 07/23/2022 05:36:10 AM - INFO: d_prime: 1.986981
    # 07/23/2022 05:36:10 AM - INFO: train_loss: 0.014289
    # 07/23/2022 05:36:10 AM - INFO: valid_loss: 0.017917
    test_sampler(NeuralSamplerAvgMaxPool)
    test_sampler(NeuralSamplerAvgPool)
    # 07/23/2022 06:42:49 AM - INFO: mAP: 0.221230
    # 07/23/2022 06:42:49 AM - INFO: AUC: 0.916620
    # 07/23/2022 06:42:49 AM - INFO: Avg Precision: 0.014316
    # 07/23/2022 06:42:49 AM - INFO: Avg Recall: 0.926914
    # 07/23/2022 06:42:49 AM - INFO: d_prime: 1.955419
    # 07/23/2022 06:42:49 AM - INFO: train_loss: 0.013721
    # 07/23/2022 06:42:49 AM - INFO: valid_loss: 0.017955
    test_sampler(NeuralSamplerNoFakeSoftmax)
    test_sampler(NeuralSamplerPosEmbSumLearnable)
    # 07/23/2022 09:52:42 AM - INFO: mAP: 0.191025
    # 07/23/2022 09:52:42 AM - INFO: AUC: 0.832402
    # 07/23/2022 09:52:42 AM - INFO: Avg Precision: 0.009599
    # 07/23/2022 09:52:42 AM - INFO: Avg Recall: 0.841954
    # 07/23/2022 09:52:42 AM - INFO: d_prime: 1.362878
    # 07/23/2022 09:52:42 AM - INFO: train_loss: 0.014404
    # 07/23/2022 09:52:42 AM - INFO: valid_loss: 0.028033
    test_sampler(NeuralSamplerPosEmbSum)
    # 07/23/2022 08:41:02 AM - INFO: mAP: 0.220362
    # 07/23/2022 08:41:02 AM - INFO: AUC: 0.918260
    # 07/23/2022 08:41:02 AM - INFO: Avg Precision: 0.014869
    # 07/23/2022 08:41:02 AM - INFO: Avg Recall: 0.926952
    # 07/23/2022 08:41:02 AM - INFO: d_prime: 1.970651
    # 07/23/2022 08:41:02 AM - INFO: train_loss: 0.013516
    # 07/23/2022 08:41:02 AM - INFO: valid_loss: 0.018110
    test_sampler(NeuralSamplerPosEmbLearnable)
    # 07/23/2022 07:41:49 AM - INFO: mAP: 0.218144
    # 07/23/2022 07:41:49 AM - INFO: AUC: 0.899066
    # 07/23/2022 07:41:49 AM - INFO: Avg Precision: 0.012408
    # 07/23/2022 07:41:49 AM - INFO: Avg Recall: 0.909120
    # 07/23/2022 07:41:49 AM - INFO: d_prime: 1.804883
    # 07/23/2022 07:41:49 AM - INFO: train_loss: 0.013924
    # 07/23/2022 07:41:49 AM - INFO: valid_loss: 0.018248
    test_sampler(NeuralSamplerPosEmb)
    # 07/22/2022 11:47:59 PM - INFO: mAP: 0.220006
    # 07/22/2022 11:47:59 PM - INFO: AUC: 0.900948
    # 07/22/2022 11:47:59 PM - INFO: Avg Precision: 0.012848
    # 07/22/2022 11:47:59 PM - INFO: Avg Recall: 0.911499
    # 07/22/2022 11:47:59 PM - INFO: d_prime: 1.820051
    # 07/22/2022 11:47:59 PM - INFO: train_loss: 0.014196
    # 07/22/2022 11:47:59 PM - INFO: valid_loss: 0.018107
    test_sampler(NeuralSampler)
    