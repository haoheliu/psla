from turtle import forward
from unicodedata import bidirectional
import torch.nn as nn
import torch
import math
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer
from .pooling import Pooling_layer
import logging
import os

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
    test_sampler(NeuralSamplerAvgMaxPool)
    test_sampler(NeuralSamplerAvgPool)
    test_sampler(NeuralSamplerNoFakeSoftmax)
    test_sampler(NeuralSamplerPosEmbSumLearnable)
    test_sampler(NeuralSamplerPosEmbSum)
    test_sampler(NeuralSamplerPosEmbLearnable)
    test_sampler(NeuralSamplerPosEmb)
    test_sampler(NeuralSampler)
    