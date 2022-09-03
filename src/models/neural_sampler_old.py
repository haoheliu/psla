import sys
sys.path.append("/media/Disk_HDD/haoheliu/projects/psla/src")
sys.path.append("/Volumes/nobackup-scratch4weeks/hl01486/project/psla/src")
sys.path.append("/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla/src")


import torch
from turtle import forward
from unicodedata import bidirectional
import torch.nn as nn
import torch
import math
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer
from functorch import vmap
import logging
import os

# from HigherModels import *
# from neural_sampler import *
# from pooling import Pooling_layer

from models.HigherModels import *
from models.neural_sampler import *
from models.pooling import Pooling_layer

RESCALE_INTERVEL_MIN=1e-4
RESCALE_INTERVEL_MAX=1-1e-4
# GROUP_BATCH=2

def init_gru(rnn):
    """Initialize a GRU layer. """

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in: (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

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

# Use Non-NN method 
class NeuralSamplerLargeEnergy(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NeuralSamplerLargeEnergy, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.alpha = alpha
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = True
        print("ALPHA %s" % self.alpha)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)
    
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX**self.alpha, score > RESCALE_INTERVEL_MIN**self.alpha) # TODO here 0.1 or 0.01
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
        ret['energy']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)
# Use DNN
class NewAlgoEnergyv2(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NewAlgoEnergyv2, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        score = energy
        ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        ret['x']=x
        energy,_=self.score_norm(energy, self.output_seq_length)
        ret['energy']=energy
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def locate_first_and_last_position(self, mask):
        """Locate the first non-negative in a row, and the element before the last non-negative element in a row

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        bs, orig_len, target_len = mask.size()
        
        assert orig_len >= target_len

        weight = torch.tensor([-1.0,1.0]).expand(target_len,-1).to(mask.device)
        weight = weight.unsqueeze(1)
        value = torch.nn.functional.conv1d(mask.permute(0,2,1).float(), weight, bias=None, stride=1, padding=0, dilation=1, groups=target_len)
        value = torch.nn.functional.pad(value, (1,0))
        value = value.permute(0,2,1)
        return value == 1, value == -1

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        
        # Normalize the socre value
        score, total_length = self.score_norm(score, total_length)

        # Monotonic Expansion
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # Get the masked weight
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask

        # Make the sum or each row to one
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        one_minus_weight_sum = 1-weight_sum
        one_minus_weight_sum_cumsum = torch.cumsum(one_minus_weight_sum, dim=2)
        need_minus, need_add = self.locate_first_and_last_position(mask)
        need_minus = need_minus[:,:,1:] * one_minus_weight_sum_cumsum[:,:,:-1]
        need_minus = torch.nn.functional.pad(need_minus,(1,0))
        need_add = need_add * one_minus_weight_sum_cumsum
        weight = weight - need_minus + need_add

        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)

        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

# 0.64
class NewAlgoDilatedConv1dMaxPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NewAlgoDilatedConv1dMaxPool, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=1, input_size=self.input_seq_length, kernel_size=5, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        score = torch.sigmoid(self.model(x.permute(0,2,1)).permute(0,2,1))
        ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        ret['x']=x
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        return ret

    def visualize(self, ret):
        x, y, emb, score, energy,maxpool = ret['x'], ret['feature'], ret['emb'], ret['score'], ret['energy'],ret['feature_maxpool']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(611)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(612)
            plt.plot(energy[i,:,0].detach().cpu().numpy())
            plt.subplot(613)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(614)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(615)
            plt.imshow(maxpool[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(616)
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def locate_first_and_last_position(self, mask):
        """Locate the first non-negative in a row, and the element before the last non-negative element in a row

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        bs, orig_len, target_len = mask.size()
        
        assert orig_len >= target_len

        weight = torch.tensor([-1.0,1.0]).expand(target_len,-1).to(mask.device)
        weight = weight.unsqueeze(1)
        value = torch.nn.functional.conv1d(mask.permute(0,2,1).float(), weight, bias=None, stride=1, padding=0, dilation=1, groups=target_len)
        value = torch.nn.functional.pad(value, (1,0))
        value = value.permute(0,2,1)
        return value == 1, value == -1
    
    # def calculate_max_pool(self, weight, feature):
    #     # weight: [3, 1056, 105]
    #     # feature: [3, 1056, 128]
    #     tensor_list = []
    #     for i in range(weight.size(-1)):
    #         weight_row = weight[:,:,i].unsqueeze(-1)
    #         tensor = (feature * weight_row).permute(0,2,1)
    #         tensor_list.append(torch.nn.functional.max_pool1d(tensor, kernel_size=self.input_seq_length).permute(0,2,1))
    #     return torch.cat(tensor_list, dim=1)

    def calculate_max_pool_fast(self, weight, feature):
        # weight: [3, 1056, 105]
        # feature: [3, 1056, 128]
        bs, seqlen, compressed_len = weight.size()
        bs, seqlen, mel_bins = feature.size()
        expanded_feature = feature.unsqueeze(2).expand(bs, seqlen, compressed_len, mel_bins)
        expanded_feature = (expanded_feature * weight.unsqueeze(-1)).permute(0,2,1,3)
        tensor = torch.nn.functional.max_pool2d(expanded_feature, kernel_size=(self.input_seq_length, 1)).squeeze(2)
        return tensor
    
    # def calculate_max_pool_slow(self, weight, feature):
    #     bs, seqlen, compressed_len = weight.size()
    #     bs, seqlen, mel_bins = feature.size()
    #     weight = weight[0]
    #     feature = feature[0]
    #     assert bs == 1
    #     tensor_list = []
    #     for i in range(compressed_len):
    #         non_zero_feature = feature[weight[:,i] > 0]
    #         pooled_feature = torch.nn.functional.max_pool1d(non_zero_feature.permute(1,0), kernel_size=(non_zero_feature.size(0))).permute(1,0).unsqueeze(0)
    #         tensor_list.append(pooled_feature)
    #     return torch.cat(tensor_list, dim=1)
    
    def calculate_max_pool_fast_grouped(self, weight, feature):
        # weight: [3, 1056, 105]
        # feature: [3, 1056, 128]
        bs, _, _ = weight.size()
        start_idx = 0
        tensor_list = []
        while(start_idx < bs):
            weight_batch, feature_batch = weight[start_idx:start_idx+1], feature[start_idx:start_idx+1]
            if(weight_batch.size(0) == 0): break
            tensor_list.append(self.calculate_max_pool_fast(weight_batch, feature_batch))
            start_idx += 1
        return torch.cat(tensor_list, dim=0)

    # def _max_pool_dimension(self, weight, feature=None):
    #     # weight: [3, 1056, 105]
    #     # feature: [3, 1056, 128]
    #     weight_row = weight.unsqueeze(-1)
    #     tensor = (feature * weight_row).permute(0,2,1)
    #     return torch.max(tensor, dim=-1)[0]

    # def calculate_max_pool_fast_efficient(self, weight, feature):
    #     # weight: [3, 1056, 105]
    #     # feature: [3, 1056, 128]
    #     tensor = vmap(self._max_pool_dimension, in_dims=-1, out_dims=-1)(weight, feature=feature)
    #     return tensor.permute(0,2,1)

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        
        # Normalize the socre value
        score, total_length = self.score_norm(score, total_length)

        # Monotonic Expansion
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # Get the masked weight
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask

        # Make the sum or each row to one
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        one_minus_weight_sum = 1-weight_sum
        one_minus_weight_sum_cumsum = torch.cumsum(one_minus_weight_sum, dim=2)
        need_minus, need_add = self.locate_first_and_last_position(mask)
        need_minus = need_minus[:,:,1:] * one_minus_weight_sum_cumsum[:,:,:-1]
        need_minus = torch.nn.functional.pad(need_minus,(1,0))
        need_add = need_add * one_minus_weight_sum_cumsum
        weight = weight - need_minus + need_add

        # [3, 105, 128]    
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        # tensor_list_maxpool=tensor_list
        # tensor_list_maxpool = self.calculate_max_pool(weight, feature)
        tensor_list_maxpool = self.calculate_max_pool_fast_grouped(weight, feature)
        # tensor_list_maxpool = self.calculate_max_pool_fast(weight, feature)
        # tensor_list_maxpool = self.calculate_max_pool_fast_efficient(weight, feature)
        tensor_list = (tensor_list + tensor_list_maxpool) / 2
        
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        
        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['feature_maxpool']=tensor_list_maxpool
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret 


class NSDilatedConv1dMaxPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NSDilatedConv1dMaxPool, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=1, input_size=self.input_seq_length, kernel_size=5, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        score = torch.sigmoid(self.model(x.permute(0,2,1)).permute(0,2,1))
        ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        ret['x']=x
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        return ret

    def visualize(self, ret):
        x, y, emb, score, energy,maxpool = ret['x'], ret['feature'], ret['emb'], ret['score'], ret['energy'],ret['feature_maxpool']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(611)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(612)
            plt.plot(energy[i,:,0].detach().cpu().numpy())
            plt.subplot(613)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(614)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(615)
            plt.imshow(maxpool[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(616)
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    # def locate_first_and_last_position(self, mask):
    #     """Locate the first non-negative in a row, and the element before the last non-negative element in a row

    #     Args:
    #         mask (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     bs, orig_len, target_len = mask.size()
        
    #     assert orig_len >= target_len

    #     weight = torch.tensor([-1.0,1.0]).expand(target_len,-1).to(mask.device)
    #     weight = weight.unsqueeze(1)
    #     value = torch.nn.functional.conv1d(mask.permute(0,2,1).float(), weight, bias=None, stride=1, padding=0, dilation=1, groups=target_len)
    #     value = torch.nn.functional.pad(value, (1,0))
    #     value = value.permute(0,2,1)
    #     return value == 1, value == -1
    
    # def calculate_max_pool(self, weight, feature):
    #     # weight: [3, 1056, 105]
    #     # feature: [3, 1056, 128]
    #     tensor_list = []
    #     for i in range(weight.size(-1)):
    #         weight_row = weight[:,:,i].unsqueeze(-1)
    #         tensor = (feature * weight_row).permute(0,2,1)
    #         tensor_list.append(torch.nn.functional.max_pool1d(tensor, kernel_size=self.input_seq_length).permute(0,2,1))
    #     return torch.cat(tensor_list, dim=1)

    def calculate_max_pool_fast(self, weight, feature):
        # weight: [3, 1056, 105]
        # feature: [3, 1056, 128]
        bs, seqlen, compressed_len = weight.size()
        bs, seqlen, mel_bins = feature.size()
        expanded_feature = feature.unsqueeze(2).expand(bs, seqlen, compressed_len, mel_bins)
        expanded_feature = (expanded_feature * weight.unsqueeze(-1)).permute(0,2,1,3)
        tensor = torch.nn.functional.max_pool2d(expanded_feature, kernel_size=(self.input_seq_length, 1)).squeeze(2)
        return tensor
    
    # def calculate_max_pool_slow(self, weight, feature):
    #     bs, seqlen, compressed_len = weight.size()
    #     bs, seqlen, mel_bins = feature.size()
    #     weight = weight[0]
    #     feature = feature[0]
    #     assert bs == 1
    #     tensor_list = []
    #     for i in range(compressed_len):
    #         non_zero_feature = feature[weight[:,i] > 0]
    #         pooled_feature = torch.nn.functional.max_pool1d(non_zero_feature.permute(1,0), kernel_size=(non_zero_feature.size(0))).permute(1,0).unsqueeze(0)
    #         tensor_list.append(pooled_feature)
    #     return torch.cat(tensor_list, dim=1)
    
    def calculate_max_pool_fast_grouped(self, weight, feature):
        # weight: [3, 1056, 105]
        # feature: [3, 1056, 128]
        bs, _, _ = weight.size()
        start_idx = 0
        tensor_list = []
        while(start_idx < bs):
            weight_batch, feature_batch = weight[start_idx:start_idx+1], feature[start_idx:start_idx+1]
            if(weight_batch.size(0) == 0): break
            tensor_list.append(self.calculate_max_pool_fast(weight_batch, feature_batch))
            start_idx += 1
        return torch.cat(tensor_list, dim=0)

    # def _max_pool_dimension(self, weight, feature=None):
    #     # weight: [3, 1056, 105]
    #     # feature: [3, 1056, 128]
    #     weight_row = weight.unsqueeze(-1)
    #     tensor = (feature * weight_row).permute(0,2,1)
    #     return torch.max(tensor, dim=-1)[0]

    # def calculate_max_pool_fast_efficient(self, weight, feature):
    #     # weight: [3, 1056, 105]
    #     # feature: [3, 1056, 128]
    #     tensor = vmap(self._max_pool_dimension, in_dims=-1, out_dims=-1)(weight, feature=feature)
    #     return tensor.permute(0,2,1)
    
    def calculate_scatter_add(self, score, feature, out_len):
        from torch_scatter import scatter_add
        # cumsum: [3, 1056, 1]
        # feature: [3, 1056, 128]
        bs, in_seq_len, feat_dim = feature.size()
        cumsum = torch.cumsum(score, dim=1)

        int_cumsum = torch.floor(cumsum.float()).permute(0,2,1).long()
        int_cumsum = torch.clip(int_cumsum, min=0, max=out_len-1)
        out = torch.zeros((bs, feat_dim, out_len))

        # feature: [bs, feat-dim, in-seq-len]
        # int_cumsum: [bs, 1, in-seq-len]
        # out: [bs, feat-dim, out-seq-len]
        out = scatter_add((feature * score).permute(0,2,1), int_cumsum, out=out)
        return out.permute(0,2,1)

    def calculate_scatter_maxpool(self, score, feature, out_len):
        from torch_scatter import scatter_max
        # cumsum: [3, 1056, 1]
        # feature: [3, 1056, 128]
        bs, in_seq_len, feat_dim = feature.size()
        cumsum = torch.cumsum(score, dim=1)

        int_cumsum = torch.floor(cumsum.float()).permute(0,2,1).long()
        int_cumsum = torch.clip(int_cumsum, min=0, max=out_len-1)
        out = torch.zeros((bs, feat_dim, out_len))

        # feature: [bs, feat-dim, in-seq-len]
        # int_cumsum: [bs, 1, in-seq-len]
        # out: [bs, feat-dim, out-seq-len]
        out,_ = scatter_max((feature * score).permute(0,2,1), int_cumsum, out=out)
        return out.permute(0,2,1)
    
    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        bs, in_len, feat_dim = feature.size()
        _, in_len, feat_dim = self.pos_emb.size()
        
        # Normalize the socre value
        score, total_length = self.score_norm(score, total_length)


        # New method
        mean_feature = self.calculate_scatter_add(score, feature=feature, out_len=self.output_seq_length)
        max_pool_feature = self.calculate_scatter_maxpool(score, feature=feature, out_len=self.output_seq_length)
        mean_pos_enc = self.calculate_scatter_add(score, feature=self.pos_emb.expand(bs, in_len, feat_dim), out_len=self.output_seq_length)


        # Monotonic Expansion
        # cumsum_score = torch.cumsum(score, dim=1)
        # cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        # threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        # smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        # greater_mask = cumsum_weight > threshold[None, None, ...]
        # mask = torch.logical_and(smaller_mask, greater_mask)

        # # Get the masked weight
        # weight = score.expand(feature.size(0), feature.size(1), total_length)
        # weight = weight * mask
        # tensor_list = torch.matmul(weight.permute(0,2,1), feature)

        # Make the sum or each row to one
        # weight_sum = torch.sum(weight, dim=1, keepdim=True)
        # one_minus_weight_sum = 1-weight_sum
        # one_minus_weight_sum_cumsum = torch.cumsum(one_minus_weight_sum, dim=2)
        # need_minus, need_add = self.locate_first_and_last_position(mask)
        # need_minus = need_minus[:,:,1:] * one_minus_weight_sum_cumsum[:,:,:-1]
        # need_minus = torch.nn.functional.pad(need_minus,(1,0))
        # need_add = need_add * one_minus_weight_sum_cumsum
        # weight = weight - need_minus + need_add

        # [3, 105, 128]    
        # tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        # tensor_list_maxpool=tensor_list
        # tensor_list_maxpool = self.calculate_max_pool(weight, feature)
        # tensor_list_maxpool = self.calculate_max_pool_fast_grouped(weight, feature)
        # tensor_list_maxpool = self.calculate_max_pool_fast(weight, feature)
        # tensor_list_maxpool = self.calculate_max_pool_fast_efficient(weight, feature)
        # tensor_list = (tensor_list + tensor_list_maxpool) / 2
        
        # pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        
        ret['emb'] = mean_pos_enc
        ret['feature'] = torch.cat([((mean_feature + max_pool_feature) / 2).unsqueeze(1), mean_pos_enc.unsqueeze(1)], dim=1)
        ret['feature_maxpool']=max_pool_feature
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret 


# Changing the step size
class NeuralSamplerUniformPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NeuralSamplerUniformPool, self).__init__()
        self.feature_channels=1
        self.mean = mean
        self.std = std

        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = False
        self.pooling = Pooling_layer(pooling_type="uniform", factor=preserve_ratio)
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
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
                axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
                for i in range(score.size(0)):
                    if(distance_with_target_length[i] >= 1):
                        intervel = 1.0-score[i][axis[i]]
                        alpha = distance_with_target_length[i] / torch.sum(intervel) 
                        if(alpha > 1): alpha=1
                        score[i][axis[i]] += intervel * alpha
            ####################################################################
            return score, total_length

    def forward(self, x):
        ret={}
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        ret['score'],_=self.score_norm(energy, self.output_seq_length)
        feature = self.pooling(x.unsqueeze(1))
        ret['score_loss']=torch.tensor([0.0]).to(x.device)
        ret['feature']=feature
        ret['x']=x
        return ret

    def visualize(self, ret):
        x, y = ret['x'], ret['feature']
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(211)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()
            
# TODO
class BaselineAdaAvgMaxPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(BaselineAdaAvgMaxPool, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        
        self.pooling = torch.nn.AdaptiveAvgPool1d(self.output_seq_length)
        self.max_pooling = torch.nn.AdaptiveMaxPool1d(self.output_seq_length)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        ret = {}
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)

        score = torch.ones_like(x[...,0:1]).to(x.device)

        # ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        ret['feature'] = (self.pooling(x.permute(0,2,1)).permute(0,2,1).unsqueeze(1) + self.max_pooling(x.permute(0,2,1)).permute(0,2,1).unsqueeze(1)) / 2

        ret['x']=x
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        ret['score'],_=self.score_norm(score, self.output_seq_length)
        ret['score_loss']=torch.tensor([0.0]).to(x.device)
        return ret

    def visualize(self, ret):
        x, y, score, energy = ret['x'], ret['feature'], ret['score'], ret['energy']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(411)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(412)
            plt.plot(energy[i,:,0].detach().cpu().numpy())
            plt.subplot(413)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

# TODO
class BaselineAdaAvgPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(BaselineAdaAvgPool, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        
        self.pooling = torch.nn.AdaptiveAvgPool1d(self.output_seq_length)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        ret = {}
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)

        score = torch.ones_like(x[...,0:1]).to(x.device)

        # ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        ret['feature'] = self.pooling(x.permute(0,2,1)).permute(0,2,1).unsqueeze(1)

        ret['x']=x
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        ret['score'],_=self.score_norm(score, self.output_seq_length)
        ret['score_loss']=torch.tensor([0.0]).to(x.device)
        return ret

    def visualize(self, ret):
        x, y, score, energy = ret['x'], ret['feature'], ret['score'], ret['energy']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(411)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(412)
            plt.plot(energy[i,:,0].detach().cpu().numpy())
            plt.subplot(413)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length
    
# Use DNN
class NewAlgoDilatedConv1dPlusEnergyv2(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NewAlgoDilatedConv1dPlusEnergyv2, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=1, input_size=1056, kernel_size=5, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        energy,_=self.score_norm(energy, self.output_seq_length)

        score = torch.sigmoid(self.model(x.permute(0,2,1)).permute(0,2,1)) + energy 
        ret = self.select_feature_fast(x, score+energy, total_length=self.output_seq_length)
        ret['score']=score
        ret['x']=x
        ret['energy']=energy
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def locate_first_and_last_position(self, mask):
        """Locate the first non-negative in a row, and the element before the last non-negative element in a row

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        bs, orig_len, target_len = mask.size()
        
        assert orig_len >= target_len

        weight = torch.tensor([-1.0,1.0]).expand(target_len,-1).to(mask.device)
        weight = weight.unsqueeze(1)
        value = torch.nn.functional.conv1d(mask.permute(0,2,1).float(), weight, bias=None, stride=1, padding=0, dilation=1, groups=target_len)
        value = torch.nn.functional.pad(value, (1,0))
        value = value.permute(0,2,1)
        return value == 1, value == -1

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        
        # Normalize the socre value
        score, total_length = self.score_norm(score, total_length)

        # Monotonic Expansion
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # Get the masked weight
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask

        # Make the sum or each row to one
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        one_minus_weight_sum = 1-weight_sum
        one_minus_weight_sum_cumsum = torch.cumsum(one_minus_weight_sum, dim=2)
        need_minus, need_add = self.locate_first_and_last_position(mask)
        need_minus = need_minus[:,:,1:] * one_minus_weight_sum_cumsum[:,:,:-1]
        need_minus = torch.nn.functional.pad(need_minus,(1,0))
        need_add = need_add * one_minus_weight_sum_cumsum
        weight = weight - need_minus + need_add

        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)

        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

class BaselineConstantScore(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(BaselineConstantScore, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv

        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)

        score = torch.ones_like(x[...,0:1]).to(x.device)
        
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def locate_first_and_last_position(self, mask):
        """Locate the first non-negative in a row, and the element before the last non-negative element in a row

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        bs, orig_len, target_len = mask.size()
        
        assert orig_len >= target_len

        weight = torch.tensor([-1.0,1.0]).expand(target_len,-1).to(mask.device)
        weight = weight.unsqueeze(1)
        value = torch.nn.functional.conv1d(mask.permute(0,2,1).float(), weight, bias=None, stride=1, padding=0, dilation=1, groups=target_len)
        value = torch.nn.functional.pad(value, (1,0))
        value = value.permute(0,2,1)
        return value == 1, value == -1

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        
        # Normalize the socre value
        score, total_length = self.score_norm(score, total_length)

        # Monotonic Expansion
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # Get the masked weight
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask

        # Make the sum or each row to one
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        one_minus_weight_sum = 1-weight_sum
        one_minus_weight_sum_cumsum = torch.cumsum(one_minus_weight_sum, dim=2)
        need_minus, need_add = self.locate_first_and_last_position(mask)
        need_minus = need_minus[:,:,1:] * one_minus_weight_sum_cumsum[:,:,:-1]
        need_minus = torch.nn.functional.pad(need_minus,(1,0))
        need_add = need_add * one_minus_weight_sum_cumsum
        weight = weight - need_minus + need_add

        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)

        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

class NewAlgoDilatedConv1d(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NewAlgoDilatedConv1d, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=1, input_size=1056, kernel_size=5, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        score = torch.sigmoid(self.model(x.permute(0,2,1)).permute(0,2,1))
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def locate_first_and_last_position(self, mask):
        """Locate the first non-negative in a row, and the element before the last non-negative element in a row

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        bs, orig_len, target_len = mask.size()
        
        assert orig_len >= target_len

        weight = torch.tensor([-1.0,1.0]).expand(target_len,-1).to(mask.device)
        weight = weight.unsqueeze(1)
        value = torch.nn.functional.conv1d(mask.permute(0,2,1).float(), weight, bias=None, stride=1, padding=0, dilation=1, groups=target_len)
        value = torch.nn.functional.pad(value, (1,0))
        value = value.permute(0,2,1)
        return value == 1, value == -1

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        
        # Normalize the socre value
        score, total_length = self.score_norm(score, total_length)

        # Monotonic Expansion
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # Get the masked weight
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask

        # Make the sum or each row to one
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        one_minus_weight_sum = 1-weight_sum
        one_minus_weight_sum_cumsum = torch.cumsum(one_minus_weight_sum, dim=2)
        need_minus, need_add = self.locate_first_and_last_position(mask)
        need_minus = need_minus[:,:,1:] * one_minus_weight_sum_cumsum[:,:,:-1]
        need_minus = torch.nn.functional.pad(need_minus,(1,0))
        need_add = need_add * one_minus_weight_sum_cumsum
        weight = weight - need_minus + need_add

        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)

        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret 

class NewAlgoDilatedConv1dIntp(nn.Module):
    """Pool the input spectrogram first, and calculate importance score on the pooled spectrogram. The complete IS is calculate by interpolations.
    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NewAlgoDilatedConv1dIntp, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True
        
        self.pooling = torch.nn.AdaptiveAvgPool1d(self.output_seq_length)
        self.max_pooling = torch.nn.AdaptiveMaxPool1d(self.output_seq_length)
        
        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=1, input_size=1056, kernel_size=5, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        pooled = self.pool(x)
        score = torch.sigmoid(self.model(pooled.permute(0,2,1)).permute(0,2,1))
        score = self.interpolate(score.permute(0,2,1)).permute(0,2,1)
        
        ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        ret['x']=x
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        return ret

    def interpolate(self, score):
        return torch.nn.functional.interpolate(score, size=self.input_seq_length, mode='linear')

    def pool(self, x):
        return (self.pooling(x.permute(0,2,1)).permute(0,2,1) + self.max_pooling(x.permute(0,2,1)).permute(0,2,1)) / 2

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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def locate_first_and_last_position(self, mask):
        """Locate the first non-negative in a row, and the element before the last non-negative element in a row

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        bs, orig_len, target_len = mask.size()
        
        assert orig_len >= target_len

        weight = torch.tensor([-1.0,1.0]).expand(target_len,-1).to(mask.device)
        weight = weight.unsqueeze(1)
        value = torch.nn.functional.conv1d(mask.permute(0,2,1).float(), weight, bias=None, stride=1, padding=0, dilation=1, groups=target_len)
        value = torch.nn.functional.pad(value, (1,0))
        value = value.permute(0,2,1)
        return value == 1, value == -1

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        
        # Normalize the socre value
        score, total_length = self.score_norm(score, total_length)

        # Monotonic Expansion
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # Get the masked weight
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask

        # Make the sum or each row to one
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        one_minus_weight_sum = 1-weight_sum
        one_minus_weight_sum_cumsum = torch.cumsum(one_minus_weight_sum, dim=2)
        need_minus, need_add = self.locate_first_and_last_position(mask)
        need_minus = need_minus[:,:,1:] * one_minus_weight_sum_cumsum[:,:,:-1]
        need_minus = torch.nn.functional.pad(need_minus,(1,0))
        need_add = need_add * one_minus_weight_sum_cumsum
        weight = weight - need_minus + need_add

        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)

        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret 
  
# Use extra large lstm model
class NewAlgoLSTMLayerNormIntp(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NewAlgoLSTMLayerNormIntp, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True
        self.input_bn = nn.BatchNorm2d(1)
        
        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim),
        )

        self.pooling = torch.nn.AdaptiveAvgPool1d(self.output_seq_length)
        self.max_pooling = torch.nn.AdaptiveMaxPool1d(self.output_seq_length)
        
        self.layernorm_lstm_start = nn.LayerNorm([self.output_seq_length, self.input_dim])
        self.feature_lstm_1 = nn.LSTM(self.input_dim, self.latent_dim, 1, batch_first=True, bidirectional=True)
        self.layernorm_lstm_middle = nn.LayerNorm([self.output_seq_length, self.latent_dim*2])
        self.feature_lstm_2 = nn.LSTM(self.latent_dim*2, self.latent_dim, 1, batch_first=True, bidirectional=True)
        
        self.linear_lstm = nn.Sequential(
            nn.Linear(self.latent_dim*2, 1),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
        init_gru(self.feature_lstm_1)
        init_gru(self.feature_lstm_2)
        self.init_seq_linear(self.pre_linear)
        self.init_seq_linear(self.linear_lstm)
        init_bn(self.input_bn)

    def init_seq_linear(self, sequencial):
        for i in range(len(sequencial)):
            if(isinstance(sequencial[i], nn.Linear)):
                init_layer(sequencial[i])

    def interpolate(self, score):
        return torch.nn.functional.interpolate(score, size=self.input_seq_length, mode='linear')

    def pool(self, x):
        return (self.pooling(x.permute(0,2,1)).permute(0,2,1) + self.max_pooling(x.permute(0,2,1)).permute(0,2,1)) / 2

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        pooled = self.pool(x)
        
        pooled = self.input_bn(pooled.unsqueeze(1)).squeeze(1)
        pooled = self.pre_linear(pooled) + pooled
        pooled = self.layernorm_lstm_start(pooled)
        feature_1, (hn, cn) = self.feature_lstm_1(pooled)
        feature_1 = self.layernorm_lstm_middle(feature_1)
        feature_2, (hn, cn) = self.feature_lstm_2(feature_1, (hn, cn))
        score = torch.sigmoid(self.linear_lstm(feature_2))
        
        score = self.interpolate(score.permute(0,2,1)).permute(0,2,1)
        
        # TODO
        score, total_length = self.score_norm(score, self.output_seq_length)
        score += self.preserv_ratio * 0.1
        score, total_length = self.score_norm(score, self.output_seq_length)
        
        ret = self.select_feature_fast(x, score, total_length=total_length)
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def locate_first_and_last_position(self, mask):
        """Locate the first non-negative in a row, and the element before the last non-negative element in a row

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        bs, orig_len, target_len = mask.size()
        
        assert orig_len >= target_len

        weight = torch.tensor([-1.0,1.0]).expand(target_len,-1).to(mask.device)
        weight = weight.unsqueeze(1)
        value = torch.nn.functional.conv1d(mask.permute(0,2,1).float(), weight, bias=None, stride=1, padding=0, dilation=1, groups=target_len)
        value = torch.nn.functional.pad(value, (1,0))
        value = value.permute(0,2,1)
        return value == 1, value == -1

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        
        # Normalize the socre value
        

        # Monotonic Expansion
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # Get the masked weight
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask

        # Make the sum or each row to one
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        one_minus_weight_sum = 1-weight_sum
        one_minus_weight_sum_cumsum = torch.cumsum(one_minus_weight_sum, dim=2)
        need_minus, need_add = self.locate_first_and_last_position(mask)
        need_minus = need_minus[:,:,1:] * one_minus_weight_sum_cumsum[:,:,:-1]
        need_minus = torch.nn.functional.pad(need_minus,(1,0))
        need_add = need_add * one_minus_weight_sum_cumsum
        weight = weight - need_minus + need_add

        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)

        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret 

# Use extra large lstm model
class NewAlgoLSTMLayerNormIntpPlusEnergyv2(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NewAlgoLSTMLayerNormIntpPlusEnergyv2, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True
        self.input_bn = nn.BatchNorm2d(1)
        
        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim),
        )

        self.pooling = torch.nn.AdaptiveAvgPool1d(self.output_seq_length)
        self.max_pooling = torch.nn.AdaptiveMaxPool1d(self.output_seq_length)
        
        self.layernorm_lstm_start = nn.LayerNorm([self.output_seq_length, self.input_dim])
        self.feature_lstm_1 = nn.LSTM(self.input_dim, self.latent_dim, 1, batch_first=True, bidirectional=True)
        self.layernorm_lstm_middle = nn.LayerNorm([self.output_seq_length, self.latent_dim*2])
        self.feature_lstm_2 = nn.LSTM(self.latent_dim*2, self.latent_dim, self.num_layers // 2, batch_first=True, bidirectional=True)
        
        self.linear_lstm = nn.Sequential(
            nn.Linear(self.latent_dim*2, 1),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
        init_gru(self.feature_lstm_1)
        init_gru(self.feature_lstm_2)
        self.init_seq_linear(self.pre_linear)
        self.init_seq_linear(self.linear_lstm)
        init_bn(self.input_bn)

    def init_seq_linear(self, sequencial):
        for i in range(len(sequencial)):
            if(isinstance(sequencial[i], nn.Linear)):
                init_layer(sequencial[i])

    def interpolate(self, score):
        return torch.nn.functional.interpolate(score, size=self.input_seq_length, mode='linear')

    def pool(self, x):
        return (self.pooling(x.permute(0,2,1)).permute(0,2,1) + self.max_pooling(x.permute(0,2,1)).permute(0,2,1)) / 2

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        pooled = self.pool(x)
        
        pooled = self.input_bn(pooled.unsqueeze(1)).squeeze(1)
        pooled = self.pre_linear(pooled) + pooled
        pooled = self.layernorm_lstm_start(pooled)
        feature_1, (hn, cn) = self.feature_lstm_1(pooled)
        feature_1 = self.layernorm_lstm_middle(feature_1)
        feature_2, (hn, cn) = self.feature_lstm_2(feature_1, (hn, cn))
        score = torch.sigmoid(self.linear_lstm(feature_2))
        
        score = self.interpolate(score.permute(0,2,1)).permute(0,2,1) + energy

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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def locate_first_and_last_position(self, mask):
        """Locate the first non-negative in a row, and the element before the last non-negative element in a row

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        bs, orig_len, target_len = mask.size()
        
        assert orig_len >= target_len

        weight = torch.tensor([-1.0,1.0]).expand(target_len,-1).to(mask.device)
        weight = weight.unsqueeze(1)
        value = torch.nn.functional.conv1d(mask.permute(0,2,1).float(), weight, bias=None, stride=1, padding=0, dilation=1, groups=target_len)
        value = torch.nn.functional.pad(value, (1,0))
        value = value.permute(0,2,1)
        return value == 1, value == -1

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        
        # Normalize the socre value
        score, total_length = self.score_norm(score, total_length)

        # Monotonic Expansion
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # Get the masked weight
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask

        # Make the sum or each row to one
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        one_minus_weight_sum = 1-weight_sum
        one_minus_weight_sum_cumsum = torch.cumsum(one_minus_weight_sum, dim=2)
        need_minus, need_add = self.locate_first_and_last_position(mask)
        need_minus = need_minus[:,:,1:] * one_minus_weight_sum_cumsum[:,:,:-1]
        need_minus = torch.nn.functional.pad(need_minus,(1,0))
        need_add = need_add * one_minus_weight_sum_cumsum
        weight = weight - need_minus + need_add

        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)

        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret 

# 0.64
class NewAlgoDilatedConv1dMaxPoolScaleChIntp(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NewAlgoDilatedConv1dMaxPoolScaleChIntp, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=3
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        self.pooling = torch.nn.AdaptiveAvgPool1d(self.output_seq_length)
        self.max_pooling = torch.nn.AdaptiveMaxPool1d(self.output_seq_length)
        
        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=1, input_size=self.input_seq_length, kernel_size=5, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def interpolate(self, score):
        return torch.nn.functional.interpolate(score, size=self.input_seq_length, mode='linear')

    def pool(self, x):
        return (self.pooling(x.permute(0,2,1)).permute(0,2,1) + self.max_pooling(x.permute(0,2,1)).permute(0,2,1)) / 2
    
    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        pooled = self.pool(x)
        score = torch.sigmoid(self.model(pooled.permute(0,2,1)).permute(0,2,1))
        score = self.interpolate(score.permute(0,2,1)).permute(0,2,1)
        
        ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        ret['x']=x
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        return ret

    def visualize(self, ret):
        x, y, emb, score, energy,maxpool = ret['x'], ret['feature'], ret['emb'], ret['score'], ret['energy'],ret['feature_maxpool']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(611)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(612)
            plt.plot(energy[i,:,0].detach().cpu().numpy())
            plt.subplot(613)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(614)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(615)
            plt.imshow(maxpool[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(616)
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def locate_first_and_last_position(self, mask):
        """Locate the first non-negative in a row, and the element before the last non-negative element in a row

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        bs, orig_len, target_len = mask.size()
        
        assert orig_len >= target_len

        weight = torch.tensor([-1.0,1.0]).expand(target_len,-1).to(mask.device)
        weight = weight.unsqueeze(1)
        value = torch.nn.functional.conv1d(mask.permute(0,2,1).float(), weight, bias=None, stride=1, padding=0, dilation=1, groups=target_len)
        value = torch.nn.functional.pad(value, (1,0))
        value = value.permute(0,2,1)
        return value == 1, value == -1
    
    # def calculate_max_pool_fast(self, weight, feature):
    #     # weight: [3, 1056, 105]
    #     # feature: [3, 1056, 128]
    #     bs, seqlen, compressed_len = weight.size()
    #     bs, seqlen, mel_bins = feature.size()
    #     expanded_feature = feature.unsqueeze(2).expand(bs, seqlen, compressed_len, mel_bins)
    #     expanded_feature = (expanded_feature * weight.unsqueeze(-1)).permute(0,2,1,3)
    #     tensor = torch.nn.functional.max_pool2d(expanded_feature, kernel_size=(self.input_seq_length, 1)).squeeze(2)
    #     return tensor
    
    def calculate_max_pool(self, weight, feature):
        # weight: [3, 1056, 105]
        # feature: [3, 1056, 128]
        tensor_list = []
        for i in range(weight.size(-1)):
            weight_row = weight[:,:,i].unsqueeze(-1)
            tensor = (feature * weight_row).permute(0,2,1)
            tensor_list.append(torch.nn.functional.max_pool1d(tensor, kernel_size=self.input_seq_length).permute(0,2,1))
        return torch.cat(tensor_list, dim=1)

    def calculate_max_pool_slow(self, weight, feature):
        # weight: [3, 1056, 105]
        # feature: [3, 1056, 128]
        bs, seqlen, compressed_len = weight.size()
        bs, seqlen, mel_bins = feature.size()
        tensor_list = []
        for i in range(mel_bins):
            expanded_feature = feature[:,:,i:i+1].unsqueeze(2).expand(bs, seqlen, compressed_len, 1)
            expanded_feature = (expanded_feature * weight.unsqueeze(-1)).permute(0,2,1,3)
            tensor = torch.nn.functional.max_pool2d(expanded_feature, kernel_size=(self.input_seq_length, 1)).squeeze(2)
            tensor_list.append(tensor)
        return torch.cat(tensor_list,dim=-1)
    
    def calculate_max_pool_fast_grouped(self, weight, feature):
        # weight: [3, 1056, 105]
        # feature: [3, 1056, 128]
        bs, _, _ = weight.size()
        start_idx = 0
        tensor_list = []
        while(start_idx < bs):
            weight_batch, feature_batch = weight[start_idx:start_idx+1], feature[start_idx:start_idx+1]
            if(weight_batch.size(0) == 0): break
            tensor_list.append(self.calculate_max_pool_slow(weight_batch, feature_batch))
            start_idx += 1
        return torch.cat(tensor_list, dim=0)
    
    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        
        # Normalize the socre value
        score, total_length = self.score_norm(score, total_length)

        # Monotonic Expansion
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # Get the masked weight
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask

        # Make the sum or each row to one
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        one_minus_weight_sum = 1-weight_sum
        one_minus_weight_sum_cumsum = torch.cumsum(one_minus_weight_sum, dim=2)
        need_minus, need_add = self.locate_first_and_last_position(mask)
        need_minus = need_minus[:,:,1:] * one_minus_weight_sum_cumsum[:,:,:-1]
        need_minus = torch.nn.functional.pad(need_minus,(1,0))
        need_add = need_add * one_minus_weight_sum_cumsum
        weight = weight - need_minus + need_add

        # [3, 105, 128]    
        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        tensor_list_maxpool = self.calculate_max_pool(weight/self.preserv_ratio, feature)
        
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        
        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1), tensor_list_maxpool.unsqueeze(1)], dim=1)
        ret['feature_maxpool']=tensor_list_maxpool
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret 
    
# Use DNN
class NewAlgoDilatedConv1dPlusPos(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NewAlgoDilatedConv1dPlusPos, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=1, input_size=self.input_seq_length, kernel_size=5, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        score = torch.sigmoid(self.model(x.permute(0,2,1)).permute(0,2,1))
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

    def locate_first_and_last_position(self, mask):
        """Locate the first non-negative in a row, and the element before the last non-negative element in a row

        Args:
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        bs, orig_len, target_len = mask.size()
        
        assert orig_len >= target_len

        weight = torch.tensor([-1.0,1.0]).expand(target_len,-1).to(mask.device)
        weight = weight.unsqueeze(1)
        value = torch.nn.functional.conv1d(mask.permute(0,2,1).float(), weight, bias=None, stride=1, padding=0, dilation=1, groups=target_len)
        value = torch.nn.functional.pad(value, (1,0))
        value = value.permute(0,2,1)
        return value == 1, value == -1

    def select_feature_fast(self, feature, score, total_length):
        ret = {}
        
        # Normalize the socre value
        score, total_length = self.score_norm(score, total_length)

        # Monotonic Expansion
        cumsum_score = torch.cumsum(score, dim=1)
        cumsum_weight = cumsum_score.expand(feature.size(0), feature.size(1), total_length)
        threshold = torch.arange(0, cumsum_weight.size(-1)).to(feature.device).float()
        smaller_mask = cumsum_weight <= threshold[None, None, ...] + 1
        greater_mask = cumsum_weight > threshold[None, None, ...]
        mask = torch.logical_and(smaller_mask, greater_mask)

        # Get the masked weight
        weight = score.expand(feature.size(0), feature.size(1), total_length)
        weight = weight * mask

        # Make the sum or each row to one
        weight_sum = torch.sum(weight, dim=1, keepdim=True)
        one_minus_weight_sum = 1-weight_sum
        one_minus_weight_sum_cumsum = torch.cumsum(one_minus_weight_sum, dim=2)
        need_minus, need_add = self.locate_first_and_last_position(mask)
        need_minus = need_minus[:,:,1:] * one_minus_weight_sum_cumsum[:,:,:-1]
        need_minus = torch.nn.functional.pad(need_minus,(1,0))
        need_add = need_add * one_minus_weight_sum_cumsum
        weight = weight - need_minus + need_add

        tensor_list = torch.matmul(weight.permute(0,2,1), feature)
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)

        ret['emb'] = pos_emb
        ret['feature'] = tensor_list.unsqueeze(1) + pos_emb.unsqueeze(1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret 


    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(NeuralSamplerLargeEnergyNNFreezePosv4, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
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
            nn.Linear(self.input_dim*2, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.pre_lstm_bn = nn.BatchNorm2d(self.input_dim)
        self.feature_lstm = nn.LSTM(self.input_dim, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.linear_lstm = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.pre_score_bn = nn.BatchNorm2d(self.input_dim)
        self.score_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim // 2, self.input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim // 4, 1),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)
    
        init_gru(self.feature_lstm)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        pre_x = self.pre_linear(x) + x
        pre_x = self.pre_lstm_bn(pre_x.unsqueeze(1).transpose(1,3)).transpose(1,3).squeeze(1)
        
        score, (hn, cn) = self.feature_lstm(pre_x)
        score = self.linear_lstm(score) + pre_x

        score = self.pre_score_bn(score.unsqueeze(1).transpose(1,3)).transpose(1,3).squeeze(1)
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
            axis = torch.logical_and(score < 1-1e-4, score > 1e-4) # TODO here 0.1 or 0.01
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
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret

    def weight_fake_softmax(self, weight, mask):
        alpha = torch.sum(weight, dim=1, keepdim=True)
        return weight/(alpha+1e-8)

# Use large LSTM
class FrameLSTM(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(FrameLSTM, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
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
            nn.Linear(self.input_dim*2, self.input_dim),
        )
        self.feature_lstm = nn.LSTM(self.input_dim, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim*2, self.input_dim),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)
    
        init_gru(self.feature_lstm)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        ret = {}
        ret['x']=x
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        ret['score'],_=self.score_norm(energy, self.output_seq_length)
        ret['score_loss']=torch.tensor([0.0]).to(x.device)
        
        pre_x = self.pre_linear(x) + x
        score, (hn, cn) = self.feature_lstm(pre_x)
        score = self.score_linear(score) + pre_x
        ret['feature'] = score[:,::int(self.input_seq_length/self.output_seq_length),:].unsqueeze(1)
        # ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        return ret

    def visualize(self, ret):
        x, y, score, energy = ret['x'], ret['feature'], ret['score'], ret['energy']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(411)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(412)
            plt.plot(energy[i,:,0].detach().cpu().numpy())
            plt.subplot(413)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

# Use large LSTM
class MappingDNN(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(MappingDNN, self).__init__()
        self.input_dim=128; self.mean=mean; self.std=std
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.compressed_length = self.input_seq_length-self.output_seq_length
        self.use_pos_emb = True

        self.pre_linear = nn.Sequential(
            nn.Linear(self.input_seq_length, self.output_seq_length + self.compressed_length),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_seq_length + self.compressed_length, self.output_seq_length + self.compressed_length // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_seq_length + self.compressed_length // 2, self.output_seq_length + self.compressed_length // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_seq_length + self.compressed_length // 4, self.output_seq_length),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        ret = {}
        ret['x']=x
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        ret['score'],_=self.score_norm(energy, self.output_seq_length)
        ret['score_loss']=torch.tensor([0.0]).to(x.device)
        pre_x = self.pre_linear(x.permute(0,2,1)).permute(0,2,1)
        ret['feature'] = pre_x.unsqueeze(1)
        # ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        return ret

    def visualize(self, ret):
        x, y, score, energy = ret['x'], ret['feature'], ret['score'], ret['energy']
        y = y[:,0,:,:] # Ignore the positional embedding on drawing the feature
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(411)
            plt.plot(score[i,:,0].detach().cpu().numpy())
            plt.subplot(412)
            plt.plot(energy[i,:,0].detach().cpu().numpy())
            plt.subplot(413)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(414)
            plt.imshow(y[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
            for i in range(score.size(0)):
                if(distance_with_target_length[i] >= 1):
                    intervel = 1.0-score[i][axis[i]]
                    alpha = distance_with_target_length[i] / torch.sum(intervel) 
                    if(alpha > 1): alpha=1
                    score[i][axis[i]] += intervel * alpha
        ####################################################################
        return score, total_length

class DoNothing(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False, mean=-7.4106, std=6.3097):
        super(DoNothing, self).__init__()
        self.mean = mean
        self.std = std
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = False
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
    def forward(self, x):
        ret={}
        magnitude = torch.sum(((x*self.std)+self.mean).exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        ret['score'],_=self.score_norm(energy, self.output_seq_length)
        
        feature = x.unsqueeze(1)
        ret['score_loss']=torch.tensor([0.0]).to(x.device)
        ret['feature']=feature
        ret['x']=x
        return ret

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
                axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
                for i in range(score.size(0)):
                    if(distance_with_target_length[i] >= 1):
                        intervel = 1.0-score[i][axis[i]]
                        alpha = distance_with_target_length[i] / torch.sum(intervel) 
                        if(alpha > 1): alpha=1
                        score[i][axis[i]] += intervel * alpha
            ####################################################################
            return score, total_length

    def visualize(self, ret):
        x, y = ret['x'], ret['feature']
        import matplotlib.pyplot as plt
        for i in range(10):
            if(i >= x.size(0)): break
            plt.figure(figsize=(6, 8))
            plt.subplot(211)
            plt.imshow(x[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(212)
            plt.imshow(y[i,0,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            path = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
            plt.savefig(os.path.join(path, "%s.png" % i))
            plt.close()

def test_sampler(sampler, data=None):
    input_tdim = 1056
    sampler = sampler(input_seq_length=input_tdim, preserve_ratio=0.5)
    if(data is None): test_input = torch.rand([3, input_tdim, 128])
    else: test_input = data
    ret =sampler(test_input)
    assert "score" in ret.keys()
    assert "score_loss" in ret.keys()
    assert "energy" in ret.keys()
    assert "feature" in ret.keys()
    sampler.visualize(ret)
    print("Perfect!", sampler, ret["feature"].size(), ret["score_loss"].size(), ret["score_loss"])
    return ret["feature"]

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
    data = torch.rand([3, 1056, 128])
    # test_feature_single()
    # test_select_feature()
    # test_sampler(FrameLSTM)
    
    out1 = test_sampler(NSDilatedConv1dMaxPool, data=data)
    
    import ipdb; ipdb.set_trace()
    


    # test_sampler(NeuralSamplerNoFakeSoftmax)                                                # Better than NeuralSampler
    # test_sampler(NeuralSamplerPosEmbLearnable)                                              # Better than NeuralSampler

    # test_sampler(NoAction)
    