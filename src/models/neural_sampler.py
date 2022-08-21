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

from .HigherModels import *
from .neural_sampler import *
from .pooling import Pooling_layer

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
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerLargeEnergy, self).__init__()
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

class NeuralSamplerAvgPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerAvgPool, self).__init__()
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = False
        self.pooling = Pooling_layer(pooling_type="avg", factor=preserve_ratio)
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
    def forward(self, x):
        ret={}
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        ret['score'],_=self.score_norm(energy, self.output_seq_length)
        feature = self.pooling(x.unsqueeze(1))
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


# Use DNN
class NewAlgoDilatedConv1dPlusEnergyv2(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NewAlgoDilatedConv1dPlusEnergyv2, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=2, input_size=1056, kernel_size=3, stride=1)
        self.relu = nn.ReLU(inplace=True)
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        energy,_=self.score_norm(energy, self.output_seq_length)

        score = torch.sigmoid(self.model(x.permute(0,2,1)).permute(0,2,1)) * energy

        ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
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

# Use DNN
class NewAlgoEnergy(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NewAlgoEnergy, self).__init__()
        self.input_dim=128
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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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

# 0.35
class BaselineAdaAvgPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(BaselineAdaAvgPool, self).__init__()
        self.input_dim=128
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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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
    
# TODO
class BaselineAdaAvgMaxPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(BaselineAdaAvgMaxPool, self).__init__()
        self.input_dim=128
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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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
    
    
# Use DNN
class NewAlgoDilatedConv1dPlusEnergy(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NewAlgoDilatedConv1dPlusEnergy, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=2, input_size=1056, kernel_size=3, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        energy,_=self.score_norm(energy, self.output_seq_length)

        score = torch.sigmoid(self.model(x.permute(0,2,1)).permute(0,2,1)) 
        ret['score']=score
        ret = self.select_feature_fast(x, score+energy, total_length=self.output_seq_length)
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

# TODO
class BaselineConstantScoreMaxPoolScale(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(BaselineConstantScoreMaxPoolScale, self).__init__()
        self.input_dim=128
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

    def calculate_max_pool(self, weight, feature):
        # weight: [3, 1056, 105]
        # feature: [3, 1056, 128]
        tensor_list = []
        for i in range(weight.size(-1)):
            weight_row = weight[:,:,i].unsqueeze(-1)
            tensor = (feature * weight_row).permute(0,2,1)
            tensor_list.append(torch.nn.functional.max_pool1d(tensor, kernel_size=self.input_seq_length).permute(0,2,1))
        return torch.cat(tensor_list, dim=1)

    def calculate_max_pool_fast(self, weight, feature):
        # weight: [3, 1056, 105]
        # feature: [3, 1056, 128]
        bs, seqlen, compressed_len = weight.size()
        bs, seqlen, mel_bins = feature.size()
        expanded_feature = feature.unsqueeze(2).expand(bs, seqlen, compressed_len, mel_bins)
        expanded_feature = (expanded_feature * weight.unsqueeze(-1)).permute(0,2,1,3)
        tensor = torch.nn.functional.max_pool2d(expanded_feature, kernel_size=(self.input_seq_length, 1)).squeeze(2)
        return tensor
    
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
    
    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)

        score = torch.ones_like(x[...,0:1]).to(x.device)
        
        ret = self.select_feature_fast(x, score, total_length=self.output_seq_length)
        ret['x']=x
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        return ret

    def visualize(self, ret):
        x, y, emb, score, energy, feature_maxpool = ret['x'], ret['feature'], ret['emb'], ret['score'], ret['energy'], ret['feature_maxpool']
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
            plt.imshow(emb[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
            plt.subplot(616)
            plt.imshow(feature_maxpool[i,...].detach().cpu().numpy().T, aspect="auto", interpolation='none')
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
        
        # Ceil the weight, if greater than 0, be 1
        tensor_list_maxpool = self.calculate_max_pool_fast_grouped(weight / self.preserv_ratio, feature)
        tensor_list = (tensor_list + tensor_list_maxpool) / 2
        
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)

        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1)], dim=1)
        ret['feature_maxpool'] = tensor_list_maxpool
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return rets

# 0.64
class NewAlgoDilatedConv1dMaxPoolCeil(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NewAlgoDilatedConv1dMaxPoolCeil, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=2, input_size=self.input_seq_length, kernel_size=3, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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
        tensor_list_maxpool = self.calculate_max_pool_fast_grouped(weight > 0, feature)
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
    
# 0.64
class NewAlgoDilatedConv1dMaxPoolScaleCh(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NewAlgoDilatedConv1dMaxPoolScaleCh, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=3
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=2, input_size=self.input_seq_length, kernel_size=3, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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
        tensor_list_maxpool = self.calculate_max_pool_fast_grouped(weight/self.preserv_ratio, feature)
        # tensor_list_maxpool = self.calculate_max_pool_fast(weight, feature)
        # tensor_list_maxpool = self.calculate_max_pool_fast_efficient(weight, feature)
        tensor_list = tensor_list
        
        pos_emb = torch.matmul(weight.permute(0,2,1), self.pos_emb)
        
        ret['emb'] = pos_emb
        ret['feature'] = torch.cat([tensor_list.unsqueeze(1), pos_emb.unsqueeze(1), tensor_list_maxpool.unsqueeze(1)], dim=1)
        ret['feature_maxpool']=tensor_list_maxpool
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret 
    
# 0.64
class NewAlgoDilatedConv1dMaxPoolScale(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NewAlgoDilatedConv1dMaxPoolScale, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=2, input_size=self.input_seq_length, kernel_size=3, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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
        tensor_list_maxpool = self.calculate_max_pool_fast_grouped(weight/self.preserv_ratio, feature)
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

# 0.64
class NewAlgoDilatedConv1dMaxPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NewAlgoDilatedConv1dMaxPool, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=2, input_size=self.input_seq_length, kernel_size=3, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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
    
# Use DNN
class BaselineConstantScore(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(BaselineConstantScore, self).__init__()
        self.input_dim=128
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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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

# Use DNN
class NeuralSamplerDilatedConv1d(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerDilatedConv1d, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=2, input_size=1056, kernel_size=3, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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

class NewAlgoDilatedConv1d(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NewAlgoDilatedConv1d, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=2, input_size=1056, kernel_size=3, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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

# Use DNN
class NewAlgoDilatedConv1dPlusPos(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NewAlgoDilatedConv1dPlusPos, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=2, input_size=self.input_seq_length, kernel_size=3, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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

class NewAlgoDilatedConv1dPlusPosScalePos(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NewAlgoDilatedConv1dPlusPosScalePos, self).__init__()
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        from models.dilated_convolutions_1d.conv import DilatedConv
        self.model = DilatedConv(in_channels=self.input_dim, dilation_rate=2, input_size=self.input_seq_length, kernel_size=3, stride=1)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding with scale and bias")
            self.scale = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
            self.bias = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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
        pos_emb = torch.matmul(weight.permute(0,2,1), (self.pos_emb * self.scale) + self.bias)

        ret['emb'] = pos_emb
        ret['feature'] = tensor_list.unsqueeze(1) + pos_emb.unsqueeze(1)
        ret['score_loss'] = torch.mean(torch.std(score, dim=1))
        ret['score']=score
        return ret 
    

class NeuralSamplerAvgMaxPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerAvgMaxPool, self).__init__()
        self.feature_channels=1
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = False
        self.pooling = Pooling_layer(pooling_type="avg-max", factor=preserve_ratio)
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=False)
    
    def forward(self, x):
        ret={}
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        ret['energy'],_=self.score_norm(energy, self.output_seq_length)
        ret['score'],_=self.score_norm(energy, self.output_seq_length)
        feature = self.pooling(x.unsqueeze(1))
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

# Use extra large lstm model
class NeuralSamplerLargeEnergyNNFreezePosv2(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerLargeEnergyNNFreezePosv2, self).__init__()
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
            nn.Linear(self.input_dim*2, self.input_dim),
        )
        self.feature_lstm = nn.LSTM(self.input_dim, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.linear_lstm = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.input_dim),
        )
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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        pre_x = self.pre_linear(x) + x

        score, (hn, cn) = self.feature_lstm(pre_x)
        score = self.linear_lstm(score) + pre_x

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

# Use extra large lstm model
class NeuralSamplerLargeEnergyNNFreezePosv2LayerNorm(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerLargeEnergyNNFreezePosv2LayerNorm, self).__init__()
        self.input_dim=128
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
            nn.Linear(self.input_dim*2, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim*2, self.input_dim),
        )

        self.layernorm_lstm_start = nn.LayerNorm([self.input_seq_length, self.input_dim])
        self.feature_lstm_1 = nn.LSTM(self.input_dim, self.latent_dim*2, self.num_layers // 2, batch_first=True, bidirectional=True)

        self.layernorm_lstm_middle = nn.LayerNorm([self.input_seq_length, self.latent_dim*4])

        self.feature_lstm_2 = nn.LSTM(self.latent_dim*4, self.latent_dim*2, self.num_layers // 2, batch_first=True, bidirectional=True)
        self.layernorm_lstm_end = nn.LayerNorm([self.input_seq_length, self.latent_dim*4])
        self.linear_lstm = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.input_dim),
        )
        
        self.score_bn = nn.BatchNorm2d(1)
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
    
        init_gru(self.feature_lstm_1)
        init_gru(self.feature_lstm_2)
        self.init_seq_linear(self.pre_linear)
        self.init_seq_linear(self.linear_lstm)
        self.init_seq_linear(self.score_linear)
        init_bn(self.input_bn)
        init_bn(self.score_bn)

    def init_seq_linear(self, sequencial):
        for i in range(len(sequencial)):
            if(isinstance(sequencial[i], nn.Linear)):
                init_layer(sequencial[i])

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        x = self.input_bn(x.unsqueeze(1)).squeeze(1)
        pre_x = self.pre_linear(x) + x

        pre_x = self.layernorm_lstm_start(pre_x)
        feature_1, (hn, cn) = self.feature_lstm_1(pre_x)

        feature_1 = self.layernorm_lstm_middle(feature_1)

        feature_2, (hn, cn) = self.feature_lstm_2(feature_1, (hn, cn))

        feature_2 = self.layernorm_lstm_end(feature_2) + feature_1

        score = self.linear_lstm(feature_2) + pre_x
        score = self.score_bn(score.unsqueeze(1)).squeeze(1)
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

# Use extra large lstm model
class NeuralSamplerLargeEnergyNNFreezePosv2LayerNormDropout(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerLargeEnergyNNFreezePosv2LayerNormDropout, self).__init__()
        self.input_dim=128
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
            nn.Dropout(0.5),
            nn.Linear(self.input_dim*2, self.input_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.input_dim*2, self.input_dim),
        )
        
        self.layernorm_lstm_start = nn.LayerNorm([self.input_seq_length, self.input_dim])
        self.feature_lstm_1 = nn.LSTM(self.input_dim, self.latent_dim*2, self.num_layers // 2, batch_first=True, bidirectional=True)

        self.layernorm_lstm_middle = nn.LayerNorm([self.input_seq_length, self.latent_dim*4])

        self.feature_lstm_2 = nn.LSTM(self.latent_dim*4, self.latent_dim*2, self.num_layers // 2, batch_first=True, bidirectional=True)
        self.layernorm_lstm_end = nn.LayerNorm([self.input_seq_length, self.latent_dim*4])
        self.linear_lstm = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.input_dim),
        )
        
        self.score_bn = nn.BatchNorm2d(1)
        self.score_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.input_dim // 2, self.input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.input_dim // 4, 1),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)
    
        init_gru(self.feature_lstm_1)
        init_gru(self.feature_lstm_2)
        self.init_seq_linear(self.pre_linear)
        self.init_seq_linear(self.linear_lstm)
        self.init_seq_linear(self.score_linear)
        init_bn(self.input_bn)
        init_bn(self.score_bn)

    def init_seq_linear(self, sequencial):
        for i in range(len(sequencial)):
            if(isinstance(sequencial[i], nn.Linear)):
                init_layer(sequencial[i])

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        x = self.input_bn(x.unsqueeze(1)).squeeze(1)
        pre_x = self.pre_linear(x) + x

        pre_x = self.layernorm_lstm_start(pre_x)
        feature_1, (hn, cn) = self.feature_lstm_1(pre_x)

        feature_1 = self.layernorm_lstm_middle(feature_1)

        feature_2, (hn, cn) = self.feature_lstm_2(feature_1, (hn, cn))

        feature_2 = self.layernorm_lstm_end(feature_2) + feature_1

        score = self.linear_lstm(feature_2) + pre_x
        score = self.score_bn(score.unsqueeze(1)).squeeze(1)
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

# Use extra large lstm model
class NeuralSamplerLargeEnergyNNFreezePosv3(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerLargeEnergyNNFreezePosv3, self).__init__()
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
            nn.Linear(self.input_dim*2, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.pre_lstm_bn = nn.BatchNorm2d(1)
        self.feature_lstm = nn.LSTM(self.input_dim, self.latent_dim*2, self.num_layers, batch_first=True, bidirectional=True)
        self.linear_lstm = nn.Sequential(
            nn.Linear(self.latent_dim*4, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.pre_score_bn = nn.BatchNorm2d(1)
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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        pre_x = self.pre_linear(x) + x
        pre_x = self.pre_lstm_bn(pre_x.unsqueeze(1)).squeeze(1)
        
        score, (hn, cn) = self.feature_lstm(pre_x)
        score = self.linear_lstm(score) + pre_x

        score = self.pre_score_bn(score.unsqueeze(1)).squeeze(1)
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

# Use extra large lstm model
class NeuralSamplerLargeEnergyNNFreezePosv4(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerLargeEnergyNNFreezePosv4, self).__init__()
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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(FrameLSTM, self).__init__()
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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(MappingDNN, self).__init__()
        self.input_dim=128
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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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

# Use large LSTM
class NeuralSamplerLargeEnergyNNFreezePos(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerLargeEnergyNNFreezePos, self).__init__()
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)
    
        init_gru(self.feature_lstm)

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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
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

# Use small LSTM
class NeuralSamplerEnergyNNFreezePos(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerEnergyNNFreezePos, self).__init__()
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
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_lstm = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.score_linear = nn.Sequential(
            nn.Linear(self.latent_dim*2, self.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, 1),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)
    
        init_gru(self.feature_lstm)

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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
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

# Use DNN
class NeuralSamplerDNNFreezePosInit(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerDNNFreezePosInit, self).__init__()
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
            nn.Linear(self.input_dim*2, self.input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim, 1),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

        self.init_seq_linear(self.pre_linear)
    
    def init_seq_linear(self, sequencial):
        for i in range(len(sequencial)):
            if(isinstance(sequencial[i], nn.Linear)):
                init_layer(sequencial[i])
                
    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
        
        score = torch.sigmoid(self.pre_linear(x))
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

# Changing the step size
class NeuralSamplerUniformPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerUniformPool, self).__init__()
        self.feature_channels=1

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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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

# Using the pooling method
class NeuralSamplerSpecPool(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerSpecPool, self).__init__()
        self.feature_channels=1
        
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.use_pos_emb = False
        self.pooling = Pooling_layer(pooling_type="spec", factor=preserve_ratio)
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
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
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
        ret['score_loss']=torch.tensor([0.0]).to(x.device)
        return ret

    def visualize(self, ret):
        return

# Standard model
class NeuralSamplerPosEmbLearnableLargeEnergyNN(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
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
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)
    
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
            axis = torch.logical_and(score < RESCALE_INTERVEL_MAX, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
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

class NeuralSamplerNNResUNet6_0_0625(nn.Module):
    def __init__(self, input_seq_length, preserve_ratio, alpha=1.0, learn_pos_emb=False):
        super(NeuralSamplerNNResUNet6_0_0625, self).__init__()
        from models.unet.resunet import UNetResComplex_100Mb_6
        self.input_dim=128
        self.latent_dim=64
        self.feature_dim=128
        self.num_layers=2
        self.feature_channels=2
        self.preserv_ratio=preserve_ratio
        self.input_seq_length = input_seq_length
        self.output_seq_length = int(self.input_seq_length * self.preserv_ratio)
        self.use_pos_emb = True

        self.unet = UNetResComplex_100Mb_6(channels=1, running_mean=self.input_dim, scale=0.0625)
        self.score_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_dim // 2, 1),
        )
        
        if(self.use_pos_emb):
            emb_dropout=0.0
            logging.info("Use positional embedding")
            pos_emb_y = PositionalEncoding(d_model=self.input_dim, dropout=emb_dropout, max_len=self.input_seq_length)(torch.zeros((1,self.input_seq_length, self.input_dim))) 
            self.pos_emb = nn.Parameter(pos_emb_y, requires_grad=learn_pos_emb)

    def forward(self, x):
        # torch.Size([96, 1056, 128])
        magnitude = torch.sum(x.exp(), dim=2, keepdim=True)
        energy = magnitude/torch.max(magnitude)
    
        unet_x = self.unet(x.unsqueeze(1)).squeeze(1) + x
        score = torch.sigmoid(self.score_linear(unet_x))
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
            axis = torch.logical_and(score < 0.99, score > RESCALE_INTERVEL_MIN) # TODO here 0.1 or RESCALE_INTERVEL_MIN
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

def test_sampler(sampler):
    input_tdim = 1056
    sampler = sampler(input_seq_length=input_tdim, preserve_ratio=0.1)
    test_input = torch.rand([3, input_tdim, 128])
    ret =sampler(test_input)
    assert "score" in ret.keys()
    assert "score_loss" in ret.keys()
    assert "energy" in ret.keys()
    assert "feature" in ret.keys()
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
    # test_sampler(FrameLSTM)
    test_sampler(BaselineAdaAvgMaxPool)
    test_sampler(BaselineConstantScoreMaxPoolCeil)
    test_sampler(NewAlgoDilatedConv1dMaxPool)
    test_sampler(NewAlgoDilatedConv1dMaxPoolv2Ch)
    test_sampler(NewAlgoDilatedConv1dMaxPoolv2)
    test_sampler(NewAlgoDilatedConv1dMaxPoolv3Ceil)


    # test_sampler(NeuralSamplerNoFakeSoftmax)                                                # Better than NeuralSampler
    # test_sampler(NeuralSamplerPosEmbLearnable)                                              # Better than NeuralSampler

    # test_sampler(NoAction)
    