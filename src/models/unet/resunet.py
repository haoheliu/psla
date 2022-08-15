import sys
import os
sys.path.append(os.getcwd())

import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import magphase
import numpy as np
import torch


def act(x, activation):
    if activation == 'relu':
        return F.relu_(x)

    elif activation == 'leaky_relu':
        return F.leaky_relu_(x, negative_slope=0.2)

    elif activation == 'swish':
        return x * torch.sigmoid(x)

    else:
        raise Exception('Incorrect activation!')

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


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, size, activation, momentum):
        super(ConvBlockRes, self).__init__()

        self.activation = activation
        if(type(size) == type((3,4))):
            pad = size[0] // 2
            size = size[0]
        else:
            pad = size // 2
            size = size

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(size, size), stride=(1, 1),
                               dilation=(1, 1), padding=(pad, pad), bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        # self.abn1 = InPlaceABN(num_features=in_channels, momentum=momentum, activation='leaky_relu')

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(size, size), stride=(1, 1),
                               dilation=(1, 1), padding=(pad, pad), bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        # self.abn2 = InPlaceABN(num_features=out_channels, momentum=momentum, activation='leaky_relu')

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, x):
        origin = x
        x = self.conv1(F.leaky_relu_(self.bn1(x), negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(origin) + x
        else:
            return origin + x

class DecoderBlockRes1(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum):
        super(DecoderBlockRes1, self).__init__()
        size = 3
        self.activation = activation
        self.stride = stride

        self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=(size, size), stride=stride,
            padding=(0, 0), output_padding=(0, 0), bias=False, dilation=(1, 1))

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_block2 = ConvBlockRes(out_channels * 2, out_channels, size, activation, momentum)
        
    def init_weights(self):
        init_layer(self.conv1)

    def prune(self, x, size):
        """Prune the shape of x after transpose convolution.
        """
        if self.stride == (1, 2):
            x = x[:, :, 1 : -1, 0 : - 1]
        if self.stride == (3, 3):
            x = torch.nn.functional.pad(x, (0, size[-1]-x.size(-1)))
        else:
            x = x[:, :, 0 : - 1, :]
        return x

    def forward(self, input_tensor, concat_tensor):
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        x = self.prune(x, concat_tensor.size())
        # import ipdb; ipdb.set_trace()
        try:
            x = torch.cat((x, concat_tensor), dim=1)
        except:
            print("error", x.size(), concat_tensor.size())
            exit(0)
        x = self.conv_block2(x)
        return x

class EncoderBlockRes1(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum):
        super(EncoderBlockRes1, self).__init__()
        size = 3

        self.conv_block1 = ConvBlockRes(in_channels, out_channels, size, activation, momentum)
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block1(x)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder

class UNetResComplex_100Mb_4_stride_3(nn.Module):
    def __init__(self, channels, running_mean=128, scale=0.25):
        # sub4 52.041G 66.272M
        super(UNetResComplex_100Mb_4_stride_3, self).__init__()

        activation = 'relu'
        momentum = 0.01
        stride=3
        self.downsample_ratio = stride ** 4
        self.bn0 = nn.BatchNorm2d(running_mean, momentum=momentum)

        self.encoder_block1 = EncoderBlockRes1(in_channels=channels, out_channels=int(scale*2**5),
                                             downsample=(stride, stride), activation=activation, momentum=momentum)
        self.encoder_block2 = EncoderBlockRes1(in_channels=int(scale*2**5), out_channels=int(scale*2**6),
                                             downsample=(stride, stride), activation=activation, momentum=momentum)
        self.encoder_block3 = EncoderBlockRes1(in_channels=int(scale*2**6), out_channels=int(scale*2**7),
                                             downsample=(stride, stride), activation=activation, momentum=momentum)
        self.encoder_block4 = EncoderBlockRes1(in_channels=int(scale*2**7), out_channels=int(scale*2**8),
                                             downsample=(stride, stride), activation=activation, momentum=momentum)
        self.conv_block7 = EncoderBlockRes1(in_channels=int(scale*2**8), out_channels=int(scale*2**9),
                                           downsample=(1,1), activation=activation, momentum=momentum)
        self.decoder_block3 = DecoderBlockRes1(in_channels=int(scale*2**9), out_channels=int(scale*2**8),
                                              stride=(stride, stride), activation=activation, momentum=momentum)
        self.decoder_block4 = DecoderBlockRes1(in_channels=int(scale*2**8), out_channels=int(scale*2**7),
                                              stride=(stride, stride), activation=activation, momentum=momentum)
        self.decoder_block5 = DecoderBlockRes1(in_channels=int(scale*2**7), out_channels=int(scale*2**6),
                                              stride=(stride, stride), activation=activation, momentum=momentum)
        self.decoder_block6 = DecoderBlockRes1(in_channels=int(scale*2**6), out_channels=int(scale*2**5),
                                                 stride=(stride, stride), activation=activation, momentum=momentum)

        self.after_conv_block1 = EncoderBlockRes1(in_channels=int(scale*2**5), out_channels=int(scale*2**5), downsample=(1,1),
                                                  activation=activation, momentum=momentum)

        self.after_conv2 = nn.Conv2d(in_channels=int(scale*2**5), out_channels=channels,
                                     kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        # sp, cos_in, sin_in = self.f_helper.wav_to_spectrogram_phase(input)
        # Batch normalization
        x = input.transpose(1, 3)
        """(batch_size, freq_bins, time_steps, channels_num)"""
        x = self.bn0(x)  # normalization to freq bins
        """(batch_size, freq_bins, time_steps, channels_num)"""
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]  # time_steps
        pad_len = int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0: x.shape[-1] - 1]  # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        
        x_center,_ = self.conv_block7(x4_pool)  # (bs, 2048, T / 64, F / 64)

        x9 = self.decoder_block3(x_center, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)
        x,_ = self.after_conv_block1(x12)  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels, T, F)
        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0: origin_len, :]
        return x
    
class UNetResComplex_100Mb_4(nn.Module):
    def __init__(self, channels, running_mean=128, scale=0.25):
        # sub4 52.041G 66.272M
        super(UNetResComplex_100Mb_4, self).__init__()

        activation = 'relu'
        momentum = 0.01
        self.downsample_ratio = 2 ** 4
        self.bn0 = nn.BatchNorm2d(running_mean, momentum=momentum)

        self.encoder_block1 = EncoderBlockRes1(in_channels=channels, out_channels=int(scale*2**5),
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block2 = EncoderBlockRes1(in_channels=int(scale*2**5), out_channels=int(scale*2**6),
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block3 = EncoderBlockRes1(in_channels=int(scale*2**6), out_channels=int(scale*2**7),
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block4 = EncoderBlockRes1(in_channels=int(scale*2**7), out_channels=int(scale*2**8),
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.conv_block7 = EncoderBlockRes1(in_channels=int(scale*2**8), out_channels=int(scale*2**9),
                                           downsample=(1,1), activation=activation, momentum=momentum)
        self.decoder_block3 = DecoderBlockRes1(in_channels=int(scale*2**9), out_channels=int(scale*2**8),
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block4 = DecoderBlockRes1(in_channels=int(scale*2**8), out_channels=int(scale*2**7),
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block5 = DecoderBlockRes1(in_channels=int(scale*2**7), out_channels=int(scale*2**6),
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block6 = DecoderBlockRes1(in_channels=int(scale*2**6), out_channels=int(scale*2**5),
                                                 stride=(2, 2), activation=activation, momentum=momentum)

        self.after_conv_block1 = EncoderBlockRes1(in_channels=int(scale*2**5), out_channels=int(scale*2**5), downsample=(1,1),
                                                  activation=activation, momentum=momentum)

        self.after_conv2 = nn.Conv2d(in_channels=int(scale*2**5), out_channels=channels,
                                     kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        # sp, cos_in, sin_in = self.f_helper.wav_to_spectrogram_phase(input)
        # Batch normalization
        x = input.transpose(1, 3)
        """(batch_size, freq_bins, time_steps, channels_num)"""
        x = self.bn0(x)  # normalization to freq bins
        """(batch_size, freq_bins, time_steps, channels_num)"""
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]  # time_steps
        pad_len = int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0: x.shape[-1] - 1]  # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        
        x_center,_ = self.conv_block7(x4_pool)  # (bs, 2048, T / 64, F / 64)

        x9 = self.decoder_block3(x_center, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)
        x,_ = self.after_conv_block1(x12)  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels, T, F)
        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0: origin_len, :]
        return x

class UNetResComplex_100Mb_5(nn.Module):
    def __init__(self, channels, running_mean=128, scale=0.25):
        # sub4 52.041G 66.272M
        super(UNetResComplex_100Mb_5, self).__init__()

        activation = 'relu'
        momentum = 0.01
        self.downsample_ratio = 2 ** 5
        self.bn0 = nn.BatchNorm2d(running_mean, momentum=momentum)

        self.encoder_block1 = EncoderBlockRes1(in_channels=channels, out_channels=int(scale*2**5),
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block2 = EncoderBlockRes1(in_channels=int(scale*2**5), out_channels=int(scale*2**6),
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block3 = EncoderBlockRes1(in_channels=int(scale*2**6), out_channels=int(scale*2**7),
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block4 = EncoderBlockRes1(in_channels=int(scale*2**7), out_channels=int(scale*2**8),
                                             downsample=(2, 2), activation=activation, momentum=momentum)

        self.encoder_block5 = EncoderBlockRes1(in_channels=int(scale*2**8), out_channels=int(scale*2**9),
                                             downsample=(2, 2), activation=activation, momentum=momentum)

        self.conv_block7 = EncoderBlockRes1(in_channels=int(scale*2**9), out_channels=int(scale*2**10),
                                           downsample=(1,1), activation=activation, momentum=momentum)

        self.decoder_block2 = DecoderBlockRes1(in_channels=int(scale*2**10), out_channels=int(scale*2**9),
                                              stride=(2, 2), activation=activation, momentum=momentum)

        self.decoder_block3 = DecoderBlockRes1(in_channels=int(scale*2**9), out_channels=int(scale*2**8),
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block4 = DecoderBlockRes1(in_channels=int(scale*2**8), out_channels=int(scale*2**7),
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block5 = DecoderBlockRes1(in_channels=int(scale*2**7), out_channels=int(scale*2**6),
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block6 = DecoderBlockRes1(in_channels=int(scale*2**6), out_channels=int(scale*2**5),
                                                 stride=(2, 2), activation=activation, momentum=momentum)

        self.after_conv_block1 = EncoderBlockRes1(in_channels=int(scale*2**5), out_channels=int(scale*2**5), downsample=(1,1),
                                                  activation=activation, momentum=momentum)

        self.after_conv2 = nn.Conv2d(in_channels=int(scale*2**5), out_channels=channels,
                                     kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        # sp, cos_in, sin_in = self.f_helper.wav_to_spectrogram_phase(input)
        # Batch normalization
        x = input.transpose(1, 3)
        """(batch_size, freq_bins, time_steps, channels_num)"""
        x = self.bn0(x)  # normalization to freq bins
        """(batch_size, freq_bins, time_steps, channels_num)"""
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]  # time_steps
        pad_len = int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0: x.shape[-1] - 1]  # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(x4_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        x_center,_ = self.conv_block7(x5_pool)  # (bs, 2048, T / 64, F / 64)
        x8 = self.decoder_block2(x_center, x5)  # (bs, 256, T / 8, F / 8)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)
        x,_ = self.after_conv_block1(x12)  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels, T, F)
        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0: origin_len, :]
        return x


class UNetResComplex_100Mb_6(nn.Module):
    def __init__(self, channels, running_mean=128, scale=0.25):
        # sub4 52.041G 66.272M
        super(UNetResComplex_100Mb_6, self).__init__()

        activation = 'relu'
        momentum = 0.01
        self.downsample_ratio = 2 ** 6
        self.bn0 = nn.BatchNorm2d(running_mean, momentum=momentum)

        self.encoder_block1 = EncoderBlockRes1(in_channels=channels, out_channels=int(scale*2**5),
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block2 = EncoderBlockRes1(in_channels=int(scale*2**5), out_channels=int(scale*2**6),
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block3 = EncoderBlockRes1(in_channels=int(scale*2**6), out_channels=int(scale*2**7),
                                             downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block4 = EncoderBlockRes1(in_channels=int(scale*2**7), out_channels=int(scale*2**8),
                                             downsample=(2, 2), activation=activation, momentum=momentum)

        self.encoder_block5 = EncoderBlockRes1(in_channels=int(scale*2**8), out_channels=int(scale*2**9),
                                             downsample=(2, 2), activation=activation, momentum=momentum)

        self.encoder_block6 = EncoderBlockRes1(in_channels=int(scale*2**9), out_channels=int(scale*2**10),
                                             downsample=(2, 2), activation=activation, momentum=momentum)

        self.conv_block7 = EncoderBlockRes1(in_channels=int(scale*2**10), out_channels=int(scale*2**10),
                                           downsample=(1,1), activation=activation, momentum=momentum)

        self.decoder_block1 = DecoderBlockRes1(in_channels=int(scale*2**10), out_channels=int(scale*2**10),
                                              stride=(2, 2), activation=activation, momentum=momentum)

        self.decoder_block2 = DecoderBlockRes1(in_channels=int(scale*2**10), out_channels=int(scale*2**9),
                                              stride=(2, 2), activation=activation, momentum=momentum)

        self.decoder_block3 = DecoderBlockRes1(in_channels=int(scale*2**9), out_channels=int(scale*2**8),
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block4 = DecoderBlockRes1(in_channels=int(scale*2**8), out_channels=int(scale*2**7),
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block5 = DecoderBlockRes1(in_channels=int(scale*2**7), out_channels=int(scale*2**6),
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block6 = DecoderBlockRes1(in_channels=int(scale*2**6), out_channels=int(scale*2**5),
                                                 stride=(2, 2), activation=activation, momentum=momentum)

        self.after_conv_block1 = EncoderBlockRes1(in_channels=int(scale*2**5), out_channels=int(scale*2**5), downsample=(1,1),
                                                  activation=activation, momentum=momentum)

        self.after_conv2 = nn.Conv2d(in_channels=int(scale*2**5), out_channels=channels,
                                     kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.after_conv2)

    def forward(self, input):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        # sp, cos_in, sin_in = self.f_helper.wav_to_spectrogram_phase(input)
        # Batch normalization
        x = input.transpose(1, 3)
        """(batch_size, freq_bins, time_steps, channels_num)"""
        x = self.bn0(x)  # normalization to freq bins
        """(batch_size, freq_bins, time_steps, channels_num)"""
        x = x.transpose(1, 3)
        """(batch_size, chanenls, time_steps, freq_bins)"""

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]  # time_steps
        pad_len = int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0: x.shape[-1] - 1]  # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(x4_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        (x6_pool, x6) = self.encoder_block6(x5_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        x_center,_ = self.conv_block7(x6_pool)  # (bs, 2048, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6)  # (bs, 256, T / 8, F / 8)
        x8 = self.decoder_block2(x7, x5)  # (bs, 256, T / 8, F / 8)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)
        x,_ = self.after_conv_block1(x12)  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels, T, F)
        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0: origin_len, :]
        return x

if __name__ == "__main__":
    from thop import clever_format
    from thop import profile

    model = UNetResComplex_100Mb_4_stride_3(channels=1, scale=0.0625)
    print(model)
    data = torch.randn((1, 1, 1056, 128))
    
    flops, params = profile(model, inputs=(data, ))
    flops, params = clever_format([flops, params], "%.3f")

    print(flops, params)