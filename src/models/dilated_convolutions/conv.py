#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Tuple
import torch
from torch import Tensor, channel_shuffle
from torch.nn import Module, Conv2d, BatchNorm2d, ReLU

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['DilatedConvBLock']


class DilatedConvBLock(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_size: Tuple[int, int],
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 dilation: Union[int, Tuple[int, int]]) \
            -> None:
        """Dilated convolution block.

        :param in_channels: Amount of input channels.
        :type in_channels: int
        :param out_channels: Amount of output channels.
        :type out_channels: int
        :param kernel_size: Kernel shape.
        :type kernel_size: int|(int, int)
        :param stride: Stride shape.
        :type stride: int|(int, int)
        :param padding: Padding shape.
        :type padding: int|(int, int)
        :param dilation: Dilation shape.
        :type dilation: int|(int, int)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.input_size = input_size
        padding = (self.get_padding_bins(input_size[0], self.dilation[0]), self.get_padding_bins(input_size[1], self.dilation[1]))
        # print(padding)
        self.cnn1 = Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride, padding=padding,
                          dilation=dilation, bias=True)

        self.batch_norm1 = BatchNorm2d(
            num_features=out_channels)

        self.non_linearity = ReLU()

        self.cnn2 = Conv2d(in_channels=out_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride, padding=padding,
                          dilation=dilation, bias=True)

        self.batch_norm2 = BatchNorm2d(
            num_features=out_channels)

    def get_padding_bins(self, input_length, dilations):
        return int((input_length*(self.stride-1)-self.stride+dilations*(self.kernel_size-1)+1) / 2)

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the dilated\
        convolution block.

        :param x: Input.
        :type x: torch.Tensor
        :return: Output.
        :rtype: torch.Tensor
        """
        x = self.batch_norm1(self.non_linearity(self.cnn1(x)))       
        x = self.batch_norm2(self.non_linearity(self.cnn2(x)))+x
        return x

class DilatedConv(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 layers: int,
                 channel_increase_rate: float,
                 dilation_rate: int,
                 input_size: Tuple[int, int],
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]]) \
            -> None:
        """Dilated convolution block.

        :param in_channels: Amount of input channels.
        :type in_channels: int
        :param out_channels: Amount of output channels.
        :type out_channels: int
        :param kernel_size: Kernel shape.
        :type kernel_size: int|(int, int)
        :param stride: Stride shape.
        :type stride: int|(int, int)
        :param padding: Padding shape.
        :type padding: int|(int, int)
        :param dilation: Dilation shape.
        :type dilation: int|(int, int)
        """
        super().__init__()
        
        channel_increase_rate=channel_increase_rate

        self.intermediate_blocks_up = torch.nn.ModuleList()
        self.intermediate_blocks_down = torch.nn.ModuleList()

        self.up_blocks_input_channels = None
        self.down_blocks_output_channels = None

        for i in range(layers-2):
            if(i == 0): self.up_blocks_input_channels=int(channel_increase_rate**(i+3))
            self.intermediate_blocks_up.append(DilatedConvBLock(int(channel_increase_rate**(i+3)), int(channel_increase_rate**(i+4)), kernel_size=kernel_size, input_size=input_size, stride=stride, dilation=(dilation_rate**(i+1), 2)))

        down_i = list(range(layers-2))[::-1]
        
        for id, i in enumerate(down_i):
            if(id==len(down_i)-1): self.down_blocks_output_channels = int(channel_increase_rate**(i+3))
            self.intermediate_blocks_down.append(DilatedConvBLock(int(channel_increase_rate**(i+4)), int(channel_increase_rate**(i+3)), kernel_size=kernel_size, input_size=input_size, stride=stride, dilation=(dilation_rate**(i+1), 2)))

        self.in_block = DilatedConvBLock(in_channels, self.up_blocks_input_channels, kernel_size=kernel_size, input_size=input_size, stride=stride, dilation=(1,1))
        self.out_block = DilatedConvBLock(self.down_blocks_output_channels, out_channels, kernel_size=kernel_size, input_size=input_size, stride=stride, dilation=(1,1))

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the dilated\
        convolution block.

        :param x: Input.
        :type x: torch.Tensor
        :return: Output.
        :rtype: torch.Tensor
        """
        
        x = self.in_block(x)
        for blk in self.intermediate_blocks_up:
            x = blk(x) 
        for blk in self.intermediate_blocks_down:
            x = blk(x)
        x = self.out_block(x)
        return x

if __name__ == "__main__":
    from thop import clever_format
    from thop import profile

    import torch
    model = DilatedConv(1, 1, layers=6, dilation_rate=3, channel_increase_rate=2, kernel_size=3, input_size=(1056, 128), stride=1)
    data = torch.randn((3, 1, 1056, 128))
    print(model)
    res = model(data)
    flops, params = profile(model, inputs=(data, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(res.size(), flops, params)

    import ipdb; ipdb.set_trace()