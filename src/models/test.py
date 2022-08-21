import torch
import os
from functorch import vmap


def calculate_max_pool(weight, feature):
    # weight: [3, 1056, 105]
    # feature: [3, 1056, 128]
    tensor_list = []
    for i in range(weight.size(-1)):
        weight_row = weight[:,:,i].unsqueeze(-1)
        tensor = (feature * weight_row).permute(0,2,1)
        tensor_list.append(torch.nn.functional.max_pool1d(tensor, kernel_size=1056).permute(0,2,1))
    return torch.cat(tensor_list, dim=1)

def calculate_max_pool_fast(weight, feature):
    # weight: [3, 1056, 105]
    # feature: [3, 1056, 128]
    bs, seqlen, compressed_len = weight.size()
    bs, seqlen, mel_bins = feature.size()
    expanded_feature = feature.unsqueeze(2).expand(bs, seqlen, compressed_len, mel_bins)
    expanded_feature = (expanded_feature * weight.unsqueeze(-1)).permute(0,2,1,3)
    tensor = torch.nn.functional.max_pool2d(expanded_feature, kernel_size=(1056, 1)).squeeze(2)
    return tensor

def _max_pool_dimension(weight, feature=None):
    # weight: [3, 1056, 105]
    # feature: [3, 1056, 128]
    weight_row = weight.unsqueeze(-1)
    tensor = (feature * weight_row).permute(0,2,1)
    return torch.max(tensor, dim=-1)[0]

def calculate_max_pool_fast_efficient(weight, feature):
    # weight: [3, 1056, 105]
    # feature: [3, 1056, 128]
    tensor = vmap(_max_pool_dimension, in_dims=-1, out_dims=-1)(weight, feature=feature)
    return tensor.permute(0,2,1)

if __name__ == "__main__":
    # weight = torch.tensor([[[ 0.5091,  0.4272, -1.2534,  0.9485],
    #      [ 0.5488, -0.7695,  0.1448,  0.1251],
    #      [-0.9808,  0.7522,  0.7452, -1.6633],
    #      [-0.0591, -1.0193,  0.8489, -0.9049],
    #      [-0.0410, -1.8466,  1.4684, -0.0456],
    #      [-0.7897, -0.2325,  0.4542, -0.1422],
    #      [ 0.5969,  0.3517, -1.0544,  0.4722],
    #      [-1.5502,  1.1317, -0.6783, -0.1735]]])
    # weight = torch.abs(weight)
    # feature = torch.ones(1,8,2)
    weight = torch.randn((3,1056,105))
    feature = torch.randn((3,1056,128))
    
    res = calculate_max_pool(weight, feature)
    # res2 = calculate_max_pool_fast(weight, feature)
    res2 = calculate_max_pool_fast_efficient(weight, feature)
    
    import ipdb; ipdb.set_trace()
    
    print(torch.mean(torch.abs(res-res2)))
    
    
    import ipdb; ipdb.set_trace()