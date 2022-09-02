from sklearn import metrics
import numpy
import numpy as np
import pickle

from sklearn.covariance import graphical_lasso

from tqdm import tqdm
import os
import warnings
import sklearn
import torch

GRAPH_WEIGHT = None

def calculate_class_weight_reverse(target, graph_weight_path, beta=1):
    global GRAPH_WEIGHT
    if(GRAPH_WEIGHT is None):
        GRAPH_WEIGHT = torch.tensor(np.load(graph_weight_path), requires_grad=False).float(); 
        if(torch.cuda.is_available()): GRAPH_WEIGHT = GRAPH_WEIGHT.cuda()
        GRAPH_WEIGHT = (GRAPH_WEIGHT/torch.max(GRAPH_WEIGHT))
    # Get the distance between each class and samples
    weight = torch.matmul(target, GRAPH_WEIGHT**beta) 

    # Normalize the max value to 1.0; Remove this line will degrade the mAP from 0.22 to 0.17
    weight = weight/torch.max(weight, dim=1, keepdim=True)[0] # TODO do we need this?
    weight[target > 0] = 1.0
    weight = 1 - weight 
    weight = weight / torch.mean(weight)
    return weight

def calculate_class_weight(target, graph_weight_path, beta=1):
    global GRAPH_WEIGHT
    if(GRAPH_WEIGHT is None):
        GRAPH_WEIGHT = torch.tensor(np.load(graph_weight_path), requires_grad=False).float(); 
        if(torch.cuda.is_available()): GRAPH_WEIGHT = GRAPH_WEIGHT.cuda()
        GRAPH_WEIGHT = (GRAPH_WEIGHT/torch.max(GRAPH_WEIGHT))
    # Get the distance between each class and samples
    weight = torch.matmul(target, GRAPH_WEIGHT**beta) 

    # Normalize the max value to 1.0; Remove this line will degrade the mAP from 0.22 to 0.17
    weight = weight/torch.max(weight, dim=1, keepdim=True)[0] # TODO do we need this?
    weight[target > 0] = 1.0
    return weight / torch.mean(weight)

def calculate_class_weight_min(target, graph_weight_path, beta=1):
    # Target: [132, 527]
    # GRAPH_WEIGHT: [527, 527]
    global GRAPH_WEIGHT
    if(GRAPH_WEIGHT is None):
        GRAPH_WEIGHT = torch.tensor(np.load(graph_weight_path), requires_grad=False).float(); 
        if(torch.cuda.is_available()): GRAPH_WEIGHT = GRAPH_WEIGHT.cuda()
        GRAPH_WEIGHT = (GRAPH_WEIGHT/torch.max(GRAPH_WEIGHT))
        
    # Get the distance between each class and samples
    graph_weight = GRAPH_WEIGHT ** beta
    weight = []
    for i in range(target.shape[0]):
        res = target[i:i+1] * graph_weight
        res[res == 0] = torch.inf # res==0 means the element is not in the target
        weight.append(torch.min(res, dim=1)[0].unsqueeze(0))
    weight = torch.cat(weight, dim=0)
    
    # If the target only have one class, the weight on that class will be inf
    weight[weight == torch.inf] = 0.0 
    
    # Normalize the weight value
    weight = weight/torch.max(weight, dim=1, keepdim=True)[0] 
    weight[target > 0] = 1.0
    weight = weight / torch.mean(weight)
    return weight

def test_class_weight(index):

    graph_weight_path = "/mnt/fast/nobackup/scratch4weeks/hl01486/project/psla/egs/audioset/undirected_graph_connectivity_no_root.npy"
    target = torch.zeros((1, 527)).cuda()
    target[0,index] = 1.0
    
    weight_1 = calculate_class_weight_min(target, graph_weight_path=graph_weight_path, beta=0.1)
    weight_3 = calculate_class_weight_min(target, graph_weight_path=graph_weight_path, beta=0.5)
    weight_5 = calculate_class_weight_min(target, graph_weight_path=graph_weight_path, beta=0.9)
    weight_7 = calculate_class_weight_min(target, graph_weight_path=graph_weight_path, beta=1.3)
    weight_9 = calculate_class_weight_min(target, graph_weight_path=graph_weight_path, beta=2.0)
    
    # weight_1 = calculate_class_weight(target, graph_weight_path=graph_weight_path, beta=0.1)
    # weight_3 = calculate_class_weight(target, graph_weight_path=graph_weight_path, beta=0.5)
    # weight_5 = calculate_class_weight(target, graph_weight_path=graph_weight_path, beta=0.9)
    # weight_7 = calculate_class_weight(target, graph_weight_path=graph_weight_path, beta=1.3)
    # weight_9 = calculate_class_weight(target, graph_weight_path=graph_weight_path, beta=2.0)
    
    plt.plot(weight_1[0].cpu().numpy())
    # plt.plot(weight_2[0].cpu().numpy())
    plt.plot(weight_3[0].cpu().numpy())
    # plt.plot(weight_4[0].cpu().numpy())
    plt.plot(weight_5[0].cpu().numpy())
    # plt.plot(weight_6[0].cpu().numpy())
    plt.plot(weight_7[0].cpu().numpy())
    # plt.plot(weight_8[0].cpu().numpy())
    plt.plot(weight_9[0].cpu().numpy())
    
    plt.savefig("weight_beta_%s.png" % index)
    plt.close()
    # import ipdb; ipdb.set_trace()
    
    
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test_class_weight(0)