import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
torch.manual_seed(1)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import copy
import math
from torch.nn import Parameter
import scipy.io as scio
from scipy import sparse
import torch.nn.functional as F
import codecs
from numpy.matlib import repmat
from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertTokenizer,BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import AdamW
import os
from collections import OrderedDict
import random
import pickle
from model_DeepFH import DeepFH
##########################################################
file=codecs.open("./data/yeast_new.txt",'r','utf-8')
line = file.readline()
feature={}
while line:
    if(line.find(">sp")==0):
        a=line[4:-7]
    else :
        feature[a]=list(line[:-1])
    line = file.readline()
print(len(feature))

GOnames=['cc','mf','bp']
numGOs=[25,25,150]

for i in range(0,3,1):
    print('Yeast_'+GOnames[i])
    LableFile = './data/Yeast_'+GOnames[i]+'New.mat'
    rGOmat = scio.loadmat(LableFile)
    funGO = rGOmat['training_label']
    t_funGO = rGOmat['testing_label']
    train_p=rGOmat['training_protein']
    test_p=rGOmat['testing_protein']
    train_s=rGOmat['training_score']
    for ii in range(0, train_s.shape[0]):
        train_s[ii,ii]=0
    test_s=rGOmat['testing_score']
    print(funGO.shape)
    del rGOmat

    mcol=funGO.sum(axis=0)
    id_col=[k for k in range(funGO.shape[1]) if mcol[k]< numGOs[i]]#cc:25,mf:25,bp:150
    funGO = np.delete(funGO, id_col, 1)
    print(funGO.shape)
    t_funGO = np.delete(t_funGO, id_col, 1)
    print(t_funGO.shape)

    training_data1 = []
    protein_idx={}
    for j in range(0, funGO.shape[0]):
        a=str(train_p[0][j])
        protein_idx[a[2:-2]]=len(protein_idx)
        training_data1.append((protein_idx[a[2:-2]],feature[a[2:-2]],funGO[j],train_s[j]))
    training_data1 = np.array(training_data1)
    print(training_data1.shape)

    test_data = []
    protein_idxt={}
    for k in range(0, t_funGO.shape[0]):
        a=str(test_p[0][k])
        protein_idxt[a[2:-2]]=len(protein_idxt)
        test_data.append((protein_idxt[a[2:-2]],feature[a[2:-2]],t_funGO[k],test_s[k]))
    test_data = np.array(test_data)

    transMatFile = './data/'+GOnames[i]+'_Yeast_Linsim.mat' ##########
    filepro = scio.loadmat(transMatFile)
    transmat1 = np.array(filepro['transMat'])

    GOnameFile = './data/'+GOnames[i]+'_Yeast_GOnames.mat'  ##########
    DeepFH(training_data1,test_data,GOnames[i],funGO,transmat1,GOnameFile,'Yeast_')



