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
from transformers import AdamW
import os
from collections import OrderedDict
import random
import pickle
from model_DeepLAP_pretrain_gcn import Deepour_pretrain_gcn
##########################################################
GOnames=['cc','mf','bp']
numGOs=[25,25,150]
species=['yeast_','human_']

for i in range(0,2,1):
    SeqFile = './data/'+species[i]+'new.txt'#序列
    file=codecs.open(SeqFile,'r','utf-8')
    line = file.readline()
    feature={}
    while line:
        if(line.find(">sp")==0):
            a=line[4:-7]
        else :
            feature[a]=line[:-1]
        line = file.readline()
    print('proteins seq number',len(feature))
    for j in range(1,2,1):
        print(species[i]+GOnames[j])
        LableFile = './data/'+species[i]+GOnames[j]+'.mat'#标签
        rGOmat = scio.loadmat(LableFile)
        funGO = rGOmat['training_label']
        t_funGO = rGOmat['testing_label']
        train_p=rGOmat['training_protein']
        test_p=rGOmat['testing_protein']
        train_s=rGOmat['training_score']
        for ii in range(0, train_s.shape[0]):
             train_s[ii,ii]=0
        test_s=rGOmat['testing_score']
        print('raw train label',funGO.shape)
        del rGOmat

        mcol=funGO.sum(axis=0)
        id_col=[k for k in range(funGO.shape[1]) if mcol[k]< numGOs[j]]#cc:25,mf:25,bp:150
        funGO = np.delete(funGO, id_col, 1)
        print('delete train label',funGO.shape)
        t_funGO = np.delete(t_funGO, id_col, 1)
        print('delete test label',t_funGO.shape)

        x=torch.FloatTensor(funGO).transpose(0, 1)
        s=F.cosine_similarity(x.unsqueeze(1),x.unsqueeze(0),dim=2)
        print('labels cosine similarity',s.shape)

        training_data1 = []
        protein_idx={}
        for q in range(0, funGO.shape[0]):
            a=str(train_p[0][q])
            protein_idx[a[2:-2]]=len(protein_idx)
            training_data1.append((protein_idx[a[2:-2]],feature[a[2:-2]],funGO[q],train_s[q]))
        training_data1 = np.array(training_data1)
        print(training_data1.shape)

        test_data = []
        protein_idxt={}
        for p in range(0, t_funGO.shape[0]):
            a=str(test_p[0][p])
            protein_idxt[a[2:-2]]=len(protein_idxt)
            test_data.append((protein_idxt[a[2:-2]],feature[a[2:-2]],t_funGO[p],test_s[p]))
        test_data = np.array(test_data)
        print(test_data.shape)

        transMatFile = './data/'+GOnames[j]+'_'+species[i]+'Linsim.mat' ##########
        filepro = scio.loadmat(transMatFile)
        transmat1 = np.array(filepro['transMat'])
        dagMat1 = np.array(filepro['dagMat'])

        Deepour_pretrain_gcn(training_data1,test_data,GOnames[j],funGO,dagMat1,species[i],s)

