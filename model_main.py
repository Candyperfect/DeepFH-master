import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  
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
import os
from collections import OrderedDict
import random
import pickle
import esm
import tqdm
import hdf5storage
import mat73

from model_DeepLAP_pretrain_gcn import Deepour_pretrain_gcn
##########################################################
species=['Yeast','Human']
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
GOnames=['cc','mf','bp']
numGOs=[25,25,150]
batchsize=4

def get_prot_fea_transformer12(prot_seq_list):
    n_prot = len(prot_seq_list)
    model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    prot_fea_list = []
    n_batch = 2
    n_step = math.ceil(n_prot / n_batch)
    for i in tqdm.tqdm(range(n_step)):
        if i == n_step:
            buf_list = prot_seq_list[i * n_batch:]
        else:
            buf_list = prot_seq_list[i * n_batch:(i + 1) * n_batch]

        batch_seq_list = []
        for j in range(len(buf_list)):
            batch_seq_list.append(('protein{}'.format(j + 1), buf_list[j]))

        batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq_list)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])
        token_embeddings = results['representations'][12]
        for j, (_, seq) in enumerate(batch_seq_list):
            prot_fea_list.append(token_embeddings[j, 1:len(seq) + 1].mean(0).numpy())
    return prot_fea_list


for i in species:
    SeqFile = './data/'+i+'_new.txt'#protein sequences
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

    file2='./data/'+i+'_Scores.mat'#sequence score
    Score = hdf5storage.loadmat(file2)['scores']#scio hdf5storage
    for jj in range(0,3,1):
        print(i+'_'+GOnames[jj])
        LableFile = './data/'+i+'NGOA_R.mat'#label annotation
        filepro = hdf5storage.loadmat(LableFile)
        file3=GOnames[jj]+'Labels'
        label = np.array(filepro[file3])
        for ii in range(0, Score.shape[0]):
            Score[ii,ii]=0

        mcol=label.sum(axis=0)
        id_col=[k for k in range(label.shape[1]) if mcol[k]< numGOs[jj]]#cc:25,mf:25,bp:150
        label = np.delete(label, id_col, 1)
        print('label',label.shape)

        mcol=label.sum(axis=1)
        id_train=[k for k in range(label.shape[0]) if mcol[k]>0]

        Score_trainnew = sparse.lil_matrix(sparse.csc_matrix(Score)[:,id_train])
        print('Score_trainnew',Score_trainnew.shape)

        datatrain1 = []
        protein_idx={}
        input_seq=[]
        for j in range(0, len(feature)):
            if j in id_train:
                a=list(feature.keys())[j]
                protein_idx[a]=len(protein_idx)
                datatrain1.append((protein_idx[a],a,feature[a],Score_trainnew[j],label[j]))
                input_seq.append(feature[a])
        datatrain1 = np.array(datatrain1)
        print('datatrain1', datatrain1.shape)
        prot_fea = get_prot_fea_transformer12(input_seq)
        prot_fea = torch.FloatTensor(prot_fea)
        file4 = './data/'+i+'_'+GOnames[jj] + '_seqfeature'
        torch.save(prot_fea,file4)
        print('prot_fea', prot_fea.shape)
        Deepour_pretrain_gcn(datatrain1,GOnames[jj],i+'_',batchsize,prot_fea)


