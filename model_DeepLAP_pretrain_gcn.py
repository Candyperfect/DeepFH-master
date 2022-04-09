import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import copy
import math
from torch.nn import Parameter
import scipy.io as scio
import torch.nn.functional as F
import codecs
from numpy.matlib import repmat
from sklearn.model_selection import train_test_split
#from transformers import BertModel, BertTokenizer, AdamW, BertConfig
from transformers import AdamW
import os
from collections import OrderedDict
import random
from scipy import sparse
import esm
import tqdm
from models import SFGCN
from utils import *
from Evaluation import *
##########################################################
def labelOneHot(labernum):
    sigonehot = []
    labelonehot = []
    for i in range(labernum):
        sigonehot.append(0.0)
    for i in range(labernum):
        sigonehot1 = sigonehot[:]
        sigonehot1[i] = 1.0
        # print(sigonehot1)
        labelonehot.append(sigonehot1)
    return labelonehot


def get_prot_fea_transformer12(prot_seq_list):
    n_prot = len(prot_seq_list)
    model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    prot_fea_list = []
    n_batch = 2
    n_step = math.ceil(n_prot / n_batch)
    for i in tqdm.tqdm(range(n_step)):
        if i == n_step:
            buf_list = prot_seq_list[i*n_batch:]
        else:
            buf_list = prot_seq_list[i*n_batch:(i+1)*n_batch]

        batch_seq_list = []
        for j in range(len(buf_list)):
            batch_seq_list.append(('protein{}'.format(j+1), buf_list[j]))
       
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_seq_list)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[12])
        token_embeddings = results['representations'][12]
        for j, (_, seq) in enumerate(batch_seq_list):
            prot_fea_list.append(token_embeddings[j, 1:len(seq)+1].mean(0).numpy())
    return prot_fea_list


def preprocessing(data,batchsize):
    # append = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    new_data = []
    lenlist = []
    for idx, sqe, label ,score in data:
        lenlist.append(len(sqe))
        new_data.append((sqe, label))
    new_data = np.array(new_data)
    sortlen = sorted(range(len(lenlist)), key=lambda k: lenlist[k])
    new_data = new_data[sortlen]

    batch_data = []
    for start_ix in range(0, len(new_data) - batchsize + 1, batchsize):
        thisblock = new_data[start_ix:start_ix + batchsize]
        mybsize = len(thisblock)
        pro_seq = []
        for i in range(mybsize):
            pro_seq.append(thisblock[i][0])
        pro_seq = np.array(pro_seq)
        prot_fea=get_prot_fea_transformer12(pro_seq)
        yyy = []
        for ii in thisblock:
            yyy.append(ii[1])

        batch_data.append((autograd.Variable(torch.FloatTensor(prot_fea)), autograd.Variable(torch.FloatTensor(yyy))))
    return batch_data

def preprocessing1(data):
    # append = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    new_data = []
    lenlist = []
    for idx, sqe, label, score in data:
        lenlist.append(len(sqe))
        new_data.append((sqe, label))
    new_data = np.array(new_data)
    sortlen = sorted(range(len(lenlist)), key=lambda k: lenlist[k])
    new_data = new_data[sortlen]

    batch_data = []
    for start_ix in range(0,1, len(new_data)):
        thisblock = new_data[start_ix:start_ix + len(new_data)]
        mybsize = len(thisblock)
        pro_seq = []
        for i in range(mybsize):
            pro_seq.append(thisblock[i][0])
        pro_seq = np.array(pro_seq)
        prot_fea=get_prot_fea_transformer12(pro_seq)
        yyy = []
        for ii in thisblock:
            yyy.append(ii[1])

        batch_data.append((autograd.Variable(torch.FloatTensor(prot_fea)), autograd.Variable(torch.FloatTensor(yyy))))
    return batch_data

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.hidden_dim = hidden_dim
        self.BN = nn.BatchNorm1d(768, momentum=0.5)
        self.fc1 = nn.Linear(768, hidden_dim)
        self.GOX = GOXfile
        self.GOX = torch.from_numpy(self.GOX).float()
        self.GOX = self.GOX.to(DEVICE)
        self.in_channel = classnum #the number of node
        self.fc2 = nn.Linear(self.in_channel, hidden_dim)
        self.transmat = transmat1
        self.transmat = torch.from_numpy(self.transmat).float()
        self.transmat = self.transmat.to(DEVICE)
        self.score = score
        # self.score = torch.from_numpy(self.score).float()
        self.score = self.score.to(DEVICE)
        self.SFGCN= SFGCN(nfeat = self.hidden_dim,
              nhid1 = 512,
              nhid2 = hidden_dim,
              dropout = 0.5)
        # self.hidden = self.init_hidden()
        self.ff = nn.Linear(classnum,classnum)
        

    def forward(self,x):
        #sequence feature
        #1
        out = self.BN(x)
        out = self.fc1(out)

        outgo = self.fc2(self.GOX)
        att, emb1, com1, com2, emb2, emb= self.SFGCN(outgo, self.transmat, self.score)
        emb = emb.transpose(0, 1)
        output1 = torch.matmul(out, emb)
        x1 = self.ff(output1)
        x1 = torch.sigmoid(x1)
        return x1,att, emb1, com1, com2, emb2, emb.transpose(0, 1)


topk = 10

def trainmodel(model,batchtraining_data,batchval_data,loss_function,optimizer):
    print('start_training')
    modelsaved = []
    modelperform = []
    topk = 10

    bestresults = -1
    bestiter = -1
    for epoch in range(5000):
        model.train()

        lossestrain = []
        recall = []
        for mysentence in batchtraining_data:
            model.zero_grad()

            # model.hidden = model.init_hidden()
            targets = mysentence[1].cuda()
            tag_scores,att, emb1, com1, com2, emb2, emb = model(mysentence[0].cuda())
            loss_class = loss_function(tag_scores, targets)
            loss_dep = (loss_dependence(emb1, com1, classnum) + loss_dependence(emb2, com2, classnum))/2
            loss_com = common_loss(com1,com2)
            loss = loss_class + 5e-10 * loss_dep + 0.001 * loss_com

            loss.backward()
            optimizer.step()
            lossestrain.append(loss.data.mean())
        print(epoch)
        modelsaved.append(copy.deepcopy(model.state_dict()))
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        model.eval()

        recall = []
        for inputs in batchval_data:

            targets = inputs[1].cuda()
            tag_scores,_,_,_,_,_,_ = model(inputs[0].cuda())

            # loss = loss_function(tag_scores, targets)

            targets = targets.data.cpu().numpy()
            tag_scores = tag_scores.data.cpu().numpy()

            for iii in range(0, len(tag_scores)):
                temp = {}
                for iiii in range(0, len(tag_scores[iii])):
                    temp[iiii] = tag_scores[iii][iiii]
                temp1 = [(k, temp[k]) for k in sorted(temp, key=temp.get, reverse=True)]
                thistop = int(np.sum(targets[iii]))
                hit = 0.0
                for ii in temp1[0:max(thistop, topk)]:
                    if targets[iii][ii[0]] == 1.0:
                        hit = hit + 1
                if thistop != 0:
                    recall.append(hit / thistop)

        print('validation top-', topk, np.mean(recall))

        modelperform.append(np.mean(recall))
        if modelperform[-1] > bestresults:
            bestresults = modelperform[-1]
            bestiter = len(modelperform) - 1

        if (len(modelperform) - bestiter) > 5:
            print(modelperform, bestiter)
            return modelsaved[bestiter]

def testmodel(modelstate,batchtest_data,GOterm,AA):
    model = ConvNet()
    model.cuda()
    model.load_state_dict(modelstate)
    loss_function = nn.BCELoss()
    model.eval()
    recall = []
    # lossestest = []

    y_true = []
    y_scores = []

    for inputs in batchtest_data:
        # model.hidden = model.init_hidden()
        targets = inputs[1].cuda()

        tag_scores,_,_,_,_,_,_ = model(inputs[0].cuda())

        # loss = loss_function(tag_scores, targets)

        targets = targets.data.cpu().numpy()
        tag_scores = tag_scores.data.cpu().numpy()

        # lossestest.append(loss.data.mean())
        y_true.append(targets)
        y_scores.append(tag_scores)

        for iii in range(0, len(tag_scores)):
            temp = {}
            for iiii in range(0, len(tag_scores[iii])):
                temp[iiii] = tag_scores[iii][iiii]
            temp1 = [(k, temp[k]) for k in sorted(temp, key=temp.get, reverse=True)]
            thistop = int(np.sum(targets[iii]))
            hit = 0.0

            for ii in temp1[0:max(thistop, topk)]:
                if targets[iii][ii[0]] == 1.0:
                    hit = hit + 1
            if thistop != 0:
                recall.append(hit / thistop)
    y_true = np.concatenate(y_true, axis=0)
    y_scores = np.concatenate(y_scores, axis=0)
    print(main(y_true, y_scores))
    y_true = y_true.T
    y_scores = y_scores.T
    np.save('./results/y_true_Deepour_pretrain_gcn_'+AA+GOterm, y_true)
    np.save('./results/y_scores_Deepour_pretrain_gcn_'+AA+GOterm, y_scores)


def Deepour_pretrain_gcn(datatrain,datatest,GOterm,funGO,transmat,AA,s):
    training_data, val_data = train_test_split(datatrain, test_size=0.1, random_state=42)
    del datatrain
    batchsize=32
    global classnum
    classnum = funGO.shape[1]

    global hidden_dim
    hidden_dim=200

    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda
    funnum=transmat.shape[0]

    global transmat1
    transmat1=transmat
    GOXfile1 = labelOneHot(funnum)

    global score
    score=s

    global GOXfile
    GOXfile = np.array(GOXfile1)

    batchtraining_data = preprocessing(training_data,batchsize)
    print('traindata ok ')
    batchtest_data = preprocessing1(datatest)
    batchval_data = preprocessing(val_data,batchsize)
    del training_data
    del val_data

    model = ConvNet()
    model.cuda()
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    basemodel = trainmodel(model,batchtraining_data,batchval_data,loss_function,optimizer)
    print('Deepour_pretrain_gcn_model alone: ')
    testmodel(basemodel,batchtest_data,GOterm,AA)



# if __name__=="__main__":









