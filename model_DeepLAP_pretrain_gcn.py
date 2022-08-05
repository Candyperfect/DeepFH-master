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
import torch.nn.functional as F
import codecs
from numpy.matlib import repmat
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#from transformers import BertModel, BertTokenizer, AdamW, BertConfig

from pytorch_pretrained_bert import BertTokenizer,BertModel
from pytorch_pretrained_bert.optimization import BertAdam
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
import torch.utils.data as Data
from torch.autograd import Variable
##########################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda
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


class MyDataSet(Data.Dataset):
    def __init__(self, data,prot_fea):
        # print('data',data.shape)
        target_labels = []
        for id, name, sqe, score, label in data:
            target_labels.append(label)
        self.input_seq = torch.FloatTensor(prot_fea)
        self.target_labels = torch.FloatTensor(target_labels)
    
    def __len__(self):
        return len(self.input_seq)
 
    def __getitem__(self, idx):
        return self.input_seq[idx], self.target_labels[idx]


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
        out = x
        out = self.fc1(out)
        outgo = self.GOX
        att, emb1, com1, com2, emb2, emb= self.SFGCN(outgo, self.transmat, self.score)
        emb = emb.transpose(0, 1)
        output1 = torch.matmul(out, emb)
        x1 = self.ff(output1)
        x1 = torch.sigmoid(x1)
        return x1,att, emb1, com1, com2, emb2, emb.transpose(0, 1)


topk = 10

def trainmodel(model,train_loader,loss_function,optimizer,GOterm,AA,accumulation_steps = 8):
    print('start_training')
    for i in range(10):
        model.train()
        model.zero_grad()
        attention=[]
        for batch_idx, (input_seq, target_labels) in enumerate(train_loader):
            input_seq, target_labels = Variable(torch.FloatTensor(input_seq)).to(DEVICE), Variable(torch.FloatTensor(target_labels)).to(DEVICE)

            tag_scores,att, emb1, com1, com2, emb2, emb = model(input_seq)
            loss_class = loss_function(tag_scores,target_labels)
            loss_dep = (loss_dependence(emb1, com1, classnum) + loss_dependence(emb2, com2, classnum))/2
            loss_com = common_loss(com1,com2)
            loss = loss_class + 5e-10 * loss_dep + 0.001 * loss_com
            attention.append(att)
        # 梯度积累
            loss = loss/accumulation_steps
            loss.backward()

            if((batch_idx+1) % accumulation_steps) == 0:
            # 每 4 次更新一下网络中的参数
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()

            if ((batch_idx+1) % accumulation_steps) == 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    i+1, batch_idx, len(train_loader), 100. *
                    batch_idx/len(train_loader), loss.item()
                ))
    file4 = './resultatt/'+GOterm+'_'+AA+ 'att'
    torch.save(attention,file4)
    return copy.deepcopy(model.state_dict())

def testmodel(modelstate,test_loader,GOterm,fold,AA,train_index):
    model = ConvNet()
    model.cuda()
    model.load_state_dict(modelstate)
    #loss_function = nn.BCELoss()
    model.eval()
    recall = []

    y_true = []
    y_scores = []

    for batch_idx, (input_seq, target_labels) in enumerate(test_loader):

        input_seq, target_labels = torch.FloatTensor(input_seq).to(DEVICE), torch.FloatTensor(target_labels).to(DEVICE)

        tag_scores,_,_,_,_,_,_ = model(input_seq)
        # loss = loss_function(tag_scores,target_labels)

        targets = target_labels.data.cpu().numpy()
        tag_scores = tag_scores.data.cpu().numpy()

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
    print('test top-', topk, np.mean(recall))
    y_true = np.concatenate(y_true, axis=0)
    y_scores = np.concatenate(y_scores, axis=0)
    y_true = y_true.T
    y_scores = y_scores.T
    scio.savemat(file_name="./results/Deepour_pretrain_gcn"+"_fold:"+str(fold)+"_"+AA+GOterm+'.mat', mdict={'real':y_true,'predicted':y_scores,'idx':train_index})

def simTO(labels):
    # labels=labels.T
    print('labels',labels.shape)
    Tosim=torch.zeros(labels.size(0),labels.size(0))
    sumMat=labels.sum(axis=1)
    trans_Labels=labels.T
    for i in range(labels.size(0)):
        if sumMat[i]>0:
            cal=torch.nonzero(labels[i,:])
            cal=cal.squeeze(1)
            tempMat=torch.matmul(labels[i,cal],trans_Labels[cal,:]).unsqueeze(0)
            repii=[]
            repii.extend([sumMat[i]]*labels.size(0))
            repii=torch.FloatTensor(repii).unsqueeze(0)
            all=torch.cat([tempMat,repii],dim=0)
            allmax=torch.max(all,dim=0)
            tempMat=tempMat.squeeze(0)
            maxMat=allmax.values
            Tosim[i,:]=tempMat.mul(1/maxMat)
    return Tosim



def Deepour_pretrain_gcn(data,GOterm,AA,batchsize,prot_fea):
    global classnum
    classnum = data[0][4].shape[0]

    global hidden_dim
    hidden_dim = data[0][4].shape[0]

    GOXfile1 = labelOneHot(classnum)
    global GOXfile
    GOXfile = np.array(GOXfile1)

    label_data = []
    for id,name,sqe,sco,label in data:
        label_data.append(label)
    label_data = np.array(label_data)

    kf_5 = KFold(n_splits=5, shuffle=True, random_state=0)
    fold = 0
    for train_index, test_index in kf_5.split(data):
        train_loader = Data.DataLoader(MyDataSet(data[train_index],prot_fea[train_index]),
                                       batch_size=batchsize, shuffle=True, drop_last=True)
        test_loader = Data.DataLoader(MyDataSet(data[test_index],prot_fea[test_index]),
                                      batch_size=batchsize, shuffle=True, drop_last=True)
        print('processing data ok ')
        transMatFile = './data/'+GOterm+'_'+AA+'fold='+str(fold+1)+'_Linsim.mat' ##########
        filepro = scio.loadmat(transMatFile)
        transmat = np.array(filepro['dagMat'])
        global transmat1
        transmat1=transmat
        x=torch.FloatTensor(label_data[train_index]).transpose(0, 1)
        s=F.cosine_similarity(x.unsqueeze(1),x.unsqueeze(0),dim=2)
        # s=simTO(x)
        print('labels cosine similarity',s.shape)
        global score
        score=s
        model = ConvNet()
        model.cuda()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(),lr=0.0003)
        basemodel = trainmodel(model,train_loader,loss_function,optimizer,GOterm,AA)
        print('Deepour_pretrain_gcn_model alone: ',fold)
        testmodel(basemodel,test_loader,GOterm,fold,AA,train_index)
        fold = fold + 1
        torch.cuda.empty_cache()
        # exit()



# if __name__=="__main__":









