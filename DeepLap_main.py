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
import random
from numpy.matlib import repmat
from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertTokenizer,BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import AdamW
import os
from collections import OrderedDict
##########################################################
file=codecs.open("./data/Yeast_cc/yeast_new.txt",'r','utf-8')#protein sequence
line = file.readline()
feature={}
while line:
    if(line.find(">sp")==0):
        a=line[4:-7]
    else :
        feature[a]=list(line[:-1])
    line = file.readline()
print(len(feature))

LableFile = './data/Yeast_cc/Yeast_ccNew.mat'#protein functional annotations
rGOmat = scio.loadmat(LableFile)
funGO = rGOmat['training_label']
t_funGO = rGOmat['testing_label']
train_p=rGOmat['training_protein']
test_p=rGOmat['testing_protein']
train_s=rGOmat['training_score']
test_s=rGOmat['testing_score']
print(funGO.shape)
del rGOmat

mcol=funGO.sum(axis=0)
id_col=[k for k in range(funGO.shape[1]) if mcol[k]<25]#GO terms,cc:25,mf:25,bp:150
funGO = np.delete(funGO, id_col, 1)
print(funGO.shape)
t_funGO = np.delete(t_funGO, id_col, 1)
print(t_funGO.shape)

training_data1 = []
for i in range(0, funGO.shape[0]):
    a=str(train_p[0][i])
    training_data1.append((feature[a[2:-2]],funGO[i]))
training_data1 = np.array(training_data1)
print(training_data1.shape)

test_data = []
for i in range(0, t_funGO.shape[0]):
    a=str(test_p[0][i])
    test_data.append((feature[a[2:-2]],t_funGO[i]))
test_data = np.array(test_data)

training_data, val_data = train_test_split(training_data1, test_size=0.1, random_state=42)
print(training_data.shape)
print(val_data.shape)
print(test_data.shape)


transMatFile = './data/Yeast_cc/cc_Yeast_Linsim.mat' #taxonomic similarity of GO term 

filepro = scio.loadmat(transMatFile)
transmat1 = np.array(filepro['transMat'])
funnum=transmat1.shape[0]

batchsize=32
classnum = funGO.shape[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda
###

def labelbert(labernum):
    GOnameFile = './data/Yeast_cc/cc_Yeast_GOnames.mat'  #name text of GO term 
    filepro = scio.loadmat(GOnameFile)
    GOnamemat = np.array(filepro['sel_GONames'])
    GOfeature=[]
    tokenizer = BertTokenizer.from_pretrained('./biobert_v1.1_pubmed_v2.5.1_convert/vocab.txt', do_lower_case=True)
    for i in range(0,GOnamemat.shape[1]):
        a=GOnamemat[0,i][0]
        text = tokenizer.tokenize(a)
        text = ["[CLS]"] + text + ["[SEP]"] 
        text_id=tokenizer.convert_tokens_to_ids(text)
        text_id=torch.tensor(text_id).unsqueeze(0)
        GOfeature.append(text_id)
    numword=np.max([len(ii[0,]) for ii in GOfeature])
    main_matrix = np.zeros((labernum, numword), dtype= np.int)
    print(numword)
    for i in range(main_matrix.shape[0]):
        num_len=len(GOfeature[i][0,])
        main_matrix[i,0:num_len]=GOfeature[i][0,]
    return main_matrix

GOXfile = labelbert(funnum)
GOXfile = np.array(GOXfile)

bert=BertModel.from_pretrained('./biobert_v1.1_pubmed_v2.5.1_convert')
bert=bert.to(DEVICE)

GOXfile = torch.from_numpy(GOXfile)
GOXfile=GOXfile.to(DEVICE)
with torch.no_grad():
    bert_output=bert(GOXfile,output_all_encoded_layers=False)

def proSeqToOnehot(proseq):
    dict = {
    'A': '100000000000000000000',
    'G': '010000000000000000000',
    'V': '001000000000000000000',
    'I': '000100000000000000000',
    'L': '000010000000000000000',
    'F': '000001000000000000000',
    'P': '000000100000000000000',
    'Y': '000000010000000000000',
    'M': '000000001000000000000',
    'T': '000000000100000000000',
    'S': '000000000010000000000',
    'H': '000000000001000000000',
    'N': '000000000000100000000',
    'Q': '000000000000010000000',
    'W': '000000000000001000000',
    'R': '000000000000000100000',
    'K': '000000000000000010000',
    'D': '000000000000000001000',
    'E': '000000000000000000100',
    'C': '000000000000000000010',
    'X': '000000000000000000001',
    'B': '000000000000000000000',
    'U': '000000000000000000000'}
    proOnehot = []
    AlaOnehot = []
    for Ala in proseq:
        if dict[Ala]:
            for item in dict[Ala]:
                if item=='1':
                    AlaOnehot.append(1.0)
                else:
                    AlaOnehot.append(0.0)
            AlaOnehotcopy = AlaOnehot[:]
            AlaOnehot.clear()
            proOnehot.append(AlaOnehotcopy)
    return proOnehot

def preprocessing(data):
    append = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    new_data = []
    lenlist = []
    for sqe, label in data:
        lenlist.append(len(sqe))
        new_data.append((sqe, label))
    new_data = np.array(new_data)
    sortlen = sorted(range(len(lenlist)), key=lambda k: lenlist[k])
    new_data = new_data[sortlen]

    batch_data = []
    for start_ix in range(0, len(new_data) - batchsize + 1, batchsize):
        thisblock = new_data[start_ix:start_ix + batchsize]
        mybsize = len(thisblock)
        numsqe = 2000
        pro_seq = []
        for i in range(mybsize):
            pro_onehot = proSeqToOnehot(thisblock[i][0])
            if len(thisblock[i][0]) >= numsqe:
                pro_onehotcopy = pro_onehot[0:2000]
            else:
                pro_onehotcopy = pro_onehot[:]
                for i in range(2000 - len(pro_onehot)):
                    appendcopy = append[:]
                    pro_onehotcopy.append(appendcopy)
            pro_seq.append(pro_onehotcopy)
        pro_seq = np.array(pro_seq)
        yyy = []
        for ii in thisblock:
            yyy.append(ii[1])
        yyy = np.array(yyy)
        batch_data.append((autograd.Variable(torch.FloatTensor(pro_seq)), autograd.Variable(torch.FloatTensor(yyy))))
    return batch_data

batchtraining_data = preprocessing(training_data)
batchtest_data = preprocessing(test_data)
batchval_data = preprocessing(val_data)


######################################################################
#Graph
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
#model
Embeddingsize=21
hidden_dim=200

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1 = nn.Conv1d(in_channels=21, out_channels=hidden_dim, kernel_size=8)  # acid 21  21*2000

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(Embeddingsize, hidden_dim)
        self.hidden = self.init_hidden()
        self.H = nn.Linear(hidden_dim, classnum)
        self.final = nn.Linear(hidden_dim, classnum)

        self.fc1 = nn.Linear(120, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.BN = nn.BatchNorm1d(120, momentum=0.5)
        self.dro = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.ff = nn.Linear(classnum,classnum)
        
        self.in_channel = classnum #the number of node
        self.gc1 = GraphConvolution(hidden_dim, 512)
        self.gc2 = GraphConvolution(512, hidden_dim)
        self.gc3 = GraphConvolution(hidden_dim, hidden_dim)
        self.transmat = transmat1
        self.pooled_output=bert_output[0]#all_encoder_layers,pooled_output
        self.transmat = torch.from_numpy(self.transmat).float()
        self.transmat = self.transmat.to(DEVICE)
        self.pooled_output=self.pooled_output.to(DEVICE)
        self.relu = nn.LeakyReLU(0.2)
        
        self.convs1 = nn.Conv1d(768,40,3)
        self.convs2 = nn.Conv1d(768,40,4)
        self.convs3 = nn.Conv1d(768,40,5)
        self.embed_drop = nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        #Normalization

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, batchsize, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(1, batchsize, self.hidden_dim)).cuda())

    def forward(self,x,):
        x = x.transpose(1,2)
        x = self.embed_drop(x)
        out1 = self.conv1(x).transpose(1,2)
        lstm_out = out1

        thisembeddings=self.pooled_output
        thisembeddings = self.embed_drop(thisembeddings)
        thisembeddings=thisembeddings.transpose(1, 2)
        output1=self.tanh(self.convs1(thisembeddings))
        output1=nn.MaxPool1d(output1.size()[2])(output1)
        output2=self.tanh(self.convs2(thisembeddings))
        output2=nn.MaxPool1d(output2.size()[2])(output2)
        output3=self.tanh(self.convs3(thisembeddings))
        output3=nn.MaxPool1d(output3.size()[2])(output3)
        output4 = torch.cat([output1,output2,output3], 1).squeeze(2)

        out = self.BN(output4)
        out = self.fc1(out)
        out = self.dro(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)

        alpha = out.matmul(lstm_out.transpose(1, 2))
        alpha = F.softmax(alpha, dim=2)
        m = alpha.matmul(lstm_out)

        gcout = self.gc1(out, self.transmat)
        gcout = self.relu(gcout)
        gcout = self.gc2(gcout, self.transmat)
        gcout = self.relu(gcout)
        gcout = self.gc3(gcout, self.transmat)

        output1 = gcout.mul(m).sum(dim=2).add(self.final.bias)
        x = self.ff(output1)
        x = torch.sigmoid(x)
        return x


topk = 10
def trainmodel(model):
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
            #optimizer.zero_grad()
            model.hidden = model.init_hidden()
            targets = mysentence[1].cuda()
            tag_scores = model(mysentence[0].cuda())
            loss = loss_function(tag_scores, targets)
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
            tag_scores = model(inputs[0].cuda())

            loss = loss_function(tag_scores, targets)

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


model = ConvNet()
model.cuda()

loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
basemodel = trainmodel(model)
# torch.save(basemodel, 'DeepLap_model')

def testmodel(modelstate):
    model = ConvNet()
    model.cuda()
    model.load_state_dict(modelstate)
    loss_function = nn.BCELoss()
    model.eval()
    recall = []
    lossestest = []

    y_true = []
    y_scores = []

    for inputs in batchtest_data:
        model.hidden = model.init_hidden()
        targets = inputs[1].cuda()

        tag_scores = model(inputs[0].cuda())

        loss = loss_function(tag_scores, targets)

        targets = targets.data.cpu().numpy()
        tag_scores = tag_scores.data.cpu().numpy()

        lossestest.append(loss.data.mean())
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
    y_true = y_true.T
    y_scores = y_scores.T
    np.save('./results/y_true_DeepLap_Yeast_cc', y_true)
    np.save('./results/y_scores_DeepLap_Yeast_cc', y_scores)
    temptrue = []
    tempscores = []
    for col in range(0, len(y_true)):
        if np.sum(y_true[col]) != 0:
            temptrue.append(y_true[col])
            tempscores.append(y_scores[col])
    temptrue = np.array(temptrue)
    tempscores = np.array(tempscores)
    y_true = temptrue.T
    y_scores = tempscores.T
    y_pred = (y_scores > 0.5).astype(np.int)
    print('top-', topk, np.mean(recall))

print('DeepLap_model alone: ')
testmodel(basemodel)
