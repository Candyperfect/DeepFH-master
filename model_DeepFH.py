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

from pytorch_pretrained_bert import BertTokenizer,BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import AdamW
import os
from collections import OrderedDict
import random
from scipy import sparse
##########################################################
def labelbert(labernum,GOnameFile):
    filepro = scio.loadmat(GOnameFile)
    GOnamemat = np.array(filepro['sel_GONames'])
    GOfeature=[]
    tokenizer = BertTokenizer.from_pretrained('/home/xxx/BioBert/biobert_v1.1_pubmed_v2.5.1_convert/vocab.txt', do_lower_case=True)
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
    'U': '000000000000000000000',
    'Z': '000000000000000000000',
    'O': '000000000000000000000'}
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


def preprocessing1(data):
    label_data = []
    idx_data = []
    for idx,feature,label,score in data:
        label_data.append(label)
        idx_data.append(idx)
    label_data = np.array(label_data)
    idx_data = np.array(idx_data)
    return idx_data,label_data

def similiar_score(score,training_idx,Label):
    Feature1 = score[:,training_idx]
    fcol=Feature1.sum(axis=1)
    for i in range(0, len(fcol)):
        if fcol[i]==0:
            fcol[i]=1
    num1=Feature1.shape[1]
    Feature_sum=repmat(fcol,num1,1)
    Feature_sum=Feature_sum.T
    guiyiFeature=Feature1/Feature_sum
    predicted=np.dot(guiyiFeature,Label)
    return predicted

def preprocessing(data,batchsize,training_idx,training_label):
    append = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    new_data = []
    lenlist = []
    for idx, sqe, label,score in data:
        lenlist.append(len(sqe))
        new_data.append((sqe, label,score))
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

        zzz = []
        for ii in thisblock:
            sc=ii[2]
            zzz.append(sc)
        zzz = np.array(zzz)
        zzz_new = similiar_score(zzz,training_idx,training_label)
        batch_data.append((autograd.Variable(torch.FloatTensor(pro_seq)), autograd.Variable(torch.FloatTensor(yyy)),autograd.Variable(torch.FloatTensor(zzz_new))))
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
        self.conv1 = nn.Conv1d(in_channels=21, out_channels=hidden_dim, kernel_size=8)  # acid 21  21*2000

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(Embeddingsize, hidden_dim)
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
        self.pooled_output=bert_output1[0]#all_encoder_layers,pooled_output
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


    def forward(self,x,score):
        #sequence feature
        #1
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
        x = (x+score)/2
        return x


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

            targets = mysentence[1].cuda()
            tag_scores = model(mysentence[0].cuda(),mysentence[2].cuda())
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
            tag_scores = model(inputs[0].cuda(),inputs[2].cuda())

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

    y_true = []
    y_scores = []

    for inputs in batchtest_data:
        targets = inputs[1].cuda()

        tag_scores = model(inputs[0].cuda(),inputs[2].cuda())

        targets = targets.data.cpu().numpy()
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
    y_true = np.concatenate(y_true, axis=0)
    y_scores = np.concatenate(y_scores, axis=0)
    y_true = y_true.T
    y_scores = y_scores.T
    np.save('./results/y_true_DeepFH_'+AA+GOterm, y_true)
    np.save('./results/y_scores_DeepFH_'+AA+GOterm, y_scores)


def DeepFH(datatrain,datatest,GOterm,funGO,transmat,GOnameFile,AA):
    training_data, val_data = train_test_split(datatrain, test_size=0.1, random_state=42)
    del datatrain
    batchsize=32
    global classnum
    classnum = funGO.shape[1]

    global Embeddingsize
    Embeddingsize=21
    global hidden_dim
    hidden_dim=200

    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # cuda
    funnum=transmat.shape[0]

    global transmat1
    transmat1=transmat
    GOXfile = labelbert(funnum,GOnameFile)
    GOXfile = np.array(GOXfile)

    bert=BertModel.from_pretrained('/home/xxx/BioBert/biobert_v1.1_pubmed_v2.5.1_convert')
    bert=bert.to(DEVICE)

    GOXfile = torch.from_numpy(GOXfile)
    GOXfile=GOXfile.to(DEVICE)
    print('a')
    with torch.no_grad():
        bert_output=bert(GOXfile,output_all_encoded_layers=False)
    print('b')

    global bert_output1
    bert_output1=bert_output

    training_idx,training_label = preprocessing1(training_data)
    batchtraining_data = preprocessing(training_data,batchsize,training_idx,training_label)
    print('traindata ok ')
    batchtest_data = preprocessing(datatest,batchsize,training_idx,training_label)
    batchval_data = preprocessing(val_data,batchsize,training_idx,training_label)
    del training_data
    del val_data
    del training_idx
    del training_label

    model = ConvNet()
    model.cuda()
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    basemodel = trainmodel(model,batchtraining_data,batchval_data,loss_function,optimizer)
    print('deepFH_model alone: ')
    testmodel(basemodel,batchtest_data,GOterm,AA)



# if __name__=="__main__":









