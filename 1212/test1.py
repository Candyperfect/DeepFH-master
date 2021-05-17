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

# from pytorch_pretrained_bert import BertTokenizer,BertModel
# from pytorch_pretrained_bert.optimization import BertAdam
# from transformers import AdamW
import os
from collections import OrderedDict
import random
from scipy import sparse
import string
from stop_words import get_stop_words    # download stop words package from https://pypi.org/project/stop-words/
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
##########################################################
import torch.nn.functional as F
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean'else loss.sum() if reduction=='sum'else loss

def linear_combination(x, y, epsilon):
    return epsilon*x+(1-epsilon)*y


def LabelSmoothingCrossEntropy(preds,target,epsilon=0.1,reduction='mean'):
    n = preds.size()[-1]
    log_preds = F.log_softmax(preds, dim=-1)
    loss = reduce_loss(-log_preds.sum(dim=-1), reduction)
    nll = F.nll_loss(log_preds, target)
    return linear_combination(loss/n, nll, epsilon)

# class LabelSmoothingCrossEntropy(nn.Module):
    # def __init__(self, epsilon:float=0.1, reduction='mean'):
        # super().__init__()
        # self.epsilon = epsilon
        # self.reduction = reduction

    # def forward(self, preds, target):
        # n = preds.size()[-1]
        # log_preds = F.log_softmax(preds, dim=-1)
        # loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        # nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        # return linear_combination(loss/n, nll, self.epsilon)


def CrossEntropyLoss_label_smooth(outputs, targets, epsilon=0.1):
    N = targets.size(0)
    num_classes=targets.size(1)
    smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1))
 
    # targets = targets.data
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - epsilon)
    log_prob = nn.functional.log_softmax(outputs, dim=1)
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss


def smooth_one_hot(true_labels: torch.Tensor, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    classes=true_labels.size(1)
    # assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))    # torch.Size([2, 5])
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)    # 空的，没有初始化
        true_dist.fill_(smoothing / (classes - 1))
        true_dist=true_labels*confidence+true_dist+(-1)*true_labels*true_dist
    return true_dist


input=torch.rand(3,5)
target=torch.zeros(3,5)
target[0][1]=1
target[1][0:3]=1
target[2][4]=1
target[2][0]=1
# loss=CrossEntropyLoss_label_smooth(input,target)
# loss=LabelSmoothingCrossEntropy(input,target)
loss=smooth_one_hot(target)
print(input)
print(target)
print(loss)