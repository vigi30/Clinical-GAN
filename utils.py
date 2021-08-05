import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
from pandas import DataFrame

def stats(newPairs,mn =600):
    x,y = [],[]
    count,county,counts =0,0,0
    for pair in newPairs:
        if len(pair[0]) > mn:
            count =count +1

        if len(pair[1]) > mn:
            county =county +1 

        if len(pair[0])>mn and len(pair[1])  >mn:
            counts = counts +1
        x.append(len(pair[0]))
        y.append(len(pair[1]))
    return max(torch.tensor(x)),max(torch.tensor(y))

# https://github.com/mp2893/doctorai
def recallTop(y_true, y_pred, rank=[20,40,60,200]):
    recall = list()
    for i in range(len(y_pred)):
        thisOne = list()
        codes = y_true[i]
        tops = y_pred[i]
        for rk in rank:
            thisOne.append(len(set(codes).intersection(set(tops[:rk])))*1.0/len(set(codes)))
        recall.append( thisOne )
    return (np.array(recall)).mean(axis=0).tolist()  
    


# https://github.com/benhamner/Metrics/pull/54

def apk(actual, predicted, k=10):

    if len(predicted)>k:
        predicted = predicted[:k]

    sum_precision = 0.0
    num_hits = 0.0

    for i, prediction in enumerate(predicted):
        if prediction in actual[:k] and prediction not in predicted[:i]:
            num_hits += 1.0
            precision_at_i = num_hits / (i + 1.0)
            sum_precision += precision_at_i

    if num_hits == 0.0:
        return 0.0

    return sum_precision / num_hits

def mapk(actual, predicted, k=10):

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

    
# Below code is motivated from:  from https://pytorch.org/tutorials/beginner/chatbot_tutorial.html#prepare-data-for-models

def padding(pair):
    pair.sort(key=lambda x: len(x[0]), reverse=True)
    inp_batch, output_batch = [],[]
    for pair in pair:
        inp_batch.append(pair[0])
        output_batch.append(pair[1])
    inp= inputVar(inp_batch).permute(1,0)
    output = outputVar(output_batch).permute(1,0)
    return inp,output



def zeroPadding(l):
    #https://docs.python.org/3/library/itertools.html
    return list(itertools.zip_longest(*l, fillvalue=0))
    

def inputVar(inp_batch):
    lengths = torch.tensor([len(i) for i in inp_batch])
    padList = zeroPadding(inp_batch)
    padVar = torch.LongTensor(padList)
    return padVar

def outputVar(out_batch):
    max_target_len  = max([len(i) for i in out_batch])
    padList = zeroPadding(out_batch)

    padVar = torch.LongTensor(padList)
    return padVar


    
def calc_visits(inps,code =2):
    return inps.count(code)


def plot(inp, recallAcc,precisionAcc,x_name,seq = False):
    plt.plot(inp, recallAcc, color='red', marker='o',label="Recall@60")
    plt.plot(inp, precisionAcc, color='blue', marker='o',label="Precision@60")

    
    plt.xlabel(x_name+' ' +'Length', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    if seq:
        plt.xlim(0,500)
    else:
        plt.xlim(0,16)
    plt.ylim(0,1)

    plt.legend(loc='center',bbox_to_anchor=(0.5, -0.3),fancybox=True, shadow=True, ncol=3)
    plt.savefig(x_name+'.pdf', dpi=1600,bbox_inches="tight")
    plt.show()
    plt.close()

    
def visitplot(inp,act, pred,x_name):
    plt.plot(inp, act, color='red')
    plt.scatter(x=inp, y=pred, c='navy', alpha=0.6)
    plt.xlabel('Input len Visits', fontsize=14)
    plt.ylabel('Prediced Len Visits ', fontsize=14)
    plt.title(f'1(a) Actual vs Predicted Visists:', fontsize=14)
    plt.grid(True)
    plt.show()
    
def createDataframe(inp,recallAcc,precisionAcc,seq = False,SAMPLE_SIZE = 5):
    df = DataFrame(inp,columns=['Length'])
    df['recallAcc'] = recallAcc
    df['precisionAcc'] = precisionAcc
    df = df.groupby("Length").mean().reset_index()
    if seq:
        label_series = pd.Series(itertools.chain.from_iterable(itertools.repeat(x, SAMPLE_SIZE) for x in df.index))
        df = df.groupby(label_series).mean()
    return df



def createVisitDataframe(inp,act,pred):
    df1 = DataFrame(inp,columns=['inp'])
    df1['act'] = act
    df1 = df1.groupby("inp").mean().reset_index()
    
    df2 = DataFrame(inp,columns=['inp'])
    df2['pred'] = pred
    df2 = df2.groupby("inp").mean().reset_index()
    return df1,df2


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def plotLossEpoch(config):
    plt.plot(config['tLoss'],label="train loss")
    plt.plot(config['vLoss'],label="valid loss")
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)