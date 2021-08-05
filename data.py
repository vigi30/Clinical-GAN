#import torch
import pickle
import numpy as  np
import random
import os
from torch.utils.data import Dataset, DataLoader

class inputData(Dataset):
    def __init__(self,data):
        self.inp =data
        #self.inp = [i[0] for i in data]
        #self.out = [i[1] for i in data]
        
    def __len__(self):
        return len(self.inp)
        
    def __getitem__(self,idx):
        
        return self.inp[idx] #self.inp[idx],self.out[idx]
    
    
def collate(batch_pairs):
    x = batch_pairs
    return (x)  


def get_data(outFile):
    #outFile = os.path.join('outputData','TF','Inp_d_p_dr_out_d')
    '''
    outFile : absolute path and file name
    
    '''
    newPairs = pickle.load(open(outFile +'.seqs','rb'))
    types = pickle.load(open(outFile +'.types','rb'))
    reverseTypes = pickle.load(open(outFile +'.reverseTypes','rb'))
    codeType = pickle.load(open(outFile +'.codeType','rb'))
    outTypes = pickle.load(open(outFile +'.outTypes','rb'))
    reverseOutTypes = {v:k for k,v in outTypes.items()}
    
    return newPairs,types,reverseTypes,codeType,outTypes,reverseOutTypes

    
def split_data(data, test_frac=0.05, valid_frac=0.05):
    sequences = np.array(data)
    dataSize = len(sequences)
    np.random.seed(0) 
    idx = np.random.permutation(dataSize)
    nTest = int(np.ceil(test_frac * dataSize))
    nValid = int(np.ceil(valid_frac * dataSize))
    #def len_argsort(seq):
     #   return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    test_idx = idx[:nTest]
    valid_idx = idx[nTest:nTest+nValid]
    train_idx = idx[nTest+nValid:]

    train_x = sequences[train_idx]
   
    test_x = sequences[test_idx]
   
    valid_x = sequences[valid_idx]
    
    
    #sorted_index = len_argsort(test_x)
    #test_x = [test_x[i] for i in sorted_index]
    #test_y = [test_y[i] for i in sorted_index]


    

    '''    train_x = [sorted(seq) for seq in train_x]
    train_y = [sorted(seq) for seq in train_y]
    valid_x = [sorted(seq) for seq in valid_x]
    valid_y = [sorted(seq) for seq in valid_y]
    test_x = [sorted(seq) for seq in test_x]
    test_y = [sorted(seq) for seq in test_y]
    '''
    train = (train_x)
    test = (test_x)
    valid = (valid_x)
    return (train, test, valid)
    


def load_data(train,test,valid,batch_size):
    
    trainData = inputData(train)
    testData = inputData(test)
    valData = inputData(valid)
    
    trainLoader = DataLoader(dataset = trainData,batch_size =batch_size ,collate_fn = collate,shuffle =True)
    
    testLoader = DataLoader(dataset = testData,batch_size =1,collate_fn = collate,shuffle =True)
    valLoader = DataLoader(dataset = valData,batch_size =1,collate_fn = collate,shuffle =True)
    
    return trainLoader,testLoader,valLoader
    #trainLoader = DataLoader(dataset = trainData,batch_size =256 ,shuffle =True)
    #testLoader = DataLoader(dataset = testData,batch_size =256,shuffle =True)
    #valLoader = DataLoader(dataset = valData,batch_size =256,shuffle =True)

def dloader(data,batch_size=1):
    trainData = inputData(data)
    trainLoader = DataLoader(dataset = trainData,batch_size =batch_size ,collate_fn = collate,shuffle =True)
    return trainLoader
    


