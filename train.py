
import numpy as np
import pandas as pd
import os
import time

from data import *
#from optimize import *
from utils import *
from ClinicalGAN import *


import warnings
warnings.filterwarnings("ignore")
import matplotlib
import argparse
parser = argparse.ArgumentParser()


def trainWithoutTune(config,data,checkpoint_dir=None,data_dir = None):
    
    input_dim =len(data['codeMap']['types'])
    output_dim = len(data['codeMap']['outTypes'])

    hid_dim =config["hid_dim"]
    pf_dim =config["pf_dim"]
    batch_size = config['batch_size']
    isdataparallel = config["isdataparallel"]

    modelFileName = config["fileName"]
    path = os.path.join("models","ClinicalGAN",modelFileName)
    dropout = config["dropout"]
    lr = config["lr"]
    gen_layers = config["gen_layers"]
    gen_heads = config["gen_heads"]
    dis_layers = config["disc_layers"]
    dis_heads = config["disc_heads"]
    n_epochs = config['epoch']
    clip = config['clip']
    alpha = config['alpha']
    gen_clip = config['gen_clip']

    
    trainLoader,testLoader,valLoader = load_data(data['train'],data['test'],data['valid'],batch_size = batch_size)
    loader = {'trainLoader':trainLoader,'testLoader':testLoader,'valLoader':valLoader}
    
    modelHypermaters = initializeClinicalGAN(input_dim, output_dim, hid_dim,pf_dim,gen_layers,
                                             gen_heads,dis_heads,dis_layers, dropout,lr,
                                             n_epochs,alpha,clip,batch_size,loader,data,config,path,gen_clip,isdataparallel,device)
    trainCGAN(modelHypermaters, checkpoint_dir)
    
    return modelHypermaters

def run(config,data,checkpoint_dir=None):

    print("\n Training has been started...")
    modelHypermaters = trainWithoutTune(config,data,checkpoint_dir)
    print("\n Training has been finished.")
    return modelHypermaters



parser.add_argument('--learning_rate',default=4e-4, type=float,help="learning rate of the model")
parser.add_argument('--epochs',default=100, type=int,help="Total number of epochs")

parser.add_argument('--gen_layers',default=3, type=int,help="Total number of generator's Encoder and Decoder layers")
parser.add_argument('--disc_layers',default=3, type=int,help="Total number of discriminator's Encoder layers")

parser.add_argument('--dropout',default=0.1, type=float,help="Dropout value to be applied forreducing overfitting ")
parser.add_argument('--clip',default=0.1, type=float,help="Discriminator's cliping value for gradient clipping")


parser.add_argument('--gen_clip',default=1.0, type=float,help="Generator's cliping value for gradient clipping")
parser.add_argument('--alpha',default=0.3, type=float,help="alpha value for geenrators loss")

parser.add_argument('--gen_heads',default=8, type=int,help="Total number of multi-head in Generator")
parser.add_argument('--disc_heads',default=8, type=int,help="Total number of multi-head in Discriminator")
parser.add_argument('--batch_size',default=4, type=int,help="batch size to be used for training the model")

parser.add_argument('--isdataparallel',default=False, type=int,help="if you have more than two gpu's, use dataparallization")
parser.add_argument('--hid_dim',default=256, type=int,help="Embedding dimension of both Generator and discriminator")
parser.add_argument('--pf_dim',default=512, type=int,help="Hidden dimension of both Generator and discriminator")
#parser.add_argument('--istune',default=False, type=int,help="if you are trying to find optimal values, use tune function")
parser.add_argument('--warmup_steps',default=30, type=int,help="warmp up steps for learning rate")
parser.add_argument('--labelSmoothing',default=0.0, type=float,help="label smoothing value for reducing overfitting")
parser.add_argument('--factor',default=1, type=int,help="factor by which the learning rate value should increase or decrease ")
parser.add_argument('--checkpoint_dir',default=None, type=str,help="If you want to run the model for more epochs after terminating the trinaing, Provide the path of the saved model")
parser.add_argument('--valid_data_ratio',default=0.05, type=int,help="How much data should be allocated to valid set ")
parser.add_argument('--test_data_ratio',default=0.05, type=int,help="How much data should be allocated to test set ")

parser.add_argument('--scenario',default="S1", type=str,required=True,help="Which type of scenario based data needs to be loaded- S1, S2, S3. Scenarios as mentioned in the paper")
parser.add_argument('--task',default="TF", type=str,required=True,help="Two types of task SDP and TF")
parser.add_argument('--fileName',default="myAwesomeModel.pt",required=True, type=str,help="fileName for the model which is going to be stored in the 'model' folder")

def main(args):

    config = {'lr': args.learning_rate,
                'gen_layers': args.gen_layers,
                'disc_layers': args.disc_layers,
                'epoch': args.epochs,
                'dropout': args.dropout,
                'clip': args.clip,
                'gen_clip': args.gen_clip,
                'alpha': args.alpha,
                'gen_heads': args.gen_heads,
                'disc_heads': args.disc_heads,
                'batch_size': args.batch_size,
                'isdataparallel': args.isdataparallel,
                'hid_dim': args.hid_dim,
                'pf_dim': args.pf_dim,
                #'istune': args.istune,
                'warmup_steps':args.warmup_steps,
                'eps':args.labelSmoothing,
                'factor':args.factor,
                'fileName': args.fileName}
                    # where to train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("device name where the model is going to train: ",torch.cuda.get_device_name(device))

    codeFile = os.path.join('outputData','originalData')
    codeDescription = pickle.load(open(codeFile +'.description','rb'))
    if args.task =='TF':
        print(f"\n Getting the Trajectory forecasting data..")
        if args.scenario =='S1':
            outFile = os.path.join('outputData','TF','Inp_d_p_dr_out_d')
        elif args.scenario =='S2':
            outFile = os.path.join('outputData','TF','Inp_d_p_dr_out_d_p')
        else:
            outFile = os.path.join('outputData','TF','Inp_d_p_dr_out_d_p_dr')
    else:
        print(f"\n Getting the sequential disease prediction data...")
        outFile = os.path.join('outputData','SDP','Inp_d_p_dr_out_d')

    newPairs,types,reverseTypes,codeType,outTypes,reverseOutTypes = get_data(outFile)
    # get the input and output max sequence length
    inp_max_len,out_max_len =stats(newPairs[:])
    # split the data-- pass the fraction such in the function split_data(data, test_frac=0.05, valid_frac=0.05)
    train, test, valid = split_data(newPairs[:], test_frac=args.test_data_ratio, valid_frac=args.valid_data_ratio)
    #initialize the dataloader
    codeMap = {'types':types,'reverseTypes':reverseTypes,'codeType' :codeType,'outTypes' :outTypes,'reverseOutTypes':reverseOutTypes,'codeDescription':codeDescription}
    data = {'train':train,'test':test,'valid':valid, 'codeMap':codeMap , 'maxInp' : inp_max_len,'maxOut': out_max_len }
    print("\n Completed...")
    print(args.checkpoint_dir )
    modelHypermaters = run(config,data,checkpoint_dir =args.checkpoint_dir )

    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)