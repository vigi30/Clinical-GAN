import torch
import numpy as np
from utils import *

from models import *
from ClinicalGAN import *
from data import *
import os
import argparse
parser = argparse.ArgumentParser()
import warnings
warnings.filterwarnings("ignore")


def initialize(config,data,path):
    
    input_dim =len(data['codeMap']['types'])
    output_dim = len(data['codeMap']['outTypes'])
    hid_dim =config["hid_dim"]
    pf_dim =config["pf_dim"]
    dropout = config["dropout"]
    lr = config["lr"]
    gen_layers = config["gen_layers"]
    gen_heads = config["gen_heads"]

  
 
    #trainLoader,testLoader,valLoader = load_data(data['train'],data['test'],data['valid'],batch_size = batch_size)
    #loader = {'trainLoader':trainLoader,'testLoader':testLoader,'valLoader':valLoader}
    
    inp_max_len,out_max_len,pad_idx = data['maxInp'],data['maxOut'] , data['codeMap']['types']['PAD']
    enc = Encoder(input_dim,hid_dim,gen_layers,gen_heads,pf_dim,dropout,inp_max_len).to(device)
    dec = Decoder(output_dim,hid_dim,gen_layers,gen_heads,pf_dim,dropout,out_max_len).to(device)
    
    gen = Generator(enc, dec, pad_idx, pad_idx).to(device)
    
    #disc = Discriminator(input_dim, hid_dim, dis_layers, dis_heads, pf_dim, dropout,pad_idx,inp_max_len+out_max_len).to(device)
    
    gen_opt = torch.optim.Adam(gen.parameters(), lr = lr)
    #disc_opt = torch.optim.SGD(disc.parameters(), lr = lr)
    
    #gen.apply(initialize_weights)
    #disc.apply(initialize_weights)
    #print("Total parameters: ",count_parameters(gen) + count_parameters(disc))
 
    
    #gen.load_state_dict(torch.load(path))
    checkpoint = torch.load(path)
    gen.load_state_dict(checkpoint['gen_state_dict'])
    gen_opt.load_state_dict(checkpoint['gen_optimizer_state_dict'])
    
    return gen


def make_src_mask(src,src_pad_idx):

    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    return src_mask

def make_trg_mask(trg,trg_pad_idx,device):
    

    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)

    trg_len = trg.shape[1]
    
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = device)).bool()
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask


def evaluateGAN(model,Loader,types,max_len,device):
    model.eval()
    pred_trgs = []
    trgs = []
    pred_trg_words =[]
    trg_words =[]
    inps = []
    with torch.no_grad():
        for i, pair in enumerate(Loader):
            batch_size = len(pair)
            src, trg = padding(pair)
            src,trg = src.to(device),trg.to(device)
            src_mask = make_src_mask(src,types['PAD'])
            enc_src = model.encoder(src, src_mask)
            
            pred_trg = [types['SOH']]
            for i in range(max_len):
                trg_tensor = torch.LongTensor(pred_trg).unsqueeze(0).to(device)
                trg_mask = make_trg_mask(trg_tensor,types['PAD'],device)
                output, attention,_ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                #pred_token = output.argmax(2)[:,-1].item()
                output = output.squeeze(0)
                _,pred_token = torch.max(output,1)
                #print(pred_token)
                pred_token = pred_token[-1] # 
                
                pred_trg.append(pred_token.item())
                #print(pred_trg)
                if pred_token == types['EOH']:
                    break
           # pred_trg_words.append([reverseTypes[code] for code in pred_trg])
            #print(pred_trg_words,trg)
            #trg_words.append([reverseTypes[code] for code in trg])
            inp = [code for code in src]
            trg = [code for code in trg]
            pred_trgs.append(pred_trg)
            trgs.append(trg)
            inps.append(inp)
        inps = convList(inps)
        trgs = convList(trgs)
        #print(f"Recall values  : {recallTop(pred_trgs, trgs)}")
        print(f"Recall values  : {recallTop(trgs, pred_trgs)}")
        k=[100,150,250]
        #print(f"test mark :  { [mark(pred_trgs,trgs, k=i) for i in k]} ")
        print(f"test mapk : { [mapk(trgs,pred_trgs, k=i) for i in k]} ")
        #print(f"test overall precision : {recommender_precision(pred_trgs,trgs)}")
        #print(f"test overall recall : {recommender_recall(pred_trgs,trgs)}")
        print("")
        plots(inps,pred_trgs,trgs)
            #print(f'inps:{inps} , inp len {len(inps)}')
    #return pred_trgs,trgs, inps #bleu_score(pred_trg_words, trg_words),(pred_trg_words,trg_words)


def convList(values):
    acts =[]
    for codes in values:
        for code in codes:
            #print(code)
            #code = code.item()
            code =code.tolist()
            acts.append(code)
    return acts


def testEvaluate(model, Loader, criterion,device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, pair in enumerate(Loader):
            batch_size = len(pair)
            src, trg= padding(pair)
            src,trg = src.to(device),trg.to(device)
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output, _ = model(src, trg[:,:-1])
        
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            
            trg = trg[:,1:].contiguous().view(-1)
            #print(output.shape ,trg.shape )
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(Loader)

def plots(inp,pred,acts):
    inpVisitsLen= []
    predVisitLen = []
    actVisitLen = []
    inpLen = []
    recallAcc,recallAcc1 = [],[]
    precisionAcc = []
    for i,inps in enumerate(inp):
        inpLen.append(len(inps))
        #print(f' inps: {inps} \n  pred:{pred[i]} \n act :{acts[i]}')
        inpVisitsLen.append(calc_visits(inp[i]))
        predVisitLen.append(calc_visits(pred[i]))
        actVisitLen.append(calc_visits(acts[i]))
        #recallAcc.extend(recallTop([pred[i]], [acts[i]],[60]))
        recallAcc1.extend(recallTop([acts[i]], [pred[i]],[60]))
        precisionAcc.extend([mapk([acts[i]],[pred[i]],60)])
    #visitLen  = createDataframe(inpVisitsLen,recallAcc,precisionAcc)
    visitLen1  = createDataframe(inpVisitsLen,recallAcc1,precisionAcc)
    #plot(visitLen['Length'], visitLen['recallAcc'], visitLen['precisionAcc'] ,'Visit')
    plot(visitLen1['Length'], visitLen1['recallAcc'], visitLen1['precisionAcc'] ,'Visit')
    seqLen  = createDataframe(inpLen,recallAcc1,precisionAcc,seq=True,SAMPLE_SIZE=20)
    plot(seqLen['Length'], seqLen['recallAcc'],seqLen['precisionAcc'] ,'Sequence',seq = True)
    #inp_act,inp_pred = createVisitDataframe(inpVisitsLen,actVisitLen,predVisitLen)
    #visitplot(inp_act['inp'],inp_act['act'], inp_pred['pred'] ,'Sequence')  
    




parser.add_argument('--scenario',default="S1", type=str,required=True,help="Which type of scenario based data needs to be loaded- S1, S2, S3. Scenarios as mentioned in the paper")
parser.add_argument('--task',default="TF", type=str,required=True,help="Two types of task SDP and TF")
parser.add_argument('--modelFileName',default="myAwesomeModel.pt",required=True,type=str,help="Load the saved model from the 'ClinicalGAN' folder ")

parser.add_argument('--valid_data_ratio',default=0.05, type=int,help="How much data should be allocated to valid set ")
parser.add_argument('--test_data_ratio',default=0.05, type=int,help="How much data should be allocated to test set ")




def main(args):

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
    trainLoader,testLoader,valLoader = load_data(data['train'],data['test'],data['valid'],batch_size = 1)
    pad_idx = data['codeMap']['types']['PAD']
    print("\n Completed...")
   
    print("\n Loading the model")
    modelFileName = args.modelFileName
    path = os.path.join("models","ClinicalGAN",modelFileName)
    
    checkpoint = torch.load(path)
    config = checkpoint['config']   
    gen = initialize(config,data,path)
    criterion = nn.NLLLoss(ignore_index = pad_idx)
    print("\n Completed...")

    print("\n Evaluate the model")

    print("\n Calculating the loss in the test data")
    batch_size = 32
    _,testLoader,_ = load_data(data['train'],data['test'],data['valid'],batch_size = batch_size)
    test_loss = testEvaluate(gen, testLoader, criterion,device)
    # making the batch size 1 inorder to calculate the accuracy and for plotting
    batch_size = 1
    _,testLoader,_ = load_data(data['train'],data['test'],data['valid'],batch_size = batch_size)

    print(f'\n | Test Loss: {test_loss:.3f} \n\n ')

    print("\n Calculating the Recall, Precision. Plotting the SequencevsAccuracy and visitlengthvsAccuracy.")
    evaluateGAN(gen,testLoader,data['codeMap']['types'],out_max_len,device)
    print("\n Completed...")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
