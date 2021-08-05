import torch
#from utils import display_attention
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
from data import *
import os
import argparse
parser = argparse.ArgumentParser()
from utils import *
from models import *
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




def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    #cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels[1:])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor",fontsize=10)
    plt.setp(ax.get_yticklabels(),fontsize=10)
    # Turn spines off and create white grid.
    #ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel('Input',loc='center',fontsize =17,labelpad=10)
    ax.set_ylabel('Output',fontsize =17,labelpad=10)
    
    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    #print(valfmt)
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            kw.update(fontsize='small')
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            #print(valfmt(data[i, j]))
            texts.append(text)

    return texts

def display_attention(input, predicted, attention, data,n_heads = 8, n_rows = 4, n_cols = 2):
    
    types  = data['types']
    reverseTypes = {v:k for k,v in types.items()}
    reverseOutTypes = data['reverseOutTypes']
    input1,predicted1=[],[]

    
    for f in input:
        if reverseTypes[f].startswith('D9_') or reverseTypes[f].startswith('D10_') or reverseTypes[f].startswith('P9_') or reverseTypes[f].startswith('P10_'):
            input1.append(reverseTypes[f][3:])
        else:
            input1.append(reverseTypes[f])

    input = input1
    #input = [reverseTypes[f]  for f in input ]
    for g in predicted:
        if reverseTypes[reverseOutTypes[g]].startswith('D9_') or reverseTypes[reverseOutTypes[g]].startswith('D10_') or reverseTypes[reverseOutTypes[g]].startswith('P9_') or reverseTypes[reverseOutTypes[g]].startswith('P10_'):
            predicted1.append(reverseTypes[reverseOutTypes[g]][3:])
        else:
            predicted1.append(reverseTypes[reverseOutTypes[g]])
    predicted=predicted1
    #predicted1 = [ reverseTypes[reverseOutTypes[g]]  for g in predicted ]

    temp = attention.permute(1,0,2)
    
    for i in range(temp.shape[0]):
        if i ==0:
            at = (sum(temp[0],0)/n_heads).unsqueeze(0)
        else:
            at = torch.cat([at,(sum(temp[i],0)/n_heads).unsqueeze(0)],0)       
    
    #fig = plt.figure(figsize=(30,30))
    #ax = fig.add_subplot(1, 1,1)
    #at = at.permute(1,0)
    _attention =at.cpu().detach().numpy()
    _attention = np.true_divide(_attention, _attention.sum(axis=1, keepdims=True)) *100
    #_attention = np.true_divide(_attention, _attention.sum(axis=0, keepdims=True)) *100
    #print(_attention.sum(axis=1))
    #fig = plt.figure(figsize=(30,30))
    fig, ax = plt.subplots()
    print(_attention.shape,len(predicted),len(input))
    im = heatmap(_attention, predicted, input, ax=ax,
                    cmap="YlGn", cbarlabel="contribution[percentage]")
    texts = annotate_heatmap(im, valfmt="{x:.1f} %")
    fig.set_figheight(30)
    fig.set_figwidth(30)
    plt.savefig('interpretReverse.pdf',format="pdf", dpi=900,bbox_inches = 'tight',pad_inches = 0)
    plt.show()





def inferenceGAN(model,inp,trg,data,n_heads,max_len,device,att=False,isDisplayCodeDesc=False):

    model.eval()
    types  = data['types']
    codeDescription = data['codeDescription']
    reverseTypes = {v:k for k,v in types.items()}
    #reverseTypes = data['reverseTypes']
    outTypes = data['outTypes']
    reverseOutTypes = data['reverseOutTypes']
    pred_trgs = []
    trgs = []
    pred_trg_words =[]
    trg_words =[]
    src =inp
    with torch.no_grad():
        #batch_size = len(pair)
        #src, trg = padding(pair)
        src = torch.LongTensor(src).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src)
        enc_src = model.encoder(src, src_mask)

        pred_trg = [types['SOH']]
        for i in range(max_len):
            trg_tensor = torch.LongTensor(pred_trg).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            output, attention,_ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            #pred_token = output.argmax(2)[:,-1].item()
            #print(output.shape)
            output = output.squeeze(0)
            _,pred_token = torch.max(output,1)
            pred_token = pred_token[-1] # 
    
            pred_trg.append(pred_token.item())
            if pred_token == types['EOH']:
                break
       # pred_trg_words.append([reverseTypes[code] for code in pred_trg])
        #print(pred_trg_words,trg)
        #trg_words.append([reverseTypes[code] for code in trg])
        #pred_trgs.append(pred_trg)
    print(f"\n Input Data: {src}")
    print(f"\n Output Data: {pred_trg}")
    print(f"\n Actual output:{trg}")
    print(f"\n *************************************************************************************************************************************************")
    if isDisplayCodeDesc:
        InputDes,OutDes,actDes = [],[],[]
        input_Desc,output_desc, act_desc =[],[],[]
        for code in src[0]:
            InputDes.append(reverseTypes[code.item()])
        for code in pred_trg:
            OutDes.append(reverseTypes[reverseOutTypes[code]])
            #OutDes.append(reverseTypes[code])
        for code in trg:
            actDes.append(reverseTypes[reverseOutTypes[code]])
            #actDes.append(reverseTypes[code])
        for code in InputDes:
            input_Desc.append(codeDescription[code])
        for code in OutDes:
            output_desc.append(codeDescription[code])
        for code in actDes:
            act_desc.append(codeDescription[code])
    
        print("\n Numerical equivalent Medical codes")
        print(f"\n Input Data medical codes: \n {InputDes}")
        print(f"\n Output Data medical codes: \n {OutDes}")
        print(f"\n Actual output medical codes:\n {actDes}")


        print(f"\n *************************************************************************************************************************************************")
        print("\n Description of Medical codes")
        print(f"\n Input Data description: \n {input_Desc}")
        print(f"\n Output Data description: \n {output_desc}")
        print(f"\n Actual output description:\n {act_desc}")

    if att:
        #if len(pred_trg)<15:
        print("\n Generating the visualization for interpretation....")
        display_attention(inp,pred_trg,attention.squeeze(0),data,n_heads)
        print("\n Complete...")

parser.add_argument('--scenario',default="S1", type=str,required=True,help="Which type of scenario based data needs to be loaded- S1, S2, S3. Scenarios as mentioned in the paper")
parser.add_argument('--task',default="TF", type=str,required=True,help="Two types of task SDP and TF")
parser.add_argument('--modelFileName',default="myAwesomeModel.pt",required=True,type=str,help="Load the saved model from the 'ClinicalGAN' folder ")

parser.add_argument('--valid_data_ratio',default=0.05, type=int,help="How much data should be allocated to valid set ")
parser.add_argument('--test_data_ratio',default=0.05, type=int,help="How much data should be allocated to test set ")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #if torch.cuda.is_available():
        #print("device name where the model is going to train: ",torch.cuda.get_device_name(device))

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
    print("\n Loading the test data...")
    trainLoader,testLoader,valLoader = load_data(data['train'],data['test'],data['valid'],batch_size = 1)
    pad_idx = data['codeMap']['types']['PAD']
    print("\n Completed...")
   
    print("\n Loading the model")
    modelFileName = args.modelFileName
    path = os.path.join("models","ClinicalGAN",modelFileName)
    
    checkpoint = torch.load(path)
    config = checkpoint['config']   
    gen = initialize(config,data,path)
    print("\n Completed...")

    print("\n Getting the output of a random input from test set")
    for i,pair in enumerate(testLoader):
        if i==1:
            break
        
        if len(pair[0][0])<100:
            inferenceGAN(gen,pair[0][0],pair[0][1], data['codeMap'],config['gen_heads'],100,device,att=True,isDisplayCodeDesc=True)
        else:
            print("\n Please run it again.")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)