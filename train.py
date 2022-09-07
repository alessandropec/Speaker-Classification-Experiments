from ast import parse
from speech_audio_dataset import SpeechAudioDataset
from speaker_classifier import LSTM_SpeakerClassifier

import torch
from torch.utils.data import DataLoader

import librosa

import numpy as np

import matplotlib.pyplot as plt

import argparse


def get_net(load=False,input_size=128,num_layers=1,hidden_size=256,num_classes=1,dropout=0.5,model_path="./saved_models/speaker_classifier.test.pt"):
    if not load:
        net = LSTM_SpeakerClassifier(input_size=input_size,num_layers=num_layers,hidden_size=hidden_size,
                                     num_classes=num_classes,dropout=dropout)
    else:
        net = LSTM_SpeakerClassifier(input_size=input_size,num_layers=num_layers,hidden_size=hidden_size,
                                     num_classes=num_classes,dropout=dropout)
        net.load_state_dict(torch.load(model_path))
        print("\nModel loaded...\n")
    
    return net

def train(net,train_data,n_epochs=100,lr=10e-4,momentum=0.8,model_path=".saved_models/speaker_classifier.test.pt",device="cpu"):
    net.to(device)
    net.train() #set train for dropout

    #Set loss function and optimizer algorithm (TO DO: change to cross entropy for multiclass task)
    loss_function = torch.nn.CrossEntropyLoss() #nn.NLLLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)


    #Accumulate the avg score of each epoch
    avg_epoch_loss=[]
    
    for epoch in range(n_epochs):  # 
        epoch_loss=[]
        print("\nEpoch:", epoch+1,"/"+str(n_epochs)+"\n")
        for audio,sr,label in train_data:

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            optimizer.zero_grad()
    
            # Step 2. Run our forward pass.
            out = net(audio.to(device))
                     
            loss = loss_function(out, label)
            print("Out: ",out,"Label: ",label,"LOSS:",loss)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        avg_loss=np.mean(epoch_loss)
        avg_epoch_loss.append(np.mean(avg_loss))
        print("Avg epoch loss",avg_loss)
        #save the model
        print("model saved at ",model_path)
        torch.save(net.state_dict(), model_path)
    
    print("Training finished loss for each epochs plotted:")
    plt.plot(avg_epoch_loss)
    plt.show()



    

def eval_net(net,train_data):
    net.eval()
    avg_loss=[]
    loss_function = torch.nn.CrossEntropyLoss() #nn.NLLLoss()
    with torch.no_grad():
        print("\nOutput for training data")
        for audio,sr,label in train_data:
            test_out = net(audio)

            loss=loss_function(test_out,label)
            avg_loss.append(loss)
            print("Net out test: ",test_out[0],"Label: ",label,"Loss: ",loss)
        print("Avg loss for all data: ",np.mean(avg_loss))

def init_argument_parser():
    parser = argparse.ArgumentParser(
        description='Training of Speaker Classifier Net (LSTM+FC layers)'
    )
    #Path vari
    parser.add_argument('--data_dir', required=True, help='the folder containing the sample audio (name ex: 0_1_1234.wav)\nfirst digit (0) is the label speaker id')
    parser.add_argument('--model_path', required=True, help='root of the model to load/save')
    parser.add_argument('--load_net',default=False,action="store_true",help="load the net specified with --path argument. Default False")

    #Training parameters
    parser.add_argument('--n_epochs',default=10,type=int,help="Number of epochs to run in training. Default 10")
    parser.add_argument('--lr',default=10e-3,type=float,help="Learning rate in adam optmizer algorithm. Default 10e-3")
    parser.add_argument('--momentum',default=0.7,type=float,help="Momentum in adam optimizer algorithm. Default 0.7")
    parser.add_argument('--num_workers',default=1,type=float,help="Number of worker to parallelize data. Default 1")
    parser.add_argument('--train_device',default="cpu",help="Number of worker to parallelize data. Default CPU")

    #Model parameters
    parser.add_argument('--input_size',default=128,type=int,help="The size of each input in each sequence, \
                                                        i.e. the number of frame from mel sectrogram \
                                                        in each windows (fedded at each time-step). Default 128")
    parser.add_argument('--hidden_size',default=256,type=int,help="The size of the hiddden layers, \
                                                        output of lstm at each time step. Default 256")
    parser.add_argument('--num_layers',default=1,type=int,help="Number of hidden layers in lstm. Default 1")
    parser.add_argument('--num_classes',default=1,type=int,help="Number of classes (different cluster of audio). Default 1")
    parser.add_argument('--dropout',default=0.5,type=float,help="Set the probability of dropout to regularize output in training. Default 0.5")
    args=parser.parse_args()
    return args


if __name__=="__main__":

    #Retrieve argument
    args=init_argument_parser()
    print("******************************************************************************************\n\
            Welcome to Audio Classification Net training phase\n\
            The following arguments will be used...\
        \n******************************************************************************************\n"+str(args)+"\n\n")

    #Create ad hoc dataset
    dataset=SpeechAudioDataset(audios_dir=args.data_dir)

    #Build dataloader
    train_data=DataLoader(dataset, batch_size=1, shuffle=True,num_workers=args.num_workers)

    for i,el in enumerate(train_data):
        print("Mel Spectrogram random audio:",el[0])
        print("Audio shape with batch of 1: ",el[0].shape," Sample rate: ",el[1],"Label: ",el[2]) #(audio [B,in_len,seq_len],sr [1], label [1])
        break

    #Load or build net
    net=get_net(load=args.load_net,input_size=args.input_size,num_layers=args.num_layers,\
                hidden_size=args.hidden_size,num_classes=args.num_classes,dropout=args.dropout,model_path=args.model_path)
    print("Model:\n",net)

    #Eval net (only training data)
    eval_net(net,train_data=train_data)

    #Train net (in place)
    print("Start training...................")
    train(net,train_data,n_epochs=args.n_epochs,lr=args.lr,momentum=args.momentum,model_path=args.model_path,device=args.train_device)

    #Eval net (only training data)
    eval_net(net,train_data=train_data)