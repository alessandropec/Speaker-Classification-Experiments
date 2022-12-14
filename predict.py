from wsgiref.simple_server import demo_app
from speech_audio_dataset import SpeechAudioDataset
from speaker_classifier import LSTM_SpeakerClassifier

import torch

import argparse

def get_net(input_size=128,num_layers=1,hidden_size=256,num_classes=1,dropout=0.5,model_path="./saved_models/speaker_classifier.test.pt",device="cpu",embedding_size=128):
    net = LSTM_SpeakerClassifier(input_size=input_size,num_layers=num_layers,hidden_size=hidden_size,
                                    num_classes=num_classes,dropout=dropout,device=device,embedding_size=embedding_size)
    net.load_state_dict(torch.load(model_path,map_location=torch.device(device)))
    print("\nModel loaded...\n")
    print(net)
    
    return net

def init_argument_parser():
    parser = argparse.ArgumentParser(
        description='Training of Speaker Classifier Net (LSTM+FC layers)'
    )
    #Path vari
    parser.add_argument('--data_dir', required=True, help='the folder containing the sample audio (name ex: 0_1_1234.wav)\nfirst digit (0) is the label speaker id')
    parser.add_argument('--model_path', required=True, help='root of the model to load/save')
    parser.add_argument('--label_tag',default="0:Jeremy Clarkson;1:Greta Thunberg",help="Use follow syntax to declare label: 0:Speaker1;1Speaker2")
    
    #Model parameters
    #Model parameters
    parser.add_argument('--input_size',default=128,type=int,help="The size of each input in each sequence, \
                                                        i.e. the number of frame from mel sectrogram \
                                                        in each windows (fedded at each time-step). Default 128")
    parser.add_argument('--hidden_size',default=256,type=int,help="The size of the hiddden layers, \
                                                        output of lstm at each time step. Default 256")
    parser.add_argument('--num_layers',default=1,type=int,help="Number of hidden layers in lstm. Default 1")
    parser.add_argument('--num_classes',default=1,type=int,help="Number of classes (different cluster of audio). Default 1")
    parser.add_argument('--embedding_size',default=128,type=int,help="The size of the first fully connected layer. Default 128")
    
    return parser.parse_args()

def predict(net,dataset,tag_labels):
    tags_out=[]
    net.eval()
    with torch.no_grad():
        for data in dataset:
            print("Input shape: ",len(data))
            out=net(data[0].unsqueeze(0))
            print("Output: ",out)
            idx=out.argmax(dim=1)
            tags_out.append(tag_labels[idx])
    return tags_out

if __name__=="__main__":

    #Retrieve argument
    args=init_argument_parser()
    print("******************************************************************************************\n\
            Welcome to Audio Classification Net training phase\n\
            The following arguments will be used...\
        \n******************************************************************************************\n"+str(args)+"\n\n")

    #Create ad hoc dataset
    dataset=SpeechAudioDataset(audios_dir=args.data_dir)

    net=get_net(args.input_size,args.num_layers,args.hidden_size,args.num_classes,dropout=0,model_path=args.model_path,embedding_size=args.embedding_size)

    keys_values=args.label_tag.split(";")
    tags=[tag.split(":")[1] for tag in keys_values]


    tags_out=predict(net,dataset,tags)
    for i,(o,l) in enumerate(zip(tags_out,dataset.audios_name)):
        print(i,") Output: ",o,"    Label: ",l)