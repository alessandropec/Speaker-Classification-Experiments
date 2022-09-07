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
    return parser.parse_args()

def predict(net,dataset,tag_labels):
    tags_out=[]
    net.eval()
    with torch.no_grad():
        for audio,sr,label in dataset:
            print("Input shape: ",audio.shape)
            out=net(audio.unsqueeze(0))
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

    net=get_net(128,1,256,2,0.5,model_path=args.model_path,embedding_size=64)

    keys_values=args.label_tag.split(";")
    tags=[tag.split(":")[1] for tag in keys_values]


    tags_out=predict(net,dataset,tags)
    print(tags_out,dataset.audios_name)