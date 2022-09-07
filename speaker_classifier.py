import torch
import torch.nn as nn

class LSTM_SpeakerClassifier(nn.Module):
    def __init__(self,input_size=128,num_layers=1,hidden_size=256,embedding_size=128,num_classes=2,dropout=0.5,device="cpu"):
        super(LSTM_SpeakerClassifier, self).__init__()
        self.input_size=input_size
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.embedding_size=embedding_size
        self.num_classes=num_classes
        self.device=device
        

        self.lstm=nn.LSTM(input_size=self.input_size,num_layers=self.num_layers,
                          hidden_size=self.hidden_size,dropout=dropout,batch_first=True)        
        self.lstm_relu = nn.ReLU()
        
        self.fc_1 =  nn.Linear(self.hidden_size, self.embedding_size) #fully connected 1
        self.fc_1_relu = nn.ReLU()
        
        self.fc_2 = nn.Linear(self.embedding_size, self.num_classes) #fully connected last layer
        self.fc_2_softmax = nn.Softmax(dim=1)

       
       
      
        
    def forward(self, x):
        #Layer,Batch,hidden NOTE: random initialization could provide different output from same input
        h0,c0=(torch.randn(self.num_layers,x.size(0),self.hidden_size).to(self.device),torch.randn(self.num_layers,x.size(0), self.hidden_size).to(self.device)) # clean out hidden state

        #Output is the last layer for each time step [BS,seq,features] hn,cn is the last output (time step) for each layer
        output, (hn, cn) = self.lstm(x, (h0,c0))
        out = output.view(-1, self.hidden_size)[-1].unsqueeze(0) #reshaping the data for Dense layer next and get only last 
        out = self.lstm_relu(out)
        
        out = self.fc_1(out) #first Dense
        out = self.fc_1_relu(out)
        out = self.fc_2(out) #Final Output
        out = self.fc_2_softmax(out)
       
        return out