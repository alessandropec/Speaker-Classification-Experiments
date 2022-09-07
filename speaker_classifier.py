import torch
import torch.nn as nn

class LSTM_SpeakerClassifier(nn.Module):
    def __init__(self,input_size=128,num_layers=1,hidden_size=256,num_classes=1,dropout=0.5):
        super(LSTM_SpeakerClassifier, self).__init__()
        self.input_size=input_size
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.num_classes=num_classes
 
        

        self.lstm=nn.LSTM(input_size=self.input_size,num_layers=self.num_layers,
                          hidden_size=self.hidden_size,dropout=dropout,batch_first=True)        
        self.lstm_relu = nn.ReLU()
        
        self.fc_1 =  nn.Linear(self.hidden_size, self.input_size) #fully connected 1
        self.fc_1_relu = nn.ReLU()
        
        self.fc = nn.Linear(self.input_size, self.num_classes) #fully connected last layer
        self.fc_2_relu = nn.ReLU()

       
       
      
        
    def forward(self, x):
        #Layer,Batch,hidden
        h0,c0=(torch.randn(self.num_layers,x.size(0),self.hidden_size),torch.randn(self.num_layers,x.size(0), self.hidden_size)) # clean out hidden state

        #Output is the last layer for each time step [BS,seq,features] hn,cn is the last output (time step) for each layer
        output, (hn, cn) = self.lstm(x, (h0,c0))
        out = output.view(-1, self.hidden_size)[-1].unsqueeze(0) #reshaping the data for Dense layer next and get only last 
        out = self.lstm_relu(out)
        
        out = self.fc_1(out) #first Dense
        out = self.fc_1_relu(out)
        out = self.fc(out) #Final Output
        out = self.fc_2_relu(out)
       
        return out