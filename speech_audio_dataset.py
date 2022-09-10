import os
import torch
import librosa
import numpy as np

from torch.utils.data import Dataset
from torchvision import datasets


class SpeechAudioDataset(Dataset):
    def __init__(self, audios_dir="./speaker_classifier_data"):
        self.audios_name = os.listdir(audios_dir)
        self.audios_dir = audios_dir

    def __len__(self):
        return len(self.audios_name)

    

    def __getitem__(self, idx,hop_length=512,n_fft=2048):
        audio_path = os.path.join(self.audios_dir, self.audios_name[idx])
        signal,sr = librosa.load(audio_path) 
        label = torch.tensor([int(self.audios_name[idx].split("_")[0])],dtype=torch.long)
        data=self.to_mel(signal,sr,hop_length,n_fft) 
    
        return torch.reshape(torch.tensor(data),(data.shape[1],data.shape[0])),label #reshape #(mel,sr,label)
    
    def get_signal(self, idx):
        audio_path = os.path.join(self.audios_dir, self.audios_name[idx])
        signal,sr = librosa.load(audio_path) 
        label = torch.tensor(int(self.audios_name[idx].split("_")[0]))
        return signal,sr, label


    def to_mel(self,signal,sr,hop_length=512,n_fft=2048):
      mel_signal = librosa.feature.melspectrogram(y=signal, sr=sr, hop_length=hop_length, 
      n_fft=n_fft)

      mel_spect = np.abs(mel_signal)

      power_to_db = librosa.power_to_db(mel_spect, ref=np.max)
      return power_to_db