import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM
#print(torch.__version__)
#from base import BaseModel
import numpy
#from utils import prepare_device
#device, device_ids = prepare_device(config['n_gpu'])

#class SpeechEmotionModel(BaseModel):
class SpeechEmotionModel(nn.Module):
    def __init__(self, emotions):
        super(SpeechEmotionModel, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=4, stride=4),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        self.flatten = nn.Flatten(1, 2)
        self.lstm = nn.LSTM(128, 32, batch_first=True)
        self.fc = nn.Linear(32, emotions)

    def forward(self, x,device):
        #print(x.shape)
        x = self.convolutions(x)

        x = self.flatten(x)
        #print(type(x))
        x=x.cuda().data.cpu().numpy()
        
        x = numpy.swapaxes(x, 1, 2)
        
        x = torch.from_numpy(x)
        x=x.to(device)
        #print(type(x))
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x1=x
        #print(x1.shape) #25*32
        x = self.fc(x) 
        #print(x.shape)# 25*8
        #print(F.log_softmax(x, dim=1).shape)
        return x1,F.log_softmax(x, dim=1)

