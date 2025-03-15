import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
#from tcn import TemporalConvNet
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
#import model.model as module_arch


from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM
#print(torch.__version__)
#from base import BaseModel
import numpy

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    # elif classname.find('GRU') != -1 or classname.find('LSTM') != -1:
    #     m.weight.data.normal_(0.0, 0.02)
    #     m.bias.data.fill_(0.01)
    else:
        print(classname)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class BasicBlock_AU(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(1,1), stride=(1,1),maxpool=(2,2), padding=0):
        super(BasicBlock_AU, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.bn1(out)
        out = self.relu(out)
        return out
        
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), nn.InstanceNorm2d(dim), nn.ReLU(inplace=False)]
        conv_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), nn.InstanceNorm2d(dim)]
        self.conv_blocks = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_blocks(x)
        return out

class RNNModel(nn.Module):
    def __init__(self, input_size=256, hidden_size=256, num_layers=2,batch_first=True,bidirectional=True):
        super(RNNModel, self).__init__()
        #self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = num_layers
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers,
                           batch_first=True,
                           bidirectional=True)
        if bidirectional:
            self.fc1 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, inputs):

        output,_ = self.rnn(inputs)
        return output
        

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

    def forward(self, x):
        #print(x.shape)
        x = self.convolutions(x)

        x = self.flatten(x)
        #print(type(x))
        #x=x.cuda().data.cpu().numpy()
        
        x = torch.transpose(x, 1, 2)
        
        #x = torch.from_numpy(x)
        #x=x.to(device)
        #print(type(x))
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x1=x
        #print(x1.shape) #25*32
        x = self.fc(x) 
        #print(x.shape)# 25*8
        #print(F.log_softmax(x, dim=1).shape)
        return x1,F.log_softmax(x, dim=1)


class Audio2AU(nn.Module):
    def __init__(self, hidden_size=128):
        super(Audio2AU, self).__init__()
        
        # the input map is 1 x 12 x 28       
        self.block1 = BasicBlock_AU(1, 8, (1,4), stride=(1,2),maxpool=(1,1)) # 3 x 12 x 13
        self.block2 = BasicBlock_AU(8, 32, kernel_size=(1,3), stride=(1,2),maxpool=(1,1)) # 8 x 12 x 6
        self.block3 = BasicBlock_AU(32, 64, kernel_size=(1,3), stride=(1,2),maxpool=(1,2)) # 16 x 12 x 1
        self.block4 = BasicBlock_AU(64, 128, kernel_size=(3,1), stride=(2,1),maxpool=(1,1)) # 32 x 5 x 1
        self.block5 = BasicBlock_AU(128, 256, kernel_size=(3,1), stride=(2,1),maxpool=(2,1)) # 32 x 1 x 1 
        self.rnn = RNNModel(256, hidden_size)
        self.fc1 = nn.Sequential(nn.Linear(256,128), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5)) # 128
        self.fc2 = nn.Sequential(nn.Linear(128,64), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5)) # 128

              
    def forward(self, audio_inputs):
        batchsize=audio_inputs.shape[0]
        seq_len=audio_inputs.shape[1]
        audio_inputs = audio_inputs.contiguous().view(audio_inputs.shape[0]*audio_inputs.shape[1], audio_inputs.shape[2], audio_inputs.shape[3],audio_inputs.shape[4])
        out = self.block1(audio_inputs)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        out = out.contiguous().view(out.shape[0], -1)
        out = out.contiguous().view(batchsize,seq_len, -1)
        out = self.rnn(out)
        rnn_out = out.contiguous().view(batchsize*seq_len, -1)
        out = self.fc1(rnn_out)
        out = self.fc2(out)

        return rnn_out,out


class AudioEncoder(nn.Module):
    def __init__(self, num_output_length, if_tanh=False):
        super(AudioEncoder, self).__init__()
        self.if_tanh = if_tanh
        # the input map is 1 x 12 x 28
        self.block1 = BasicBlock(1, 16, kernel_size=3, stride=1) # 16 x 12 x 28
        self.block2 = BasicBlock(16, 32, kernel_size=3, stride=2) # 32 x 6 x 14
        self.block3 = BasicBlock(32, 64, kernel_size=3, stride=1) # 64 x 6 x 14
        self.block4 = BasicBlock(64, 128, kernel_size=3, stride=1) # 128 x 6 x 14
        self.block5 = BasicBlock(128, 256, kernel_size=3, stride=2) # 256 x 3 x 7
        # self.fc1 = nn.Linear(6912, 512)
        # self.batch_norm = nn.BatchNorm2d(512)
        self.fc1 = nn.Sequential(nn.Linear(5376, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(512, num_output_length)

    def forward(self, inputs):
        #print(inputs.shape)
        out = self.block1(inputs)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = out.contiguous().view(out.shape[0], -1)
        # out = F.relu(self.batch_norm(self.fc1(out)))
        out = self.fc1(out)
        out = self.fc2(out)
        if self.if_tanh:
          out = torch.tanh(out)
        #print(out.shape)
        return out
'''
class ImageEncoder(nn.Module):
    def __init__(self, size_image, num_output_length, if_tanh=False):
        super(ImageEncoder, self).__init__()
        self.if_tanh = if_tanh
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        size_mini_map = self.get_size(size_image, 4)
        self.fc = nn.Linear(size_mini_map*size_mini_map*128, num_output_length)

    def get_size(self, size_image, num_layers):
        return int(size_image/2**num_layers)

    def forward(self, inputs):
        img_e_conv1 = self.conv1(inputs)
        img_e_conv2 = self.conv2(img_e_conv1)
        img_e_conv3 = self.conv3(img_e_conv2)
        img_e_conv4 = self.conv4(img_e_conv3)
        img_e_fc_5 = img_e_conv4.contiguous().view(img_e_conv4.shape[0], -1)
        img_e_fc_5 = self.fc(img_e_fc_5)
        if self.if_tanh:
            img_e_fc_5 = torch.tanh(img_e_fc_5)
        return img_e_fc_5, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4


class ImageDecoder(nn.Module):
    def __init__(self, size_image, input_dim):
        super(ImageDecoder, self).__init__()
        self.size_mini_map = self.get_size(size_image, 4)
        self.fc = nn.Linear(input_dim, self.size_mini_map*self.size_mini_map*256)
        self.dconv1 = nn.Sequential(nn.ConvTranspose2d(384, 196, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
        self.dconv2 = nn.Sequential(nn.ConvTranspose2d(260, 128, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
        self.dconv3 = nn.Sequential(nn.ConvTranspose2d(160, 80, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
        self.dconv4 = nn.Sequential(nn.ConvTranspose2d(96, 48, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
        self.dconv5 = nn.Sequential(nn.Conv2d(48, 16, 5, stride=1, padding=2), nn.ReLU(inplace=True))
        self.dconv6 = nn.Conv2d(16, 3, 5, stride=1, padding=2)

    def get_size(self, size_image, num_layers):
        return int(size_image/2**num_layers)

    def forward(self, concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4):
        # out = torch.cat([img_z, audio_z], dim=1) # (batch_size, input_dim)
        out = self.fc(concat_z)
        # reshape 256 x 7 x 7
        out = out.contiguous().view(out.shape[0],  256, self.size_mini_map, self.size_mini_map)
        out = F.relu(out, inplace=True)
        # concate (256+128) x 7x7
        out = torch.cat([out, img_e_conv4], dim=1)
        out = self.dconv1(out)
        # concate (196+64) x 14x14
        out = torch.cat([out, img_e_conv3], dim=1)
        out = self.dconv2(out)
        # concate (128+32) x 28x28
        out = torch.cat([out, img_e_conv2], dim=1)
        out = self.dconv3(out)
        # concate (80+16) x 56x56
        out = torch.cat([out, img_e_conv1], dim=1)
        out = self.dconv4(out)
        out = self.dconv5(out)
        out = self.dconv6(out)
        return torch.tanh(out)
'''

class ImageEncoder(nn.Module):
    def __init__(self, size_image, num_output_length, if_tanh=False):
        super(ImageEncoder, self).__init__()
        self.if_tanh = if_tanh
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        size_mini_map = self.get_size(size_image, 4)
        self.fc = nn.Linear(size_mini_map*size_mini_map*128, num_output_length*4)
        self.fc1 = nn.Linear(num_output_length*4, num_output_length)

    def get_size(self, size_image, num_layers):
        return int(size_image/2**num_layers)

    def forward(self, inputs):
        img_e_conv1 = self.conv1(inputs)
        img_e_conv2 = self.conv2(img_e_conv1)
        img_e_conv3 = self.conv3(img_e_conv2)
        img_e_conv4 = self.conv4(img_e_conv3)
        img_e_fc_5 = img_e_conv4.contiguous().view(img_e_conv4.shape[0], -1)
        img_e_fc_5 = self.fc(img_e_fc_5)
        img_e_fc_6 = self.fc1(img_e_fc_5)
        if self.if_tanh:
            img_e_fc_6 = torch.tanh(img_e_fc_6)
        return img_e_fc_6, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4


class ImageDecoder(nn.Module):
    def __init__(self, size_image, input_dim):
        super(ImageDecoder, self).__init__()
        self.size_mini_map = self.get_size(size_image, 4)
        self.fc = nn.Linear(input_dim, self.size_mini_map*self.size_mini_map*256)
        self.dconv1 = nn.Sequential(nn.ConvTranspose2d(384, 196, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
        self.dconv2 = nn.Sequential(nn.ConvTranspose2d(260, 128, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
        self.dconv3 = nn.Sequential(nn.ConvTranspose2d(160, 80, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
        self.dconv4 = nn.Sequential(nn.ConvTranspose2d(96, 48, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
        self.dconv5 = nn.Sequential(nn.Conv2d(48, 16, 5, stride=1, padding=2), nn.ReLU(inplace=True))
        self.dconv6 = nn.Conv2d(16, 3, 5, stride=1, padding=2)

    def get_size(self, size_image, num_layers):
        return int(size_image/2**num_layers)

    def forward(self, concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4):
        # out = torch.cat([img_z, audio_z], dim=1) # (batch_size, input_dim)
        out = self.fc(concat_z)
        # reshape 256 x 7 x 7
        out = out.contiguous().view(out.shape[0],  256, self.size_mini_map, self.size_mini_map)
        out = F.relu(out, inplace=True)
        # concate (256+128) x 7x7
        out = torch.cat([out, img_e_conv4], dim=1)
        out = self.dconv1(out)
        # concate (196+64) x 14x14
        out = torch.cat([out, img_e_conv3], dim=1)
        out = self.dconv2(out)
        # concate (128+32) x 28x28
        out = torch.cat([out, img_e_conv2], dim=1)
        out = self.dconv3(out)
        # concate (80+16) x 56x56
        out = torch.cat([out, img_e_conv1], dim=1)
        out = self.dconv4(out)
        out = self.dconv5(out)
        out = self.dconv6(out)
        return torch.tanh(out)
'''
class ImageEncoder(nn.Module):
    def __init__(self, size_image, num_output_length, if_tanh=False):
        super(ImageEncoder, self).__init__()
        self.if_tanh = if_tanh
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 512, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        #self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        size_mini_map = self.get_size(size_image, 4)
        #self.fc = nn.Linear(29*29*128, num_output_length)
        self.fc1 = nn.Linear(7*7*512, 2048)
        self.fc2 = nn.Linear(2048, 256)

    def get_size(self, size_image, num_layers):
        return int(size_image/2**num_layers)

    def forward(self, inputs):
        img_e_conv1 = self.conv1(inputs)
 
        img_e_conv2 = self.conv2(img_e_conv1)

        img_e_conv3 = self.conv3(img_e_conv2)
   
        img_e_conv4 = self.conv4(img_e_conv3)
        img_e_conv5 = self.conv5(img_e_conv4)
        img_e_conv6 = self.conv6(img_e_conv5)
        
        #img_e_conv5 = self.conv4(img_e_conv4)

        img_e_fc_7 = img_e_conv6.contiguous().view(img_e_conv6.shape[0], -1)

        img_e_fc_8 = self.fc1(img_e_fc_7)
        img_e_fc_9 = self.fc2(img_e_fc_8)
        if self.if_tanh:
            img_e_fc_9 = torch.tanh(img_e_fc_9)
        return img_e_fc_9, img_e_conv3, img_e_conv4, img_e_conv5, img_e_conv6



class ImageDecoder(nn.Module):
    def __init__(self, size_image, input_dim):
        super(ImageDecoder, self).__init__()
        #self.size_mini_map = self.get_size(size_image, 4)
        self.size_mini_map = 7
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, self.size_mini_map*self.size_mini_map*1024)
        self.dconv1 = nn.Sequential(nn.ConvTranspose2d(1536, 512, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=False))
        self.dconv2 = nn.Sequential(nn.ConvTranspose2d(512+256, 256, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=False))
        self.dconv3 = nn.Sequential(nn.ConvTranspose2d(384, 196, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=False))
        self.dconv4 = nn.Sequential(nn.ConvTranspose2d(260, 128, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=False))
        self.dconv5 = nn.Sequential(nn.ConvTranspose2d(128, 80, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=False))
        self.dconv6 = nn.Sequential(nn.ConvTranspose2d(80, 48, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=False))
        self.dconv7 = nn.Sequential(nn.Conv2d(48, 16, 4, stride=1, padding=2), nn.ReLU(inplace=False))
        self.dconv8 = nn.Conv2d(16, 3, 4, stride=1, padding=2)

    def get_size(self, size_image, num_layers):
        return int(size_image/2**num_layers)

    def forward(self, concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4):
        # out = torch.cat([img_z, audio_z], dim=1) # (batch_size, input_dim)
        #print(concat_z.shape)
        out = self.fc1(concat_z)
        out = self.fc2(out)
        #print(self.size_mini_map)
        # reshape 256 x 7 x 7
        out = out.contiguous().view(out.shape[0],  1024, self.size_mini_map, self.size_mini_map)
        out = F.relu(out, inplace=True)
        # concate (256+128) x 7x7
        out = torch.cat([out, img_e_conv4], dim=1)
        out = self.dconv1(out)
        # concate (196+64) x 14x14
        out = torch.cat([out, img_e_conv3], dim=1)
        out = self.dconv2(out)
        # concate (128+32) x 28x28
        out = torch.cat([out, img_e_conv2], dim=1)
        out = self.dconv3(out)
        # concate (80+16) x 56x56
        out = torch.cat([out, img_e_conv1], dim=1)
        out = self.dconv4(out)
        #print(out.shape)
        out = self.dconv5(out)
        #print(out.shape)
        out = self.dconv6(out)
        #print(out.shape)
        out = self.dconv7(out)
        #print(out.shape)
        out = self.dconv8(out)
        #print(out.shape)
        return torch.tanh(out)
'''

class RNNModel_G(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type, num_layers=1):
        super(RNNModel_G, self).__init__()
        self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = num_layers
        if rnn_type=='GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=1,  batch_first=True)

    def forward(self, inputs):

        output = self.rnn(inputs)
        return output


    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                    weight.new_zeros(self.nlayers, batch_size, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, batch_size, self.nhid)


class LipGeneratorRNN(nn.Module):
    def __init__(self, audio_encoder_type, img_encoder_type, img_decoder_type, rnn_type, size_image, num_output_length, hidden_size=1024, if_tanh=False):
        super(LipGeneratorRNN, self).__init__()
        '''
        if audio_encoder_type=='reduce':
            self.audio_encoder = AudioEncoder(num_output_length, if_tanh)
        elif audio_encoder_type=='bmvc':
            self.audio_encoder = AudioEncoderBMVC(num_output_length, if_tanh)
        elif audio_encoder_type =='hk':
            self.audio_encoder = AudioEncoder_hk(35, 12)
        '''
        if img_encoder_type=='reduce':
            #self.image_encoder = ImageEncoder(size_image, num_output_length, if_tanh)
            self.image_encoder1 = ImageEncoder(size_image, num_output_length, if_tanh)
        elif img_encoder_type=='FCN':
            self.image_encoder = ImageEncoderFCN(size_image, num_output_length, if_tanh)
        if img_decoder_type=='reduce':
            self.image_decoder = ImageDecoder(size_image, hidden_size)
        elif img_decoder_type=='residual':
            self.image_decoder = ImageDecoderResidual(size_image, hidden_size)
        if rnn_type=='GRU':
            #self.rnn = RNNModel_G(2*num_output_length+64*5, hidden_size, rnn_type)
            self.rnn = RNNModel_G(320, hidden_size, rnn_type)
        elif rnn_type=='TCN':
            num_channels=[512,512,512,1024]
            #self.tcn = TemporalConvNet(2*num_output_length+64*5+32, num_channels)
            #self.tcn = TemporalConvNet(2*num_output_length+32, num_channels)
            self.tcn = TemporalConvNet(2*num_output_length, num_channels)
              
      
        #self.audio_encoder_type = audio_encoder_type
        self.img_encoder_type = img_encoder_type
        self.img_decoder_type = img_decoder_type

        # initialize weights
        #self.audio_encoder.apply(weights_init)
        self.image_encoder1.apply(weights_init)
        self.image_decoder.apply(weights_init)

        
        if rnn_type=='GRU':
            self.rnn.apply(weights_init)

    # image_inputs shape: batch_size, seq_len, c, h, w
    
    def forward(self, image_inputs1, audio_inputs, teacher_forcing_ratio=0.5):
        # reshape inputs to (seq_len*batch_size, ...)
        batch_size = 1
        #seq_len = image_inputs1.shape[1]
        
        # new model
        '''
        input_wavs = input_wavs.contiguous().view(input_wavs.shape[1],input_wavs.shape[0], input_wavs.shape[2], input_wavs.shape[3])
        input_wavs = input_wavs.type(torch.FloatTensor)
        input_wavs=input_wavs.to(device)
        ser_model = module_arch.SpeechEmotionModel(8)
        ser_model = ser_model.to(device)
        x1,s=ser_model(input_wavs,device)
        #print(s)
        '''
        audio_inputs = audio_inputs.unsqueeze(0).expand(batch_size, -1)

        
        #image_inputs1 = np.transpose(image_inputs1.cpu().detach().numpy(),(2,0,1))
        #image_inputs1 = torch.tensor(image_inputs1,requires_grad=True)
        image_inputs1 = image_inputs1.permute(2,0,1)
        image_inputs1 = image_inputs1.unsqueeze(0).expand(batch_size, image_inputs1.shape[0], image_inputs1.shape[1], image_inputs1.shape[2])
        #image_inputs1 = image_inputs1.contiguous().view(batch_size, image_inputs1.shape[3], image_inputs1.shape[1], image_inputs1.shape[2])
        #image_inputs2 = image_inputs2.contiguous().view(batch_size, image_inputs2.shape[3], image_inputs2.shape[1], image_inputs2.shape[2])
        #audio_inputs = audio_inputs.contiguous().view(seq_len*batch_size, audio_inputs.shape[2], audio_inputs.shape[3], audio_inputs.shape[4])

        #audio_z = self.audio_encoder(audio_inputs)
        #print(image_inputs1.shape)
        #print(image_inputs1.contiguous().view(batch_size,image_inputs1.shape[2], image_inputs1.shape[1], image_inputs1.shape[3]) == image_inputs1)

        image_z1,  img_e_conv11, img_e_conv21, img_e_conv31, img_e_conv41 = self.image_encoder1(image_inputs1)
        #print(image_z1.shape)
        
        #image_z2,  img_e_conv12, img_e_conv22, img_e_conv32, img_e_conv42 = self.image_encoder(image_inputs2)
        
                
        

        if self.img_encoder_type=='FCN':
            audio_z = audio_z.unsqueeze(-1).unsqueeze(-1)
            audio_z = audio_z.repeat(1,1, image_z.shape[2], image_z.shape[3])
        #concat_z = torch.cat([image_z, audio_z,aus_inputs,input_wavs], dim=1)
        # concat_z = torch.cat([image_z, audio_z,input_wavs], dim=1)
        concat_z = torch.cat([image_z1,audio_inputs], dim=1)
        #concat_z = torch.cat([image_z1], dim=1)
        #print(concat_z.shape) 25*1372 yuan:25*1344
        # reshape z to (batch_size, seq_len, ...)
        seq_len=1
        if self.img_encoder_type=='FCN':
            concat_z = concat_z.contiguous().view(batch_size, seq_len, concat_z.shape[1], concat_z.shape[2], concat_z.shape[3])
        else:
            concat_z = concat_z.contiguous().view(1,batch_size,  concat_z.shape[1])
        

        '''
        # fed concat_z to RNN, output size: (batch_size, seq_len, hidden_size)
        #concat_z = pack_padded_sequence(concat_z, valid_len, batch_first=True)
        #hidden = self.rnn.init_hidden(batch_size)
        concat_z = concat_z.contiguous().view(batch_size,  concat_z.shape[2],seq_len)
        #print(concat_z.shape) 1*1376*25 yuan:1*1344*25
        
        tcn_output = self.tcn(concat_z)
        #tcn_output, _ = pad_packed_sequence(tcn_output, batch_first=True, total_length=seq_len)


        # reshap rnn output to (seq_len*batch_size, hidden_size)
        tcn_output = tcn_output.contiguous().view(batch_size, tcn_output.shape[2],tcn_output.shape[1])
        tcn_output = tcn_output.contiguous().view(batch_size*seq_len, tcn_output.shape[2])
        # decoder
        G = self.image_decoder(tcn_output, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4)
        G = G.contiguous().view(batch_size, seq_len, G.shape[1], G.shape[2], G.shape[3])
        return G
        '''
        valid_len=[1]

        # fed concat_z to RNN, output size: (batch_size, seq_len, hidden_size)
        concat_z = pack_padded_sequence(concat_z, torch.Tensor(valid_len).to('cpu'), batch_first=True)
        #hidden = self.rnn.init_hidden(batch_size)
        rnn_output, _ = self.rnn(concat_z)
        rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True, total_length=seq_len)
        #print("rnn_output",rnn_output.shape)


        # reshap rnn output to (seq_len*batch_size, hidden_size)
        #rnn_output = rnn_output.contiguous().view(batch_size, rnn_output.shape[2])
        rnn_output = rnn_output[0]

        # decoder
        G = self.image_decoder(rnn_output, img_e_conv11, img_e_conv21, img_e_conv31, img_e_conv41)
        #print(G.shape)
        #G = G.contiguous().view(batch_size,G.shape[1], G.shape[2], G.shape[3])
        return image_z1,G
        

    def model_type(self):
        return 'RNN'



def load_ckpt(model, ckpt_path, prefix=None):
    old_state_dict = torch.load(ckpt_path)
    cur_state_dict = model.state_dict()
    for param in cur_state_dict:
        if prefix is not None:
            old_param = param.replace(prefix, '')
        else:
            old_param = param
        if old_param in old_state_dict and cur_state_dict[param].size()==old_state_dict[old_param].size():
            print("loading param: ", param)
            model.state_dict()[param].data.copy_(old_state_dict[old_param].data)
        else:
            print("warning cannot load param: ", param)






