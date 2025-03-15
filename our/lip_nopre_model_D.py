import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchvision import models
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    # elif classname.find('GRU') != -1 or classname.find('LSTM') != -1:
    #     m.weight.data.normal_(0.0, 0.02)
    #     m.bias.data.fill_(0.01)
    else:
        print(classname)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), nn.InstanceNorm2d(dim), nn.ReLU(inplace=True)]
        conv_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), nn.InstanceNorm2d(dim)]
        self.conv_blocks = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_blocks(x)
        return out

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [ h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class classifier_AU(nn.Module):
    def __init__(self):
        super(classifier_AU, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=1), nn.InstanceNorm2d(16),nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, stride=1, padding=1), nn.InstanceNorm2d(32),nn.LeakyReLU(0.2, True))
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 5, stride=2, padding=1), nn.InstanceNorm2d(64),nn.LeakyReLU(0.2, True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=1, padding=1), nn.InstanceNorm2d(128),nn.LeakyReLU(0.2, True))
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 5, stride=2, padding=1), nn.InstanceNorm2d(64),nn.LeakyReLU(0.2, True))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 512, 5, stride=1, padding=1), nn.InstanceNorm2d(128),nn.LeakyReLU(0.2, True))
        self.maxpool3 = nn.MaxPool2d(2,2)
        self.linear1  = nn.Sequential(nn.Linear(5*5*512 ,5*512), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        self.linear2  = nn.Sequential(nn.Linear(5*512 ,512), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        self.linear3  = nn.Sequential(nn.Linear(512 ,256), nn.LeakyReLU(0.2, True),nn.Dropout(p=0.5))
        self.linear4 = nn.Linear(256 ,1)
        #self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, inputs):
        out = self.conv1(inputs)
        #print(out.shape)    
        out = self.conv2(out)
        #print(out.shape)
        out = self.maxpool1(out)
        out = self.conv3(out)
        #print(out.shape)
        out = self.conv4(out)
        #print(out.shape)
        out = self.maxpool2(out)
        #print(out.shape)
        out = self.conv5(out)
        #print(out.shape)
        out = self.conv6(out)
        #print(out.shape)
        out = self.maxpool3(out)
        #print(out.shape)
        out = out.contiguous().view(out.shape[0], -1)
        #print(out.shape)
        out = self.linear1(out)
        out = self.linear2(out)
        linear3_out = self.linear3(out)
        out = self.linear4(linear3_out)
        out = self.sigmoid(out)
        #out = self.softmax(out)
        return linear3_out,out

class DiscriminatorFrame(nn.Module):
    def __init__(self):
        super(DiscriminatorFrame, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.InstanceNorm2d(16), nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True))
        self.conv5 = nn.Conv2d(128, 1, 1)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.discriminator = DiscriminatorFrame()

    def forward(self, inputs):
        if len(inputs.shape)==5:
            inputs = inputs.contiguous().view(inputs.shape[0]*inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4])
        return self.discriminator(inputs)




