from numpy import true_divide
import torch
from torch import nn
from torch.nn import functional as F
from utils import utils as utils
from matplotlib import pyplot as plt
import torchvision


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


class My_Model(torch.nn.Module):
    def __init__(self):
        super(My_Model, self).__init__()
        pretrained_net = torchvision.models.resnet50(pretrained=True)
        #copies all the pretrained layers in the ResNet-18 except for 
#the final global average pooling layer and the fully-connected layer that are closest to the output
        self.net = nn.Sequential(*list(pretrained_net.children())[:-2])
        self.num_classes = 1
        self.net.add_module('final_conv', nn.Conv2d(2048, self.num_classes, kernel_size=1))
        self.net.add_module('transpose_conv', nn.ConvTranspose2d(self.num_classes, self.num_classes,
                                            kernel_size=64, padding=16, stride=32))
        self.net.add_module('activation',nn.Sigmoid())
        W = bilinear_kernel(self.num_classes, self.num_classes, 64)
        self.net.transpose_conv.weight.data.copy_(W)

    def forward(self, x):
        return  self.net(x)
    
    def save(self, path):
        torch.save(self.net.state_dict(), path)






 
