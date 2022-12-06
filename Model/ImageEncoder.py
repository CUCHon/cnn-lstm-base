import random
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import init
import torch.nn.functional as Func
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class ImageEncoder(nn.Module): #图像encoder
    def __init__(self):
        super(ImageEncoder,self).__init__()
        Resnet = models.resnet152(pretrained=True)
        local_features_module=list(Resnet.children())[0:8]#前八层是特征提取
        global_features_module=list(Resnet.children())[8]#第九层是池化，最后一层是softmax分类
        self.resnet_local = nn.Sequential(* local_features_module)
        self.resnet_global = nn.Sequential(global_features_module)
    def forward(self,Image):
        with torch.no_grad():#不更新resnet
            local_features=self.resnet_local(Image)
            global_features=self.resnet_global(local_features)
        return  global_features