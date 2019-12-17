import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

class MyModel(nn.Module):
    def __init__(self, img_size=64, no_of_classes=2, ksize=5, dim=64):
        self.no_of_classes = no_of_classes
        self.img_size = img_size
        self.ksize = ksize
        self.dim = dim
        
        super().__init__()

        self.conv1 = nn.Conv2d(1,dim,ksize)
        self.conv2 = nn.Conv2d(dim,dim*2,ksize)
        self.conv3 = nn.Conv2d(dim*2,dim*4,ksize)

        self.linput_shape = None
        #self.forward(torch.randn(self.img_size,self.img_size).view(-1,1,self.img_size,self.img_size))
        self.convpass(torch.randn(self.img_size,self.img_size).view(-1,1,self.img_size,self.img_size))


        self.linear1 = nn.Linear(self.linput_shape,512)
        self.linear2 = nn.Linear(512,self.no_of_classes)
    
    def convpass(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self.linput_shape is None:
            self.linput_shape = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x

    def forward(self,x):
        x = self.convpass(x)        
        x = x.view(-1, self.linput_shape)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        return F.softmax(x, dim=1)