from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
plt.ion()   # interactive mode
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
from torchvision.models.resnet import Bottleneck
import torchvision.models as models
import math
import torch.utils.model_zoo as model_zoo
# from gensim.models import word2vec
# from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
# import torchmetrics 
     
class HELLMEMECLASS(nn.Module):
 
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(HELLMEMECLASS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fctext = nn.Linear(768, 5)
        self.fcface = nn.Linear(128, 5)
        self.fc1 = nn.Linear(2058,300)
        self.relu1 = nn.ReLU()
        self.bn11 = nn.BatchNorm1d(300 )
        self.fc7 = nn.Linear(300,2,bias =True)
        self.sigmoid = nn.Sigmoid()
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x,y,z):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        #新加層的forward
        # x = x.view(x.size(0), -1)
        # x = self.convtranspose1(x)
        # x = self.maxpool2(x)
        # x = x.view(x.size(0), -1)
        # x = self.fclass(x)
        
        
        x = torch.flatten(x, 1)
        y = y.to(torch.float)
        z = z.to(torch.float)
        y = self.fctext(y)
        # y = self.relu(y)
        z = self.fcface(z)
        # z = self.relu(z)
        # y = self.relu(y)
        # x = self.fcvision(x)
        x = torch.cat([x,y],1) # concatenate two tensor according to second dimension
        x = torch.cat([x,z],1) # concatenate two tensor according to second dimension
        
        # print(y)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn11(x)

     
        # x = self.relu4(x)



        x = self.fc7(x)
        x = self.sigmoid(x)
        

        return x


          

