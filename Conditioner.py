import torch.nn as nn
import torch.nn.functional as F
from CONST_VARS import CONST

class ConditionerBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel, stride) 
        self.batchnorm = nn.BatchNorm2d(out_dim)


    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return F.relu(x)

class Conditioner(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = ConditionerBlock(1, 16, 3,2)
        self.conv1 = ConditionerBlock(16, 32,   3,2)
        self.conv2 = ConditionerBlock(32, 64,   3,2)
        self.conv3 = ConditionerBlock(64, 128,  3,2)
        self.conv4 = ConditionerBlock(128, CONST.latent_dim, 3,2)

    def forward(self, x): #! torch.Size([16, 1, 64, 72])
        # x = x.view(-1, CONST.n_measures * CONST.measure_resolution, CONST.n_pitches) 
        x = self.conv0(x) #! torch.Size([16, 16, 31, 35])
        x = self.conv1(x) #! torch.Size([16, 32, 15, 17])
        x = self.conv2(x) #! torch.Size([16, 64, 7, 8])
        x = self.conv3(x) #! torch.Size([16, 128, 3, 3])
        x = self.conv4(x) #! torch.Size([16, 128, 1, 1])
        x = x.view(-1, CONST.latent_dim)  

        return x
    
