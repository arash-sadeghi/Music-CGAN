import torch.nn as nn
import torch.nn.functional as F
from CONST_VARS import CONST

class ConditionerBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, kernel, stride) 
        self.batchnorm = nn.BatchNorm3d(out_dim)


    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return F.relu(x)

class Conditioner(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = ConditionerBlock(1, 16, (1, 1, 12), (1, 1, 12))
        self.conv1 = ConditionerBlock(16, 32, (1, 4, 1), (1, 4, 1))
        self.conv2 = ConditionerBlock(32, 64, (1, 1, 3), (1, 1, 1))
        self.conv3 = ConditionerBlock(64, 128, (1, 1, 4), (1, 1, 4))
        self.conv4 = ConditionerBlock(128, 256, (1, 4, 1), (1, 4, 1))
        self.conv5 = ConditionerBlock(256 , CONST.latent_dim , (4, 1, 1), (4, 1, 1))


    def forward(self, x): 
        x = x.view(CONST.BATCH_SIZE ,1, CONST.n_measures , CONST.measure_resolution, CONST.n_pitches) 
        x = self.conv0(x) 
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) 
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, CONST.latent_dim)  

        return x
    
