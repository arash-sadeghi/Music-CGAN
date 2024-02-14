import torch.nn as nn
import torch.nn.functional as F
from CONST_VARS import CONST

class GeneraterBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride ,output_padding = 0):
        super().__init__()
        self.transconv = nn.ConvTranspose2d(in_dim, out_dim, kernel, stride ,output_padding = output_padding)
        self.batchnorm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x) 
        return F.relu(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__() #! layer norms are adjusted as such to generate an ouput of lenght 64x72
        self.transconv0 = GeneraterBlock(CONST.latent_dim*2, 128, 3,2, output_padding=(0,0))
        self.transconv1 = GeneraterBlock(128, 64,               3,2,output_padding=(0,1))
        self.transconv2 = GeneraterBlock(64, 32,                3,2,output_padding=(0,0))
        self.transconv3 = GeneraterBlock(32, 16,                3,2,output_padding=(0,0))
        self.transconv4 = GeneraterBlock(16, 1,                 3,2,output_padding=(1,1))

    def forward(self, x): #! torch.Size([1, 128])
        # layer = 0
        x = x.view(-1, CONST.latent_dim*2, 1, 1) #! torch.Size([1, 128, 1, 1])
        # print(f"layer {layer} size {x.shape}");layer+=1        

        x = self.transconv0(x) #! torch.Size([1, 128, 4, 4]) #! torch.Size([1, 128, 3, 3])
        # print(f"layer {layer} size {x.shape}");layer+=1        

        x = self.transconv1(x) #! torch.Size([1, 64, 10, 10]) #! torch.Size([1, 64, 7, 7])
        # print(f"layer {layer} size {x.shape}");layer+=1        

        x = self.transconv2(x) #! torch.Size([1, 32, 22, 22]) #! torch.Size([1, 32, 15, 15])
        # print(f"layer {layer} size {x.shape}");layer+=1        

        x = self.transconv3(x) #! torch.Size([1, 16, 46, 46]) #! torch.Size([1, 16, 31, 31])
        # print(f"layer {layer} size {x.shape}");layer+=1        

        x = self.transconv4(x) #! torch.Size([1, 1, 94, 94]) #! torch.Size([1, 1, 63, 63])
        # print(f"layer {layer} size {x.shape}");layer+=1        

        x = x.view(-1, 1, CONST.n_measures * CONST.measure_resolution, CONST.n_pitches) 

        return x
