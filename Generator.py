import torch.nn as nn
import torch.nn.functional as F
from CONST_VARS import CONST

class GeneraterBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = nn.ConvTranspose3d(in_dim, out_dim, kernel, stride) #? what is 3d transposed convolution
        self.batchnorm = nn.BatchNorm3d(out_dim)

    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return F.relu(x)

class Generator(nn.Module):
    """A convolutional neural network (CNN) based generator. The generator takes
    as input a latent vector and outputs a fake sample."""
    def __init__(self):
        super().__init__()
        self.transconv0 = GeneraterBlock(CONST.latent_dim, 256, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneraterBlock(256, 128, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneraterBlock(128, 64, (1, 1, 4), (1, 1, 4))
        self.transconv3 = GeneraterBlock(64, 32, (1, 1, 3), (1, 1, 1))
        self.transconv4 = nn.ModuleList([
            GeneraterBlock(32, 16, (1, 4, 1), (1, 4, 1))
            for _ in range(CONST.n_tracks) #? what is this for loop for?
        ])
        self.transconv5 = nn.ModuleList([
            GeneraterBlock(16, 1, (1, 1, 12), (1, 1, 12))
            for _ in range(CONST.n_tracks)
        ])

    def forward(self, x):
        x = x.view(-1, CONST.latent_dim, 1, 1, 1)  #! torch.Size([1, 128]) -> torch.Size([1, 128, 1, 1, 1])
        x = self.transconv0(x) #! torch.Size([1, 128, 1, 1, 1]) -> torch.Size([1, 256, 4, 1, 1])
        x = self.transconv1(x) #! torch.Size([1, 256, 4, 1, 1]) -> torch.Size([1, 128, 4, 4, 1])
        x = self.transconv2(x) #! torch.Size([1, 128, 4, 4, 1]) -> torch.Size([1, 64, 4, 4, 4])
        x = self.transconv3(x) #! torch.Size([1, 64, 4, 4, 4]) -> torch.Size([1, 32, 4, 4, 6])
        x = [transconv(x) for transconv in self.transconv4] #! torch.Size([1, 32, 4, 4, 6]) -> list with 5 element, each torch.Size([1, 16, 4, 16, 6]) #? what? does this correspond to any architecture form paper? 
        x = CONST.torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv5)], 1)#! list with 5 element, each torch.Size([1, 16, 4, 16, 6]) -> torch.Size([1, 5, 4, 16, 72]) #? what? why?


        #! in single sample example for overfitting: torch.Size([1, 5, 4, 16, 72]) torch.Size([1, 5, 64, 72])
        x = x.view(-1, CONST.n_tracks, CONST.n_measures * CONST.measure_resolution, CONST.n_pitches) #! torch.Size([1, 5, 4, 16, 72]) -> torch.Size([1, 5, 64, 72])
        #! in single sample example for overfitting: torch.Size([1, 5, 64, 72])

        return x
    




# Generate Fake Data: output like real data [1, 28, 28] and values -1, 1
class GeneratorMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2) # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2) # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # [n, 1, 28, 28]


    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)  #256

        # Upsample (transposed conv) 16x16 (64 feature maps)
        x = self.ct1(x)
        x = F.relu(x)

        # Upsample to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)

        # Convolution to 28x28 (1 feature map)
        return self.conv(x)