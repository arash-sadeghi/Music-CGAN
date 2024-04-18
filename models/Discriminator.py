import torch.nn as nn
import torch.nn.functional as F
from CONST_VARS import CONST

class LayerNorm(nn.Module):
    """An implementation of Layer normalization that does not require size
    information. Copied from https://github.com/pytorch/pytorch/issues/1959."""
    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(CONST.torch.Tensor(n_features).uniform_())
            self.beta = nn.Parameter(CONST.torch.zeros(n_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y
    
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = nn.Conv3d(in_dim, out_dim, kernel, stride)
        self.layernorm = LayerNorm(out_dim)

    def forward(self, x):
        x = self.transconv(x)
        x = self.layernorm(x)
        return nn.functional.leaky_relu(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        embedding_size = CONST.n_pitches
        num_embeddings = len(CONST.genre_code)  
        self.embedder = nn.Embedding(num_embeddings, embedding_size)
        self.conv0 = DiscriminatorBlock(3, 16, (1, 1, 12), (1, 1, 12)) 
        self.conv1 = DiscriminatorBlock(16, 16, (1, 4, 1), (1, 4, 1)) 
        self.conv2 = DiscriminatorBlock(16, 64, (1, 1, 3), (1, 1, 1))
        self.conv3 = DiscriminatorBlock(64, 64, (1, 1, 4), (1, 1, 4))
        self.conv4 = DiscriminatorBlock(64, 128, (1, 4, 1), (1, 4, 1))
        self.conv5 = DiscriminatorBlock(128, 128, (2, 1, 1), (1, 1, 1))
        self.conv6 = DiscriminatorBlock(128, 256, (3, 1, 1), (3, 1, 1))
        self.dense = nn.Linear(256, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, genre):
        genre = self.embedder(genre)
        genre = genre.unsqueeze(1).repeat(1,64,1).unsqueeze(1)
        x = CONST.torch.cat((x,genre),axis = 1)
        x = x.view(-1, 3,  CONST.n_measures , CONST.measure_resolution, CONST.n_pitches)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 256)
        x = self.dense(x)
        # x = self.sigmoid(x)
        return x
