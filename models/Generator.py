import torch.nn as nn
import torch.nn.functional as F
from CONST_VARS import CONST
class GeneraterBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride ,output_padding = 0):
        super().__init__()
        self.transconv = nn.ConvTranspose3d(in_dim, out_dim, kernel, stride ,output_padding = output_padding)
        self.batchnorm = nn.BatchNorm3d(out_dim)

    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x) 
        # return F.relu(x)
        return F.sigmoid(x)

class ConditionerBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, kernel, stride) 
        self.batchnorm = nn.BatchNorm3d(out_dim)


    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        # return F.relu(x)
        return F.leaky_relu(x)

class Generator(nn.Module):
    latent_depth_chanel = 3*CONST.latent_dim
    def __init__(self):
        super().__init__() #! layer norms are adjusted as such to generate an ouput of lenght 64x72

        embedding_size = CONST.latent_dim
        num_embeddings = len(CONST.genre_code)  
        self.embedder = nn.Embedding(num_embeddings, embedding_size)
        
        self.conv0 = ConditionerBlock(1, 16, (1, 1, 12), (1, 1, 12))
        self.conv1 = ConditionerBlock(16, 32, (1, 4, 1), (1, 4, 1))
        self.conv2 = ConditionerBlock(32, 64, (1, 1, 3), (1, 1, 1))
        self.conv3 = ConditionerBlock(64, 128, (1, 1, 4), (1, 1, 4))
        self.conv4 = ConditionerBlock(128, 256, (1, 4, 1), (1, 4, 1))
        self.conv5 = ConditionerBlock(256 , CONST.latent_dim , (4, 1, 1), (4, 1, 1))

        self.transconv0 = GeneraterBlock(Generator.latent_depth_chanel, 256, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneraterBlock(256*2, 128, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneraterBlock(128*2, 64, (1, 1, 4), (1, 1, 4))
        self.transconv3 = GeneraterBlock(64*2, 32, (1, 1, 3), (1, 1, 1))
        self.transconv4 = GeneraterBlock(32*2, 16, (1, 4, 1), (1, 4, 1))
        self.transconv5 = GeneraterBlock(16*2, 1, (1, 1, 12), (1, 1, 12))


    def forward(self, x , condition , genre): #! torch.Size([1, 128])

        condition = condition.view(-1,1, CONST.n_measures , CONST.measure_resolution, CONST.n_pitches) 
        # x = x.view(CONST.BATCH_SIZE ,1, CONST.n_measures , CONST.measure_resolution, CONST.n_pitches) 
        condition0 = self.conv0(condition) 
        condition1 = self.conv1(condition0)
        condition2 = self.conv2(condition1)
        condition3 = self.conv3(condition2) 
        condition4 = self.conv4(condition3)
        condition5 = self.conv5(condition4)
        condition5 = condition5.view(-1, CONST.latent_dim)        
        
        genre = self.embedder(genre)

        x = CONST.torch.cat((x, condition5,genre),axis=1)
        
        x = x.view(-1, Generator.latent_depth_chanel, 1, 1, 1) 
        
        x0 = self.transconv0(x) 
        x1 = self.transconv1(CONST.torch.cat((x0,condition4),axis = 1)) 
        x2 = self.transconv2(CONST.torch.cat((x1,condition3),axis = 1)) 
        x3 = self.transconv3(CONST.torch.cat((x2,condition2),axis = 1)) 
        x4 = self.transconv4(CONST.torch.cat((x3,condition1),axis = 1)) 
        x5 = self.transconv5(CONST.torch.cat((x4,condition0),axis = 1)) 
        x5 = x5.view(-1, 1, CONST.n_measures * CONST.measure_resolution, CONST.n_pitches)
        return x5
