import itertools
import numpy as np
import torch
import torch.nn as nn



def get_refinement_block(model='automap_scae', in_channel=40, out_channel=1):
    if model == 'automap_scae':
        return nn.Sequential(nn.Conv2d(in_channel, 64, 3, 1, 2), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 2), nn.ReLU(True),
                             nn.ConvTranspose2d(64, out_channel, 3, 1, 3))
    elif model == 'simple':
        return nn.Sequential(nn.Conv2d(in_channel, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.Conv2d(64, out_channel, 3, 1, 1))
    elif model == 'my':
        return nn.Sequential()
    else:
        raise NotImplementedError



class AUTOMAP(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(AUTOMAP,self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_reshape = int(np.prod(self.input_shape))
        self.output_reshape = int(np.prod(self.output_shape))
        self.domain_transform = nn.Linear(self.input_reshape, self.output_reshape*2)
        self.domain_transform2 = nn.Linear(self.output_reshape*2, self.output_reshape*4)
        self.sparse_convolutional_autoencoder = get_refinement_block('automap_scae', 4, output_shape[0])

    def forward(self, x):
        """Expects input_shape (batch_size, 2, ndim, ndim)"""
        batch_size = len(x) #[10,1,128,128]
        x = x.reshape(batch_size, int(np.prod(self.input_shape))) #[10,1,16384]
        x = torch.tanh(self.domain_transform(x))
        x = torch.tanh(self.domain_transform2(x))
        x = x.reshape(batch_size,-1, 128,128)
        x = self.sparse_convolutional_autoencoder(x)
        # nn.Conv2d(in_channel, 64, 3, 1, 2), nn.ReLU(True),
        # nn.Conv2d(64, 64, 3, 1, 2), nn.ReLU(True),
        # nn.ConvTranspose2d(64, out_channel, 3, 1, 3)
        return x