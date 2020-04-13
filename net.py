import itertools
import numpy as np
import torch
import torch.nn as nn



def get_refinement_block(model='automap_scae', in_channel=1, out_channel=1):
    if model == 'automap_scae':
        return nn.Sequential(nn.Conv2d(in_channel, 64, 5, 1, 2), nn.ReLU(True),
                             nn.Conv2d(64, 64, 5, 1, 2), nn.ReLU(True),
                             nn.ConvTranspose2d(64, out_channel, 7, 1, 3))
    elif model == 'simple':
        return nn.Sequential(nn.Conv2d(in_channel, 32, kernel_size=(3,3),stride=(1,1)), nn.ReLU(True),
                             nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1)), nn.ReLU(True),
                             #nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             #nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
                             nn.ConvTranspose2d(32, out_channel,kernel_size=(3,3), stride=(1,1)))
    else:
        raise NotImplementedError



class AUTOMAP(nn.Module):
    def __init__(self,input_shape,output_shape):
        super(AUTOMAP,self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_reshape = int(np.prod(self.input_shape))
        self.output_reshape = int(np.prod(self.output_shape))
        self.domain_transform = nn.Linear(self.input_reshape, self.output_reshape)
        self.domain_transform2 = nn.Linear(self.output_reshape, self.output_reshape)
        self.sparse_convolutional_autoencoder = get_refinement_block('automap_scae', output_shape[0], output_shape[0])

    def forward(self, x):
        """Expects input_shape (batch_size, 2, ndim, ndim)"""
        batch_size = len(x)
        x = x.reshape(batch_size, int(np.prod(self.input_shape)))
        x = torch.tanh(self.domain_transform(x))
        x = torch.tanh(self.domain_transform2(x))
        x = x.reshape(-1, *self.output_shape)
        x = self.sparse_convolutional_autoencoder(x)
        return x