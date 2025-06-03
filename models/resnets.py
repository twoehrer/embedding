import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """
    Residual Block that includes a batch normalization layer and a skip connection with adjustable skip parameter. 
    skip_param = 0 means no skip connection, skip_param = 1 means standard skip connection.
    The activation function can be set to 'relu', 'tanh', or 'id' (identity).
    """ 
    
    def __init__(self, features, skip_param = 1, activation = 'relu', batchnorm = True):

        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(features, features)
        if batchnorm: #batchnorm is important to stabilize the training for deeper networks
            self.bn = nn.BatchNorm1d(features)
            
        if activation == 'relu':
            self.activation = nn.ReLU()
        if activation == 'tanh':
            self.activation = nn.Tanh()
        if activation == 'id':
            self.activation = nn.Identity()
        self.skip_param = skip_param
    def forward(self, x):
        identity = x #cont here
        out = self.fc(x)
        if hasattr(self, 'bn'):
            out = self.bn(out)
        out = self.activation(out)
        out = out + self.skip_param * identity
        return out

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden, skip_param = 1, activation = 'relu'):
        
        super(ResNet, self).__init__()
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.skip_param = skip_param
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        if activation == 'tanh':
            self.activation = nn.Tanh()
        if activation == 'id':
            self.activation = nn.Identity()
            
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, skip_param=skip_param, activation = activation) for _ in range(num_hidden)]
        )
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, output_layer = True):
        x = self.activation(self.input_fc(x))
        x = self.res_blocks(x)
        if output_layer:
            x = self.output_fc(x)
            x = torch.sigmoid(x)
        return x
    
    '''
    sub_model is used to access a partial network of the input to output network
    layers are counted from 0 (input to hidden dim) until nth layer ( (n-1)th hidden layer to output layer)
    from_layer is the starting layer that is included in the sub_model
    to_layer is the final layer included in the sub_model
    e.g. from_layer = 1, to_layer = 2 includes the first two ResBlocks
    initial layer is counted as layer 0
    hidden to output layer is counted as final layer
    '''
    def sub_model(self, x, from_layer, to_layer):
        if to_layer > self.num_hidden + 1:
            print('Error: to_layer is larger than existing number of layers')
            return
        if from_layer > to_layer:
            print('Error: to_layer cannot be larger than from_layer')
        
        
        if from_layer == 0:
            x = self.activation(self.input_fc(x))
            from_layer += 1 #if from_layer = 0 I need to increase the from_layer count

        if to_layer > 0 and from_layer < self.num_hidden + 1:
            reduced_block = self.res_blocks[from_layer - 1 : to_layer] #from layer 1 to 2 means hidden layer 0 to hidden layer 1
            x = reduced_block(x)
        if to_layer == self.num_hidden + 1:
            x = self.output_fc(x)
        return x
            
