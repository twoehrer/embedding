import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiagonalLinear(nn.Module):
    def __init__(self, in_features, out_features, fixed_w00 = None):
        super().__init__()
        self.dim = min(in_features, out_features)
        self.out_features = out_features
        if fixed_w00 is None:
            self.weight = nn.Parameter(torch.ones(self.dim))
            self.fixed = None
        else:
            # first weight fixed (buffer), others trainable
            self.register_buffer("fixed", torch.tensor([float(fixed_w00)]))  # shape [1]
            self.weight_rest = nn.Parameter(torch.ones(max(self.dim - 1, 0)))
                                
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        
        
    def _weight_vec(self, x):
        # build [w00_fixed, w2, ..., w_dim] on the fly (correct device/dtype)
        if self.fixed is None:
            return self.weight
        return torch.cat((self.fixed.to(x.device, x.dtype), self.weight_rest), dim=0)

    def forward(self, x):
        w = self._weight_vec(x)                   # length = self.dim
        out = x[..., :self.dim] * w               # diagonal scaling
        pad = self.out_features - out.shape[-1]   # pad to out_features if needed
        if pad > 0:
            out = F.pad(out, (0, pad))
        return out + self.bias


class ResidualBlock(nn.Module):
    """
    Residual Block that includes a batch normalization layer and a skip connection with adjustable skip parameter. 
    skip_param = 0 means no skip connection, skip_param = 1 means standard skip connection.
    The activation function can be set to 'relu', 'tanh', or 'id' (identity).
    """ 
    
    def __init__(self, features, skip_param = 1, sara_param = 1, activation = 'relu', batchnorm = True):

        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(features, features)
        
        # --- Xavier initialization (good for tanh, id; works ok for relu if you prefer simpler setup)
        nn.init.xavier_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        
        if batchnorm: #batchnorm is helpful to stabilize the training for deeper networks
            self.bn = nn.BatchNorm1d(features)
            
        if activation == 'relu':
            self.activation = nn.ReLU()
        if activation == 'tanh':
            self.activation = nn.Tanh()
        if activation == 'id':
            self.activation = nn.Identity()
        self.skip_param = skip_param
        self.sara_param = sara_param
    def forward(self, x):
        identity = x #cont here
        out = self.fc(x)
        if hasattr(self, 'bn'):
            out = self.bn(out)
        out = self.activation(out)
        out = self.sara_param * out + self.skip_param * identity
        return out

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden, skip_param = 1,  sara_param = 1, activation = 'relu', final_sigmoid = True, batchnorm = True, input_layer = True, input_layer_diagonal = False, fixed_w00 = None):
        
        super(ResNet, self).__init__()
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.skip_param = skip_param
        self.sara_param = sara_param
        self.input_layer_exists = input_layer
        self.input_layer_diagonal = input_layer_diagonal
        if activation == 'relu':
            self.activation = nn.ReLU()
        if activation == 'tanh':
            self.activation = nn.Tanh()
        if activation == 'id':
            self.activation = nn.Identity()
        self.final_sigmoid = final_sigmoid
            
        if self.input_layer_exists: #if i stay in the dimension of the input i do not need an input layer, so this is optional for e.g. the 1d to 1d example
            if self.input_layer_diagonal:
                self.input_fc = DiagonalLinear(self.input_dim, self.hidden_dim, fixed_w00 = fixed_w00)
            else:
                self.input_fc = nn.Linear(input_dim, hidden_dim)
                
            self.input_layer = nn.Sequential(
                self.input_fc,
                self.activation
            )
            
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, skip_param=skip_param, sara_param = sara_param, activation = activation, batchnorm=batchnorm) for _ in range(num_hidden)]
        )
        if final_sigmoid:
            self.output_fc = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),nn.Sigmoid())
        else:
            self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, output_layer = True):
        if self.input_layer_exists:
            x = self.input_layer(x)
        x = self.res_blocks(x)
        if output_layer:
            x = self.output_fc(x)
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
            from_layer += 1 #if from_layer = 0 I need to increase the from_layer count for the next if statements

        if to_layer > 0 and from_layer < self.num_hidden + 1:
            reduced_block = self.res_blocks[from_layer - 1 : to_layer] #from layer 1 to 2 means hidden layer 0 to hidden layer 1
            x = reduced_block(x)
        if to_layer == self.num_hidden + 1:
            x = self.output_fc(x) #notice that the model output can also include a final sigmoid activation for normalization
        return x
            

    def sub_model_new(self, x, from_layer, to_layer):
        import itertools
        
        #generates a list that includes all realized layers of the model
        realized_sequence = []
        if self.input_layer_exists:
            realized_sequence.append(self.input_layer)
        for block in self.res_blocks:
            realized_sequence.append(block)  
        realized_sequence.append(self.output_fc)
        
        #slices the layers wanted in the sub_model
        sub_model = realized_sequence[from_layer:to_layer + 1]
        sub_model = nn.Sequential(*sub_model)
        return sub_model(x)