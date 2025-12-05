import numpy as np
import torch
import torch.nn as nn
import math

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1) # in_dim
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
    return m

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
    return m



class FourierFeature(nn.Module):
    def __init__(self, coor_in_dim = 3, coor_emb_dim = 512):
        super().__init__()
        fourier_basis = torch.randn(coor_in_dim, coor_emb_dim // 2) * 10
        self.register_buffer('_fourier_basis', fourier_basis)
        
    def forward(self,x): # x -> ( n , coor_in_dim=3 )
        x = 2*math.pi*x @ self._fourier_basis  # matrix multiplication ( n , coor_emb_dim//2 )
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1) # ( n , coor_emb_dim//2 + coor_emb_dim//2 )

class Sine(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * x)

class Siren(nn.Module):
    def __init__(self, depth=8, coor_in_dim=3, coor_emb_dim=512 ,hidden_size=768):
        super().__init__()
        # map
        self.mapper_ff = FourierFeature(coor_in_dim, coor_emb_dim)
        # layers
        layers = []
        layers.append(first_layer_sine_init(nn.Linear(coor_emb_dim,hidden_size))) # sine init first layer
        layers.append(Sine()) # sine activation
        for i in range(depth-2):
            layers.append(sine_init(nn.Linear(hidden_size,hidden_size)))
            layers.append(Sine())
        layers.append(nn.Linear(hidden_size,1)) # output layer (hidden_size, 1)
        self.layers = nn.Sequential(*layers)
    def forward(self,x,mode=None):
        x = self.mapper_ff(x) # map to higher dim (n,coor_in_dim) -> (n,coor_emb_dim)
        if mode is None:
            return torch.sin(30*self.layers(x)) # (-1,1)
        if mode == 'sigmoid':
            return torch.sigmoid(self.layers(x)) # (0,1), generally
        if mode == 'linear':
            return self.layers(x)

