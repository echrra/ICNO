import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import trunc_normal_
from model.Embedding import timestep_embedding
from model.WF_Attention import  *
from model.C_Attention import *
ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,'prelu': nn.PReLU}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x
    
class CrossAtt_block(nn.Module):


    def __init__(
            self,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            H=85,
            W=85
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.H=H
        self.W=W
        
        self.Attn = Single_LinearAttention_Galerkin(n_dim=hidden_dim, n_head = 8)


        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
        

    def forward(self, fx):

        fx_ln = self.ln_1(fx)    # [B, N, C]

        B, N, C = fx_ln.shape
        # [B, C, H, W]
        assert N == self.H * self.W, "need N = H*W"
        x = fx_ln.permute(0, 2, 1).reshape(B, C, self.H, self.W)

        x = self.Attn(x)  # [B, C, H, W]

        # reshape 
        x = x.reshape(B, C, N).permute(0, 2, 1)  # => [B, N, C]

        fx = x + fx

        # MLP 
        fx_ln2 = self.ln_2(fx)
        x2 = self.mlp(fx_ln2)
        fx = fx + x2

        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx

class WFAtt_block(nn.Module):

    def __init__(
            self,
            gate:float,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            H=85,
            W=85
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.H=H
        self.W=W
        
        self.Attn = Wavelet_Mixer(dim=hidden_dim,gate=gate)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
        

    def forward(self, fx):


        fx_ln = self.ln_1(fx)    # [B, N, C]

        B, N, C = fx_ln.shape
        # [B, C, H, W]
        assert N == self.H * self.W, "need N = H*W"
        x = fx_ln.permute(0, 2, 1).reshape(B, C, self.H, self.W)
        x = self.Attn(x)  # [B, C, H, W]

        x = x.reshape(B, C, N).permute(0, 2, 1)  # => [B, N, C]

        fx = x + fx

        # MLP 
        fx_ln2 = self.ln_2(fx)
        x2 = self.mlp(fx_ln2)
        fx = fx + x2

        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=64,
                 dropout=0.0,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 gate=-1,
                 H=85,
                 W=85,
                 ):
        super(Model, self).__init__()
        self.__name__ = 'ICNO_2D'
        self.H = H
        self.W = W
        
        self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))


        self.blocks =nn.ModuleList()
        alpha = 8
        rate = 3 
        for i in range(n_layers):

            if (i+1) % (rate+1) == 0:
                layer =CrossAtt_block(hidden_dim=n_hidden,dropout=dropout,act=act,mlp_ratio=mlp_ratio,out_dim=out_dim,H=H,W=W,last_layer=(i == n_layers - 1))
                self.blocks.append(layer)
            else:
                layer = WFAtt_block(gate=gate, hidden_dim=n_hidden,dropout=dropout,act=act,mlp_ratio=mlp_ratio,out_dim=out_dim,H=H,W=W,last_layer=(i == n_layers - 1))
                self.blocks.append(layer)

                                   
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx, T=None):

        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)

        return fx
