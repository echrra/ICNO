import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, reduce
from pytorch_wavelets import DWTForward, DWTInverse
import math

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,'PReLU': nn.PReLU}

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()


        act = ACTIVATION[act]
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        return x
    

class Single_LinearAttention_Galerkin(torch.nn.Module):
    def __init__(self, n_dim, n_head, ):
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.dim_head = self.n_dim // self.n_head

        self.to_qkv = torch.nn.Linear(n_dim, n_dim*3, bias = False)
        self.project_out = (not self.n_head == 1)
        self.Wq = torch.nn.Linear(self.n_dim, self.n_dim)
        self.Wk = torch.nn.Linear(self.n_dim, self.n_dim)
        self.Wv = torch.nn.Linear(self.n_dim, self.n_dim)
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(self.n_dim, self.n_dim),
            torch.nn.GELU()
        ) if self.project_out else torch.nn.Identity()

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_head)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_head)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_head)

        dots = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, dots) * (1./q.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return out
              

