import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function

from pytorch_wavelets import DWTForward, DWTInverse


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU,'prelu': nn.PReLU}

class MLP(nn.Module):
    def __init__(self, channel):
        super(MLP, self).__init__()
        self.linear = nn.Linear(channel, channel, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        rs1 = self.linear(x.permute(0, 2, 3, 1).reshape(B, -1, C))
        rs2 = self.sigmoid(rs1).permute(0, 2, 1).reshape(B, C, H, W)
        return rs2

class resblock(nn.Module):
    def __init__(self, channel):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 1, 1, 0, bias=True)
        self.act = nn.PReLU(num_parameters=channel, init=0.01)

    def forward(self, x):
        rs1 = self.act(self.conv1(x))
        rs2 = self.conv2(rs1) + x
        return rs2
    
    

class EnhancedFourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels,gate, groups=1):
        super(EnhancedFourierUnit, self).__init__()
        self.groups = groups
        self.gate = gate
        self.relu = nn.ReLU(inplace=True)
        if gate == 0 :
            self.q_weights = nn.Parameter(torch.randn(in_channels*2, out_channels*2) * 0.05)
            self.k_weights = nn.Parameter(torch.randn(in_channels*2, out_channels*2) * 0.05)
        else :
            self.q_weight = nn.Parameter(torch.randn(in_channels, out_channels) * 0.05)
            self.k_weight = nn.Parameter(torch.randn(in_channels, out_channels) * 0.05)
        

    def forward(self, q, k, v):

        batch, c, h, w = q.size()

        if self.gate == 0 :

            qffted = torch.fft.rfft2(q, norm='ortho')
            q_fft_real = torch.unsqueeze(torch.real(qffted), dim=-1)  # [B, C, H, W/2+1, 1]
            q_fft_imag = torch.unsqueeze(torch.imag(qffted), dim=-1)  # [B, C, H, W/2+1, 1]
            qffted = torch.cat((q_fft_real, q_fft_imag), dim=-1)       # [B, C, H, W/2+1, 2]

            kffted = torch.fft.rfft2(k, norm='ortho')
            k_fft_real = torch.unsqueeze(torch.real(kffted), dim=-1)  # [B, C, H, W/2+1, 1]
            k_fft_imag = torch.unsqueeze(torch.imag(kffted), dim=-1)  # [B, C, H, W/2+1, 1]
            kffted = torch.cat((k_fft_real, k_fft_imag), dim=-1)       # [B, C, H, W/2+1, 2]

            #  [B, C*2, H, W/2+1] 
            #  (B, C, 2, H, W/2+1) ->  (B, C*2, H, W/2+1)
            qffted = qffted.permute(0, 1, 4, 2, 3).contiguous()
            qffted = qffted.view(batch, -1, h, (w // 2) + 1)

            kffted = kffted.permute(0, 1, 4, 2, 3).contiguous()
            kffted = kffted.view(batch, -1, h, (w // 2) + 1)

            qffted = torch.einsum('bchw,co->bohw', qffted, self.q_weights)
            kffted = torch.einsum('bchw,co->bohw', kffted, self.k_weights)

            ffted = qffted * kffted  

            # reshape  [B, C, H, W/2+1, 2]
            ffted = ffted.view(batch, -1, 2, h, (w // 2) + 1)
            ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()  # [B, C, H, W/2+1, 2]

            ffted = torch.view_as_complex(ffted)  

            output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        
        #shared weight
        else :

            qffted = torch.fft.rfft2(q, norm='ortho')
            kffted = torch.fft.rfft2(k, norm='ortho')

            q_real, q_imag = qffted.real, qffted.imag
            k_real, k_imag = kffted.real, kffted.imag

            q_real = torch.einsum('bchw,co->bohw', q_real, self.q_weight)
            q_imag = torch.einsum('bchw,co->bohw', q_imag, self.q_weight)

            k_real = torch.einsum('bchw,co->bohw', k_real, self.k_weight)
            k_imag = torch.einsum('bchw,co->bohw', k_imag, self.k_weight)

            out_real = q_real * k_real - q_imag * k_imag
            out_imag = q_real * k_imag + q_imag * k_real

            ffted = torch.complex(out_real, out_imag)

            output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')

        output = output * v

        return output
    

class ImprovedFFN_1(nn.Module):
    def __init__(self, in_channel, FFN_channel, out_channel):
        super(ImprovedFFN_1, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, FFN_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(FFN_channel, FFN_channel, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(FFN_channel, out_channel, 1, 1, 0, bias=True)
        self.act = nn.PReLU(num_parameters=FFN_channel, init=0.01)

    def forward(self, x):
        rs1 = self.conv1(x)
        rs2 = self.act(self.conv2(rs1))+ rs1
        rs3 = self.conv3(rs2) 

        return rs3
    
class Wavelet_Mixer(nn.Module):

    def __init__(self, dim, gate):
        super(Wavelet_Mixer, self).__init__()
        self.dim = dim
        self.gate = gate 
        self.wd_ll_conv = MLP(channel=dim)
        self.wd_lh_conv = MLP(channel=dim)
        self.wd_hl_conv = MLP(channel=dim)
        self.wd_hh_conv = MLP(channel=dim)

        self.resblock = resblock(channel=dim)
        self.resblock_1 = resblock(channel=dim)


        self.conv_x = ImprovedFFN_1(in_channel=dim, FFN_channel=dim // 2, out_channel=dim)
        self.conv_v = ImprovedFFN_1(in_channel=dim, FFN_channel=dim // 2, out_channel=dim)

        self.FFC3 = EnhancedFourierUnit(in_channels=dim, out_channels=dim, gate=gate)

        self.dwt = DWTForward(J=1, wave='haar')
        self.idwt = DWTInverse(wave='haar')

    def forward(self, x, band_shuffle: bool=False):

        B, C, H, W = x.shape    
 
        x_low, x_high_list = self.dwt(x)
        _, _, H2, W2 = x_low.shape
        wd_ll = x_low
        wd_lh = x_high_list[0][:, :, 0, :, :]
        wd_hl = x_high_list[0][:, :, 1, :, :]
        wd_hh = x_high_list[0][:, :, 2, :, :]


        v = self.resblock(wd_ll)
        v_ll = self.FFC3(q=wd_ll, k=wd_ll, v=v)
        v_lh = self.FFC3(q=wd_lh, k=wd_ll, v=v)
        v_hl = self.FFC3(q=wd_hl, k=wd_ll, v=v)
        v_hh = self.FFC3(q=wd_hh, k=wd_ll, v=v)

        if band_shuffle:
            perm = torch.randperm(3, device=x.device)
            outs = [v_lh, v_hl, v_hh]
            v_lh, v_hl, v_hh = [outs[i] for i in perm]

        v_idwt = self.idwt((v_ll, [torch.cat([v_lh,v_hl,v_hh],dim=1).view(B, C , 3, H2, W2)]))
        x_idwt = self.idwt((self.wd_ll_conv(wd_ll),
            [torch.cat([self.wd_lh_conv(wd_lh), self.wd_hl_conv(wd_hl), self.wd_hh_conv(wd_hh)],dim=1).view(B, C , 3, H2, W2)]))
        

        v_idwt = v_idwt[..., :H, :W] 
        x_idwt = x_idwt[..., :H, :W] 

        x_1 = self.conv_x(x_idwt) + self.conv_v(v_idwt)
        
        x = self.resblock_1(x_1)

        return x


