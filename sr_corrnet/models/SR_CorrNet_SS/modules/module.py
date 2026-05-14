import torch
import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
import numpy as np

from sr_corrnet.utils.decorators import logger_wraps
from .network import TransBlock, TransDecoderBlock, CS_TransBlock



class Encoder(nn.Module):
    def __init__(self, d_model: int, kernel_sizes: list, num_mics: int, taps_freq: list, taps_frame: list):
        super().__init__()

        self.L_p, self.L_f = taps_frame[0], taps_frame[1]
        self.L_h, self.L_l = taps_freq[0], taps_freq[1]
        self.L_time = self.L_p + self.L_f + 1 
        self.L_freq = self.L_h + self.L_l + 1
        d_input = 2*(num_mics*self.L_time*self.L_freq)

        self.embed = nn.Sequential(
                                    nn.Conv2d(d_input, 4*d_model, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
                                    nn.SiLU(),
                                    nn.Conv2d(4*d_model, d_model, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2),
                                    )
        self.layer_norm = nn.RMSNorm(d_model)
        
    def forward(self, x: torch.tensor, x_mf: torch.tensor):

        h = self.get_correlation(x, x_mf) # B, T, F, C
        h = h.permute(0, 3, 1, 2) # B, C, T, F
        h = self.embed(h)
        h = h.permute(0, 2, 3, 1) # B, T, F, C'

        h = self.layer_norm(h)

        return h

    @torch.no_grad()
    def get_correlation(self, x, x_mf): 
        # x : B, F, T
        # x_mf : B, M, L, F, T
        x_rms = torch.sqrt(torch.mean(x.abs()**2, dim=2, keepdim=True) + 1.0e-12) # B, F, 1
        x = x / x_rms
        x_mf = x_mf / x_rms.reshape(x.shape[0],1,1,-1,1)

        x_abs = x.abs()
        x_mf_abs = x_mf.abs()
        corr = torch.einsum("...ft,...mlft->...tflm", x, x_mf.conj()) # B, T, F, L_tf, M
        scot = torch.einsum("...ft,...mlft->...tflm", x_abs, x_mf_abs) # B, T, F, L_tf, M
        corr = corr / torch.sqrt(scot + 1.0e-12)
        B, T, F = corr.shape[:3]
        corr = torch.view_as_real(corr) # B, T, F, L_t*L_t, L_f*L_f, M*M, 2
        corr = corr.reshape(B, T, F, -1) # B, T, F, L_t*L_t, L_f*L_f, M*M
        return corr
        
    


class CrossSpkBlock(nn.Module):
    def __init__(self, d_model, d_hidden, n_head, dropout_rate):
        super().__init__()
        self.block = CS_TransBlock(d_model, d_hidden, n_head, dropout_rate)
        
    def forward(self, x: torch.tensor, speaker_mask=None):
        
        '''
        input : B, T, F, C
        output : B, T, F, C
        '''
        B, N, T, F, C = x.shape
        x = x.permute(0,2,3,1,4).reshape(B*T*F, N, C)
        x = self.block(x)
            
        x = x.reshape(B, T, F, N, C).permute(0,3,1,2,4)
            
        return x


class TF_Block(nn.Module):

    def __init__(self, channel_module: dict,  spectral_module: dict, rope):
        """Construct an EncoderLayer object."""
        super(TF_Block, self).__init__()

        class FreqModule(nn.Module):

            def __init__(self, d_model: int, d_hidden: int, n_head: int, kernel_size: int, dropout_rate: float, rope):
                super().__init__()

                self.block = TransBlock(d_model, d_hidden, n_head, kernel_size, dropout_rate, rope)

            def forward(self, x: torch.tensor):
                '''
                input : B, T, F, C
                output : B, T, F, C
                '''
                
                B, T, F, C = x.shape
                x = x.reshape(B*T, F, C)
                x = self.block(x)
                x = x.reshape(B, T, F, C)
                
                return x

        class TimeModule(nn.Module):
            def __init__(self, d_model: int, d_hidden: int, n_head: int, kernel_size: int, dropout_rate: float, rope):
                super().__init__()

                self.block = TransBlock(d_model, d_hidden, n_head, kernel_size, dropout_rate, rope)

            def forward(self, x: torch.tensor):
                '''
                input : B, T, F, C
                output : B, T, F, C
                '''
                B, T, F, C = x.shape
                x = x.permute(0, 2, 1, 3) #* B, F, T, C
                x = x.reshape(B*F, T, C)
                x = self.block(x)
                x = x.reshape(B, F, T, C)
                x = x.permute(0, 2, 1, 3) #* B, T, F, C
                
                return x

        self.freq_block = FreqModule(**spectral_module, rope=rope)
        self.time_block = TimeModule(**channel_module, rope=rope)


    def forward(self, x: torch.tensor):
        x = self.freq_block(x)
        x = self.time_block(x)
        
        return x


class FixedSplit(nn.Module):
    def __init__(self, d_model, max_n_spks):
        super().__init__()
        self.spk_split = nn.Sequential(nn.Linear(d_model, max_n_spks*d_model*4),
                                       nn.SiLU(),
                                       nn.Linear(max_n_spks*d_model*4, max_n_spks*d_model))
        self.max_n_spks = max_n_spks

    def forward(self,x):
        B, T, F, C = x.shape
        x = self.spk_split(x)  # B*T, F, NC
        x = x.reshape(B, T, F, C, self.max_n_spks) # B, T, F, C, N
        x = x.permute(0, 4, 1, 2, 3) # B, N, T, F, C
        return x
    
    
    
class AttractorSplit(nn.Module):
    def __init__(self, d_model, d_freq, n_head, max_n_spks, dropout_rate):
        super().__init__()
        
        class AttractorDecoder(nn.Module):
            def __init__(self, d_model, n_head, dropout_rate):
                super().__init__()
                self.net1 = TransDecoderBlock(d_model, 4*d_model, n_head, dropout_rate)
                self.net2 = TransDecoderBlock(d_model, 4*d_model, n_head, dropout_rate)

            def forward(self, q, kv):
                # q: B, K, C
                # kv: B, T, C
                q = self.net1(q, kv) # B, K, C
                q = self.net2(q, kv) # B, K, C

                return q

        class SplitModule(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.fusion = nn.Sequential(nn.Linear(2*d_model, 4*d_model),
                                            nn.SiLU(),
                                            nn.Linear(4*d_model, d_model))
                
            def forward(self, x, att):
                # x: B, T, F, C
                # att: B, K, C
                B, T, F, C = x.shape
                K = att.shape[1]
                x = x.unsqueeze(1).expand(-1, K, -1, -1, -1) # B, K, T, F, C
                att = att.unsqueeze(2).unsqueeze(2).expand(-1, -1, T, F, -1) # B, K, T, F, C
                x = torch.cat([x, att], dim=-1) # B, K, T, F, 2C
                x = self.fusion(x) # B, K, T, F, C
                return x
            

        self.layernorm = nn.LayerNorm(d_model)
        self.spk_query = nn.Parameter(torch.randn(1, max_n_spks+2, d_model))
        self.dec = AttractorDecoder(d_model, n_head, dropout_rate)
        self.pres_linear = nn.Linear(d_model, 1)
        self.split = SplitModule(d_model)
        self.pe_tf = self.get_2d_sinusoids(5000, 1500, d_model)
                

    def forward(self, x, n_spks=None, prob_thres=0.5):
        # x: B, T, F, C
        # Example usage:
        #   - n_spks=None: estimate speakers based on probability (inference mode, B=1)
        #   - n_spks=2: all samples have 2 speakers (training mode or inference mode with fixed number of speakers)
        #   - n_spks=torch.tensor([2, 3, 1]): batch samples have 2, 3, 1 speakers respectively

        kv = self.layernorm(x)
        B, T, F, C = kv.shape
        # Prepare query depending on n_spks
        q = self.spk_query.expand(B, -1, -1) # B, K+1, C

        # add 2d positional encoding and decode for attractor
        kv = kv + self.pe_tf[:T, :F, :].unsqueeze(0).to(kv.device) # B, T, F, C
        kv = kv.reshape(B, -1, C)

        att = self.dec(q, kv) # B, K+1, C
        att = att / (att.norm(dim=-1, keepdim=True) + 1e-8)
        logits = self.pres_linear(att).squeeze(-1) # B, K+1, F
        probs = torch.sigmoid(logits) # B, K+1
        
        x = self.split(x, att) # B, K, C
        # x = self.split(x, att[:,:-1]) # B, K, C
        speaker_mask, x_residual = None, None
        if n_spks is None: # 1. Inference mode: estimate number of speakers based on probability
            assert x.size(0) == 1, "Inference mode (n_spks=None) assumes batch size of 1" # Assuming batch size is 1
            active_spks = (probs[0,1:] > prob_thres).nonzero(as_tuple=True)[0]
            x = x[:, 1:len(active_spks)+1] # B, estimated_K, T, F, C
        else:
            assert isinstance(n_spks, torch.Tensor), "n_spks must be None or torch.Tensor"  # Training mode with variable speakers per batch
            B, Kp2 = probs.shape
            if B == 1:
                x_residual = torch.cat([x[:,:1], x[:, int(n_spks.item())+1:]], dim=1)
                x = x[:, 1:int(n_spks.item())+1] # B, K, T, F, C
            else:
                K = Kp2 - 2  # Number of speakers (excluding presence token)
                speaker_mask = torch.zeros(B, K, device=x.device, dtype=torch.bool)
                for i, n_spk in enumerate(n_spks):
                    speaker_mask[i, :int(n_spk)] = True

        return x, {"logits":logits, "probs":probs, "split_res":x_residual}, speaker_mask

    def sinusoids(self, length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)    

    def get_2d_sinusoids(self, T, F, channels, max_timescale=10000):

        channels_per_axis = channels // 2
        t_sinusoid = self.sinusoids(T, channels_per_axis, max_timescale)
        f_sinusoid = self.sinusoids(F, channels_per_axis, max_timescale)
        
        t_sinusoid = t_sinusoid.unsqueeze(1).expand(-1, F, -1) # T, F, C/2
        f_sinusoid = f_sinusoid.unsqueeze(0).expand(T, -1, -1) # T, F, C/2
        
        return torch.cat([t_sinusoid, f_sinusoid], dim=-1) # T, F, C
    

class FilterEstimator(nn.Module):
    def __init__(self, d_model, num_mics, taps_freq, taps_frame, MIMO=True):
        super().__init__()

        self.L_p, self.L_f = taps_frame[0], taps_frame[1]
        self.L_h, self.L_l = taps_freq[0], taps_freq[1]
        self.L_time = self.L_p + 1 + self.L_f
        self.L_freq = self.L_h + 1 + self.L_l
        self.out_mics = num_mics if MIMO else 1

        class MaskEstim(nn.Module):
            def __init__(self, d_model, num_mics, out_mics, L_time, L_freq):
                super().__init__()
                self.net = nn.Linear(d_model, 3*num_mics*out_mics*L_time*L_freq)
                self.mag_act = nn.Softplus()
                self.com_act = nn.Tanh()

            def forward(self, x):
                B, N, T, F, C  = x.shape
                x = self.net(x)
                x = x.reshape(B*N, T, F, -1, 3)
                x = self.mag_act(x[...,[0]])*self.com_act(x[...,1:]) # B*N, T, F, C, 2
                x = x.permute(0, 3, 1, 2, 4) # B*N, C, T, F, 2
                return x

        self.mask = MaskEstim(d_model, num_mics, self.out_mics, self.L_time, self.L_freq)
        self.num_mics = num_mics
        self.frame_win_len = taps_frame

        
    def forward(self, x):
        # x : B, T, F, C 
        B, N, T, F, C = x.shape
        
        mask = self.mask(x) # B, N, MMW, T, F, 2
        M = self.num_mics
        M_o = self.out_mics
        L_time = self.L_time
        L_freq = self.L_freq
        mask = mask.reshape(B, N, M_o, M, L_freq*L_time, T, F, 2)
        mask = mask.permute(0, 2, 3, 4, 6, 5, 1, 7) # B, M_o, M, L_freq*L_time, F, T, N, 2
        mask = torch.complex(mask[...,0], mask[...,1])  # B, M_o, M, W, F, T, N
        mask = list(torch.unbind(mask, dim=-1))

        return mask