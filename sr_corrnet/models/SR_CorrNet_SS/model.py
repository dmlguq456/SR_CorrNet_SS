import torch
import warnings
warnings.filterwarnings('ignore')
import numpy as np

import torch.nn as nn
from sr_corrnet.utils.decorators import logger_wraps
from .modules.module import Encoder, TF_Block, FilterEstimator, CrossSpkBlock, FixedSplit, AttractorSplit
from rotary_embedding_torch import RotaryEmbedding


# @logger_wraps()
class Model(nn.Module):
    def __init__(
                 self,
                 ref_ch: int,
                 taps_freq: list,
                 taps_frame: list,
                 d_model: int,
                 input_embedding: dict,
                 RoPE: dict,
                 multi_path_block: dict,
                 is_var_spks: bool,
                 spk_split: dict,
                 cross_spk: dict,
                 N_Enc: int,
                 N_Dec: int,
                 mask_estimator: dict,
                 apply_pos_enc: bool = True
                 ):
        super().__init__()

        self.L_h, self.L_l = taps_freq[0], taps_freq[1]
        self.L_p, self.L_f = taps_frame[0], taps_frame[1]
        self.ref_ch = ref_ch
        self.encoder = Encoder(**input_embedding)
        rope = RotaryEmbedding(RoPE['d_model'] // RoPE['n_head'])
        self.enc_block = nn.Sequential(*[TF_Block(**multi_path_block, rope=rope) for _ in range(N_Enc)])
        self.dec_block = nn.Sequential(*[TF_Block(**multi_path_block, rope=rope) for _ in range(N_Dec)])
        self.dec_cs = nn.Sequential(*[CrossSpkBlock(**cross_spk) for _ in range(N_Dec)])
        self.is_var_spks = is_var_spks
        self.spk_split = AttractorSplit(**spk_split['varying']) if is_var_spks else FixedSplit(**spk_split['fixed'])
        self.filter_estim = FilterEstimator(**mask_estimator)
        self.filter_estim_aux = nn.ModuleList([FilterEstimator(**mask_estimator) for _ in range(N_Dec)])
        self.layernorm = nn.LayerNorm(d_model) if N_Dec > 0 else nn.Identity()
        self.apply_pos_enc = apply_pos_enc

    def forward(self, x, aux_loss=False, n_spks=None):
        # x : (B), 2M, F, T

        # prepare input
        if len(x.shape) == 3: # When No Batch Dimension
            x = x.unsqueeze(0)
        x_r, x_i = x.chunk(2, dim=1)
        x = torch.complex(x_r,x_i)  # B, M, F, T
        x_mf = self._multiframe(x, L_p=self.L_p, L_f=self.L_f, L_h=self.L_h, L_l=self.L_l)  # B, L, M, F, T
        
        # input embedding & positional encoding
        x_enc = self.encoder(x[:, self.ref_ch], x_mf)
        B, T, F, C = x_enc.shape
        if self.apply_pos_enc:
            x_enc = x_enc + self.sinusoids(F, C).reshape(1, 1, F, C).to(x.device)

        # separation encoder
        x_enc = self.enc_block(x_enc)

        # speaker split
        if self.is_var_spks:
            x_sep, pres, spk_mask = self.spk_split(x_enc, n_spks)
            if pres["split_res"] is not None:
                m_res = self.filter_estim_aux[0](pres["split_res"])
                pres["split_res"] = [self.filtering(m_res_, x_mf) for m_res_ in m_res]  # silent output for residual
            if x_sep.shape[1] == 0:
                return [], [], {'logits':[], 'probs':[]}
        else:
            x_sep = self.spk_split(x_enc)
            pres, spk_mask = None, None
        x_sep = self.layernorm(x_sep)
            
        # reconstruction decoder
        if spk_mask is not None:
            x_dec, x_dec_h = self.batch_wise_dec(x_sep, spk_mask)
        else:
            x_dec, x_dec_h = self.decoder_forward(x_sep)
                
        # filter
        m = self.filter_estim(x_dec) # B, M_o, M, W, F, T, N
        out = [self.filtering(m_, x_mf) for m_ in m]

        # auxiliary filter
        if aux_loss:
            out_aux = []
            for i in range(len(self.filter_estim_aux)):
                m_aux = self.filter_estim_aux[i](x_dec_h[i])
                out_aux_i = [self.filtering(m_aux_, x_mf) for m_aux_ in m_aux]
                out_aux.append(out_aux_i)  # [[B, M, L]*N]*len(aux)
        else:
            out_aux = None

        return out, out_aux, pres

    def batch_wise_dec(self, x_sep, speaker_mask):
        B = x_sep.shape[0]
        x_dec = []
        x_dec_h = [[] for _ in range(len(self.filter_estim_aux))]
        for b in range(B):
            x_b = x_sep[b].unsqueeze(0)  # 1, N, T, F, C
            out_b = torch.zeros_like(x_b)
            out_b_aux = [torch.zeros_like(x_b) for _ in range(len(self.filter_estim_aux))]
            active_idx = speaker_mask[b].nonzero(as_tuple=True)[0]
            if len(active_idx) > 0:
                x_active, x_dec_h_active = self.decoder_forward(x_b[:, active_idx])
                out_b[:, active_idx] = x_active
                for i, x_aux in enumerate(x_dec_h_active): out_b_aux[i][:, active_idx] = x_aux
            x_dec.append(out_b)
            for i in range(len(self.filter_estim_aux)): x_dec_h[i].append(out_b_aux[i])
        x_dec = torch.cat(x_dec, dim=0)
        for i in range(len(self.filter_estim_aux)): x_dec_h[i] = torch.cat(x_dec_h[i], dim=0)
        return x_dec, x_dec_h


    def decoder_forward(self, x):
        B, N, T, F, C = x.shape
        x_dec_h = []
        for sha, cs in zip(self.dec_block, self.dec_cs):
            x_dec_h.append(x)
            x = x.reshape(B*N, T, F, C)
            x = sha(x)
            x = x.reshape(B, N, T, F, C)
            x = cs(x)
        return x, x_dec_h

    def sinusoids(self, length, channels, max_timescale=10000):
        """Returns sinusoids for positional embedding"""
        assert channels % 2 == 0
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)    


    def _multiframe(self, x, L_p, L_f, L_h, L_l):
        # x: B, M, F, T
        L_time = L_p + L_f + 1
        L_freq = L_h + L_l + 1
        B, M, F, T = x.shape
        x_pad = nn.functional.pad(x, (L_p, L_f, L_h, L_l)) # left, right (Time), top, bottom (Freq)
        windows = x_pad.unfold(dimension=2, size=L_freq, step=1) # B, M, F, T, L_freq
        windows = windows.unfold(dimension=3, size=L_time, step=1) # B, M, F, T, L_freq, L_time
        x_mf = windows.reshape(B, M, F, T, L_time*L_freq).permute(0, 1, 4, 2, 3) # B, M, L(L_time*L_freq), F, T
        return x_mf  # B, M, L, F, T


    def filtering(self, filt, mixture):
        # filt : B, M_o, M, L, F, T
        # mixture : B, M, L, F, T
        out = (mixture.unsqueeze(1) * filt).sum(dim=(2, 3))  # (B, M_o, F, T)
        out = torch.stack([torch.real(out), torch.imag(out)],dim=-1)
        return out
    

if __name__ == "__main__":
    import os
    from sr_corrnet.utils import util_system
    from ptflops import get_model_complexity_info
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/1ch_WSJ_fix_2spk.yaml")
    yaml_dict = util_system.parse_yaml(yaml_path)
    wandb_run = util_system.wandb_setup(yaml_dict)
    config = wandb_run.config
    
    nnet = Model(**config["model"])
    
    MACs_ptflops, params_ptflops = get_model_complexity_info(nnet, (125,257,14,), print_per_layer_stat=False, verbose=False) # (num_samples,)
    MACs_ptflops, params_ptflops = MACs_ptflops.replace(" MMac", ""), params_ptflops.replace(" M", "")
    logger.info(f"ptflops: MACs: {MACs_ptflops}, Params: {params_ptflops}")


