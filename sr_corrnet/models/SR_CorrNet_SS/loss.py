import torch
import torch.nn as nn
import numpy as np

from math import ceil
from itertools import permutations
from dataclasses import dataclass, field, fields
from loguru import logger
from sr_corrnet.utils.decorators import logger_wraps
from sr_corrnet.utils import util_stft


# Utility functions
def l2norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim)

def l1norm(mat, keepdim=False):
    return torch.norm(mat, dim=-1, keepdim=keepdim, p=1)


       
@logger_wraps()
class PIT_SISNR_mag(nn.Module):
    def __init__(self, scale_inv: bool, device: torch.device):
        super().__init__()
        self.device = device
        self.scale_inv = scale_inv
        self.stft = util_stft.STFT(frame_length=512, frame_shift=256, device=self.device, normalize=True)

    def forward(self, estims, targets, eps=1.0e-10, prior_idx=None):
        assert len(estims) == len(targets), "The number of estimated sources and target sources must be the same."
        n_spks = len(estims)

        def _SDR_loss(permute, batch_idx=None):
            loss_for_permute = []
            for s, t in enumerate(permute):
                est = estims[s]
                src = targets[t]

                # If batch_idx is provided, select specific batch samples
                if batch_idx is not None:
                    est = est[[batch_idx]]
                    src = src[[batch_idx]]

                est_zm = est - torch.mean(input=est, dim=-1, keepdim=True)
                src_zm = src - torch.mean(input=src, dim=-1, keepdim=True)

                if self.scale_inv:
                    scale_factor = torch.sum(est * src, dim=-1, keepdim=True) / (l2norm(l2norm(src, keepdim=True),keepdim=True)**2 + eps)
                    src_zm_scale = scale_factor * src_zm

                est_stft = self.stft(est_zm, cplx=True)
                src_stft = self.stft(src_zm_scale, cplx=True)
                est_mag = torch.sqrt(est_stft.real**2 + est_stft.imag**2)
                src_mag = torch.sqrt(src_stft.real**2 + src_stft.imag**2)

                utt_loss = - 20 * torch.log10(eps + l2norm(l2norm(src_mag)) / (l2norm(l2norm(est_mag - src_mag)) + eps))
                utt_loss = torch.clamp(utt_loss, min=-30)
                
                loss_for_permute.append(utt_loss)
            return sum(loss_for_permute)/n_spks
        if prior_idx is not None:
            # Handle both single permutation and batch-wise permutations
            if isinstance(prior_idx, list) and len(prior_idx) > 0 and isinstance(prior_idx[0], list):
                # Batch-wise permutations: compute loss for each batch with its specific permutation
                batch_losses = [_SDR_loss(perm, batch_idx=b) for b, perm in enumerate(prior_idx)]
                min_perutt = torch.stack(batch_losses)
            else:
                # Single permutation for all batches
                min_perutt = _SDR_loss(prior_idx)
        else:
            pscore = torch.cat([_SDR_loss(p) for p in permutations(range(n_spks))])
            min_perutt, _ = torch.min(pscore, dim=0)

        return torch.mean(min_perutt)


@logger_wraps()
class PIT_SISNR_time(nn.Module):
    def __init__(self, scale_inv: bool, device: torch.device):
        super().__init__()
        self.device = device
        self.scale_inv = scale_inv

    def forward(self, estims, targets, eps=1.0e-10, return_perm_idx=False):
        assert len(estims) == len(targets), "The number of estimated sources and target sources must be the same."
        n_spks = len(estims)
        def _SDR_loss(permute):
            loss_for_permute = []
            for s, t in enumerate(permute):
                est = estims[s]
                src = targets[t]
                est_zm = est - torch.mean(input=est, dim=-1, keepdim=True)
                src_zm = src - torch.mean(input=src, dim=-1, keepdim=True)
                src_zm_scale = src_zm
                if self.scale_inv:
                    scale_factor = torch.sum(est_zm * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True)**2 + eps)
                    src_zm_scale = scale_factor * src_zm

                utt_loss = - 20 * torch.log10(eps + l2norm(src_zm_scale) / (l2norm(est_zm - src_zm_scale) + eps))
                utt_loss = torch.clamp(utt_loss, min=-30)
                
                loss_for_permute.append(utt_loss)
            return sum(loss_for_permute)/n_spks
        
        pscore = torch.stack([_SDR_loss(p) for p in permutations(range(n_spks))])
        indices = [list(p) for p in permutations(range(n_spks))]
        min_perutt, min_idx = torch.min(pscore, dim=0)
        if return_perm_idx:
            # Handle batch-wise indexing for permutation indices
            if min_idx.dim() > 0:  # batch size > 1
                batch_indices = [indices[idx.item()] for idx in min_idx]
                return torch.mean(min_perutt), batch_indices
            else:  # batch size = 1
                return torch.mean(min_perutt), indices[min_idx.item()]
        else:
            return torch.mean(min_perutt)


@logger_wraps()
class PIT_SISNRi(nn.Module):
    def __init__(self, scale_inv: bool, device: torch.device):
        super().__init__()
        self.device = device
        self.scale_inv = scale_inv
    
    def forward(self, estims, targets, input, eps=1.0e-20):
        assert len(estims) == len(targets), "The number of estimated sources and target sources must be the same."
        n_spks = len(estims)
        input_zm = input - torch.mean(input, dim=-1, keepdim=True)
        
        def _SDR_loss(permute):
            loss_for_permute = []
            for s, t in enumerate(permute):
                est = estims[s]
                src = targets[t]
                est_zm = est - torch.mean(est, dim=-1, keepdim=True)
                src_zm = src - torch.mean(src, dim=-1, keepdim=True)
                
                src_zm_s = src_zm
                if self.scale_inv:
                    factor = torch.sum(est_zm * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True)**2 + eps)
                    src_zm_s = factor * src_zm
                
                utt_loss_est = 20 * torch.log10(eps + l2norm(src_zm_s) / (l2norm(est_zm - src_zm_s) + eps))
                
                src_zm_x = src_zm
                if self.scale_inv:
                    src_zm_x = torch.sum(input_zm * src_zm, dim=-1, keepdim=True) / (l2norm(src_zm, keepdim=True)**2 + eps) * src_zm
                utt_loss_in = 20 * torch.log10(eps + l2norm(src_zm_x) / (l2norm(input_zm - src_zm_x) + eps))
                loss_for_permute.append(utt_loss_est - utt_loss_in)
            return sum(loss_for_permute)/n_spks 
        
        pscore = torch.stack([_SDR_loss(p) for p in permutations(range(n_spks))],dim=0)
        max_perutt, max_idx = torch.max(pscore, dim=0)
        return torch.mean(max_perutt)
