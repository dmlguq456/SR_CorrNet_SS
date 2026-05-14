import os
import numpy as np
import torch
import torch.nn as nn
import time
import soundfile as sf
from pathlib import Path

from loguru import logger
from tqdm import tqdm
from sr_corrnet.utils import util_engine, util_stft
from sr_corrnet.utils.util_engine import format_pbar, format_epoch_log, log_scalars_to_tb, PBAR_FMT, BestModelTracker
from sr_corrnet.utils.decorators import logger_wraps
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss_logit
from .loss import PIT_SISNR_time, PIT_SISNR_mag, PIT_SISNRi
from mir_eval.separation import bss_eval_sources


# @logger_wraps()
class Engine(object):
    def __init__(self, args, config, model, dataloaders, gpuid, device):
        
        ''' Default setting '''
        self.config = config
        self.device = device
        self.model = model.to(self.device)
        self.loader_config = config["dataloader"]
        self.ref_ch = config["dataset"]["ref_ch"]
        self.max_n_spks = config["max_n_spks"]
        self.fs = config["dataset"]["synthesis_config"]["sampling_rate"]
        if args.engine_mode == "train":
            self.train_loaders = dataloaders.pop("train")
        self.dev_loaders = dataloaders
        self.use_SepRe = True if config["N_Dec"] > 0 else False
        self.subset_conf = {}
        self.subset_conf["train"] = config["dataset"]["subset_conf"]["train"]
        self.subset_conf["valid"] = config["dataset"]["subset_conf"]["valid"]

        self.loss = PIT_SISNR_time(scale_inv=True, device=self.device)
        self.loss_mag = PIT_SISNR_mag(scale_inv=True, device=self.device)
        self.sisnri = PIT_SISNRi(scale_inv=True, device=self.device)
        self.main_optimizer, self.warmup_scheduler, self.main_scheduler = util_engine.setup_optimizer_and_scheduler(self.model, config)

        self.stft = util_stft.STFT(**config['stft'], device=self.device, normalize=True)
        self.istft = util_stft.iSTFT(**config['stft'], device=self.device, normalize=True)
        
        self.config_name = args.config if hasattr(args, 'config') else 'default'
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_path, self.writer, self.start_epoch = util_engine.setup_logging(
            self.config_name, base_dir, self.model, self.main_optimizer, self.device,
            engine_mode=args.engine_mode, sr=self.fs, n_fft=256, n_hop=128)

        self.best_tracker = BestModelTracker(mode="min")
        self.best_tracker.restore(self.checkpoint_path)

        # Logging
        util_engine.model_params_mac_summary(
            model=self.model,
            input_shape=self.config["engine"]["check_compute_len"],
            metrics=['ptflops', 'thop', 'torchinfo'],
            device=self.device
        )
        torch.cuda.empty_cache()

        logger.info(f"Clip gradient by 2-norm {self.config['engine']['clip_norm']}")


    # @logger_wraps()
    def _extractor(self, mixture, target, presence=None, n_spks=None):
        # mixture: B, L, M or B, L
        if len(mixture.shape) == 2:
            mixture = mixture.unsqueeze(1)  # B, M, L
        elif len(mixture.shape) == 3:
            mixture = mixture.permute(0, 2, 1) # B, M, L
        mixture /= mixture.std(axis=-1, keepdim=True)
        target_stft = [self.stft(t.to(self.device), cplx=True) for t in target] # [B, F, T]
        mixture_stft = self.stft(mixture.to(self.device), cplx=True) # B, F, T
        model_input = torch.cat([torch.real(mixture_stft), torch.imag(mixture_stft)],dim=1) # B, M*2, F, T
        if presence is not None:
            presence = presence.to(self.device)
        if n_spks is not None:
            n_spks = n_spks.to(self.device)

        return {'mix_stft': mixture_stft, 'target_stft': target_stft,
                'model_input': model_input, 'presence': presence, 'n_spks': n_spks}


    def calculate_loss(self, 
                       estim_stft, 
                       target_stft, 
                       mixture_stft,
                       estim_stft_aux_list=None,
                       estim_pres=None,
                       presence=None,
                       n_spks=None,
                       cal_SDRi=False
                       ):
        cur_acc = None
        cur_loss_aux = None
        cur_loss_pres = None

        estim_stft = [torch.complex(e[...,0], e[...,1]) for e in estim_stft]  # [B, F, T]
        estim_src = [self.istft(e.squeeze(1), cplx=True) for e in estim_stft]
        target_src = [self.istft(t, cplx=True).to(self.device) for t in target_stft]
        cur_loss_main, perm_idx = self.loss(estim_src, target_src, return_perm_idx=True)
        
        if estim_pres is not None:
            cur_loss_pres = bce_loss_logit(estim_pres["logits"], presence)
            estim_num = (estim_pres["probs"] > 0.5).int().sum(-1)
            cur_acc = (estim_num == n_spks).float().mean()
            if estim_pres["split_res"] is not None:
                estim_res = [torch.complex(e[...,0], e[...,1]) for e in estim_pres["split_res"]]  # [B, F, T]
                estim_res = [self.istft(e.squeeze(1), cplx=True) for e in estim_res]
                cur_loss_c_res = [estim_res_.abs().mean() for estim_res_ in estim_res]
                cur_loss_c_res = sum(cur_loss_c_res) / len(cur_loss_c_res)
                cur_loss_pres += cur_loss_c_res

        if self.use_SepRe and estim_stft_aux_list is not None:
            cur_loss_aux = []
            for estim_stft_aux in estim_stft_aux_list:
                estim_stft_aux = [torch.complex(e[...,0], e[...,1]) for e in estim_stft_aux]  # [B, F, T]
                estim_src_aux = [self.istft(e.squeeze(1), cplx=True) for e in estim_stft_aux]
                cur_loss_aux.append(self.loss_mag(estim_src_aux, target_src, prior_idx=perm_idx))
            cur_loss_aux = sum(cur_loss_aux)/len(cur_loss_aux)

        input = self.istft(mixture_stft[:,self.ref_ch], cplx=True)
        cur_sisnri = self.sisnri(estim_src, target_src, input)

        def calculate_SDR():
            estims = torch.stack(estim_src, dim=0).squeeze(1)
            targets = torch.stack(target_src, dim=0).squeeze(1)
            estims = estims.cpu().data.numpy()
            targets = targets.cpu().data.numpy()
            inputs = torch.cat([input, input], dim=0)
            inputs = inputs.cpu().data.numpy()
            sdr_out, _, _, _ = bss_eval_sources(targets, estims)
            sdr_in, _, _, _ = bss_eval_sources(targets, inputs)                
            
            return sdr_out.mean(), sdr_in.mean()
        
        sdr_out, sdr_in = calculate_SDR() if cal_SDRi else (None, None)

        return cur_loss_main, cur_loss_aux, cur_sisnri, cur_loss_pres, cur_acc, sdr_out, sdr_in
               

            
    

    def logging_sample(self, estim_stft, target_stft, mixture, epoch, n_spks):
        estim_stft = [torch.complex(e[...,0], e[...,1]) for e in estim_stft]  # [B, F, T]
        target_src = [self.istft(t, cplx=True).to(self.device) for t in target_stft]
        estim_src = [self.istft(e.squeeze(1), cplx=True) for e in estim_stft]
        mixture_sample = mixture[0,:,self.ref_ch]  if len(mixture.shape) == 3 else mixture[0]
        self.writer.log_wav2spec(mixture_sample, "noisy", epoch)
        self.writer.log_audio(mixture_sample, 'noisy_audio', epoch)
        for i in range(n_spks[0].item() if isinstance(n_spks, torch.Tensor) else n_spks):
            self.writer.log_wav2spec(target_src[i][0], f"clean_{i}", epoch)
            self.writer.log_audio(target_src[i][0], f'clean_audio_{i}', epoch)
            self.writer.log_wav2spec(estim_src[i][0], f"estim_{i}", epoch)
            self.writer.log_audio(estim_src[i][0], f'estim_audio_{i}', epoch)


    def _train(self, _dataloader, epoch):
        dataloader = util_engine.create_dataloader_with_sampler(_dataloader, self.subset_conf["train"], self.loader_config)

        self.model.train()
        tot_loss = tot_loss_aux = tot_loss_pres = 0
        tot_acc_pres = n_batch = 0
        pbar = tqdm(total=len(dataloader), unit='batch', desc='TRAIN', colour="YELLOW", dynamic_ncols=True, bar_format=PBAR_FMT)
        for batch in dataloader:
            key, mixture, target = batch['key'], batch['mixture'], batch['target']
            n_spks, presence = batch['n_spks'], batch['pres_label']
            # Scheduler learning rate for warm-up (Iteration-based update for transformers)
            if epoch == 1: self.warmup_scheduler.step()
            w_aux = 0.5*(0.95)**(epoch-100) if epoch > 100 else 0.5

            # feature pre-processing
            feat = self._extractor(mixture, target, presence, n_spks)
            mixture_stft, target_stft, model_input = feat['mix_stft'], feat['target_stft'], feat['model_input']
            presence, n_spks = feat['presence'], feat['n_spks']
            self.main_optimizer.zero_grad()
            # network forward
            estim_stft, estim_stft_aux_list, estim_pres = self.model(model_input, aux_loss=self.use_SepRe, n_spks=n_spks)

            cur_loss_list = self.calculate_loss(estim_stft, target_stft, mixture_stft, estim_stft_aux_list, 
                                                estim_pres=estim_pres, presence=presence, n_spks=n_spks)
            cur_loss_main, cur_loss_aux, cur_sisnri, cur_loss_pres, cur_acc, sdr_out, sdr_in = cur_loss_list
            tot_loss += cur_loss_main.item()
            if estim_pres is not None:
                tot_loss_pres += cur_loss_pres.item()
                tot_acc_pres += cur_acc.item()
            if self.use_SepRe:
                tot_loss_aux += cur_loss_aux.item()
                cur_loss = cur_loss_main + w_aux*cur_loss_aux + (cur_loss_pres if estim_pres is not None else 0)
            else:
                cur_loss = cur_loss_main + (cur_loss_pres if estim_pres is not None else 0)
            # update the parameters
            cur_loss.backward()
            if self.config['engine']['clip_norm']: 
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['engine']['clip_norm'])
            self.main_optimizer.step()
            # update the progress bar
            torch.cuda.synchronize()
            n_batch += 1
            pbar.update(1)
            pbar_dict = {'L_se': tot_loss/n_batch, 'L_se_aux': tot_loss_aux/n_batch,
                         'L_pres': tot_loss_pres/n_batch, 'pres_acc': tot_acc_pres/n_batch}
            pbar.set_postfix(format_pbar(pbar_dict))
        pbar.close()

        dict_loss = {'L_se': tot_loss / n_batch}
        if self.use_SepRe:
            dict_loss['L_se_aux'] = tot_loss_aux / n_batch
        if estim_pres is not None:
            dict_loss['L_pres'] = tot_loss_pres / n_batch
            dict_loss['pres_acc'] = tot_acc_pres / n_batch
        return dict_loss, n_batch

    
    def _validate(self, _dataloader, epoch, is_test=False, save_dir=None):
        if is_test:
            dataloader = _dataloader
            pbar_desc, pbar_colour = 'TEST', 'WHITE'
        else:
            dataloader = util_engine.create_dataloader_with_sampler(_dataloader, self.subset_conf["valid"], self.loader_config)
            pbar_desc, pbar_colour = 'VALID', 'RED'

        cal_SDRi = is_test and self.config.get("eval_SDRi", False)
        test_unknown = is_test and self.config.get("test_unknown_n_spks", False)

        self.model.eval()
        tot_loss = tot_sisnri = tot_loss_pres = 0
        tot_acc_pres = n_batch = 0
        tot_sdr = tot_sdri = 0
        random_sample_idx = 10
        pbar = tqdm(total=len(dataloader), unit='batch', desc=pbar_desc, colour=pbar_colour, dynamic_ncols=True, bar_format=PBAR_FMT)
        with torch.inference_mode():
            for batch in dataloader:
                key, mixture, target = batch['key'], batch['mixture'], batch['target']
                n_spks, presence = batch['n_spks'], batch['pres_label']
                # feature pre-processing
                feat = self._extractor(mixture, target, presence, n_spks)
                mixture_stft, target_stft, model_input = feat['mix_stft'], feat['target_stft'], feat['model_input']
                presence, n_spks = feat['presence'], feat['n_spks']
                # network forward
                n_spks_prior = None if test_unknown else n_spks
                estim_stft, _, estim_pres = self.model(model_input, n_spks=n_spks_prior)

                # adjust n_spks pair when unknown n_spks test
                if test_unknown:
                    if len(estim_stft) > len(target_stft):
                        estim_stft = estim_stft[:len(target_stft)]
                    elif len(estim_stft) < len(target_stft):
                        for i in range(len(estim_stft), len(target_stft)):
                            B, F, T = target_stft[i].shape
                            zeros_tensor = torch.ones(B, F, T, 2, device=target_stft[i].device, dtype=estim_stft[0].dtype) * 1.0e-8
                            estim_stft.append(zeros_tensor)
                elif is_test:
                    assert len(estim_stft) == len(target_stft), "Estimated source number is different from target source number."

                # loss
                cur_loss_list = self.calculate_loss(estim_stft, target_stft, mixture_stft,
                                                    estim_pres=estim_pres, presence=presence, n_spks=n_spks, cal_SDRi=cal_SDRi)
                cur_loss_main, cur_loss_aux, cur_sisnri, cur_loss_pres, cur_acc, sdr_out, sdr_in = cur_loss_list
                tot_loss += cur_loss_main.item()
                if estim_pres is not None:
                    tot_loss_pres += cur_loss_pres.item()
                    tot_acc_pres += cur_acc.item()
                tot_sisnri += cur_sisnri.item()
                if sdr_out is not None:
                    tot_sdri += sdr_out - sdr_in
                    tot_sdr += sdr_out

                # save enhanced audio
                if save_dir is not None:
                    estim_stft_c = [torch.complex(e[...,0], e[...,1]) for e in estim_stft]
                    estim_wav = [self.istft(e.squeeze(1), cplx=True)[0].cpu().numpy() for e in estim_stft_c]
                    key_stem = Path(key[0]).stem
                    os.makedirs(save_dir, exist_ok=True)
                    for spk_idx, wav in enumerate(estim_wav):
                        out_path = os.path.join(save_dir, f"{key_stem}_spk{spk_idx}.wav")
                        sf.write(out_path, wav, self.fs)

                # TensorBoard sample logging (validation only)
                if not is_test and n_batch == random_sample_idx:
                    self.logging_sample(estim_stft, target_stft, mixture, epoch, n_spks)

                # update the progress bar
                n_batch += 1
                pbar.update(1)
                pbar_dict = {'L_se': tot_loss/n_batch, 'SISNRi': tot_sisnri/n_batch,
                             'L_pres': tot_loss_pres/n_batch, 'pres_acc': tot_acc_pres/n_batch}
                if cal_SDRi:
                    pbar_dict['SDR'] = tot_sdr/n_batch
                    pbar_dict['SDRi'] = tot_sdri/n_batch
                pbar.set_postfix(format_pbar(pbar_dict))
            pbar.close()

        dict_loss = {'L_se': tot_loss / n_batch, 'SISNRi': tot_sisnri / n_batch}
        if estim_pres is not None:
            dict_loss['L_pres'] = tot_loss_pres / n_batch
            dict_loss['pres_acc'] = tot_acc_pres / n_batch
        if sdr_out is not None:
            dict_loss['SDR'] = tot_sdr / n_batch
            dict_loss['SDRi'] = tot_sdri / n_batch
        return dict_loss, n_batch



    @logger_wraps()
    def run_eval(self):
        save_cfg = self.config["engine"].get("save_audio", {})
        if save_cfg.get("enabled", False):
            save_root = save_cfg.get("path", "log/eval_audio")
            if not os.path.isabs(save_root):
                save_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_root)
        else:
            save_root = None
        with torch.cuda.device(self.device):
            for part, loader in self.dev_loaders.items():
                save_dir = os.path.join(save_root, part) if save_root else None
                t0 = time.time()
                td, tn = self._validate(loader, self.start_epoch-1, is_test=True, save_dir=save_dir)
                t1 = time.time()
                logger.info(format_epoch_log(self.config_name, self.start_epoch-1, f"EVAL-{part.upper()}", td, t1-t0))
            if save_root:
                logger.info(f"Enhanced audio saved to {save_root}")
        logger.info(f"Evaluating done!")


    @logger_wraps()
    def run(self):
        with torch.cuda.device(self.device):
            start_time = time.time()
            logger.info(f"\n⚡⚡⚡⚡⚡⚡⚡⚡⚡ [CONFIG] {self.config_name} ⚡⚡⚡⚡⚡⚡⚡⚡⚡")
            if self.start_epoch > 10:
                for part, loader in self.dev_loaders.items():
                    t0 = time.time()
                    init_dict, init_n = self._validate(loader, self.start_epoch-1)
                    t1 = time.time()
                    logger.info(format_epoch_log(self.config_name, self.start_epoch-1, f"INIT-{part.upper()}", init_dict, t1-t0))

            for epoch in range(self.start_epoch, self.config['engine']['max_epoch']):

                # training
                train_start_time = time.time()
                train_dict, train_n_batch = self._train(self.train_loaders, epoch)
                train_end_time = time.time()
                torch.cuda.empty_cache()

                logger.info(format_epoch_log(self.config_name, epoch, "TRAIN", train_dict, train_end_time - train_start_time))

                # validation & test
                valid_results = {}
                for part, loader in self.dev_loaders.items():
                    t0 = time.time()
                    vd, vn = self._validate(loader, epoch)
                    t1 = time.time()
                    torch.cuda.empty_cache()
                    logger.info(format_epoch_log(self.config_name, epoch, part.upper(), vd, t1-t0))
                    valid_results[part] = vd

                util_engine.step_scheduler(self.main_scheduler, epoch, self.config['engine']['start_scheduling'], val_loss=vd['L_se'])

                is_best = util_engine.save_checkpoint_optimized(
                    epoch, self.model, self.main_optimizer, self.checkpoint_path,
                    val_metric=vd['L_se'], best_tracker=self.best_tracker)
                util_engine.sync_checkpoint_to_home(self.checkpoint_path, "SS", self.config_name)

                logger.info(f"[{self.config_name}][Epoch {epoch:3d}] LR={self.main_optimizer.param_groups[0]['lr']:.2e}")

                # TensorBoard logging
                tb_metrics = {"Loss/train": train_dict['L_se']}
                for part, vd in valid_results.items():
                    tb_metrics[f"Loss/{part}"] = vd['L_se']
                    tb_metrics[f"SISNRi/{part}"] = vd['SISNRi']
                    if 'L_pres' in vd:
                        tb_metrics[f"Loss_pres/{part}"] = vd['L_pres']
                        tb_metrics[f"Pres_acc/{part}"] = vd['pres_acc']
                tb_metrics["LR"] = self.main_optimizer.param_groups[0]['lr']
                log_scalars_to_tb(self.writer, tb_metrics, epoch)
                self.writer.flush()

        logger.info(f"Training for {self.config['engine']['max_epoch']} epochs done!")
