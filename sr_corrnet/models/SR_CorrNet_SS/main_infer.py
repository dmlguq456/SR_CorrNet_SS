import os
import time
import torch
import soundfile as sf
from pathlib import Path
from loguru import logger
from .model import Model
from .engine_infer import EngineInfer
from .dataset import get_dataloaders, get_multi_dataloaders
from sr_corrnet.utils import util_system, util_engine
from sr_corrnet.utils.decorators import logger_wraps
from sr_corrnet.utils.util_engine import PBAR_FMT
from tqdm import tqdm

# Setup logger
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/system_log.log")
logger.add(log_file_path, level="DEBUG", mode="w")


def setup_inference(args):
    """Load config, create model, load checkpoint, return EngineInfer + dataloaders."""
    config_name = getattr(args, 'config', 'baseline.yaml')
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", config_name)
    yaml_dict = util_system.parse_yaml(yaml_path)
    config = yaml_dict["config"]
    logger.info(f"Using config file: {config_name}")

    # Create model
    model = Model(**config["model"])

    # Setup device
    gpuid = tuple(map(int, args.gpuid.split(',')))
    device = torch.device(f'cuda:{gpuid[0]}')
    model = model.to(device)

    # Load checkpoint
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base_dir, f"log/log_{Path(config_name).stem}", "scratch_weights")

    start_epoch = util_engine.load_last_checkpoint_n_get_epoch(ckpt_path, model, location=device)
    logger.info(f"Loaded checkpoint from {ckpt_path}, epoch {start_epoch - 1}")

    # Create inference engine
    engine = EngineInfer(config, model, device)

    # Create dataloaders (test partitions only)
    dataloaders = setup_dataloader(config)

    return engine, config, config_name, start_epoch, dataloaders


def setup_dataloader(config):
    """Create test dataloaders for SS inference.

    Replicates main.py L29-39 dataloader creation with test partitions only.
    Uses get_multi_dataloaders or get_dataloaders based on config['is_var_spks'].
    """
    partitions = config["partitions"]["test"]
    if config["is_var_spks"]:
        return get_multi_dataloaders(partitions, config)
    else:
        return get_dataloaders(partitions, config)


def save_outputs(dump_dir, key, result, fs=8000):
    """Save SS inference results — per-speaker wav files.

    SS pattern: {dump_dir}/{key}_spk{idx}.wav
    """
    os.makedirs(dump_dir, exist_ok=True)
    waveforms = result["waveforms"]
    for idx, wav in enumerate(waveforms):
        file_path = os.path.join(dump_dir, f"{key}_spk{idx}.wav")
        sf.write(file_path, wav.cpu().data.numpy(), fs)


def run_inference(engine, dataloaders, output_dir=None, config_name=None, start_epoch=1):
    """Run inference on all dataloaders and save separated wav files.

    Iterates all partitions, processes each with per-sample inference.
    """
    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "log", f"log_{Path(config_name).stem}",
                                  f"inference_epoch{start_epoch - 1}")

    with torch.cuda.device(engine.device):
        for part, loader in dataloaders.items():
            dump_dir = os.path.join(output_dir, part)
            t0 = time.time()
            n_utts = _infer_dataloader(engine, loader, dump_dir)
            t1 = time.time()
            logger.info(f"[{Path(config_name).stem}] INFERENCE-{part.upper()}: "
                        f"{n_utts} files saved to {dump_dir} ({t1-t0:.1f}s)")

    logger.info("Inference done!")


def _infer_dataloader(engine, dataloader, dump_dir):
    """Process inference from dataloader — per-sample with n_spks forwarding."""
    n_utts = 0
    pbar = tqdm(total=len(dataloader), unit='batch', desc='INFERENCE',
                colour="CYAN", dynamic_ncols=True, bar_format=PBAR_FMT)

    for batch in dataloader:
        key, mixture, n_spks = batch['key'], batch['mixture'], batch['n_spks']
        n_spks_prior = None if engine.test_unknown_n_spks else n_spks

        # Per-sample processing
        for b in range(len(key)):
            waveform_b = mixture[b]  # (L,) or (L, M)
            if waveform_b.dim() == 2:
                waveform_b = waveform_b.T  # (L, M) -> (M, L)
            elif waveform_b.dim() == 1:
                waveform_b = waveform_b.unsqueeze(0)  # (L,) -> (1, L)
            n_spks_b = n_spks_prior[b] if n_spks_prior is not None else None
            # SS std normalization
            waveform_b = waveform_b / (waveform_b.std(dim=-1, keepdim=True) + 1e-8)
            # STFT -> single-pass forward -> iSTFT
            stft = engine.stft(waveform_b.to(engine.device), cplx=True)
            seg_result = engine._single_pass_session(stft, n_spks=n_spks_b)
            stft_out = seg_result["stft_out"]  # (N, F, T)
            waveforms = [
                engine.istft(stft_out[i], cplx=True, squeeze=True)
                for i in range(stft_out.shape[0])
            ]
            result = {"waveforms": waveforms, "vad": None, "doa": None}
            save_outputs(dump_dir, key[b], result, fs=engine.fs)

        n_utts += len(key)
        pbar.update(1)

    pbar.close()
    return n_utts


@logger_wraps()
def main_infer(args):
    """CLI entry point — called from run.py when engine_mode=='inference'."""
    engine, config, config_name, start_epoch, dataloaders = setup_inference(args)
    run_inference(engine, dataloaders, output_dir=args.output,
                  config_name=config_name, start_epoch=start_epoch)
