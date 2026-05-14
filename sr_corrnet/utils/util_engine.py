import json
import os
import re
from pathlib import Path

import torch
import numpy as np
from loguru import logger

from sr_corrnet.utils.migrate_checkpoint import migrate_state_dict



_EPOCH_PATTERN = re.compile(r'^epoch\.(\d+)\.(pt|pth|pkl)$')


# ── Best model tracking ─────────────────────────────────────────────────────

class BestModelTracker:
    """Track the best validation metric across epochs.

    Persisted as a lightweight JSON file alongside checkpoints so that
    training resume can restore the best-so-far state.

    Args:
        mode: 'min' for losses (lower is better), 'max' for metrics like SISNRi.
        initial_best: Starting threshold. If None, the first update always wins.
    """

    def __init__(self, mode: str = "min", initial_best: float = None):
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got '{mode}'"
        self.mode = mode
        self.best_metric = initial_best
        self.best_epoch = -1

    def update(self, epoch: int, metric: float) -> bool:
        """Return True if *metric* is a new best."""
        if self.best_metric is None:
            is_best = True
        elif self.mode == "min":
            is_best = metric < self.best_metric
        else:
            is_best = metric > self.best_metric

        if is_best:
            self.best_metric = metric
            self.best_epoch = epoch
        return is_best

    # -- serialisation --------------------------------------------------------

    _FILENAME = "best_tracker.json"

    def state_dict(self) -> dict:
        return {"best_metric": self.best_metric, "best_epoch": self.best_epoch, "mode": self.mode}

    def load_state_dict(self, state: dict) -> None:
        self.best_metric = state["best_metric"]
        self.best_epoch = state["best_epoch"]
        self.mode = state.get("mode", self.mode)

    def save(self, directory: str) -> None:
        path = os.path.join(directory, self._FILENAME)
        with open(path, "w") as f:
            json.dump(self.state_dict(), f)

    def restore(self, directory: str) -> bool:
        """Restore from *directory*. Returns True on success, False if missing."""
        path = os.path.join(directory, self._FILENAME)
        if not os.path.exists(path):
            return False
        with open(path) as f:
            self.load_state_dict(json.load(f))
        logger.info(f"Restored BestModelTracker: best_epoch={self.best_epoch}, "
                     f"best_metric={self.best_metric:.4e}")
        return True


# ── Checkpoint optimisation helpers ──────────────────────────────────────────

def strip_checkpoint(src_path, dst_path=None, migrate_keys=True):
    """Create an inference-optimized checkpoint (model weights only).

    Removes optimizer state and epoch metadata. Optionally migrates legacy
    state-dict keys and strips ``_orig_mod.`` prefixes from compiled models.

    The output is a bare ``OrderedDict`` (not wrapped in a dict with a
    ``model_state_dict`` key). This is compatible with
    ``_BaseInference.from_pretrained`` which accepts both formats.

    Args:
        src_path: Path to the source training checkpoint.
        dst_path: Output path. Defaults to ``<src_path>.stripped.pt``.
        migrate_keys: Apply key migration and compiled-model fix if True.

    Returns:
        str: The output path.
    """
    if dst_path is None:
        dst_path = str(src_path) + ".stripped.pt"
    raw = torch.load(str(src_path), map_location="cpu", weights_only=True)
    state_dict = raw["model_state_dict"] if "model_state_dict" in raw else raw
    if migrate_keys:
        state_dict = _fix_compiled_state_dict(state_dict)
        state_dict = _migrate_state_dict(state_dict)
    torch.save(state_dict, dst_path)
    logger.debug(f"Stripped checkpoint saved to {dst_path}")
    return dst_path


def _prune_old_checkpoints(checkpoint_dir, keep_last_n=5, keep_every_n=10):
    """Keep the last *keep_last_n* epoch checkpoints plus every *keep_every_n*-th epoch.

    Files named ``best_model.pth`` and ``best_tracker.json`` are never removed.

    Args:
        checkpoint_dir: Directory containing epoch checkpoint files.
        keep_last_n: Number of most recent epoch checkpoints to keep (default 5).
        keep_every_n: Also keep checkpoints at every N-th epoch (default 10).
                      Set to 0 or None to disable.
    """
    result = _find_latest_checkpoint(checkpoint_dir)
    if result is None:
        return
    _, _, all_files, all_epochs = result
    paired = sorted(zip(all_epochs, all_files))
    if len(paired) <= keep_last_n:
        return

    keep_set = set(e for e, _ in paired[-keep_last_n:])
    if keep_every_n:
        keep_set.update(e for e, _ in paired if e % keep_every_n == 0)

    for ep, fname in paired:
        if ep not in keep_set:
            path = os.path.join(checkpoint_dir, fname)
            os.remove(path)
            logger.debug(f"Pruned old checkpoint: {fname}")


def save_checkpoint_optimized(
    epoch,
    model,
    optimizer,
    checkpoint_path,
    val_metric,
    best_tracker,
    keep_last_n=5,
):
    """Save training checkpoint, update best model, and prune old files.

    This is the drop-in replacement for ``save_checkpoint_per_nth``.

    Args:
        epoch: Current epoch number.
        model: The model whose state is saved.
        optimizer: The optimizer whose state is saved.
        checkpoint_path: Directory for saving checkpoints.
        val_metric: Scalar validation metric for best-model comparison.
        best_tracker: A ``BestModelTracker`` instance.
        keep_last_n: Number of recent epoch checkpoints to keep (default 5).

    Returns:
        bool: True if this epoch is a new best.
    """
    # 1. Save full training checkpoint (optimizer included for resume)
    ckpt_file = os.path.join(checkpoint_path, f"epoch.{epoch:04}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_file,
    )

    # 2. Best model tracking
    is_best = best_tracker.update(epoch, val_metric)
    if is_best:
        best_path = os.path.join(checkpoint_path, "best_model.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            best_path,
        )
        logger.info(f"New best model at epoch {epoch} (metric={val_metric:.4e})")
    best_tracker.save(checkpoint_path)

    # 3. Prune old epoch checkpoints
    _prune_old_checkpoints(checkpoint_path, keep_last_n=keep_last_n)

    return is_best


class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            return 1.0
        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)



def _find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file and return (path, epoch, all_files, all_epochs).

    Only files matching the epoch.<N>.(pt|pth|pkl) naming convention are
    considered. Non-checkpoint files in the directory are silently ignored.

    Returns None if no matching checkpoint files are found.
    """
    all_files = os.listdir(checkpoint_dir)
    matches = [(f, int(m.group(1))) for f in all_files if (m := _EPOCH_PATTERN.match(f))]
    if not matches:
        return None
    matches.sort(key=lambda x: x[1])
    max_epoch = matches[-1][1]
    latest_path = os.path.join(checkpoint_dir, matches[-1][0])
    checkpoint_files = [f for f, _ in matches]
    epochs = [e for _, e in matches]
    return latest_path, max_epoch, checkpoint_files, epochs


def _fix_compiled_state_dict(state_dict):
    """Remove '_orig_mod.' prefix from compiled model state_dict keys."""
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        logger.info("Detected compiled state_dict format. Auto-fixing keys...")
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


def _migrate_state_dict(state_dict):
    """Migrate old layer names to new names in state_dict keys.

    This function is idempotent: applying it to already-migrated keys
    is a no-op because old patterns do not appear in new key names.
    """
    new_sd, log = migrate_state_dict(state_dict)
    if log:
        logger.info(f"Migrated {len(log)} state_dict keys to new naming scheme.")
    return new_sd


def load_last_checkpoint_n_get_epoch(checkpoint_dir, model, optimizer=None, location='cuda'):
    """Load the latest checkpoint (model + optimizer state) from a given directory.

    Args:
        checkpoint_dir (str): Directory containing the checkpoint files.
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        location (str): Device location for loading the checkpoint. Defaults to 'cuda'.

    Returns:
        int: The next epoch number (loaded epoch + 1).
             If no checkpoint is found, returns 1.
    """
    result = _find_latest_checkpoint(checkpoint_dir)
    if result is None:
        return 1
    latest_path, _, _, _ = result
    logger.info(f"Loaded Pretrained model from {latest_path} .....")
    checkpoint_dict = torch.load(latest_path, map_location=location, weights_only=True)
    state_dict = _fix_compiled_state_dict(checkpoint_dict['model_state_dict'])
    state_dict = _migrate_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    return checkpoint_dict['epoch'] + 1


def sync_checkpoint_to_home(checkpoint_path, variant_short, config_name):
    """Export a stripped checkpoint to sr_corrnet/checkpoints/{variant}/{config_stem}/model.pt.

    Called automatically after each checkpoint save during training.
    Creates the target directory on first run if it does not exist.
    The exported file contains model weights only (no optimizer state).
    Silently skips if the source checkpoint is not found.

    Args:
        checkpoint_path: Source checkpoint directory (e.g., log/log_UMA_7ch/scratch_weights/).
        variant_short: Short variant name ("SS").
        config_name: Config filename (e.g., "UMA_7ch_varying_0_3spk.yaml").
    """
    # Prefer best_model.pth if available, otherwise use the latest epoch
    best_path = os.path.join(checkpoint_path, "best_model.pth")
    if os.path.exists(best_path):
        src_path = best_path
    else:
        result = _find_latest_checkpoint(checkpoint_path)
        if result is None:
            return
        src_path = result[0]

    config_stem = Path(config_name).stem
    home_dir = Path(__file__).resolve().parent.parent / "checkpoints" / variant_short / config_stem
    home_dir.mkdir(parents=True, exist_ok=True)
    dest = home_dir / "model.pt"
    strip_checkpoint(src_path, str(dest), migrate_keys=True)
    logger.info(f"Checkpoint exported: {src_path} -> {dest}")

    # Write metadata alongside the exported checkpoint
    import json
    from datetime import date
    src_name = Path(src_path).name
    epoch_match = _EPOCH_PATTERN.match(src_name)
    if epoch_match:
        epoch = int(epoch_match.group(1))
    else:
        # best_model.pth etc. — read epoch from inside the checkpoint
        raw = torch.load(str(src_path), map_location="cpu", weights_only=False)
        epoch = raw.get("epoch")
    metadata = {
        "variant": variant_short,
        "config": config_stem,
        "source_checkpoint": src_name,
        "epoch": epoch,
        "export_date": date.today().isoformat(),
    }
    meta_path = home_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.debug(f"Metadata saved to {meta_path}")


def step_scheduler(scheduler, epoch, start_scheduling, **kwargs):
    """Step the scheduler only when epoch > start_scheduling."""
    if epoch <= start_scheduling:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(kwargs.get('val_loss'))
    else:
        scheduler.step()

def model_params_mac_summary(model, input_shape, metrics, device):
    from torchinfo import summary as summary_
    from ptflops import get_model_complexity_info
    from thop import profile

    # ptflpos
    if 'ptflops' in metrics:
        MACs_ptflops, params_ptflops = get_model_complexity_info(model, (*input_shape,), print_per_layer_stat=False, verbose=False) # (num_samples,)
        MACs_ptflops, params_ptflops = MACs_ptflops.replace(" MMac", ""), params_ptflops.replace(" M", "")
        logger.info(f"ptflops: MACs: {MACs_ptflops}, Params: {params_ptflops}")

    # thop
    if 'thop' in metrics:
        input = torch.randn(1, *input_shape).to(device)
        MACs_thop, params_thop = profile(model, inputs=input, verbose=False)
        MACs_thop, params_thop = MACs_thop/1e9, params_thop/1e6
        logger.info(f"thop: MACs: {MACs_thop} GMac, Params: {params_thop}")
    
    # torchinfo
    if 'torchinfo' in metrics:
        model_profile = summary_(model, input_size=(1, *input_shape), verbose=0)
        MACs_torchinfo, params_torchinfo = model_profile.total_mult_adds/1e9, model_profile.total_params/1e6
        logger.info(f"torchinfo: MACs: {MACs_torchinfo} GMac, Params: {params_torchinfo}")


    # MEASURE PERFORMANCE
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 500
    # repetitions2 = 500
    # timings = np.zeros((repetitions,1))
    # torch.set_num_threads(1)
    # with torch.no_grad():
    #     for rep in range(repetitions+repetitions2):
    #         if rep > repetitions:
    #             starter.record()
    #             _ = model(dummy_input)
    #             ender.record()
    #             # WAIT FOR GPU SYNC
    #             torch.cuda.synchronize()
    #             curr_time = starter.elapsed_time(ender)
    #             timings[rep-repetitions2] = curr_time
    # logger.info(f"Timing: {timings.mean()}")


def create_sampler(dataset_size, num_samples):
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    sampled_indices = indices[:num_samples]
    return torch.utils.data.SubsetRandomSampler(sampled_indices)


def setup_logging(config_name, base_dir, model, optimizer, device, engine_mode="train", **writer_kwargs):
    """Create log directories, checkpoint path, and TensorBoard writer.

    - train: loads from scratch_weights/ (includes optimizer state for resume).
    - test/inference: loads from checkpoints/{variant}/{config}.pt (stripped
      weights synced every epoch or downloaded from HF Hub).
    """
    from sr_corrnet.utils import util_writer
    log_base = f"log/log_{Path(config_name).stem}"
    ckpt_path = os.path.join(base_dir, log_base, "scratch_weights")
    os.makedirs(ckpt_path, exist_ok=True)
    tb_path = os.path.join(base_dir, log_base, "tensorboard")
    writer = util_writer.MyWriter(logdir=tb_path, **writer_kwargs)

    if engine_mode == "train":
        start_epoch = load_last_checkpoint_n_get_epoch(ckpt_path, model, optimizer, location=device)
    else:
        variant_short = Path(base_dir).name.replace("SR_CorrNet_", "")
        config_stem = Path(config_name).stem
        exported = Path(__file__).resolve().parent.parent / "checkpoints" / variant_short / config_stem / "model.pt"

        if exported.exists():
            logger.info(f"Loading checkpoint from {exported}")
            checkpoint_dict = torch.load(str(exported), map_location=device, weights_only=True)
            state_dict = _fix_compiled_state_dict(checkpoint_dict)
            state_dict = _migrate_state_dict(state_dict)
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(f"No checkpoint at {exported}, falling back to scratch_weights/")
            load_last_checkpoint_n_get_epoch(ckpt_path, model, optimizer, location=device)
        start_epoch = 1

    return ckpt_path, writer, start_epoch


def setup_optimizer_and_scheduler(model, config):
    """Create optimizer, warmup scheduler, and main scheduler from config."""
    engine_cfg = config["engine"]
    optim_cls = getattr(torch.optim, engine_cfg["optimizer"]["name"])
    sched_cls = getattr(torch.optim.lr_scheduler, engine_cfg["scheduler"]["name"])
    optimizer = optim_cls(model.parameters(),
                          **engine_cfg["optimizer"].get(engine_cfg["optimizer"]["name"], {}))
    warmup = WarmupConstantSchedule(optimizer, **engine_cfg["scheduler"]["WarmupConstantSchedule"])
    scheduler = sched_cls(optimizer, **engine_cfg["scheduler"].get(engine_cfg["scheduler"]["name"], {}))
    return optimizer, warmup, scheduler


def create_dataloader_with_sampler(_dataloader, subset_conf, loader_config):
    """Return a new DataLoader with subset sampler if enabled, otherwise return the original."""
    if subset_conf["subset"]:
        sampler = create_sampler(len(_dataloader.dataset), subset_conf["num_per_epoch"])
        return torch.utils.data.DataLoader(
            dataset=_dataloader.dataset, collate_fn=_dataloader.collate_fn,
            sampler=sampler, **loader_config)
    return _dataloader


# ── Logging utilities ────────────────────────────────────────────────────────

_LOG_SCALE_KEYS = {'SISNRi', 'SDR', 'SDRi', 'PESQ', 'STOI'}

PBAR_FMT = '{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'


def format_pbar(dict_loss):
    """Format dict for tqdm postfix: .2e default, .3f for log-scale (dB) metrics."""
    return {
        k: (f"{v:.3f}" if k in _LOG_SCALE_KEYS else f"{v:.2e}")
        for k, v in dict_loss.items()
    }


def log_scalars_to_tb(writer, metric_dict, epoch):
    """Log each key in metric_dict as an individual TensorBoard scalar.

    Keys should use 'group/partition' format (e.g., 'Loss/train', 'PESQ/valid')
    so TensorBoard auto-groups them into separate charts.
    """
    for tag, value in metric_dict.items():
        writer.add_scalar(tag, value, epoch)
    writer.flush()


def format_epoch_log(config_name, epoch, stage, metrics, elapsed_sec, suffix=""):
    """Build a single epoch log line.

    Args:
        config_name: YAML config stem for identification.
        epoch: Current epoch number.
        stage: 'TRAIN', 'VALID', etc.
        metrics: dict of {key: value} to display.
        elapsed_sec: Wall-clock seconds for this stage.
        suffix: Optional trailing text such as ' (improved)'.
    """
    header = f"[{config_name}]\n[Epoch {epoch}] {stage} ({elapsed_sec:.1f}s){suffix}"
    metric_parts = []
    for k, v in metrics.items():
        if k in _LOG_SCALE_KEYS:
            metric_parts.append(f"{k}={v:.3f}")
        else:
            metric_parts.append(f"{k}={v:.2e}")
    metric_str = " | ".join(metric_parts)
    return f"{header}{metric_str}"
