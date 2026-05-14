"""Checkpoint export utility.

Copies trained checkpoints from the internal log/ directory structure
to an external location (default: sr_corrnet/checkpoints/{variant}/).
Training code continues to save to log/ as before -- this utility provides
a clean way to "publish" a trained checkpoint for library API consumption.

CLI usage::

    python sr_corrnet/export.py --variant SS --config 1ch_WSJ_fix_2spk.yaml
    python sr_corrnet/export.py --variant SS --config 1ch_WSJ_fix_2spk.yaml --output /path/to/dir/

Library usage::

    from sr_corrnet.export import export_checkpoint
    ckpt_path = export_checkpoint("SS", "1ch_WSJ_fix_2spk.yaml")
"""
import argparse
import importlib
import os
from pathlib import Path

from sr_corrnet._config import _VARIANT_MAP
from sr_corrnet.utils.util_engine import (
    _EPOCH_PATTERN,
    _find_latest_checkpoint,
)


_DEFAULT_CKPT_HOME = Path(__file__).resolve().parent / "checkpoints"

# Map full variant names to short names used in the default output directory
_VARIANT_SHORT = {
    "SR_CorrNet_SS": "SS",
}

# Build variant package mapping: accept both short ("SS") and full ("SR_CorrNet_SS") names
_VARIANT_PACKAGE = {**_VARIANT_MAP}
for short, pkg in _VARIANT_MAP.items():
    full_name = pkg.split(".")[-1]  # "SR_CorrNet_SS"
    _VARIANT_PACKAGE[full_name] = pkg

# Reverse map: short name -> full name (used for output_dir naming)
_SHORT_TO_FULL = {v: k for k, v in _VARIANT_SHORT.items()}


def _normalize_variant(variant):
    """Normalize variant name to the full form (e.g. 'SS' -> 'SR_CorrNet_SS').

    Args:
        variant (str): Either a short name ('SS') or a full name ('SR_CorrNet_SS').

    Returns:
        str: Full variant name.

    Raises:
        ValueError: If the variant name is not recognized.
    """
    if variant not in _VARIANT_PACKAGE:
        valid = sorted(_VARIANT_PACKAGE.keys())
        raise ValueError(
            f"Unknown variant '{variant}'. Valid options: {valid}"
        )
    # If already full name, return as-is; otherwise expand via reverse map
    if variant in _VARIANT_SHORT:
        return variant  # already full name
    return _SHORT_TO_FULL[variant]


def _resolve_source_dir(variant_full, config_name):
    """Locate the checkpoint directory inside the variant's log/ tree.

    The variant package is imported to find its installed location, then
    the standard log directory layout is used::

        <variant_dir>/log/log_<config_stem>/scratch_weights/

    Prefers best_model.pth if available, otherwise falls back to the
    latest epoch checkpoint.

    Args:
        variant_full (str): Full variant name, e.g. 'SR_CorrNet_SS'.
        config_name (str): Config filename, e.g. 'UMA_7ch.yaml'.

    Returns:
        Path: Path to the checkpoint file or directory.

    Raises:
        ImportError: If the variant package cannot be imported.
        FileNotFoundError: If no matching checkpoint directory is found.
    """
    pkg_name = _VARIANT_PACKAGE[variant_full]
    try:
        pkg = importlib.import_module(pkg_name)
    except ImportError as e:
        raise ImportError(
            f"Could not import variant package '{pkg_name}': {e}"
        ) from e

    variant_dir = Path(pkg.__file__).parent
    config_stem = Path(config_name).stem
    scratch_path = variant_dir / "log" / f"log_{config_stem}" / "scratch_weights"

    def _has_checkpoints(path):
        """Return True if path exists and contains at least one epoch checkpoint file."""
        if not path.is_dir():
            return False
        return any(_EPOCH_PATTERN.match(f) for f in os.listdir(path))

    best = scratch_path / "best_model.pth"
    if best.is_file():
        return best

    if _has_checkpoints(scratch_path):
        return scratch_path

    raise FileNotFoundError(
        f"No checkpoint files found in:\n"
        f"  {scratch_path}\n"
        f"Train the model first or verify the config name."
    )


def _validate_config(variant_short, config_name):
    """Validate that config_name exists in the variant's configs/ directory."""
    from sr_corrnet._config import resolve_config
    config_file = config_name if config_name.endswith(".yaml") else config_name + ".yaml"
    resolve_config(variant_short, config_file)  # raises FileNotFoundError if invalid


def export_checkpoint(variant, config_name):
    """Export the latest trained checkpoint to checkpoints/{variant}/{config_stem}/model.pt.

    Locates the latest checkpoint under the variant's internal
    log/log_<config>/scratch_weights/ directory and exports a stripped
    copy (model weights only, no optimizer state) via sync_checkpoint_to_home.

    Args:
        variant (str): Model variant. Accepts both short form ('SS') and
                       full form ('SR_CorrNet_SS').
        config_name (str): Config filename used during training, e.g.
                           'UMA_7ch.yaml'.

    Returns:
        Path: Absolute path to the exported checkpoint file.

    Raises:
        ValueError: If variant is not recognized.
        FileNotFoundError: If config_name is invalid or no checkpoint found.

    Example::

        from sr_corrnet.export import export_checkpoint
        dest = export_checkpoint("SS", "1ch_WSJ_fix_2spk.yaml")
        print(f"Exported to: {dest}")
    """
    from sr_corrnet.utils.util_engine import sync_checkpoint_to_home

    variant_full = _normalize_variant(variant)
    variant_short = _VARIANT_SHORT[variant_full]
    _validate_config(variant_short, config_name)

    # Resolve source checkpoint directory
    source = _resolve_source_dir(variant_full, config_name)
    source_dir = str(Path(source).parent) if Path(source).is_file() else str(source)

    sync_checkpoint_to_home(source_dir, variant_short, config_name)

    config_stem = Path(config_name).stem
    dest_path = _DEFAULT_CKPT_HOME / variant_short / config_stem / "model.pt"
    print(f"Exported to: {dest_path}")
    return dest_path


def _file_hash(path):
    """Compute SHA256 hash of a file."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_file_path(ckpt_dir):
    """Return the path to the .upload_hash file in a checkpoint directory."""
    return Path(ckpt_dir) / ".upload_hash"


def upload_to_hub(
    variant,
    config_name,
    repo_id=None,
    private=True,
    token=None,
    force=False,
):
    """Upload an exported checkpoint (with its config YAML) to HF Hub.

    The checkpoint must already exist in checkpoints/ (run export first).
    Skips upload if the checkpoint hasn't changed since last upload
    (tracked via .upload_hash file). Use force=True to override.

    Args:
        variant (str): Model variant ('SS' or 'SR_CorrNet_SS').
        config_name (str): Config filename, e.g. 'UMA_7ch_varying_0_3spk.yaml'.
        repo_id (str, optional): HF repo ID. Auto-generated if None.
        private (bool): Create private repo (default True).
        token (str, optional): HF token. Uses cached token if None.
        force (bool): Upload even if checkpoint hasn't changed (default False).

    Returns:
        str or None: URL of the HF repo, or None if skipped.

    Raises:
        FileNotFoundError: If checkpoint not found in checkpoints/.
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for HF Hub uploads. "
            "Install with: uv sync --extra hub  (or: pip install sr-corrnet[hub])"
        )

    # 1. Locate exported checkpoint
    variant_full = _normalize_variant(variant)
    variant_short = _VARIANT_SHORT[variant_full]
    config_stem = Path(config_name).stem
    ckpt_path = _DEFAULT_CKPT_HOME / variant_short / config_stem / "model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"No exported checkpoint at {ckpt_path}\n"
            f"Export first: python sr_corrnet/export.py --variant {variant_short} --config {config_name}"
        )

    # 2. Check if changed since last upload
    current_hash = _file_hash(ckpt_path)
    hash_file = _hash_file_path(ckpt_path.parent)
    if not force and hash_file.exists() and hash_file.read_text().strip() == current_hash:
        print(f"[SKIP] {variant}/{Path(config_name).stem} — unchanged since last upload")
        return None

    # 3. Resolve config YAML path
    from sr_corrnet._config import resolve_config
    yaml_path = resolve_config(variant_short, config_name)

    # 4. Auto-generate repo_id if not provided
    if repo_id is None:
        config_stem = Path(config_name).stem
        slug = config_stem.lower().replace("_", "-")
        repo_id = f"shinuh/sr-corrnet-{variant_short.lower()}-{slug}"

    # 5. Upload to HF Hub
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(ckpt_path),
        path_in_repo="model.pt",
        repo_id=repo_id,
    )
    api.upload_file(
        path_or_fileobj=str(yaml_path),
        path_in_repo="config.yaml",
        repo_id=repo_id,
    )

    # 6. Save hash
    hash_file.write_text(current_hash)

    url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded to: {url}")
    return url


def upload_all(private=True, token=None, force=False):
    """Upload all checkpoints under sr_corrnet/checkpoints/ to HF Hub.

    Only uploads checkpoints that have changed since last upload.

    Args:
        private (bool): Create private repos (default True).
        token (str, optional): HF token.
        force (bool): Upload all regardless of hash (default False).

    Returns:
        dict: {repo_id: url_or_None} for each checkpoint.
    """
    results = {}
    ckpt_root = _DEFAULT_CKPT_HOME
    if not ckpt_root.exists():
        print("No checkpoints found.")
        return results

    for variant_dir in sorted(ckpt_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant_short = variant_dir.name
        for config_dir in sorted(variant_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            model_pt = config_dir / "model.pt"
            if not model_pt.exists():
                continue
            config_name = config_dir.name + ".yaml"
            try:
                url = upload_to_hub(variant_short, config_name,
                                     private=private, token=token, force=force)
                slug = config_dir.name.lower().replace("_", "-")
                repo_id = f"shinuh/sr-corrnet-{variant_short.lower()}-{slug}"
                results[repo_id] = url
            except Exception as e:
                print(f"[ERROR] {variant_short}/{config_dir.name}: {e}")
                results[f"{variant_short}/{config_dir.name}"] = None

    uploaded = sum(1 for v in results.values() if v is not None)
    skipped = sum(1 for v in results.values() if v is None)
    print(f"\nDone: {uploaded} uploaded, {skipped} skipped/failed")
    return results


def download_from_hub(variant, config_name, repo_id=None, output_dir=None, token=None):
    """Download a checkpoint from HF Hub for local use with run.py.

    Downloads ``model.pt`` from the given HF repo and saves to
    ``sr_corrnet/checkpoints/{variant}/{config_stem}/model.pt``.

    Args:
        variant (str): Model variant ('SS' or 'SR_CorrNet_SS').
        config_name (str): Config filename, e.g. ``"UMA_7ch_varying_0_3spk.yaml"``.
        repo_id (str, optional): HF repo ID. Auto-generated if None.
        output_dir (str or Path, optional): Destination directory. Defaults to
                   ``sr_corrnet/checkpoints/{variant}/{config_stem}/``.
        token (str, optional): HF token. Uses cached token if None.

    Returns:
        dict: ``{"checkpoint": Path, "config": Path}`` pointing to local files.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for HF Hub downloads. "
            "Install with: uv sync --extra hub  (or: pip install sr-corrnet[hub])"
        )

    variant_full = _normalize_variant(variant)
    variant_short = _VARIANT_SHORT[variant_full]

    _validate_config(variant_short, config_name)

    config_stem = Path(config_name).stem

    # Auto-generate repo_id if not provided
    if repo_id is None:
        slug = config_stem.lower().replace("_", "-")
        repo_id = f"shinuh/sr-corrnet-{variant_short.lower()}-{slug}"

    # Download checkpoint and config
    ckpt_path = Path(hf_hub_download(repo_id=repo_id, filename="model.pt", token=token))
    config_path = Path(hf_hub_download(repo_id=repo_id, filename="config.yaml", token=token))

    # Copy to output directory
    if output_dir is None:
        output_dir = _DEFAULT_CKPT_HOME / variant_short / config_stem
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    import shutil
    dest_ckpt = output_dir / "model.pt"
    dest_config = output_dir / "config.yaml"
    shutil.copy2(ckpt_path, dest_ckpt)
    shutil.copy2(config_path, dest_config)

    print(
        f"Downloaded from: https://huggingface.co/{repo_id}\n"
        f"  Checkpoint: {dest_ckpt}\n"
        f"  Config:     {dest_config}"
    )
    return {"checkpoint": dest_ckpt, "config": dest_config}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export, upload, or download SR_CorrNet checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Export local checkpoint\n"
            "  python sr_corrnet/export.py --variant SS --config 1ch_WSJ_fix_2spk.yaml\n"
            "\n"
            "  # Upload to HF Hub\n"
            "  python sr_corrnet/export.py --upload --variant SS --config 1ch_WSJ_fix_2spk.yaml\n"
            "\n"
            "  # Upload all changed checkpoints to HF Hub\n"
            "  python sr_corrnet/export.py --upload-all\n"
            "\n"
            "  # Download from HF Hub\n"
            "  python sr_corrnet/export.py --download --variant SS --config 1ch_WSJ_fix_2spk.yaml\n"
        ),
    )
    parser.add_argument(
        "--variant",
        default=None,
        choices=["SS", "SR_CorrNet_SS"],
        help=(
            "Model variant. Accepts short form (SS) or full form (SR_CorrNet_SS). "
            "Required for export/upload, optional for download (auto-detected)."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Config filename used during training, e.g. 'UMA_7ch.yaml'. Required for export/upload.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory for the exported/downloaded checkpoint. "
            f"Defaults to sr_corrnet/checkpoints/{{variant}}/."
        ),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the exported checkpoint to Hugging Face Hub after export.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download a checkpoint from Hugging Face Hub.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="HF repo ID for upload/download (auto-generated for upload if omitted).",
    )
    parser.add_argument(
        "--upload-all",
        action="store_true",
        help="Upload all changed checkpoints to HF Hub.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force upload even if checkpoint hasn't changed.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create a public HF repo (default is private).",
    )

    cli_args = parser.parse_args()

    if cli_args.upload_all:
        upload_all(private=not cli_args.public, force=cli_args.force)
    elif cli_args.download:
        if cli_args.variant is None or cli_args.config is None:
            parser.error("--download requires --variant and --config")
        result = download_from_hub(
            variant=cli_args.variant,
            config_name=cli_args.config,
            repo_id=cli_args.repo_id,
            output_dir=cli_args.output,
        )
        print(f"Done: {result['checkpoint']}")
    elif cli_args.upload:
        if cli_args.variant is None or cli_args.config is None:
            parser.error("--upload requires --variant and --config")
        url = upload_to_hub(
            variant=cli_args.variant,
            config_name=cli_args.config,
            repo_id=cli_args.repo_id,
            private=not cli_args.public,
        )
        print(f"Done: {url}")
    else:
        if cli_args.variant is None or cli_args.config is None:
            parser.error("export requires --variant and --config")
        exported_path = export_checkpoint(
            variant=cli_args.variant,
            config_name=cli_args.config,
        )
        print(f"Done: {exported_path}")
