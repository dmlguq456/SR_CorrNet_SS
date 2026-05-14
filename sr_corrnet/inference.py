"""Public inference API for SR_CorrNet SS (Speech Separation).

Provides SSInference, which wraps the SS variant's internal EngineInfer
with a clean, user-facing interface.

Waveform-level API::

    from sr_corrnet import SSInference

    model = SSInference.from_pretrained(
        config="path/to/1ch_WSJ_fix_2spk.yaml",
        checkpoint_path="path/to/checkpoints/",
        device="cuda:0",
    )

    # Process a waveform tensor
    import torch
    waveform = torch.randn(1, 160000)
    result = model.process_waveform(waveform, n_spks=torch.tensor(2))
    # result["waveforms"] -> list of per-speaker waveform tensors

    # Process a wav file and save outputs
    result = model.process_file("input.wav", output_dir="output/")

STFT-level API (for frame-by-frame integration)::

    # Access the model's STFT/iSTFT transforms
    stft_fn  = model.stft   # waveform -> complex spectrogram
    istft_fn = model.istft  # complex spectrogram -> waveform

    # Process a single STFT frame chunk
    result = model.process_stft_chunk(stft_frames)  # single chunk
    # stft_frames: (channels, freq_bins, time_frames) complex tensor

    # Process full STFT (single-pass)
    result = model.process_stft(stft_frames)
    # result["stft_out"] -> (N, F, T) complex tensor
"""
import importlib
from pathlib import Path

import librosa
import torch
from loguru import logger

from sr_corrnet.export import _DEFAULT_CKPT_HOME
from sr_corrnet.utils import util_system
from sr_corrnet.utils.util_engine import (
    _fix_compiled_state_dict, _migrate_state_dict, _find_latest_checkpoint
)


class _BaseInference:
    """Base class for public inference wrappers.

    Subclasses set ``_VARIANT`` to the variant package path (e.g.,
    ``"sr_corrnet.models.SR_CorrNet_SS"``). All logic lives here;
    subclasses only provide the ``_VARIANT`` class attribute.

    Public API:
        from_pretrained(config, checkpoint_path, device="cuda:0", **kwargs)
        process_file(input_path, output_dir=None, n_spks=None) -> dict
        process_waveform(waveform, n_spks=None) -> dict
        process_stft(stft_input, n_spks=None) -> dict
        process_stft_chunk(stft_chunk, n_spks=None) -> dict
    """

    _VARIANT: str  # overridden by each subclass

    @staticmethod
    def _is_hf_repo_id(path_or_id):
        """Return True if *path_or_id* looks like a HF Hub repo ID (owner/name)."""
        if path_or_id is None:
            return False
        s = str(path_or_id)
        if Path(s).exists():
            return False
        parts = s.split("/")
        if len(parts) != 2:
            return False
        # HF repo IDs don't end in file extensions
        return not any(p.endswith((".pt", ".pth", ".yaml", ".pkl")) for p in parts)

    @classmethod
    def _download_from_hub(cls, repo_id, config, variant_short):
        """Download model.pt and config.yaml from a HF Hub repo.

        Args:
            repo_id: e.g. "shinuh/sr-corrnet-css-uma-7ch"
            config: current config (str/Path/dict or None)
            variant_short: "SS"

        Returns:
            (checkpoint_path, config) where both point to local cached files.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for HF Hub downloads. "
                "Install with: uv sync --extra hub  (or: pip install sr-corrnet[hub])"
            )

        ckpt_path = Path(hf_hub_download(repo_id=repo_id, filename="model.pt"))

        # Download config from repo if user didn't provide a local config
        if config is None or (
            isinstance(config, str)
            and not Path(config).exists()
            and not config.endswith(".yaml")
        ):
            config = str(Path(hf_hub_download(repo_id=repo_id, filename="config.yaml")))

        return ckpt_path, config

    @classmethod
    def from_pretrained(cls, config=None, checkpoint_path=None, device="cuda:0", **kwargs):
        """Create an inference engine from a config and a checkpoint.

        Args:
            config: Path to a YAML config file (str or Path), a pre-loaded
                config dict, a config name (e.g. ``"1ch_DNS"``), or ``None``
                when *checkpoint_path* is a HF Hub repo ID that includes
                ``config.yaml``.
            checkpoint_path: One of:
                - Local file path to a ``.pt``/``.pth`` checkpoint
                - Local directory containing checkpoint files
                - HF Hub repo ID (e.g. ``"shinuh/sr-corrnet-css-uma-7ch"``)
                - ``None`` (falls back to ``sr_corrnet/checkpoints/{variant}/``)
            device: PyTorch device string (e.g., ``"cuda:0"``, ``"cpu"``).
            **kwargs: Override values for ``config["inference"]``.

        Returns:
            Initialized instance of the calling class.

        Raises:
            FileNotFoundError: If ``checkpoint_path`` is None and the default
                path does not exist, or if a provided path does not exist.
            KeyError: If the config dict is missing required sections.
        """
        # Suppress internal logging during library setup.
        # Note: logger.disable/enable affects global loguru state. In
        # multi-threaded scenarios, logging from other threads will also be
        # suppressed during this call. For thread-safe usage, callers should
        # manage logging themselves and not rely on the auto-suppress behavior.
        logger.disable("sr_corrnet")

        try:
            return cls._from_pretrained_impl(config, checkpoint_path, device, **kwargs)
        finally:
            logger.enable("sr_corrnet")

    @classmethod
    def _from_pretrained_impl(cls, config, checkpoint_path, device, **kwargs):
        variant_short = cls._VARIANT.split(".")[-1].replace("SR_CorrNet_", "")

        # ── 0. HF Hub detection ──────────────────────────────────────────────
        if checkpoint_path is not None and cls._is_hf_repo_id(str(checkpoint_path)):
            checkpoint_path, config = cls._download_from_hub(
                str(checkpoint_path), config, variant_short
            )

        # ── 1. Load config ────────────────────────────────────────────────────
        if isinstance(config, (str, Path)):
            config_str = str(config)
            # If not an existing file path, try resolving as a config name
            # e.g. "1ch_DNS.yaml" or "1ch_DNS" → resolve via _config.py
            if not Path(config_str).exists():
                from sr_corrnet._config import resolve_config
                config_name = config_str if config_str.endswith(".yaml") else config_str + ".yaml"
                config_str = resolve_config(variant_short, config_name)
                config = config_str  # update for config_stem resolution later
            yaml_dict = util_system.parse_yaml(config_str)
            # parse_yaml returns the full YAML dict; inner config is under "config" key
            cfg = yaml_dict["config"]
        elif isinstance(config, dict):
            # Accept either the full YAML dict or the inner config dict
            if "config" in config:
                cfg = config["config"]
            else:
                cfg = config
        else:
            raise TypeError(
                f"config must be a file path (str/Path) or a dict, got {type(config)}"
            )

        # ── 2. Apply kwargs overrides to config["inference"] ─────────────────
        if kwargs:
            infer_section = cfg.setdefault("inference", {})
            for key, value in kwargs.items():
                infer_section[key] = value

            # SS-specific: sampling_rate lives in dataset config, not inference
            if "sampling_rate" in kwargs:
                try:
                    cfg["dataset"]["synthesis_config"]["sampling_rate"] = (
                        kwargs["sampling_rate"]
                    )
                except KeyError:
                    pass  # SS dataset section absent — non-fatal

        # ── 3. Resolve checkpoint path ────────────────────────────────────────
        if checkpoint_path is None:
            config_stem = Path(config).stem if isinstance(config, (str, Path)) else None
            default_dir = Path(_DEFAULT_CKPT_HOME) / variant_short / config_stem if config_stem else Path(_DEFAULT_CKPT_HOME) / variant_short

            if not default_dir.exists():
                raise FileNotFoundError(
                    f"Default checkpoint directory does not exist: {default_dir}\n"
                    f"Populate it with: export_checkpoint('{variant_short}', "
                    f"'<config_name>.yaml')"
                )
            checkpoint_path = default_dir
        else:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint path does not exist: {checkpoint_path}"
                )

        # ── 4. Build model ────────────────────────────────────────────────────
        model_module = importlib.import_module(f"{cls._VARIANT}.model")
        Model = model_module.Model

        torch_device = torch.device(device)
        model = Model(**cfg["model"]).to(torch_device)

        # ── 5. Load weights ───────────────────────────────────────────────────
        if checkpoint_path.is_dir():
            config_stem = Path(config).stem if isinstance(config, (str, Path)) else None
            exported_pt = checkpoint_path / f"{config_stem}.pt" if config_stem else None

            if exported_pt is not None and exported_pt.exists():
                ckpt_file = exported_pt
            else:
                result = _find_latest_checkpoint(str(checkpoint_path))
                if result is None:
                    raise FileNotFoundError(
                        f"No checkpoint found in: {checkpoint_path}\n"
                        f"Expected: {exported_pt or 'any .pt/.pth/.pkl file'}\n"
                        f"Run export_checkpoint('{variant_short}', '<config>.yaml') "
                        f"to populate the default path, or specify checkpoint_path "
                        f"as a file path."
                    )
                ckpt_file = Path(result[0])
        else:
            ckpt_file = checkpoint_path

        raw = torch.load(str(ckpt_file), map_location=torch_device, weights_only=True)
        state_dict = raw["model_state_dict"] if "model_state_dict" in raw else raw
        state_dict = _fix_compiled_state_dict(state_dict)
        state_dict = _migrate_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded checkpoint: {ckpt_file}")

        # ── 6. Build EngineInfer ──────────────────────────────────────────────
        engine_module = importlib.import_module(f"{cls._VARIANT}.engine_infer")
        EngineInfer = engine_module.EngineInfer

        engine = EngineInfer(cfg, model, torch_device)

        # ── 7. Build instance ─────────────────────────────────────────────────
        instance = cls.__new__(cls)
        instance.engine = engine
        instance._config = cfg
        # Store num_mics/mic_idx from config rather than engine attributes
        # because SS EngineInfer does not set these attributes.
        instance._num_mics = cfg.get("num_mics", 1)
        instance._mic_idx = cfg.get("mic_idx", [0])
        return instance

    @property
    def stft(self):
        """Return the STFT module from the underlying engine.

        The returned object is a ``ConvSTFT`` instance whose ``__call__`` accepts
        ``(waveform, cplx=True/False)``.
        """
        return self.engine.stft

    @property
    def istft(self):
        """Return the iSTFT module from the underlying engine.

        The returned object is a ``ConviSTFT`` instance.
        """
        return self.engine.istft

    @property
    def device(self):
        """Return the device of the underlying engine."""
        return self.engine.device

    def process_stft_chunk(self, stft_chunk, n_spks=None):
        """Process a single STFT chunk through the model.

        Thin wrapper around ``engine.infer_chunk()``.  Takes a complex-valued
        STFT tensor and returns the structured result dict.

        Args:
            stft_chunk: Complex STFT tensor of shape ``(M, F, T)`` on the
                correct device (use ``self.device``).
            n_spks: Optional speaker count hint (``torch.Tensor``).

        Returns:
            dict with keys:
            - ``"stft_out"``: ``(N, M_o, F, T)`` — all output channels
              (chunk-level, before ref_ch extraction).
            - ``"vad"``, ``"doa"``: Optional auxiliary outputs.

        Note:
            **SS variant**: The SS model expects std-normalized waveforms.
            Apply normalization before STFT:
            ``waveform / (waveform.std(dim=-1, keepdim=True) + 1e-8)``
            then ``model.stft(waveform, cplx=True)``.
        """
        return self.engine.infer_chunk(stft_chunk, n_spks=n_spks)

    @torch.inference_mode()
    def process_stft(self, stft_input, n_spks=None):
        """Process a full STFT input (STFT in, STFT out), single-pass.

        Accepts and returns STFT-domain data.  Apply ``model.istft()`` to
        convert the output back to waveforms.  For long inputs requiring
        chunk-based processing, call ``process_stft`` on segments directly.

        Args:
            stft_input: Complex STFT tensor ``(M, F, T)``.  Automatically
                moved to ``self.device`` if not already there.
            n_spks: Optional speaker count hint (``torch.Tensor``).

        Returns:
            dict with keys:
              ``"stft_out"`` — ``(N, F, T)`` complex tensor (ref_ch extracted)
              ``"vad"``      — ``(N, T)`` or ``None``
              ``"doa"``      — ``(N, T)`` or ``None``

        Note:
            **SS variant**: This method bypasses std normalization.  Ensure
            your STFT input was computed from a std-normalized waveform:
            ``waveform / (waveform.std(dim=-1, keepdim=True) + 1e-8)``
        """
        mixture_stft = stft_input.to(self.device)

        if self._VARIANT.endswith("_SS"):
            logger.warning(
                "SS variant: process_stft bypasses std normalization. "
                "Ensure your STFT input was computed from a std-normalized waveform."
            )

        return self.engine._single_pass_session(mixture_stft, n_spks=n_spks)

    def process_waveform(self, waveform, n_spks=None):
        """Process a raw waveform tensor.

        Applies SS-specific std normalization internally before inference.
        This release does not support CSS chunking; for long-form audio,
        process in segments.

        Args:
            waveform: ``(M, L)`` or ``(L,)`` float tensor. Values should be
                normalised to ``[-1, 1]`` (raw PCM float).
            n_spks: Optional speaker count hint (``torch.Tensor``).

        Returns:
            dict with keys:
              ``"waveforms"`` — list of 1-D waveform tensors, one per speaker
              ``"vad"``       — ``None``
              ``"doa"``       — ``None``
        """
        # Dim handling
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # SS-specific std normalization
        waveform = waveform / (waveform.std(dim=-1, keepdim=True) + 1e-8)
        # STFT -> single-pass -> iSTFT
        mixture_stft = self.engine.stft(waveform.to(self.device), cplx=True)
        result = self.engine._single_pass_session(mixture_stft, n_spks=n_spks)
        stft_out = result["stft_out"]  # (N, F, T)
        waveforms = [
            self.engine.istft(stft_out[i], cplx=True, squeeze=True)
            for i in range(stft_out.shape[0])
        ]
        return {"waveforms": waveforms, "vad": None, "doa": None}

    def process_file(self, input_path, output_dir=None, n_spks=None):
        """Process a single audio file.

        Loads the file, selects microphone channels according to the config,
        runs inference, and optionally saves outputs using the variant's
        native ``save_outputs()`` function.  This release does not support
        CSS chunking; for long-form audio, process in segments.

        Args:
            input_path: Path to a ``.wav`` file (str or Path).
            output_dir: Directory for saving output files. When ``None``,
                inference results are returned but not written to disk.
            n_spks: Optional speaker count hint (``torch.Tensor``).

        Returns:
            dict with keys:
              ``"waveforms"`` — list of 1-D waveform tensors, one per speaker
              ``"vad"``       — ``None``
              ``"doa"``       — ``None``
        """
        input_path = Path(input_path)

        # Load audio — librosa returns (L,) for mono or (M, L) for multi-channel
        mixture, fs_in = librosa.load(str(input_path), mono=False, sr=None)
        mixture = torch.tensor(mixture)

        # Ensure channel dimension is present: (L,) -> (1, L)
        if mixture.dim() == 1:
            mixture = mixture.unsqueeze(0)

        # Select microphone channels if the recorded channel count mismatches
        if mixture.shape[0] != self._num_mics:
            mixture = mixture[self._mic_idx]

        # Warn when input sample rate differs from configured rate
        fs_config = self.engine.fs
        if fs_in != fs_config:
            logger.warning(
                f"Input sample rate {fs_in} Hz differs from config {fs_config} Hz. "
                "Results may be degraded."
            )

        result = self.process_waveform(mixture, n_spks=n_spks)

        if output_dir is not None:
            output_dir = str(output_dir)
            key = input_path.stem

            # Import variant-specific save_outputs and call with positional args
            # (not keyword args) to avoid dump_dir vs output_dir name mismatch
            # SS uses dump_dir for output.
            main_infer_module = importlib.import_module(
                f"{self._VARIANT}.main_infer"
            )
            main_infer_module.save_outputs(output_dir, key, result, fs_in)

        return result


class SSInference(_BaseInference):
    """Inference wrapper for the Speech Separation (SS) variant.

    Note: SS models expect n_spks to be provided unless the model was
    trained with ``is_var_spks=True`` / ``test_unknown_n_spks=True``.
    Pass ``n_spks=torch.tensor(2)`` to ``process_waveform()`` / ``process_file()``
    when the speaker count is known in advance.
    """

    _VARIANT = "sr_corrnet.models.SR_CorrNet_SS"
