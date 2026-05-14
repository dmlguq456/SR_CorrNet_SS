"""SS inference engine — tensor-in/tensor-out API for Speech Separation.

Library usage example::

    from sr_corrnet.models.SR_CorrNet_SS.engine_infer import EngineInfer
    from sr_corrnet.models.SR_CorrNet_SS.model import Model
    from sr_corrnet.utils import util_system, util_engine

    config = util_system.parse_yaml("path/to/config.yaml")["config"]
    model = Model(**config["model"])
    util_engine.load_last_checkpoint_n_get_epoch("path/to/weights/", model, location="cuda:0")

    device = torch.device("cuda:0")
    engine = EngineInfer(config, model.to(device), device)

    # STFT -> infer_chunk -> iSTFT
    stft_chunk = engine.stft(waveform.to(device), cplx=True)  # (M, F, T)
    result = engine.infer_chunk(stft_chunk, n_spks=torch.tensor(2))
    # result["stft_out"] -> (N, M_o, F, T) complex per-speaker STFT
"""
import torch

from sr_corrnet.utils import util_stft


class EngineInfer:
    """Inference engine for SS (Speech Separation) variant.

    Provides a clean tensor-in / tensor-out API separated from I/O concerns.
    Entry point: infer_chunk — single STFT chunk → model forward → structured dict.

    Standard-normalization is applied at the SSInference layer (inference.py).
    Model returns (out_list, out_aux, pres) — out_list is list of N tensors.
    n_spks is forwarded to the model for variable-speaker-count support.
    No VAD/DOA output.
    """

    def __init__(self, config, model, device):
        """Lightweight init — no optimizer, no TensorBoard, no MAC computation.

        Args:
            config: dict — full YAML config (needs 'stft', 'model', 'dataset',
                    'max_n_spks', and 'inference' keys)
            model: nn.Module — already loaded with weights
            device: torch.device
        """
        self.model = model
        self.model.eval()
        self.device = device

        self.ref_ch = config["model"].get("ref_ch", 0)
        self.max_n_spks = config["max_n_spks"]
        self.test_unknown_n_spks = config.get("test_unknown_n_spks", False)
        self.fs = config["dataset"]["synthesis_config"]["sampling_rate"]

        self.stft = util_stft.STFT(**config["stft"], device=device, normalize=True)
        self.istft = util_stft.iSTFT(**config["stft"], device=device, normalize=True)

    # ── Layer 1: Chunk-level ──────────────────────────────────────────────────

    @torch.inference_mode()
    def infer_chunk(self, stft_chunk, n_spks=None):
        """Process a single STFT chunk through the SS model.

        Args:
            stft_chunk: (M, F, T) complex STFT tensor — already on self.device
            n_spks: speaker count tensor (from dataloader batch) or None.
                    Forwarded to model; None means model infers speaker count.

        Returns:
            dict with:
              'stft_out': (N, M_o, F, T) complex tensor — N speakers
              'vad':      None — SS model has no VAD estimator
              'doa':      None — SS model has no DOA estimator
              'pres':     dict with 'probs'/'logits' etc, or None
        """
        # Real/imag concatenation along channel dim: (M, F, T) → (2M, F, T)
        model_input = torch.cat(
            [torch.real(stft_chunk), torch.imag(stft_chunk)], dim=0
        )

        # Model adds batch dim internally when input is 3-D (model.py line 55-56)
        # out: list of N tensors, each (B=1, M_o, F, T, 2) — real/imag at last dim
        out, out_aux, pres = self.model(model_input, n_spks=n_spks)

        # Handle variable-speaker case where model returns empty list
        if len(out) == 0:
            # Return empty tensor with correct shape for downstream handling
            empty = torch.zeros(
                0, 1, stft_chunk.shape[-2], stft_chunk.shape[-1],
                dtype=torch.complex64, device=self.device
            )
            return {"stft_out": empty, "vad": None, "doa": None, "pres": pres}

        # Convert each speaker's output from real/imag to complex: (B=1, M_o, F, T)
        estim_stft = [
            torch.complex(e[..., 0], e[..., 1]) for e in out
        ]
        # Stack along speaker dimension: list of (1, M_o, F, T) → (N, M_o, F, T)
        estim_stft = torch.cat(estim_stft, dim=0)

        return {"stft_out": estim_stft, "vad": None, "doa": None, "pres": pres}

    def _single_pass_session(self, mixture_stft, n_spks=None):
        """Single forward pass: forwards the entire STFT through the model and
        extracts ref_ch from the per-speaker output. Used by
        ``process_waveform``/``process_stft`` for finite-mixture separation.

        Args:
            mixture_stft: (M, F, T) complex STFT — already on self.device
            n_spks: speaker count tensor or None — forwarded to model

        Returns:
            dict with:
              'stft_out': (N, F, T) complex STFT — ref_ch extracted, N speakers
              'vad': None
              'doa': None
        """
        chunk_result = self.infer_chunk(mixture_stft, n_spks=n_spks)
        stft_out = chunk_result["stft_out"]  # (N, M_o, F, T)
        # Select ref_ch: (N, M_o, F, T) → (N, F, T)
        stft_out = stft_out[:, self.ref_ch, :, :]
        return {"stft_out": stft_out, "vad": None, "doa": None}
