"""SS (Speech Separation) inference — separate fixed number of speakers.

Two ways to load the model:
    # Option A: from HF Hub (downloads model.pt + config.yaml automatically)
    model = SSInference.from_pretrained("shinuh/sr-corrnet-ss-1ch-wsj-fix-2spk", device=device)

    # Option B: from local config name (checkpoint must exist in sr_corrnet/checkpoints/SS/)
    model = SSInference.from_pretrained(config="1ch_WSJ_fix_2spk", device=device)
"""
import argparse
import torch
from sr_corrnet import SSInference

SAMPLING_RATE = 8000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/path/to/input.wav",
                        help="Path to a mono 8 kHz wav (e.g. WSJ0-2mix mixture)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--n_spks", type=int, default=2, help="0 = auto")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Load from HF Hub — config.yaml is downloaded automatically alongside model.pt
    model = SSInference.from_pretrained("shinuh/sr-corrnet-ss-1ch-wsj-fix-2spk", device=device)
    n_spks = torch.tensor(args.n_spks) if args.n_spks > 0 else None
    result = model.process_file(args.input, output_dir=args.output, n_spks=n_spks)

    for i, wav in enumerate(result["waveforms"]):
        print(f"Speaker {i}: {wav.shape[0]} samples ({wav.shape[0]/SAMPLING_RATE:.2f}s)")

if __name__ == "__main__":
    main()
