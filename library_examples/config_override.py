"""Config override patterns for SSInference.from_pretrained kwargs."""
import torch
from sr_corrnet import SSInference
from sr_corrnet.utils import util_system

DEVICE = "cuda:0"
CONFIG = "1ch_WSJ_fix_2spk.yaml"

waveform = torch.randn(1, 16000)  # replace with real 8 kHz audio
n_spks = torch.tensor(2)

# --- 1. Process a waveform with explicit speaker count ---
model = SSInference.from_pretrained(CONFIG, device=DEVICE)
result = model.process_waveform(waveform, n_spks=n_spks)
print(f"Separated {len(result['waveforms'])} speakers")

# --- 2. Dict-based config loading with field override ---
from sr_corrnet._config import resolve_config
yaml_dict = util_system.parse_yaml(resolve_config("SS", "1ch_WSJ_fix_2spk.yaml"))
yaml_dict["config"]["max_n_spks"] = 2
model = SSInference.from_pretrained(config=yaml_dict, device=DEVICE)

# --- 3. Process a file and save separated outputs ---
# result = model.process_file("input.wav", output_dir="output/")

