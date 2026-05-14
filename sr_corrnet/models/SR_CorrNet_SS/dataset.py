import os
import torch
import torch.nn as nn
import random
import librosa
import numpy as np

from sr_corrnet.utils import util_dataset
from sr_corrnet.utils.decorators import logger_wraps
from loguru import logger
from torch.utils.data import Dataset, DataLoader, ConcatDataset



@logger_wraps()
def get_multi_dataloaders(partitions, config):
    # create dataset object for each partition
    dataset_config = config["dataset"]
    loader_config = config["dataloader"]
    dataloaders = {}
    for partition in partitions:
        dynamic_mixing = dataset_config["dynamic_mixing"] if partition == 'train' else False
        dataset_sub = {}
        for sub_dir in dataset_config["subset_dir"]:
            d_dir = dataset_config[sub_dir][partition]
            scp_config_mix = os.path.join(dataset_config["scp_dir"], sub_dir, d_dir['mixture'])
            scp_config_spk = [os.path.join(dataset_config["scp_dir"], sub_dir, spk_key) for spk_key in d_dir['spk']]
            scp_config_noise = os.path.join(dataset_config["scp_dir"], sub_dir, d_dir['noise']) if 'noise' in d_dir else None
            dataset = MyDataset(
                dataset_config,
                partition = partition,
                wave_scp_srcs = scp_config_spk,
                wave_scp_srcs_reverb = None,
                wave_scp_mix = scp_config_mix,
                wave_scp_noise = scp_config_noise,
                dynamic_mixing = dynamic_mixing)
            dataset_sub[sub_dir] = dataset
        if partition == 'test':
            for sub_dir in dataset_config["subset_dir"]:
                dataloaders[partition+'_'+sub_dir] = DataLoader(
                    dataset = dataset_sub[sub_dir],
                    batch_size = 1,
                    shuffle = False,
                    pin_memory = loader_config["pin_memory"],
                    num_workers = loader_config["num_workers"],
                    drop_last = False,
                    collate_fn = _collate)
        else:
            dataloaders[partition] = DataLoader(
                dataset = ConcatDataset(dataset_sub.values()),
                batch_size = 1,
                shuffle = True,
                pin_memory = loader_config["pin_memory"],
                num_workers = loader_config["num_workers"],
                drop_last = loader_config["drop_last"],
                collate_fn = _collate)
    return dataloaders


@logger_wraps()
def get_dataloaders(partitions, config):
    # create dataset object for each partition
    dataset_config = config["dataset"]
    loader_config = config["dataloader"]
    dataloaders = {}
    for partition in partitions:
        scp_config_mix = os.path.join(dataset_config["scp_dir"], dataset_config[partition]['mixture'])
        scp_config_spk = [os.path.join(dataset_config["scp_dir"], spk_key) for spk_key in dataset_config[partition]['spk']]
        if getattr(dataset_config[partition], 'spk_reverb', False):
            scp_config_spk_reverb = [os.path.join(dataset_config["scp_dir"], spk_key) for spk_key in dataset_config[partition]['spk_reverb']]
        else:
            scp_config_spk_reverb = None
        scp_config_noise = os.path.join(dataset_config["scp_dir"], dataset_config[partition]['noise']) if 'noise' in dataset_config[partition] else None
        dynamic_mixing = dataset_config["dynamic_mixing"] if partition == 'train' else False
        dataset = MyDataset(
                dataset_config,
                partition = partition,
                wave_scp_srcs = scp_config_spk,
                wave_scp_srcs_reverb = scp_config_spk_reverb,
                wave_scp_mix = scp_config_mix,
                wave_scp_noise = scp_config_noise,
                dynamic_mixing = dynamic_mixing)
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = 1 if partition == 'test' else loader_config["batch_size"],
            shuffle = False  if partition == 'test' else loader_config["shuffle"], # only train: (partition == 'train') / all: True
            pin_memory = loader_config["pin_memory"],
            num_workers = loader_config["num_workers"],
            drop_last = loader_config["drop_last"],
            collate_fn = _collate)
        dataloaders[partition] = dataloader
    return dataloaders


@logger_wraps()
class MyDataset(Dataset):
    def __init__(self, config, partition, wave_scp_srcs, wave_scp_srcs_reverb, wave_scp_mix, wave_scp_noise, dynamic_mixing):
        self.partition = partition
        for wave_scp_src in wave_scp_srcs:
            if not os.path.exists(wave_scp_src): raise FileNotFoundError(f"Could not find file {wave_scp_src}")

        self.fs = config['synthesis_config']['sampling_rate']
        self.max_len = config['synthesis_config']['max_len'] if partition != "test" else 100*self.fs
        self.ref_ch = config['ref_ch']
        self.max_n_spks = config['max_n_spks']
        self.wave_dict_srcs = [util_dataset.parse_scps(wave_scp_src) for wave_scp_src in wave_scp_srcs]
        if wave_scp_srcs_reverb is not None:
            self.wave_dict_srcs_reverb = [util_dataset.parse_scps(wave_scp_src_reverb) for wave_scp_src_reverb in wave_scp_srcs_reverb]
        else:
            self.wave_dict_srcs_reverb = None
        self.wave_dict_mix = util_dataset.parse_scps(wave_scp_mix)
        self.wave_dict_noise = util_dataset.parse_scps(wave_scp_noise) if wave_scp_noise else None
        self.wave_keys = list(self.wave_dict_mix.keys())
        logger.info(f"Create MyDataset for {wave_scp_mix} with {len(self.wave_dict_mix)} utterances")
        self.dynamic_mixing = dynamic_mixing
    
    def __len__(self):
        return len(self.wave_dict_mix)
    
    def __contains__(self, key):
        return key in self.wave_dict_mix

    def _dynamic_mixing(self, key):
        def __match_length(wav, len_data): 
            leftover = len(wav) - len_data
            idx = random.randint(0,leftover)
            wav = wav[idx:idx+len_data]
            return wav
        
        samps_src, samps_src_reverb = [], []
        src_len = [self.max_len]
        # dyanmic source choice        
        # checking whether it is the same speaker
        while True:
            key_random = random.choice(list(self.wave_dict_srcs[0].keys()))
            tmp1 = key.split('_')[1][:3] != key_random.split('_')[3][:3]
            tmp2 = key.split('_')[3][:3] != key_random.split('_')[1][:3]
            if tmp1 and tmp2: break
        idx1, idx2 = (0, 1) if random.random() > 0.5 else (1, 0)
        files = [self.wave_dict_srcs[idx1][key], self.wave_dict_srcs[idx2][key_random]]

        n_spks = len(files)
        pres_list = [0] + n_spks*[1] + (self.max_n_spks-n_spks+1)*[0]

        # load
        for idx, file in enumerate(files):
            if not os.path.exists(file):
                raise FileNotFoundError("Input file {} do not exists!".format(file))
            samps_tmp, _ = librosa.load(file, sr=self.fs, mono=False)
            if len(samps_tmp.shape) == 2:
                samps_tmp = samps_tmp[self.ref_ch]

            gain = pow(10,-random.uniform(-3,3)/20)
            # Speed Augmentation
            samps_src.append(gain*samps_tmp)
            
            src_len.append(len(samps_tmp))

        # truncate
        min_len = min(src_len)
        samps_src = [__match_length(s, min_len) for s in samps_src]
        samps_mix = sum(samps_src) 
        
        samps_mix = np.transpose(samps_mix) if len(samps_mix.shape) == 2 else samps_mix

        return samps_mix, samps_src, n_spks, pres_list
    

    def _direct_load(self, key):
        samps_src = []
        files = [src[key] for src in self.wave_dict_srcs]
        n_spks = len(files)
        pres_list = [0] + n_spks*[1] + (self.max_n_spks-n_spks+1)*[0]
        # files = [wave_dict_src[key] for wave_dict_src in self.wave_dict_srcs]
        for file in files:
            if not os.path.exists(file): raise FileNotFoundError(f"Input file {file} do not exists!")
            samps_tmp, _ = librosa.load(file, sr=self.fs, mono=False)
            if len(samps_tmp.shape) == 2:
                samps_tmp = samps_tmp[self.ref_ch]
            samps_src.append(samps_tmp)
        
        file = self.wave_dict_mix[key]    
        if not os.path.exists(file): raise FileNotFoundError(f"Input file {file} do not exists!")
        samps_mix, _ = librosa.load(file, sr=self.fs, mono=False)
        samps_mix = np.transpose(samps_mix) if len(samps_mix.shape) == 2 else samps_mix
        if self.partition != "test":
            if len(samps_mix) > self.max_len:
                start = random.randint(0,len(samps_mix)-self.max_len)
                samps_mix = samps_mix[start:start+self.max_len]
                samps_src = [s[start:start+self.max_len] for s in samps_src]
        
        return samps_mix, samps_src, n_spks, pres_list
    
    def __getitem__(self, index):
        key = self.wave_keys[index]
        if any(key not in self.wave_dict_srcs[i] for i in range(len(self.wave_dict_srcs)-2)) or key not in self.wave_dict_mix: raise KeyError(f"Could not find utterance {key}")
        if self.dynamic_mixing:
            samps_mix, samps_src, n_spks, pres_list = self._dynamic_mixing(key)
        else:
            samps_mix, samps_src, n_spks, pres_list = self._direct_load(key)

        return {"num_sample" : samps_mix.shape[0], "mix": samps_mix, "src": samps_src, "n_spks":n_spks, "pres":pres_list, "key":key}
    
    
def _collate(egs):

    def __prepare_target_rir(dict_lsit, index):
        return nn.utils.rnn.pad_sequence([torch.tensor(d["src"][index], dtype=torch.float32)  for d in dict_lsit], batch_first=True)

    if type(egs) is not list: raise ValueError("Unsupported index type({})".format(type(egs)))
    dict_list = sorted([eg for eg in egs], key=lambda x: x['num_sample'], reverse=True)

    key = [d['key'] for d in dict_list]
    n_spks = torch.tensor([d['n_spks'] for d in dict_list], dtype=torch.int32)
        
    mixture = nn.utils.rnn.pad_sequence([torch.tensor(d['mix'], dtype=torch.float32) for d in dict_list], batch_first=True)
    src = [__prepare_target_rir(dict_list, index) for index in range(n_spks[0])]
    pre_list = torch.tensor([d['pres'] for d in dict_list], dtype=torch.float32)
    return {'key': key, 'mixture': mixture, 'target': src, 'n_spks': n_spks, 'pres_label': pre_list}

