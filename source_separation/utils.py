import os
from typing import Any, Optional
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from model import DiffWave
from config import Config


def get_pow(s): 
    return np.mean(np.abs(s)**2)


def view_as_complex(x: torch.Tensor): 
    return torch.view_as_complex(x.transpose(2, 1).contiguous()).numpy()


def random_shift(data: torch.Tensor, target_len: int):
    rand_start_idx = np.random.randint(len(data) - target_len, size=(1,))
    idxs = rand_start_idx + np.arange(0, target_len)
    return np.take_along_axis(data, idxs, axis=-1), rand_start_idx


def random_phase(data: np.ndarray):
    rand_phase = np.random.rand()
    return data * np.exp(1j * 2 * np.pi * (rand_phase + 0j)), rand_phase


class SeparationDataset(Dataset):
    def __init__(
        self,
        soi_type: str, 
        interference_type: str, 
        target_len: Optional[int] = None,
    ):
        super().__init__()
        self.qpsk_root_dir = f"../dataset/separation/{soi_type}"
        self.interference_root_dir = f"../dataset/separation/{interference_type}"
        self.data1 = [s for s in sorted(os.listdir(
            self.qpsk_root_dir)) if s.endswith(".npy") and s != "cov.npy"]
        self.data2 = [s for s in sorted(os.listdir(
            self.interference_root_dir)) if s.endswith(".npy") and s != "cov.npy"]
        self.interference_type = interference_type
        self.target_len = target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        np.random.seed(31415 + idx)
        idx1 = np.random.randint(1000)
        idx2 = np.random.randint(1000 if "ofdm" in self.interference_type else 50) 

        qpsk_sample = np.load(os.path.join(self.qpsk_root_dir, self.data1[idx1]))
        interference_sample = np.load(os.path.join(self.interference_root_dir, self.data2[idx2]))
        interference_sample, shift = random_shift(interference_sample, self.target_len)
        interference_sample, phase = random_phase(interference_sample)

        return {
            "qpsk_sample": torch.view_as_real(torch.tensor(qpsk_sample)).transpose(0, 1).float(),
            "interference_sample": torch.view_as_real(torch.tensor(interference_sample)).transpose(0, 1).float(),
            "shift": torch.tensor(shift).float(),
            "phase": torch.tensor(phase).float(),
        }


def load_separation_dataset(
        soi_type: str, interference_type: str, signal_length: int):
    dataset = SeparationDataset(soi_type, interference_type, signal_length)
    return dataset


def load_models(
        soi_type: str, 
        interference_type: str, 
        device: Any,
):
    assert soi_type == "qpsk"  # programmer error
    qpsk_checkpoint_dir = "../checkpoints/updated/qpsk_100000_160_20_03_09/weights-360000.pt"
    if interference_type == "ofdm_qpsk":
        interference_checkpoint_dir = "../checkpoints/updated/qam_ofdm_500000_37_09_29_55/weights-340000.pt"
    elif interference_type == "ofdm_bpsk":
        interference_checkpoint_dir = "../checkpoints/updated/pam_ofdm_200000_37_19_30_17/weights-220000.pt"
    elif interference_type == "commsignal2":
        interference_checkpoint_dir = "../checkpoints/updated/comm_signal_2_trimmed_16_53_25/weights-14000.pt"
            
    checkpoint = torch.load(qpsk_checkpoint_dir, map_location="cpu")
    cfg_qpsk = OmegaConf.create(checkpoint["cfg"])
    model_qpsk = DiffWave(cfg_qpsk.model).to(device)
    model_qpsk.load_state_dict(checkpoint["model"])
    model_qpsk.eval()

    checkpoint = torch.load(interference_checkpoint_dir, map_location="cpu")
    cfg_interference = OmegaConf.create(checkpoint["cfg"])
    model_ofdm = DiffWave(cfg_interference.model).to(device)
    model_ofdm.load_state_dict(checkpoint["model"])
    model_ofdm.eval()

    # Set fast sampling parameters and inference noise schedule
    cfg_qpsk.trainer.fast_sampling = True
    cfg_interference.trainer.fast_sampling = True
    return model_qpsk, cfg_qpsk, model_ofdm, cfg_interference 


def diffusion_denoising(
        model: torch.nn.Module, 
        cfg: Config, 
        input_sample: torch.Tensor, 
        inference_noise_schedule: Optional[np.ndarray] = None,
):
    if cfg.trainer.fast_sampling:
        assert inference_noise_schedule is not None
    if inference_noise_schedule is not None:
        assert cfg.trainer.fast_sampling

    with torch.no_grad():
        training_noise_schedule = np.array(
            cfg.model.noise_schedule.train_noise_schedule)
        inference_noise_schedule = np.array(
            inference_noise_schedule) if cfg.trainer.fast_sampling else training_noise_schedule

        train_alpha = 1 - training_noise_schedule
        train_alpha_cum = np.cumprod(train_alpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if train_alpha_cum[t + 1] <= alpha_cum[s] <= train_alpha_cum[t]:
                    delta = (train_alpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                        train_alpha_cum[t] ** 0.5 - train_alpha_cum[t+1] ** 0.5)
                    T.append(t + delta)
                    break
        T = np.array(T, dtype=np.float32)

        sample = input_sample
        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
            sample = c1 * \
                (sample - c2 * model(sample,
                                     torch.tensor([T[n]], device=sample.device)))
            if n > 0:
                sigma = ((1.0 - alpha_cum[n-1]) /
                         (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                sample += sigma * torch.randn_like(sample)
        sample = sample.cpu()

    return sample