# Adapted from https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/learner.py

import os
from dataclasses import asdict
import traceback
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import RFDatasetBase, get_train_val_dataset
from model import DiffWave


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


class DiffWaveLearner:
    def __init__(self, cfg: Config, model: nn.Module, rank: int):
        self.cfg = cfg

        # Store some import variables
        self.model_dir = cfg.model_dir
        self.distributed = cfg.distributed.distributed
        self.world_size = cfg.distributed.world_size
        self.rank = rank
        self.log_every = cfg.trainer.log_every
        self.validate_every = cfg.trainer.validate_every
        self.save_every = cfg.trainer.save_every
        self.infer_every = cfg.trainer.infer_every
        self.fast_sampling = cfg.trainer.fast_sampling
        self.max_steps = cfg.trainer.max_steps
        self.build_dataloaders()

        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.trainer.learning_rate)
        self.autocast = torch.cuda.amp.autocast(enabled=cfg.trainer.fp16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.trainer.fp16)
        self.step = 0

        self.train_noise_schedule = np.array(
            cfg.model.noise_schedule.train_noise_schedule)  # \beta
        self.inference_noise_schedule = np.array(
            cfg.model.noise_schedule.inference_noise_schedule)
        self.noise_level = torch.tensor(np.cumprod(
            1 - self.train_noise_schedule).astype(np.float32))  # \alpha
        self.loss_fn = nn.MSELoss()
        self.writer = SummaryWriter(self.model_dir)

    @property
    def is_master(self):
        return self.rank == 0

    def build_dataloaders(self):
        self.dataset = RFDatasetBase(
            root_dir=self.cfg.data.root_dir,
            target_len=self.cfg.data.target_len,
            augmentation=self.cfg.data.augmentation,
            mean=self.cfg.data.mean,
            std=self.cfg.data.std,
        )
        self.train_dataset, self.val_dataset = get_train_val_dataset(
            self.dataset, self.cfg.data.train_fraction)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=not self.distributed,
            num_workers=self.cfg.data.num_workers if self.distributed else 0,
            sampler=DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank) if self.distributed else None,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=not self.distributed,
            num_workers=self.cfg.data.num_workers if self.distributed else 0,
            pin_memory=True,
        )

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()},
            'cfg': asdict(self.cfg),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)

        if os.path.islink(link_name):
            os.unlink(link_name)
        os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self):
        device = next(self.model.parameters()).device

        while True:
            for i, features in enumerate(tqdm(self.train_dataloader, desc=f"Training ({self.step} / {self.max_steps})")):
                features = _nested_map(features, lambda x: x.to(
                    device) if isinstance(x, torch.Tensor) else x)
                loss = self.train_step(features)

                # Check for NaNs
                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f'Detected NaN loss at step {self.step}.')

                if self.is_master:
                    if self.step % self.log_every == 0:
                        self.writer.add_scalar('train/loss', loss, self.step)
                        self.writer.add_scalar(
                            'train/grad_norm', self.grad_norm, self.step)
                    if self.step % self.validate_every == 0:
                        self.validate()
                    if self.step % self.infer_every == 0:
                        self.infer_and_log_samples()
                    if self.step % self.save_every == 0:
                        self.save_to_checkpoint()

                if self.distributed:
                    dist.barrier()

                self.step += 1

                if self.step == self.max_steps:
                    if self.is_master and self.distributed:
                        self.save_to_checkpoint()
                        print("Ending training...")
                    dist.barrier()
                    exit(0)

    def train_step(self, features: Dict[str, torch.Tensor]):
        for param in self.model.parameters():
            param.grad = None

        sample = features["sample"]

        N, _, _ = sample.shape
        device = sample.device
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            t = torch.randint(0, len(self.train_noise_schedule), [
                              N], device=sample.device)
            noise_scale = self.noise_level[t].reshape(-1, 1, 1)
            noise_scale_sqrt = noise_scale ** 0.5
            noise = torch.randn_like(sample)
            noisy_audio = noise_scale_sqrt * sample + \
                (1.0 - noise_scale) ** 0.5 * noise

            predicted = self.model(noisy_audio, t)
            loss = self.loss_fn(noise, predicted.squeeze(1))

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.trainer.max_grad_norm or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    @torch.no_grad()
    def validate(self):
        device = next(self.model.parameters()).device
        self.model.eval()

        loss = 0
        for features in tqdm(self.val_dataloader, desc=f"Running validation after step {self.step}"):
            features = _nested_map(features, lambda x: x.to(
                device) if isinstance(x, torch.Tensor) else x)
            sample = features["sample"]
            N, _, _ = sample.shape
            device = sample.device
            self.noise_level = self.noise_level.to(device)

            with self.autocast:
                t = torch.randint(0, len(self.cfg.model.noise_schedule.train_noise_schedule), [
                    N], device=sample.device)
                noise_scale = self.noise_level[t].reshape(-1, 1, 1)
                noise_scale_sqrt = noise_scale ** 0.5
                noise = torch.randn_like(sample)
                noisy_audio = noise_scale_sqrt * sample + \
                    (1.0 - noise_scale) ** 0.5 * noise

                # Use the underlying module to get the losses
                if self.distributed:
                    predicted = self.model.module(noisy_audio, t)
                else:
                    predicted = self.model(noisy_audio, t)
                loss += self.loss_fn(noise, predicted.squeeze(1)
                                     ) * sample.shape[0]
        loss = loss / len(self.val_dataset)

        self.writer.add_scalar('val/loss', loss, self.step)
        self.model.train()

        return loss

    @torch.no_grad()
    def infer(self):
        device = next(self.model.parameters()).device
        self.model.eval()

        training_noise_schedule = self.train_noise_schedule
        inference_noise_schedule = self.inference_noise_schedule if self.fast_sampling else training_noise_schedule

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

        sample = torch.randn(
            1, 2, self.cfg.trainer.infer_target_len, device=device)
        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
            if self.distributed:
                sample = c1 * \
                    (sample - c2 * self.model.module(sample,
                                                     torch.tensor([T[n]], device=sample.device)))
            else:
                sample = c1 * \
                    (sample - c2 * self.model(sample,
                                              torch.tensor([T[n]], device=sample.device)))
            if n > 0:
                sigma = ((1.0 - alpha_cum[n-1]) /
                         (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                sample += sigma * torch.randn_like(sample)

        self.model.train()

        return sample

    def infer_and_log_samples(self):

        num_infer_samples = self.cfg.trainer.num_infer_samples
        num_infer_shifts, num_rows, num_cols = self.cfg.trainer.infer_shifts

        if "ofdm" in self.cfg.data.root_dir:
            for i in range(num_infer_samples):
                sample = self.infer().detach().cpu().squeeze(0).numpy()
                sample = sample[0, ...] + 1j * sample[1, ...]
                fig, axs = plt.subplots(
                    nrows=num_rows, ncols=num_cols, figsize=(num_cols * 2, num_rows * 2))
                for j in range(num_infer_shifts):
                    row = j // num_cols
                    col = j % num_cols
                    shifted = sample[j:j+400].reshape(-1, num_infer_shifts)[
                        :, num_infer_shifts//5:]
                    sym = np.fft.fft(shifted)
                    sym = sym.flatten()
                    axs[row, col].scatter(sym.real, sym.imag, marker='x')
                    axs[row, col].axis('equal')
                plt.tight_layout()
                self.writer.add_figure(f'infer/image_{i}', fig, self.step)
        elif "comm_signal_2" in self.cfg.data.root_dir:
            fig1, axs1 = plt.subplots(
                nrows=2, ncols=num_infer_samples, figsize=(num_infer_samples * 4, 4))
            fig2, axs2 = plt.subplots(
                nrows=2, ncols=num_infer_samples, figsize=(num_infer_samples * 4, 4))
            for i in range(num_infer_samples):
                sample = self.infer().detach().cpu().squeeze(0).numpy()
                sample = sample[0, ...] + 1j * sample[1, ...]
                axs1[0, i].plot(np.real(sample).reshape(-1, ))
                axs1[0, i].set_title("real")
                axs1[1, i].plot(np.imag(sample).reshape(-1, ))
                axs1[1, i].set_title("imag")
                axs2[0, i].plot(np.real(sample).reshape(-1, )[500: 600])
                axs2[0, i].set_title("real")
                axs2[1, i].plot(np.imag(sample).reshape(-1, )[500: 600])
                axs2[1, i].set_title("imag")
            plt.tight_layout()
            self.writer.add_figure('infer/images', fig1, self.step)
            self.writer.add_figure('infer/images_zoomed', fig2, self.step)
        else:
            fig, axs = plt.subplots(
                nrows=1, ncols=num_infer_samples, figsize=(num_infer_samples * 2, 2))
            for i in range(num_infer_samples):
                sample = self.infer().detach().cpu().squeeze(0).numpy()
                axs[i].scatter(sample[0, ...], sample[1, ...], marker='x')
            plt.tight_layout()
            self.writer.add_figure('infer/images', fig, self.step)


def _train_impl(rank: int, model: nn.Module, cfg: Config):
    torch.backends.cudnn.benchmark = True

    learner = DiffWaveLearner(cfg, model, rank)
    learner.restore_from_checkpoint()
    learner.train()


def train(cfg: Config):
    """Training on a single GPU."""
    model = DiffWave(cfg.model).cuda()
    _train_impl(0, model, cfg)


def init_distributed(rank: int, world_size: int, port: str):
    """Initialize distributed training on multiple GPUs."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group(
        'nccl', rank=rank, world_size=world_size)


def train_distributed(rank: int, world_size: int, port, cfg: Config):
    """Training on multiple GPUs."""
    init_distributed(rank, world_size, port)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    model = DiffWave(cfg.model).to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    _train_impl(rank, model, cfg)
