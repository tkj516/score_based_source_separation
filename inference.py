import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from config import Config
from model import DiffWave


def batch_sampler(args):
    # Load the model from the model_dir
    if not os.path.exists(args.model_dir):
        raise ValueError(f"The path {args.model_dir} does not exist.")

    device = torch.device(args.device)

    if os.path.isdir(args.model_dir):
        checkpoint = torch.load(os.path.join(args.model_dir, "weights.pt"))
    else:
        checkpoint = torch.load(args.model_dir)

    cfg: Config = OmegaConf.create(checkpoint["cfg"])
    # Override certain arguments
    cfg.trainer.fast_sampling = args.fast_sampling

    model = DiffWave(cfg.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        training_noise_schedule = np.array(
            cfg.model.noise_schedule.train_noise_schedule)
        inference_noise_schedule = np.array(
            args.inference_noise_schedule) if cfg.trainer.fast_sampling else training_noise_schedule

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
            args.num_samples, 2, args.signal_length, device=device)
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
        sample = sample.cpu().numpy()

    # Generate figures
    num_infer_shifts, num_rows, num_cols = args.infer_shifts
    figs = []
    if "ofdm" in cfg.data.root_dir:
        for idx in range(sample.shape[0]):
            sample_c = sample[idx, 0, ...] + 1j * sample[idx, 1, ...]
            fig, axs = plt.subplots(
                nrows=num_rows, ncols=num_cols, figsize=(num_cols * 2, num_rows * 2))
            for j in range(num_infer_shifts):
                row = j // num_cols
                col = j % num_cols
                shifted = sample_c[j:j+400].reshape(-1, num_infer_shifts)[:, num_infer_shifts//5:]
                sym = np.fft.fft(shifted)
                sym = sym.flatten()
                axs[row, col].scatter(sym.real, sym.imag, marker='x')
                axs[row, col].axis('equal')
            plt.tight_layout()
            figs.append(fig)
    elif "comm_signal_2" in cfg.data.root_dir:
        fig, axs = plt.subplots(
            nrows=4, ncols=sample.shape[0], figsize=(sample.shape[0] * 4, 8))
        for idx in range(sample.shape[0]):
            sample = sample[idx, 0, ...] + 1j * sample[idx, 1, ...]
            axs[0].plot(np.real(sample).reshape(-1, ))
            axs[0].set_title("real")
            axs[1].plot(np.imag(sample).reshape(-1, ))
            axs[1].set_title("imag")
            axs[2].plot(np.real(sample).reshape(-1, )[500: 600])
            axs[2].set_title("real")
            axs[3].plot(np.imag(sample).reshape(-1, )[500: 600])
            axs[3].set_title("imag")
        plt.tight_layout()
        figs.append(fig)
    else:
        fig, axs = plt.subplots(
            nrows=1, ncols=sample.shape[0], figsize=(sample.shape[0] * 2, 2))
        for idx in range(sample.shape[0]):
            axs[idx].scatter(sample[idx, 0, ...], sample[idx, 1, ...], marker='x')
        plt.tight_layout()
        figs.append(fig)

    return figs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference using RF-DiffWave")
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory containing checkpoints or a full path to weights.pt file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate."
    )
    parser.add_argument(
        "--signal_length",
        type=int,
        help="Length of generated signal."
    )
    parser.add_argument(
        "--inference_noise_schedule",
        nargs="+",
        default=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
        help="Noise schedule to use with fast sampling."
    )
    parser.add_argument(
        "--fast_sampling",
        type=bool,
        default=False,
        help="If True, use fast sampling."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on."
    )
    parser.add_argument(
        "--infer_shifts",
        nargs="+",
        default=[20, 4, 5],
        help="Number of shifts during testing."
    )
    parser.add_argument(
        "--save_samples",
        type=bool,
        default=True,
        help="Save plots to disk",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save plots."
    )
    args = parser.parse_args()

    figs = batch_sampler(args)

    if args.save_samples:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        for i, fig in enumerate(figs):
            fig.savefig(os.path.join(args.save_dir, f"sample_{i}.png"))
