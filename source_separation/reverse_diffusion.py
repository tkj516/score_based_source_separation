import functools
import sys

from config import Config
sys.path.append("..")

import random
import numpy as np
import tensorflow as tf

import rfcutils2.qpsk_helper_fn as qpskfn

# This must be used before loading PyTorch
tf.config.set_visible_devices([], "GPU")
import torch
import torch.nn.functional as F
from typing import Any, Tuple

from utils import diffusion_denoising, view_as_complex
from analytical_qpsk_score_model import matchedfilter_remod, qpsk_score_model

random.seed(912345)
np.random.seed(912345)
torch.manual_seed(912345)

device = torch.device("cuda:0")


def reverse_diffusion_separation(
        mixture: torch.Tensor, 
        coeff: float, 
        qpsk: torch.Tensor, 
        interference: torch.Tensor,
        model_interference: torch.nn.Module,
        cfg_interference: Config,
        dataset_stats: Tuple[torch.Tensor, torch.Tensor],
        device: Any,
        **kwargs,
):
    """Perform source separation using reverse diffusion on the interference.

    Args:
        mixture: Inputs mixture (y = s + \kappa * b)
        coeff: \kappa
        qpsk: The SOI (s)
        interference: The interference (b)
        model_qpsk: The qpsk diffusion model
        model_interference: The interference diffusion model
        cfg_interference: The config for the interference model
        dataset_stats: Dataset mean and standard deviation if interference is 
            not zero mean and unit variance.
    """

    del kwargs
    # Set up the reverse diffusion chain
    interference_reverse_diffusion = functools.partial(
        diffusion_denoising, 
        model=model_interference, 
        cfg=cfg_interference, 
        inference_noise_schedule=np.linspace(1e-4, 1e-3, 10),
    )

    y_std = (mixture.to(device) - dataset_stats[0]) / dataset_stats[1]
    interference_est = (
        dataset_stats[1].cpu() * interference_reverse_diffusion(input_sample=y_std.to(device)) \
            + dataset_stats[0].cpu()
    )
    s_est = coeff * (mixture - interference_est)

    s_hat = s_est.to(device)
    b_hat = interference_est.to(device)
    bit_true, _ = qpskfn.qpsk_matched_filter_demod(view_as_complex(qpsk))
    bit_pred, _ = qpskfn.qpsk_matched_filter_demod(view_as_complex(s_hat.detach().cpu()))
    bit_error = np.mean(bit_true != bit_pred)
    print(
        f"MSE(s_hat-s): {F.mse_loss(qpsk, s_hat.detach().cpu()):.4f},"
        f" MSE(b_hat-b): {F.mse_loss(interference, b_hat.detach().cpu()):.6f},"
        f" BER: {bit_error:.4f}"
    )
    return s_est