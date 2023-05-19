import sys
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

from utils import view_as_complex
from analytical_qpsk_score_model import matchedfilter_remod, qpsk_score_model

random.seed(912345)
np.random.seed(912345)
torch.manual_seed(912345)

device = torch.device("cuda:0")


def modified_basis_separation(
        mixture: torch.Tensor, 
        coeff: float, 
        num_iters: int,
        use_trained_model: bool,
        qpsk: torch.Tensor, 
        interference: torch.Tensor,
        model_qpsk: torch.nn.Module,
        model_interference: torch.nn.Module,
        dataset_stats: Tuple[torch.Tensor, torch.Tensor],
        use_alpha_posterior: bool,
        device: Any,
        **kwargs,
):
    """Perform source separation using modified BASIS.

    Augments the BASIS algorithm using the hard constraint y = s + \kappa b
    instead of using y = s + \kappa b + w.  The latter requires modeling the 
    likelihood as a spherical Gaussian and hence Langevin dyanmics is required 
    to updated both the estimate of s and the estimate of b at different
    noise levels.  We still use an annealing noise schedule but replace the 
    Gaussian likelihood with the hard constraint b = (y - s) / \kappa according
    to the MAP rule in our paper.  Thus, only s needs to be updated using
    Langevin dynamics directly.  Additional provisions are added to use the
    alpha posterior weighting rule (\omega=\kappa^2).

    Args:
        mixture: Inputs mixture (y = s + \kappa * b)
        coeff: \kappa
        num_iters: Number of iterations to run Langevin dynamics
        use_trained_model: If True, use the learned diffusion model to compute
            the SOI score
        qpsk: The SOI (s)
        interference: The interference (b)
        model_qpsk: The qpsk diffusion model
        model_interference: The interference diffusion model
        use_alpha_posterior: If True, use the \\alpha-posterior weighting
            to scale the interference score
        dataset_stats: Dataset mean and standard deviation if interference is 
            not zero mean and unit variance.
    """

    del kwargs
    # Same noise schedule as in our diffusion models. See learner.py
    noise_schedule = np.linspace(1e-4, 0.05, 50)
    noise_level = torch.tensor(
        np.cumprod(1 - noise_schedule).astype(np.float32)
    ).to(device)

    # Compute the matched filtering solution given the mixture
    s_mf = matchedfilter_remod(view_as_complex(coeff * mixture))
    s_est = torch.view_as_real(torch.tensor(s_mf.numpy())).transpose(1, 2).float()
    
    s_est = s_est
    b_est = mixture.to(device) - 1 / coeff * s_est.to(device)
    
    for n in range(len(noise_level) - 1, -1, -1):
        n1, n2 = max(n,0), max(n,0)
        t1 = torch.tensor([n1]).to(device)
        t2 = torch.tensor([n2]).to(device)
        noise_scale1 = noise_level[n1]
        noise_scale2 = noise_level[n2]
        eta1 = (2e-6 * (1-noise_scale1) / (1-noise_level[0])).cpu()
        sigma1 = (1 - noise_scale1.detach().cpu().numpy()) ** 0.5 
        sigma2 = (1 - noise_scale2.detach().cpu().numpy()) ** 0.5 
        
        for _ in range(num_iters):    
            if use_trained_model:
                score_t1 = model_qpsk(s_est.to(device), t1)
                score_t1 = -score_t1.detach().cpu() / (sigma1)
            else:
                score_t1 = qpsk_score_model(s_est, noise_scale1).to(device)
                score_t1 = score_t1.detach().cpu()

            b_est_std = (
                b_est.to(device) - dataset_stats[0].to(device)
                ) / dataset_stats[1].to(device)
            score_t2 = model_interference(b_est_std.to(device), t2)
            score_t2 *= 1 / dataset_stats[1].to(device)
            score_t2 = -score_t2.detach().cpu() / (sigma2)
    
            noise1 = torch.randn_like(s_est)
            if use_alpha_posterior:
                g = coeff * score_t2 - score_t1
            else:
                g = score_t2 - coeff * score_t1
            s_est = s_est - eta1 * g + np.sqrt(eta1) * noise1
            b_est = mixture - 1 / coeff * s_est
            
        s_hat = s_est.to(device)
        b_hat = b_est.to(device)
        bit_true, _ = qpskfn.qpsk_matched_filter_demod(view_as_complex(qpsk))
        bit_pred, _ = qpskfn.qpsk_matched_filter_demod(view_as_complex(s_hat.detach().cpu()))
        bit_error = np.mean(bit_true != bit_pred)
        print(
            f"{n:>4}: - MSE(s_hat-s): {F.mse_loss(qpsk, s_hat.detach().cpu()):.4f},"
            f" MSE(b_hat-b): {F.mse_loss(interference, b_hat.detach().cpu()):.6f},"
            f" BER: {bit_error:.4f}"
        )
    return s_est