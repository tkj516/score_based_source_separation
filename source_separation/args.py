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
from analytical_qpsk_score_model import (
    matchedfilter_remod, get_rrc_matrix, qpsk_score_model
)

random.seed(912345)
np.random.seed(912345)
torch.manual_seed(912345)

# Min and max timesteps from pre-trained diffusion model 
# noise schedule
MIN_STEP, MAX_STEP = 1, 49


def args_separation(
        mixture: torch.Tensor, 
        coeff: float, 
        scaling: float, 
        num_iters: int, 
        use_trained_model: bool, 
        qpsk: torch.Tensor, 
        interference: torch.Tensor,
        model_qpsk: torch.nn.Module,
        model_interference: torch.nn.Module,
        learning_rate_range: Tuple[float, float],
        dataset_stats: Tuple[torch.Tensor, torch.Tensor],
        device: Any,
        **kwargs,
):
    """Perform source separation using \\alpha-RGS.

    Args:
        mixture: Inputs mixture (y = s + \kappa * b)
        coeff: \kappa
        scaling: \\alpha-posterior hyperparameter (\omega)
        num_iters: Number of iterations to run \\alpha-RGS (N)
        use_trained_model: If True, use the learned diffusion model to compute
            the SOI score
        qpsk: The SOI (s)
        interference: The interference (b)
        model_qpsk: The qpsk diffusion model
        model_interference: The interference diffusion model
        learning_rate_range: Learning rate schedule range (\eta_max, \eta_min)
        dataset_stats: Dataset mean and standard deviation if interference is 
            not zero mean and unit variance.
    """

    del kwargs
    eta_max, eta_min = learning_rate_range
    # Same noise schedule as in our diffusion models. See learner.py
    noise_schedule = np.linspace(1e-4, 0.05, 50)
    noise_level = torch.tensor(
        np.cumprod(1 - noise_schedule).astype(np.float32)
    ).to(device)

    # Compute the matched filtering solution given the mixture
    s_mf = matchedfilter_remod(view_as_complex(coeff*mixture))
    s_est = torch.view_as_real(torch.tensor(s_mf.numpy())).transpose(1, 2).float()
    mixture = mixture.to(device)
    
    for i in range(num_iters):
        # Update learning rate using cosine annealing learning rate schedule
        eta = eta_min + 0.5 * (eta_max -  eta_min)*(1 + np.cos(np.pi * (i / num_iters)))

        # Sample t and 1 - \alpha_t
        t1 = torch.randint(MIN_STEP, max(1, int(MAX_STEP)) + 1, [1], device=device)
        noise_scale1 = noise_level[t1]
        # Sample u and 1 - \alpha_u
        t2 = torch.randint(MIN_STEP, max(1, int(MAX_STEP)) + 1, [1], device=device)
        noise_scale2 = noise_level[t2]
        
        noise1 = torch.randn_like(s_est)
        if not use_trained_model:
            rrc_mtx, _ = get_rrc_matrix()
            complex_noise = 1 / np.sqrt(2) * (np.random.randn(1, rrc_mtx.shape[-1]) \
                                              + 1j*np.random.randn(1, rrc_mtx.shape[-1]))
            rrc_noise = np.matmul(rrc_mtx, complex_noise.T).T
            noise1 = torch.view_as_real(torch.tensor(rrc_noise)).transpose(1, 2).float().to(device)
        noise2 = torch.randn_like(s_est)
        
        # Apply Gaussian smoothing to the estimates for s
        st_est = noise_scale1 ** 0.5 * s_est.to(device) + (1 - noise_scale1) ** 0.5 * noise1.to(device)
        # Apply Gaussian smoothing to the estimate for b
        b_est = mixture - 1 / coeff * s_est.to(device)
        b_est = (b_est - dataset_stats[0]) / dataset_stats[1]
        bu_est = noise_scale2 ** 0.5 * b_est.to(device) + (1 - noise_scale2) ** 0.5 * noise2.to(device)
        
        # Compute the score of the smoothened SOI
        sigma1 = (1 - noise_scale1.detach().cpu().numpy()) ** 0.5 
        if use_trained_model:
            score_t1 = model_qpsk(st_est, t1)
            score_t1 = score_t1 - noise1.to(device)
            score_t1 = -score_t1.detach().cpu()/(sigma1)
        else:
            score_t1 = qpsk_score_model(st_est, noise_scale1).to(device)
            score_t1 = score_t1.detach().cpu()
            score_t1 = score_t1 + (1 / sigma1) * noise1.detach().cpu().numpy()
        score_t1 *= (1 - sigma1 ** 2) ** 0.5
        
        # Compute the score of the smoothened interference
        sigma2 = (1 - noise_scale2.detach().cpu().numpy()) ** 0.5 
        score_t2 = model_interference(bu_est, t2)
        score_t2 = score_t2 - noise2.to(device)
        score_t2 = -score_t2.detach().cpu()/(sigma2)
        score_t2 *= (1 - sigma2 ** 2) ** 0.5
        
        s_est = s_est.detach().cpu()
        g = scaling * score_t2 - score_t1

        # Update the estimate using stochastic gradient descent (SGD)
        s_est = s_est - eta * g

        if (i+1) % 1000  == 0:
            s_hat = s_est.to(device)
            b_hat = mixture.to(device) - 1/coeff * s_est.to(device)
            bit_true, _ = qpskfn.qpsk_matched_filter_demod(view_as_complex(qpsk))
            bit_pred, _ = qpskfn.qpsk_matched_filter_demod(view_as_complex(s_hat.detach().cpu()))
            bit_error = np.mean(bit_true != bit_pred)
            print(
                f"{i:>4}: - MSE(s_hat-s): {F.mse_loss(qpsk, s_hat.detach().cpu()):.4f},"
                f" MSE(b_hat-b): {F.mse_loss(interference, b_hat.detach().cpu()):.6f},"
                f" BER: {bit_error:.4f}"
            )
    return s_est