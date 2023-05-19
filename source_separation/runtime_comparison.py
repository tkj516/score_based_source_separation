import sys
sys.path.append("..")

import time
import click
import random
import pickle
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import torch
import functools

NUM_SAMPLES = 1000
SIGNAL_LENGTH = 2560
DEVICE = torch.device("cuda:0")

from typing import Tuple, Union

from utils import load_separation_dataset, load_models
from args import args_separation
from modified_basis import modified_basis_separation

random.seed(912345)
np.random.seed(912345)
torch.manual_seed(912345)


@click.command()
@click.option("--soi_type", default="qpsk", type=str)
@click.option("--interference_type", required=True, 
              type=click.Choice([
                  "ofdm_bpsk", "ofdm_qpsk", "commsignal2",
              ]))
@click.option("--sir_db", required=True, type=int)
@click.option("--use_trained_model/--no-use_trained_model", default=True)
@click.option("--learning_rate_range", nargs=2, 
              type=click.Tuple([float, float]), default=(5e-3, 1e-6))
@click.option("--num_iters", type=int, default=20000)
def main(
    soi_type: str,
    interference_type: str,
    sir_db: int, 
    use_trained_model: bool, 
    learning_rate_range: Tuple[float, float],
    num_iters: int,
):
    """Times source separation using \\alpha-RGS and Modified BASIS.

    Args:
        soi_type: The SOI (s) signal type
        interference_type: The interference (b) signal type
        sir_db: The SIR level of the mixture
        use_trained_model: If True, use the learned diffusion model to compute
            the SOI score
        learning_rate_range: Set the cosine annealing learning rate range
            (\eta_max, \eta_min)
        num_iters: Number of iterations to run \\alpha-RGS (N) or number of
            internal iterations for BASIS/ModifiedBASIS
    """

    # Load the inference dataset and pre-trained diffusion models
    dataset = load_separation_dataset(soi_type, interference_type, SIGNAL_LENGTH)
    model_qpsk, _, model_interference, _ = load_models(
        soi_type, interference_type, DEVICE
    )
    if interference_type == "commsignal2":
        dataset_mean = torch.tensor(
            np.array([ 1.3608e-05, -1.5107e-06]).reshape(1, 2, 1)
        ).float()
        dataset_std =  torch.tensor(
            np.array([0.7634, 0.7634]).reshape(1, 2, 1)
        ).float()
    else:
        dataset_mean = torch.zeros(1, 2, 1)
        dataset_std = torch.ones(1, 2, 1)

    # Set the value of \kappa
    coeff = np.sqrt(10 ** (-sir_db / 10))
    if "ofdm" in interference_type:
        # Only used for plotting, not required for source separation
        coeff = coeff * np.sqrt(64/56)  # 56/64 is the compensating power for OFDM

    args_separation_fn = functools.partial(
        args_separation,
        coeff=coeff,
        scaling=coeff,
        num_iters=num_iters,
        use_trained_model=use_trained_model,
        model_qpsk=model_qpsk,
        model_interference=model_interference,
        learning_rate_range=learning_rate_range,
        dataset_stats=(dataset_mean.to(DEVICE), dataset_std.to(DEVICE)),
        device=DEVICE,
    )

    modified_basis_separation_fn = functools.partial(
        modified_basis_separation,
        coeff=coeff,
        num_iters=num_iters // 50,
        use_trained_model=use_trained_model,
        model_qpsk=model_qpsk,
        model_interference=model_interference,
        dataset_stats=(dataset_mean.to(DEVICE), dataset_std.to(DEVICE)),
        use_alpha_posterior=True,
        device=DEVICE,
    )
    
    all_timing_args = []
    all_timing_basis = []
    for idx in range(100):
        qpsk = dataset[idx]["qpsk_sample"].unsqueeze(0)
        interference = dataset[idx]["interference_sample"].unsqueeze(0)

        y = 1 / coeff * qpsk + interference
        with torch.no_grad():
            t0 = time.time()
            args_separation_fn(
                mixture=y, 
                qpsk=qpsk,
                interference=interference,
            )
            t1 = time.time()
            time_args = t1 - t0
            
            t0 = time.time()
            modified_basis_separation_fn(
                mixture=y,
                qpsk=qpsk,
                interference=interference
            )
            t1 = time.time()
            time_basis = t1 - t0
            
            all_timing_args.append(time_args)
            all_timing_basis.append(time_basis)
            print(f"aRGS={all_timing_args[-1]}, BASIS={all_timing_basis[-1]}")
        pickle.dump(
            (
                all_timing_args, 
                all_timing_basis,
            ), open(f"timing_comparisons.pkl","wb")
        )
        

if __name__ == "__main__":
    main()
