import sys
sys.path.append("..")

import os
import click
import random
import pickle
import functools
import numpy as np
import tensorflow as tf

import rfcutils2.qpsk_helper_fn as qpskfn
# This must be used before loading PyTorch
tf.config.set_visible_devices([], "GPU")
import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import Tuple, Union

from utils import (
    load_separation_dataset, load_models, view_as_complex
)
from args import args_separation
from basis import basis_separation
from modified_basis import modified_basis_separation
from reverse_diffusion import reverse_diffusion_separation

random.seed(912345)
np.random.seed(912345)
torch.manual_seed(912345)

# Signal lengths for separation
SIGNAL_LENGTH = 2560
DEVICE = torch.device("cuda:0")


SOURCE_SEPARATORS = {
    "aRGS": args_separation,
    "BASIS": basis_separation,
    "ModifiedBASIS": modified_basis_separation,
    "ReverseDiffusion": reverse_diffusion_separation,
}


@click.command()
@click.option("--method", required=True,
              type=click.Choice([
                  "aRGS", "BASIS", "ModifiedBASIS", "ReverseDiffusion"
              ]))
@click.option("--soi_type", default="qpsk", type=str)
@click.option("--interference_type", required=True, 
              type=click.Choice([
                  "ofdm_bpsk", "ofdm_qpsk", "commsignal2",
              ]))
@click.option("--sir_db", required=True, type=int)
@click.option("--scaling", default="kappa")
@click.option("--use_trained_model/--no-use_trained_model", default=True)
@click.option("--learning_rate_range", nargs=2, 
              type=click.Tuple([float, float]), default=(5e-3, 1e-6))
@click.option("--num_iters", type=int, default=20000)
def main(
    method: str,
    soi_type: str,
    interference_type: str,
    sir_db: int, 
    scaling: Union[str, float], 
    use_trained_model: bool, 
    learning_rate_range: Tuple[float, float],
    num_iters: int,
):
    """Runs source separation using the methods provided.

    Args:
        method: The source separation method to run
        soi_type: The SOI (s) signal type
        interference_type: The interference (b) signal type
        sir_db: The SIR level of the mixture
        scaling: The \\alpha-posterior hyperparameter value
        use_trained_model: If True, use the learned diffusion model to compute
            the SOI score
        learning_rate_range: Set the cosine annealing learning rate range
            (\eta_max, \eta_min)
        num_iters: Number of iterations to run \\alpha-RGS (N) or number of
            internal iterations for BASIS/ModifiedBASIS
    """

    # Load the inference dataset and pre-trained diffusion models
    dataset = load_separation_dataset(soi_type, interference_type, SIGNAL_LENGTH)
    model_qpsk, _, model_interference, cfg_interference = load_models(
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

    # Set the value of \alpha-posterior parameter \alpha=\omega
    scaling_str = scaling
    use_alpha_posterior = False
    if isinstance(scaling, str):
        if scaling == "invkappa":
            scaling = 1 / coeff
        elif scaling == "kappa":
            scaling = coeff
            use_alpha_posterior = True
        else:
            raise ValueError("Unexpected string identifier for scaling value.")

    separation_fn = functools.partial(
        SOURCE_SEPARATORS[method],
        coeff=coeff,
        scaling=scaling,
        num_iters=num_iters,
        use_trained_model=use_trained_model,
        model_qpsk=model_qpsk,
        model_interference=model_interference,
        cfg_interference=cfg_interference,
        learning_rate_range=learning_rate_range,
        dataset_stats=(dataset_mean.to(DEVICE), dataset_std.to(DEVICE)),
        use_alpha_posterior=use_alpha_posterior,
        device=DEVICE,
    )

    # Create an output file to save results 
    output_folder = f"metrics/{soi_type}_{interference_type}_{method}"
    if method in ["aRGS", "ModifiedBASIS"]:
        scaling_str = f"Scale_{scaling_str}"
        output_folder += f"_{scaling_str}"
    if not method == "ReverseDiffusion":
        trained_model_str = "TrainedModel" if use_trained_model else "AnalyticalModel"
        output_folder += f"_{trained_model_str}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Experiments for {output_folder} at {sir_db} dB SINR")

    for idx in tqdm(range(100, 200)):
        qpsk = dataset[idx]["qpsk_sample"].unsqueeze(0)
        interference = dataset[idx]["interference_sample"].unsqueeze(0)

        # Normalize the mixture by \kappa, i.e., y / \kappa
        y = 1 / coeff * qpsk + interference
        with torch.no_grad():
            # SOI estimate using source separator
            s_pred = separation_fn(
                mixture=y, 
                qpsk=qpsk, 
                interference=interference,
            )
            s_pred = s_pred.to(DEVICE)

        # Dump all the results into the output folder
        pickle.dump(
            (y, qpsk, interference, s_pred), 
            open(f"{output_folder}/sample_{sir_db}dB_{idx}.pkl","wb")
        )

        bit_true, _ = qpskfn.qpsk_matched_filter_demod(view_as_complex(qpsk))
        bit_pred, _ = qpskfn.qpsk_matched_filter_demod(view_as_complex(s_pred.cpu()))

        print("=================================================")
        print(f"SIR [dB] = {sir_db}, sample number {idx - 100}")
        print(f"BER ({method}) = {np.mean(bit_true != bit_pred)}")
        print(f"MSE ({method}) = {F.mse_loss(qpsk, s_pred.detach().cpu()).numpy()}")
        print("=================================================")
        print()


if __name__ == "__main__":
    main()
