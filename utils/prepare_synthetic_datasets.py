import sys
sys.path.append("..")

import argparse
import functools
import os
import rfcutils2.shortqpsk_helper_fn as short_qpskfn
import rfcutils2.qpsk_helper_fn as qpskfn
import rfcutils2.shortofdm_helper_fn as short_ofdmfn
import rfcutils2.ofdm_helper_fn as ofdmfn
import rfcutils2.ofdm_bpsk_helper_fn as ofdmfn_bpsk
import rfcutils2.ofdm_qpsk_helper_fn as ofdmfn_qpsk
import numpy as np
import tensorflow as tf


def generate_dataset(args, generation_fn):
    with tf.device('cpu'):
        sig, _, _, _ = generation_fn(args.num_samples, args.signal_length)
    sig = sig.numpy()

    if not os.path.exists(os.path.join("../dataset", args.root_dir)):
        os.makedirs(os.path.join("../dataset", args.root_dir), exist_ok=True)

    savedir = os.path.join("../dataset", args.root_dir,
                           f"{args.signal_name}_{args.num_samples}_{args.signal_length}")
    if os.path.exists(savedir):
        raise ValueError("Data directory already exists!")
    else:
        os.makedirs(savedir)

    for i in range(sig.shape[0]):
        np.save(os.path.join(savedir, f"sig_{i}.npy"), sig[i, :])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for generating RF datasets.")
    parser.add_argument("--num_samples", type=int, default=200000,
                        help="Number of samples to generate.")
    parser.add_argument("--signal_length", type=int, default=1280,
                        help="Number of bits in the time domain signal.")
    parser.add_argument("--root_dir", type=str,
                        help="Root directory to store new dataset.")
    parser.add_argument("--signal_name", type=str, help="Name of the signal.")
    subparsers = parser.add_subparsers(
        help="Subparsers for different data types")

    parser_short_qpsk = subparsers.add_parser(
        "short_qpsk", help="Generate short QPSK dataset.")
    parser_short_qpsk.set_defaults(func=functools.partial(
        generate_dataset, generation_fn=short_qpskfn.generate_qpsk_signal))

    parser_qpsk = subparsers.add_parser(
        "qpsk", help="Generate QPSK dataset.")
    parser_qpsk.set_defaults(func=functools.partial(
        generate_dataset, generation_fn=qpskfn.generate_qpsk_signal))
    
    parser_ofdm_bpsk = subparsers.add_parser(
        "ofdm_bpsk", help="Generate OFDM (BPSK) dataset.")
    parser_ofdm_bpsk.set_defaults(func=functools.partial(
        generate_dataset, generation_fn=ofdmfn_bpsk.generate_ofdm_signal))
    
    parser_ofdm_qpsk = subparsers.add_parser(
        "ofdm_qpsk", help="Generate OFDM (QPSK) dataset.")
    parser_ofdm_qpsk.set_defaults(func=functools.partial(
        generate_dataset, generation_fn=ofdmfn_qpsk.generate_ofdm_signal))
    
    parser_short_ofdm = subparsers.add_parser(
        "short_ofdm", help="Generate general short OFDM dataset.")
    parser_short_ofdm.set_defaults(func=functools.partial(
        generate_dataset, generation_fn=short_ofdmfn.generate_ofdm_signal))

    parser_ofdm = subparsers.add_parser(
        "ofdm", help="Generate general OFDM dataset.")
    parser_ofdm.set_defaults(func=functools.partial(
        generate_dataset, generation_fn=ofdmfn.generate_ofdm_signal))

    args = parser.parse_args()
    args.func(args)