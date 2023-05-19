import sys
sys.path.append("..")

import random
import pickle
import numpy as np
import tensorflow as tf

import rfcutils2.qpsk_helper_fn as qpskfn
# This must be used before loading PyTorch
tf.config.set_visible_devices([], "GPU")
import torch
from utils import view_as_complex

random.seed(912345)
np.random.seed(912345)
torch.manual_seed(912345)


def matchedfilter_remod(sig):
    b, _ = qpskfn.qpsk_matched_filter_demod(sig)
    s, _, _, _ = qpskfn.modulate_qpsk_signal(b)
    return s


def get_rrc_matrix():
    rrc_mtx = pickle.load(open("rrcmtx_2560.pkl", "rb"))
    irrc_mtx = 1 / 16 * rrc_mtx.T
    return rrc_mtx, irrc_mtx


def qpsk_score_model(sig, noise_scale):
    # Get QPSK symbols and RRC filter matrices
    qpsk_symbols = 1 / np.sqrt(2) * np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j])
    rrc_mtx, irrc_mtx = get_rrc_matrix()

    sigma = (1 - noise_scale.detach().cpu().numpy()) ** 0.5 
    sig_complex = view_as_complex(sig.detach().cpu())
    sym_complex = np.matmul(irrc_mtx, sig_complex.T).T
    sym_prob = -1 / sigma ** 2 * np.abs(np.expand_dims(sym_complex, axis=-1) \
                                        - np.sqrt(1 - sigma**2)*qpsk_symbols.reshape(1, 1, -1)) ** 2
    sym_prob -= sym_prob.max(axis=-1, keepdims=True)
    sym_prob = np.exp(sym_prob)
    sym_prob /= sym_prob.sum(axis=-1, keepdims=True)
    expec_sym = np.sum(sym_prob * np.sqrt(1 - sigma ** 2) * qpsk_symbols.reshape(1, 1, -1), axis=-1)
    score_theta = 1 / sigma ** 2 * (-sig_complex + np.matmul(rrc_mtx, expec_sym.T).T)
    return torch.view_as_real(torch.tensor(score_theta)).transpose(1, 2).float()