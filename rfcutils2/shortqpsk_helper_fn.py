import sionna as sn
import numpy as np
import tensorflow as tf

# Binary source to generate uniform i.i.d. bits
binary_source = sn.utils.BinarySource()

samples_per_symbol = 4 # Number of samples per symbol, i.e., the oversampling factor
beta = 0.5 # Roll-off factor
span_in_symbols = 4 # Filter span in symbold
rrcf = sn.signal.RootRaisedCosineFilter(span_in_symbols, samples_per_symbol, beta)

# 4-QAM constellation
NUM_BITS_PER_SYMBOL = 2
constellation = sn.mapping.Constellation("qam", NUM_BITS_PER_SYMBOL, trainable=False) # The constellation is set to be NOT trainable

# Mapper and demapper
mapper = sn.mapping.Mapper(constellation=constellation)
demapper = sn.mapping.Demapper("app", constellation=constellation)

# AWGN channel
awgn_channel = sn.channel.AWGN()

def generate_qpsk_signal(batch_size, num_symbols, ebno_db=None):
    bits = binary_source([batch_size, num_symbols*NUM_BITS_PER_SYMBOL]) # Blocklength
    return modulate_qpsk_signal(bits, ebno_db)

def get_psf():
    return rrcf
    # return rrcf._coefficients_source

def matched_filter(sig):
    x_mf = rrcf(sig, padding="same")
    return x_mf

def qpsk_matched_filter_demod(sig, no=1e-4, soft_demod=False):
    x_mf = matched_filter(sig)
    num_symbols = sig.shape[-1]//samples_per_symbol
    ds = sn.signal.Downsampling(samples_per_symbol, samples_per_symbol//2, num_symbols)
    x_hat = ds(x_mf)
    x_hat = x_hat / tf.math.sqrt(tf.cast(samples_per_symbol, tf.complex64))
    llr = demapper([x_hat,no])
    if soft_demod:
        return llr, x_hat
    return tf.cast(llr > 0, tf.float32), x_hat

def modulate_qpsk_signal(info_bits, ebno_db=None):
    x = mapper(info_bits)
    us = sn.signal.Upsampling(samples_per_symbol)
    x_us = us(x)
    x_us = tf.pad(x_us, tf.constant([[0, 0,], [samples_per_symbol//2, 0]]), "CONSTANT")
    x_us = x_us[:, :-samples_per_symbol//2]
    x_rrcf = rrcf(x_us, padding="same")
    if ebno_db is None:
        y = x_rrcf
    else:
        no = sn.utils.ebnodb2no(ebno_db=ebno_db,
                        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
                        coderate=1.0) # Coderate set to 1 as we do uncoded transmission here
        y = awgn_channel([x_rrcf, no])
    y = y * tf.math.sqrt(tf.cast(samples_per_symbol, tf.complex64))
    return y, x, info_bits, constellation