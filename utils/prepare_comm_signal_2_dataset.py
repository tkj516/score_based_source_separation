import os
import numpy as np
import h5py
from tqdm import tqdm

SIGNAL_LENGTH = 2560

if __name__ == "__main__":
    if not os.path.exists("../dataset/CommSignal2_raw_data.h5"):
        raise FileNotFoundError(
            "Please first download the commsignal2 dataset from "
            "https://www.dropbox.com/s/clh4xq7u3my3nx6/rfc_dataset.zip?dl=0"
        )
    
    with h5py.File('../dataset/CommSignal2_raw_data.h5', 'r') as data_h5file:
        sig_data = np.array(data_h5file.get('dataset'))
        sig_type_info = data_h5file.get('sig_type')[()]
        if isinstance(sig_type_info, bytes):
            sig_type_info = sig_type_info.decode("utf-8")
    assert sig_data.dtype == np.complex64

    data = sig_data[:100, 5000: 40000]

    root_dir = "../dataset/commsignal2"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Create the training set for diffusion model learning
    count = 0
    for i in tqdm(range(data.shape[0])):
        for j in range(0, data.shape[1] - SIGNAL_LENGTH, 20):
            np.save(os.path.join(
                root_dir, f"sig_{count}.npy"), data[i, j: j + SIGNAL_LENGTH].reshape(SIGNAL_LENGTH, ))
            count +=  1

    # Create the test set for source separation
    data = sig_data[100:, :]
    for i in tqdm(range(data.shape[0])):
        np.save(os.path.join(
            "../dataset/separation/commsignal2", f"sig_{count}.npy"
        ), data[i, :].reshape(-1, ))
