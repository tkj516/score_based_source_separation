# Score-based Source Separation with Applications to Digital Communication Signals
This directory contains code for our paper titled "Score-based Source
Separation with Applications to Digital Communication Signals".  Currently 
the directory is organized into different scripts to enable the entire pipeline 
to work.

**Note:** Currently the datasets and checkpoints we used in our models are not
included for space considerations.  Upon publication of the paper, we will
upload all relevant model parameters and data.  Regardless, one can re-generate 
the datasets and train the models from scratch using the provided code.  Due to
file size limits we are unable to attach covariance matrices required for LMMSE
estimation, however all other baselines can be run at the moment (just comment
out the LMMSE baseline in the notebook under `source_separations/notebooks`).

### Overall Pipeline with our Source Separation Solution

![My Image](figures/digital_comm_github.png)

### Results on Various RF Mixtures (Lowest (Best) BER and MSE is from our method)

![My Image](figures/github_results.png)

## Directory Structure

    .                           # Scripts for diffusion model training using the DiffWave architecture
    ├── source_separation       # Proposed method and all SCSS baselines
    │   └── notebooks           # Example notebook to plot source separation results
    ├── configs                 # Configuration files for training diffusion models on different signals
    ├── rfcutils2               # Utility files to generate synthetic QPSK and OFDM signals
    ├── dataset                 # The generated datasets will reside here
    ├── utils                   # Tools and utilities for generating the different RF datasets
    └── README.md

## Dependencies

We recommend use Anaconda and setting a virtual environment.  This environemnt
can be created easily using the provided `environment.yaml` file.

## Defining the Diffusion Model Config
In order to run an experiment, the parameters for loading the data, defining the
model and setting up the training pipeline are organized into structured configs. 
Our implementation is based off a publicly available DiffWave implementation [1].
The configs are managed by a tool called [OmegaConf](https://omegaconf.readthedocs.io/en/2.2_branch/)
which helps type check input arguments and merge different configs togther.
Refer to `configs.py` for the definition of the config (`class Config`):

```python
@dataclass
class Config:
    model_dir: str = MISSING

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig(root_dir="dataset/short_ofdm")
    distributed: DistributedConfig = DistributedConfig()
    trainer: TrainerConfig = TrainerConfig()
```

The experiment specific config is defined in a YAML file.  For example, consider
the config in `configs/ofdm_bpsk.yaml`:

```yaml
model_dir: ${datetime:"checkpoints/ofdm_bpsk"}
model:
  residual_layers: 30
  residual_channels: 128
  dilation_cycle_length: 10
data:
  root_dir: dataset/ofdm/bpsk
  augmentation: true
  target_len: 2560
  batch_size: 128
  num_workers: 2
  train_fraction: 0.9
distributed:
  distributed: true
  world_size: 2
trainer:
  learning_rate: 5e-4
  max_steps: 1_000_000
  log_every: 50
  save_every: 2000
  validate_every: 1000
  infer_every: 500
  num_infer_samples: 1
  fast_sampling: false
  infer_shifts:
    - 80
    - 8
    - 10
```

This config defines the experiment by setting the (hyper)parameters that the
`Config` class in `config.py` requires.  Any parameter not set in YAML file 
assumes it's default value.

The YAML config is passed to the main training script to launch the experiment:

```python
python main.py --config configs/ofdm_bpsk.yaml
```

Sometimes, you might want to run the same experiment with a change in a single 
field within the config.  For example, consider running the same experiment with
`trainer.fast_sampling` turned off and with a batch size of 64.  Instead of
editing the YAML you can override values in the CLI command via,

```python
python main.py --config configs/ofdm_bpsk.yaml "trainer.fast_sampling=false::data.batch_size=64"
```

where `::` must be used to demarcate user specified fields.

## Generating the Datasets
The dataset is generated with the help of the tools and utilies in the `utils` 
folder.  At the current moment we are unable to attach our datasets due to file
size limits for the supplementary material.  You may instead re-generate datasets
using the commands provide next.

### Synthetic Datasets (QPSK and OFDM)

To generate a dataset of QPSK signals of length 2560, and save them under
`dataset/qpsk` the following command can be used:

```python
cd utils
python prepare_synthetic_datasets.py \
                --num_samples 200200 \
                --signal_length 160 \
                --root_dir qpsk \
                --signal_name qpsk \
                qpsk
```

Similar commands can be used to generated OFDM (BPSK) and OFDM (QPSK) synthetic
signals:

```python
cd utils
python prepare_synthetic_datasets.py \
                --num_samples 200200 \
                --signal_length 37 \
                --root_dir ofdm \
                --signal_name bpsk \
                ofdm_bpsk
```

```python
cd utils
python prepare_synthetic_datasets.py \
                --num_samples 200200 \
                --signal_length 37 \
                --root_dir ofdm \
                --signal_name qpsk \
                ofdm_qpsk
```

After generating these datasets, copy over the first 200 samples in each dataset
to create a test set, under `dataset/separation/<signal_name>`.  The first 100
samples from this test set can be used for hyperparameter validation experiments
and the latter 100 samples can be used for inference in the source separation 
experiments.

### Real-world Dataset: CommSignal2

A single script creates the training and test set for CommSignal2.  Be sure to
download the dataset from https://www.dropbox.com/s/clh4xq7u3my3nx6/rfc_dataset.zip?dl=0
before proceeding.  Place this dataset under `dataset`.

```python
cd utils
python prepare_comm_signal_2_dataset.py
```

The datasets are loaded into the training pipeline using a PyTorch Dataset defined
in dataset.py.  The data can be augmented by shifting and a rotation by setting
`data.augmentation=true` in the config YAML.

## Trainer Setup and Workflow
The trainer is defined via the `class TrainerConfig` in `config.py`.  The model
can either be trained on a single GPU or on multiple GPUs using `DistributedDataParallel`.
To train in a distributed fashion set `distributed.distributed=true` in the 
config (see `class DistributedConfig`).

If trained in a distributed fashion, the model is copied to each GPU and the 
dataset is split evenly amongst the different GPUs.  The GPUs coordinate and
sync periodically.  The input dataset is split into a train and validation set.  
The validation, inference and logging frequencies are set in the `TrainerConfig`.
All checkpoints and Tensorboard logs can be found under the `checkpoints/` directory.

The training pipeline is defined in `learner.py`.  Some features of the 
trainer include:
- Training in a distributed fashion
- Logging results in real time on Tensorboard
- Resume training easily by specifying an already existing `model_dir`

## Training Commands

The trainer is wrapped with the script `main.py` that calls `learner.py` and 
`model.py` to train the diffusion models.  Refer to the previous sections to 
understand more about the model configs and training pipeline.  The diffusion 
models in our paper can be trained by executing four simple commands

### QPSK
```python
python main.py --config configs/qpsk.yaml
```

### OFDM (BPSK)
```python
python main.py --config configs/ofdm_bpsk.yaml
```

### OFDM (QPSK)
```python
python main.py --config configs/ofdm_qpsk.yaml
```

### Commsignal 2
```python
python main.py --config configs/commsignal2.yaml
```

## Source Separation Experiments

With the pre-trained models in `checkpoints` one can now run different source
separation experiments on the RF mixtures.  We provide implementations for the 
following methods:

- $\alpha$-RGS (Proposed Method)
- BASIS [2] (Original version from the paper)
- Modified BASIS 
- Reverse Diffusion 
- Matched Filtering
- LMMSE Estimation
- A supervised learning benchmark that uses UNet [3]

All the source separation scripts can be found in the `source_separation` folder:

    .                           
    ├── args.py                                           # Proposed method
    ├── basis.py                                          # BASIS separation 
    ├── modified_basis.py                                 # A modified version of BASIS to match our problem setup
    ├── reverse_diffusion.py                              # Runs reverse diffusion using interference diffusion model
    ├── unet_model.py                                     # Reference supervised learning model with joint and paired training data
    ├── utils.py                                          # Tools and utilities
    ├── run_separation.py                                 # Runs a source separation method and saves results
    ├── validate_alpha_posterior_hyperparameter.py        # Validates choice of $\omega = \kappa^2$
    ├── runtime_comparisons.py                            # Compares runtime of $\alpha$-RGS and Modified BASIS
    └── notebooks                                         # Example notebook that demonstrates how to visualize results

### Run a source separation experiment

All source separation experiments can be run using `run_separation.py`.  The 
source separation dataloader can be found in `utils.py`.  Please update the
model checkpoints in this file before running the separation experiments. All
separation experiments are run with 100 mixtures per SIR value. An example
on how to visualize the results is under `notebooks`. This is
the expected argument template for `run_separation.py`:

    Usage: run_separation.py [OPTIONS]

    Runs source separation using the methods provided.

    Args:     
    method: The source separation method to run     
    soi_type: The SOI (s) signal type     
    interference_type: The interference (b) signal type     
    sir_db: The SIR level of the mixture     
    scaling: The \alpha-posterior hyperparameter value     
    use_trained_model: If True, use the learned diffusion model to compute the SOI score
    learning_rate_range: Set the cosine annealing learning rate range (\eta_max, \eta_min)     
    num_iters: Number of iterations to run \alpha-RGS (N) or number of internal iterations for BASIS/ModifiedBASIS

    Options:
    --method [aRGS|BASIS|ModifiedBASIS|ReverseDiffusion]
                                    [required]
    --soi_type TEXT
    --interference_type [ofdm_bpsk|ofdm_qpsk|commsignal2]
                                    [required]
    --sir_db INTEGER                [required]
    --scaling TEXT
    --use_trained_model / --no-use_trained_model
    --learning_rate_range <FLOAT FLOAT>...
    --num_iters INTEGER
    --help                          Show this message and exit.

To reproduce the curves in our paper please refer to the following example commands. 
Note that when using $\alpha$-RGS (aRGS) with CommSignal2, the learning rate range 
should be set to (2e-3, 1e-6).  The SIR must be provided as a number, e.g., -10 for
-10 dB.  Furthermore, the scaling parameter refers to the value of $\frac{\omega}{\kappa}$.
Thus, when using the optimal $\alpha$-posterior ($\omega=\kappa^2$), `scaling=kappa`,
while using the vanilla MAP ($\omega=1$) refers to `scaling=invkappa`. The user may
also choose their own arbitrary scaling value as desired by setting `scaling=0.5`
for example.

#### $\alpha$-RGS

```python
# aRGS Trained SOI Model + OFDM (BPSK) at <SIR> dB
python run_separation.py --method=aRGS --interference_type=ofdm_bpsk --sir_db=<SIR> --scaling=kappa --learning_rate_range 5e-3 1e-6 --num_iters 20000
```

```python
# aRGS Trained SOI Model + OFDM (QPSK) at <SIR> dB
python run_separation.py --method=aRGS --interference_type=ofdm_qpsk --sir_db=<SIR> --scaling=kappa --learning_rate_range 5e-3 1e-6 --num_iters 20000
```

```python
# aRGS Trained SOI Model + Commsignal2 at <SIR> dB
python run_separation.py --method=aRGS --interference_type=commsignal2 --sir_db=<SIR> --scaling=kappa --learning_rate_range 2e-3 1e-6 --num_iters 20000
```

```python
# aRGS Analytical SOI Model + OFDM (BPSK) at <SIR> dB
python run_separation.py --method=aRGS --interference_type=ofdm_bpsk --sir_db=<SIR> --scaling=kappa --learning_rate_range 5e-3 1e-6 --num_iters 20000 --no-use_trained_model
```

```python
# aRGS Analytical SOI Model + OFDM (QPSK) at <SIR> dB
python run_separation.py --method=aRGS --interference_type=ofdm_qpsk --sir_db=<SIR> --scaling=kappa --learning_rate_range 5e-3 1e-6 --num_iters 20000 --no-use_trained_model
```

```python
# aRGS Analytical SOI Model + Commsignal2 at <SIR> dB
python run_separation.py --method=aRGS --interference_type=commsignal2 --sir_db=<SIR> --scaling=kappa --learning_rate_range 2e-3 1e-6 --num_iters 20000 --no-use_trained_model
```

#### BASIS (Original)

```python
# BASIS Analytical SOI Model + <INTERFERENCE_TYPE> at <SIR> dB
python run_separation.py --method=BASIS --interference_type=<INTERFERENCE_TYPE> --sir_db=<SIR> --num_iters 100 --no-use_trained_model
```

#### Modified BASIS

```python
# Modified BASIS Analytical SOI Model + <INTERFERENCE_TYPE> at <SIR> dB, no $\alpha$-posterior
python run_separation.py --method=ModifiedBASIS --interference_type=<INTERFERENCE_TYPE> --sir_db=<SIR> --scaling=invkappa --num_iters 100 --no-use_trained_model
```

```python
# Modified BASIS Analytical SOI Model + <INTERFERENCE_TYPE> at <SIR> dB with $\alpha$-posterior
python run_separation.py --method=ModifiedBASIS --interference_type=<INTERFERENCE_TYPE> --sir_db=<SIR> --scaling=kappa --num_iters 100 --no-use_trained_model
```

#### Reverse Diffusion (with Interference model)

```python
# Reverse Diffusion using <INTERFERENCE_TYPE> model at <SIR> dB
python run_separation.py --method=ReverseDiffusion --interference_type=<INTERFERENCE_TYPE> --sir_db=<SIR>
```

### Run hyperparameter validation and comparing runtimes

The choice of $\omega=\kappa^2$ can be validated by toggling the ratio of 
$\frac{\omega}{\kappa^2}$.  We experiment with this ratio set to 0.01, 0.1, 1, 10 and 100.
This experiment, for a particular ratio, may be run to compute average BER and MSE
across 100 different mixtures.  Note these validation experiments use a different
set of 100 mixtures compared to the ones used for our main results.

```python
# Hyperparameter validation with ratio <R> using Analytical SOI Model + <INTERFERENCE_TYPE> model at <SIR> dB.  Learning rate must also be set appropriately. See above or paper.
python run_separation.py --method=aRGS --interference_type=<INTERFERENCE_TYPE> --sir_db=<SIR> --omega_div_kappa2=<R> --learning_rate_range <MAX> <MIN>
```

Additionally we can compare runtimes of our method and modified BASIS via:

```python
# Hyperparameter validation with ratio <R> using Analytical SOI Model + <INTERFERENCE_TYPE> model at <SIR> dB.  Learning rate must also be set appropriately for $\alpha$-RGS. See above or paper.
python runtime_comparison.py --interference_type=<INTERFERENCE_TYPE> --sir_db=<SIR> --learning_rate_range <MAX> <MIN> --num_iters=20000
```
 ### Visualizing Results

 All results will be stored in the `metrics` folder under `source_separation`. 
 With these results saved, the BER and MSE plots (with optional error bars) can
 be visualized by following the example notebook provided under `notebooks`. 
 Within this notebook additional baselines such as matched filtering, LMMSE 
 estimation and UNet separation are also run.

 ## References

 [1] https://github.com/lmnt-com/diffwave <br>
 [2] https://github.com/jthickstun/basis-separation <br>
 [3] https://github.com/RFChallenge/SCSS_CSGaussian