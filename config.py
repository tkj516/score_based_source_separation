from dataclasses import MISSING, asdict, dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver(
    "datetime", lambda s: f'{s}_{datetime.now().strftime("%H_%M_%S")}', replace=True)


@dataclass
class NoiseScheduleConfig:
    train_noise_schedule: List[float] = field(
        default_factory=lambda: np.linspace(1e-4, 0.05, 50).tolist())
    inference_noise_schedule: List[float] = field(
        default_factory=lambda: [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5])


@dataclass
class ModelConfig:
    input_channels: int = 2
    residual_layers: int = 30
    residual_channels: int = 64
    dilation_cycle_length: int = 10
    noise_schedule: NoiseScheduleConfig = NoiseScheduleConfig()


@dataclass
class DataConfig:
    root_dir: str = MISSING
    target_len: int = 2560
    augmentation: bool = True
    batch_size: int = 16
    num_workers: int = 4
    train_fraction: float = 0.8
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None


@dataclass
class DistributedConfig:
    distributed: bool = False
    world_size: int = 2


@dataclass
class TrainerConfig:
    learning_rate: float = 2e-4
    max_steps: int = 1000
    max_grad_norm: Optional[float] = None
    fp16: bool = False

    log_every: int = 50
    save_every: int = 2000
    validate_every: int = 100
    infer_every: int = 500
    infer_target_len: int = 2560
    num_infer_samples: int = 4
    fast_sampling: bool = True
    # If unshifting during test time
    infer_shifts: Tuple[int] = field(default_factory=lambda: (20, 4, 5))


@dataclass
class Config:
    model_dir: str = MISSING

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig(root_dir="dataset/short_ofdm")
    distributed: DistributedConfig = DistributedConfig()
    trainer: TrainerConfig = TrainerConfig()


def parse_configs(cfg: DictConfig, cli_cfg: Optional[DictConfig] = None) -> DictConfig:
    base_cfg = OmegaConf.structured(Config)
    merged_cfg = OmegaConf.merge(base_cfg, cfg)
    if cli_cfg is not None:
        merged_cfg = OmegaConf.merge(merged_cfg, cli_cfg)
    return merged_cfg


if __name__ == "__main__":
    base_config = OmegaConf.structured(Config)
    config = OmegaConf.load("configs/short_ofdm.yaml")
    config = OmegaConf.merge(base_config, OmegaConf.from_cli(), config)
    config = Config(**config)

    print(asdict(config))
