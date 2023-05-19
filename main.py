import socketserver
import sys
from argparse import ArgumentParser
from ast import parse
from typing import List

from omegaconf import OmegaConf
from torch.cuda import device_count
from torch.multiprocessing import spawn

from config import Config, parse_configs

from learner import train, train_distributed


def _get_free_port():
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]


def main(argv: List[str]):
    parser = ArgumentParser(description="Train a Diffwave model.")
    parser.add_argument("--config", type=str, required=True,
                        help="Configuration file for model.")
    args = parser.parse_args(argv[1:-1])

    # First create the base config
    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli(argv[-1].split("::")) if argv[-1] != "" else None
    cfg: Config = Config(**parse_configs(cfg, cli_cfg))

    # Setup training
    world_size = device_count()
    if world_size != cfg.distributed.world_size:
        raise ValueError(
            "Requested world size is not the same as number of visible GPUs.")
    if cfg.distributed.distributed:
        if world_size < 2:
            raise ValueError(
                f"Distributed training cannot be run on machine with {world_size} device(s).")
        if cfg.data.batch_size % world_size != 0:
            raise ValueError(
                f"Batch size {cfg.data.batch_size} is not evenly divisble by # GPUs = {world_size}.")
        cfg.data.batch_size = cfg.data.batch_size // world_size
        port = _get_free_port()
        spawn(train_distributed, args=(world_size, port, cfg), nprocs=world_size, join=True)
    else:
        train(cfg)


if __name__ == "__main__":
    argv = sys.argv
    if len(sys.argv) == 3:
        argv = argv + [""]
    main(argv)
