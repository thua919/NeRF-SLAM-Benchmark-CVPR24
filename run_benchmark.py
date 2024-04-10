
# This file is modified from ESLAM
import argparse
import torch
import numpy as np
import random
import os
import json
from torch.utils.tensorboard import SummaryWriter
from src import config
from src.standard_SLAM import standard_SLAM

def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running standard benchmark SLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/benchmark_standard.yaml')
    
    save_path = cfg["data"]["output"]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))
    
    writer = SummaryWriter(save_path)
    _set_random_seed(919)
    
    nerf_slam = standard_SLAM(cfg, writer=writer)

    nerf_slam.run()

if __name__ == '__main__':
    main()
