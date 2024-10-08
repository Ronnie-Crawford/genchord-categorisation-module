# MIT License
# Copyright (c) 2024 Genome Research Ltd.
# Author: Ronnie Crawford <rc30@sanger.ac.uk>

# Set up environment to manage PyTorch memory
#import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Standard modules
import argparse

# Third-party modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Local modules
from config import config
from helpers import get_device
from datasets import get_datasets
from models import set_up_model

def main(device: str):

    if device == "cuda":

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    DEVICE = get_device(device)

    datasets = get_datasets()
    print(datasets[0][1])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type = str, help = "Manually overides the automatic device detection to use specific device from CPU, MPS or CUDA (GPU).")

    args = parser.parse_args()

    main(device = args.device)
