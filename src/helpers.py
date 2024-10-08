# MIT License
# Copyright (c) 2024 Genome Research Ltd.
# Author: Ronnie Crawford <rc30@sanger.ac.uk>

# Third-party modules
import torch

def get_device(device: str):

    """
    Determines the best available device (GPU, MPS, or CPU).

    Returns:

        - torch.device: The best available device.
    """

    if device == None:

        if torch.cuda.is_available():

            print("Using device: CUDA.")
            return torch.device("cuda")

        elif torch.backends.mps.is_available():

            print("Using device: MPS.")
            return torch.device("mps")

        else:

            print("Using device: CPU.")
            return torch.device("cpu")

    else:

        device = device.lower()
        print(f"Manual overide, using device: {device.upper()}")
        return torch.device(device)
