# MIT License
# Copyright (c) 2024 Genome Research Ltd.
# Author: Ronnie Crawford <rc30@sanger.ac.uk>

# Third-party modules
import pandas as pd
import torch
from torch.utils.data import Dataset

# Local modules
from config import config
from preprocessing import preprocess_dataset

class CopyTreeDataset(Dataset):

    def __init__(self, header_dict: dict, data: pd.DataFrame):

        """
        Set up PyTorch dataset by reading in

        Parameters:
            - path (str): Path to the csv/tsv file containing the dataset.
        """

        self.chromosomes = data["chromosome"]
        self.starts = data["starts"]
        self.ends = data["ends"]
        self.copy_numbers = data["copy_number"]

    def __len__(self):

        return len(self.starts)

    def __getitem__(self, index: int):

        chromosome = self.chromosomes[index]
        start = self.starts[index]
        end = self.ends[index]
        copy_number = self.copy_numbers[index]

        region = {
            "chromosome": chromosome,
            "start": start,
            "end": end,
            "copy_number": copy_number
        }

        return region

def get_datasets():

    """
    Sets up the dataset with the given name using the CopyTreeDataset class.

    Returns:
        - datasets (list): A list of datasets that have been initialised.
    """

    datasets = []

    for dataset_name in config["DATASETS_IN_USE"]:

        try:

            header_dict, data = preprocess_dataset(config["DATASETS"][dataset_name]["PATH"])
            dataset = CopyTreeDataset(header_dict, data)

        except Exception as error: raise Exception(f"Could not initialise dataset: {dataset_name}")

        datasets.append(dataset)

    return datasets
