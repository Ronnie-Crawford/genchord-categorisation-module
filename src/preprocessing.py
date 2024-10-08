# MIT License
# Copyright (c) 2024 Genome Research Ltd.
# Author: Ronnie Crawford <rc30@sanger.ac.uk>

# Third-party modules
import numpy as np
import pandas as pd

def preprocess_dataset(file_path):

    # Read the first 8 lines into a dictionary
    with open(file_path, 'r') as tree_file:

        header_lines = [next(tree_file) for _ in range(8)]

    header_dict = {}
    for line in header_lines:

        if ": " in line:

            key, value = line.strip().split(": ", 1)
            header_dict[key] = value

    # Read the remaining data into a DataFrame
    data = pd.read_csv(file_path, sep=" ", skiprows=8, header=0, names = ["chromosome", "starts", "copy_number"])
    data["ends"] = data["starts"].shift(-1, fill_value = 0)
    data["ends"] = data["ends"].replace(0, np.nan)

    return header_dict, data
