import os
import sys
import glob
import argparse

import numpy as np



# Import Common
current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from prettytable import ALL
from prettytable import PrettyTable
from description_loader import DescriptionLoader

# Get the Data
parser = argparse.ArgumentParser(description="Generates a table with the number of in and out of ODD datapoints there are")
parser.add_argument('--annotator',
                    type=str,
                    choices=['Human', 'ChatGPT_Base', 'Llama_Base', 'Vicuna_Base', 'Llama_Plus', 'Vicuna_Plus'],
                    required=True,
                    help="The annotator to use. Choose between 'Human', 'ChatGPT_Base', 'Llama_Base', 'Vicuna_Base', 'Llama_Plus', 'Vicuna_Plus'.")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")
# Get the dataset directory
DATASET_DIRECTORY = args.dataset_directory

# Create a table with the specified columns
table = PrettyTable()
table.hrules = ALL

# Add field names
table.field_names = ["Dataset Filter", "Dataset Name", "In ODD Count", "Out ODD Count"]

# Get all the available datasets
available_datasets_paths = glob.glob(f"{DATASET_DIRECTORY}/*")
available_datasets = [os.path.basename(dset) for dset in available_datasets_paths]
available_datasets = sorted(available_datasets)

# List the types we want to find
description_filter_types = ["Full Dataset", "Failing Data", "Passing Data"]

# For the different data filters
for data_filter in description_filter_types:
    # For each of the datasets
    for dataset in available_datasets:

        # Get all the files based on the filter:
        if data_filter == "Full Dataset":
            descriptions = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{args.annotator}/*_output.txt")
        elif data_filter == "Failing Data":
            descriptions = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{args.annotator}/fail_*_output.txt")
        elif data_filter == "Passing Data":
            descriptions = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{args.annotator}/pass_*_output.txt")

        # Load the coverage vector
        dl = DescriptionLoader(descriptions)
        
        # Count the number in ODD/ out ODD/ and unknown
        in_odd_count  = np.sum(np.all(dl.coverage_vector == 0, axis=1))
        out_odd_count = np.sum(np.all(np.isin(dl.coverage_vector, [0, 1]), axis=1) & ~np.all(dl.coverage_vector == 0, axis=1))
        unknown_count = np.sum(np.any(dl.coverage_vector == -1, axis=1))
        assert unknown_count == 0, "Human dataset should not have any unknowns"

        # Add the table data
        table.add_row((data_filter, dataset.replace("_", " "), in_odd_count, out_odd_count))
        

# Align columns
table.align["Dataset Filter"] = "l"
table.align["Dataset Name"] = "l"
table.align["In ODD Count"] = "r"
table.align["Out ODD Count"] = "r"

# Your existing code to print the table
print(table)