import cv2
import os
import glob
import argparse

# Get the Data
parser = argparse.ArgumentParser(description="Displays the number of failure inducing inputs")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

# Get the dataset directory
DATASET_DIRECTORY = f"{args.dataset_directory}"

# Get all the available datasets
available_datasets_paths = glob.glob(f"{DATASET_DIRECTORY}/*")
available_datasets = [os.path.basename(dset) for dset in available_datasets_paths]
available_datasets = sorted(available_datasets)