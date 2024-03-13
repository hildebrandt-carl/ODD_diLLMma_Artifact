import os
import sys
import glob
import copy
import argparse

from math import pi

import numpy as np
import matplotlib.pyplot as plt


# Import Common
current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from constants import ODD
from description_loader import DescriptionLoader


# Get the Data
parser = argparse.ArgumentParser(description="Displays a comparison between different ODD dimensions")
parser.add_argument('--description_filter',
                    choices=['Both', 'Pass', 'Fail'],
                    required=True,
                    help="Filter results by status: 'Both' to include both passing and failing, 'Pass' for passing only, or 'Fail' for failing only.")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")
# Get the dataset directory
DATASET_DIRECTORY = args.dataset_directory

# Get all the available datasets
available_datasets_paths = glob.glob(f"{DATASET_DIRECTORY}/*")
available_datasets = [os.path.basename(dset) for dset in available_datasets_paths]
available_datasets = sorted(available_datasets)

# Get all the available annotator
available_annotators_paths = glob.glob(f"{DATASET_DIRECTORY}/*/5_Descriptions/*")
available_annotators       = [os.path.basename(annotator) for annotator in available_annotators_paths]
available_annotators       = [annotator for annotator in available_annotators if "Human" not in annotator]
available_annotators       = list(set(available_annotators))
available_annotators       = sorted(available_annotators)

# Load the human baseline data
if args.description_filter == "Both":
    descriptions = sorted(glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/Human/*_output.txt"))
elif args.description_filter == "Pass":
    descriptions = sorted(glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/Human/pass_*_output.txt"))
elif args.description_filter == "Fail":
    descriptions = sorted(glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/Human/fail_*_output.txt"))
baseline_dl = DescriptionLoader(descriptions)

f1_score_per_dimension = []
for annotator in available_annotators:

    all_annotator_descriptions = []
    all_human_descriptions = []
    for dataset in available_datasets:
        # Get all the files based on the filter:
        if args.description_filter == "Both":
            annotator_descriptions  = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{annotator}/*_output.txt")
            human_descriptions      = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/Human/*_output.txt")
        elif args.description_filter == "Pass":
            annotator_descriptions  = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{annotator}/pass_*_output.txt")
            human_descriptions      = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/Human/pass_*_output.txt")
        elif args.description_filter == "Fail":
            annotator_descriptions  = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{annotator}/fail_*_output.txt")
            human_descriptions      = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/Human/fail_*_output.txt")
        all_annotator_descriptions += annotator_descriptions
        all_human_descriptions += human_descriptions
    
    # Sort the data
    all_annotator_descriptions = sorted(all_annotator_descriptions)
    all_human_descriptions = sorted(all_human_descriptions)
    assert len(all_annotator_descriptions) == len(all_human_descriptions), "Annotator and human's description files are not the same length"

    # Make sure they align
    for f1, f2 in zip(all_annotator_descriptions, all_human_descriptions):
        f1_base = os.path.basename(f1)
        f2_base = os.path.basename(f2)
        assert f1_base == f2_base, f"Filenames don't match {f1_base} - {f2_base}"

    # Get the coverage vectors
    annotator_dl    = DescriptionLoader(all_annotator_descriptions)
    human_dl        = DescriptionLoader(all_human_descriptions)

    # Get the data
    true = copy.deepcopy(human_dl.coverage_vector)
    pred = copy.deepcopy(annotator_dl.coverage_vector)

    # Compute some metrics
    exact_match_vectors = np.all(true == pred, axis=1)
    matching_rows_count = np.sum(exact_match_vectors)

    # Find the true in and out of ODD
    true_out_odd = np.any(true == 1, axis=1)
    true_out_odd_count = np.sum(true_out_odd)
    true_in_odd = np.all(true == 0, axis=1)
    true_in_odd_count = np.sum(true_in_odd)

    # Find how many predictions were correct
    correct_out_odd_pred_count = np.sum(np.any(pred[true_out_odd] == 1, axis=1))
    correct_in_odd_pred_count  = np.sum(np.all(pred[true_in_odd] == 0, axis=1))
    
    # Total in and out of ODD predictions
    correct_in_out_pred_count = correct_out_odd_pred_count + correct_in_odd_pred_count

    # Print the data
    print(f"Annotator: {annotator}")
    print(f"Total annotations: {np.shape(true)[0]}")
    print(f"Correctly identified (In/Out of ODD): {correct_in_out_pred_count}/{np.shape(true)[0]}")
    print(f"In ODD correctly identified: {correct_in_odd_pred_count}/{true_in_odd_count}")
    print(f"Out of ODD correctly identified: {correct_out_odd_pred_count}/{true_out_odd_count}")
    print(f"Exact ODD dimensions match: {matching_rows_count}/{np.shape(true)[0]}")
    print("")


# ChatGPT_Base == 586/1500
# ChatGPT_Base in ODD 294/739
# ChatGPT_Base out ODD 657/761
# ChatGPT_Base total match 951/1500

# Llama_Base == 167/1500
# Llama_Base in ODD 148/739
# Llama_Base out ODD 607/761
# Llama_Base total match 755/1500

# Vicuna_Base == 577/1500
# Vicuna_Base in ODD 573/739
# Vicuna_Base out ODD 111/761
# Vicuna_Base total match 684/1500
