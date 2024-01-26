import os
import glob
import shutil
import argparse

from tqdm import tqdm
from scipy.stats import mode

import numpy as np

# Declare constants
DATASET_DIRECTORY = "../1_Datasets/Data"

# Define the questions and labels
ODD = [("q00", "Poor Visibility"),
       ("q01", "Image Obstructed"),
       ("q02", "Sharp Curve"),
       ("q03", "On-off Ramp"),
       ("q04", "Intersection"),
       ("q05", "Restricted Lane"),
       ("q06", "Construction"),
       ("q07", "Bright Light"),
       ("q08", "Narrow Road"),
       ("q09", "Hilly Road")]    
ODD_LENGTH = len(ODD)

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Performs clustering.")
parser.add_argument('--human_models', 
                    type=str,
                    default="",
                    help="The human model as a comma separated list use (research_a, research_b, researcher_c)")
parser.add_argument('--dataset',
                    type=str,
                    choices=['OpenPilot_2k19', 'External_jutah', 'OpenPilot_2016'],
                    help="The dataset you want to process (OpenPilot_2k19, External_jutah, OpenPilot_2016)")
parser.add_argument('--size',
                    type=int,
                    default=-1,
                    help="The size of the dataset you want to use")
parser.add_argument('--selection_strategy',
                    type=str,
                    choices=['all_match', 'majority_match', 'worst_case'],
                    default="all_match",
                    help="Either require all vectors to match, or have a majority match, or to select the worst case")
args = parser.parse_args()

# Make sure you have set a dataset size
assert args.size > 0, "Dataset size can not be less than or equal to 0"

# Get the list of databases
datasets = os.listdir(DATASET_DIRECTORY)
assert args.dataset in datasets, f"The dataset was not found in `{DATASET_DIRECTORY}`"

# Define the base path
BASE_PATH = f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions/{args.size}"

# Get the list of models
human_models = [f"human_{m}" for m in args.human_models.split(", ")]

# Make sure there are all the humans models
all_models = os.listdir(f"{BASE_PATH}")
for human_model in human_models:
    assert human_model in all_models, (
    f"{human_model} was not found in: {BASE_PATH}/"
)

# Get all the base files which have been processed
all_files = []
for human_model in human_models:
    files = glob.glob(f"{BASE_PATH}/{human_model}/q00/*_output.txt")
    basenames = [os.path.basename(name) for name in files]
    basenames = sorted(basenames)
    all_files.append(basenames)

# Compute the list all all common file names
common_files = set(all_files[0])
for lst in all_files[1:]:
    common_files &= set(lst)

# Make sure we have some data
assert len(common_files) > 0, f"There are no files in common"

# Turn into a list
common_files = sorted(list(common_files))

# Create the human dataset directory
for i in range(ODD_LENGTH+1):
    output_dir = f"{BASE_PATH}/human/q{i:02d}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Keep track of matching files
matching_files_count = 0

# Loop through all the files
for i, filename in enumerate(common_files):

    # Holds the vector data
    vector = np.full((len(human_models), ODD_LENGTH), -1)

    # For each of the files:
    for human_index, human_model in enumerate(human_models):

        # Load the vector
        for odd_index in range(10):
            file = f"{BASE_PATH}/{human_model}/q{odd_index:02d}/{filename}"
            # Load the first vector
            with open(file, "r") as f:
                # Read the data and convert it to upper case
                file_data = f.read()
                if file_data == "YES":
                    vector[human_index][odd_index] = 1
                elif file_data == "NO":
                    vector[human_index][odd_index] = 0
                else:
                    print("Something is wrong")

    # used to hold the finally selected vector
    selected_vector = None
    save_file = False

    if args.selection_strategy == "majority_match":
        # Calculate the mode (most common row) of the vector
        unique_rows, counts     = np.unique(vector, axis=0, return_counts=True)
        all_rows_match_majority = np.any(counts > len(vector) / 2)
        if all_rows_match_majority:
            majority_row_index = np.argmax(counts)
            selected_vector = unique_rows[majority_row_index]
            save_file = True
        
    if args.selection_strategy == "all_match":
        # Check if all rows match
        all_rows_match = not np.any(~np.all(vector == vector[0, :], axis=1))

        # Count matching files
        if all_rows_match:
            selected_vector = vector[0]
            save_file = True

    if args.selection_strategy == "worst_case":
        # Find the vector with the largest number of violations
        highest_index = np.argmax(np.sum(vector, axis=1))
        selected_vector = vector[highest_index]
        save_file = True

    # Display results for review
    if ("fail" in filename) and not save_file:
        print(filename)
        print(f"Matched: {save_file}")
        for human_index, human_model in enumerate(human_models):
            selected_items = [ODD[i] for i, v in enumerate(vector[human_index]) if v == 1]
            print(f"{human_model}: {selected_items}")
        print("")

    # Save the data from the matches
    if save_file:
        matching_files_count+=1
        # Save the output
        for odd_index in range(ODD_LENGTH):
            # Open the OR for writing
            with open(f"{BASE_PATH}/human/q{odd_index:02d}/{filename}", "w") as file:
                if selected_vector[odd_index] == 1:               
                    file.write("YES")
                elif selected_vector[odd_index] == 0:
                    file.write("NO")
                else:
                    print("Something went wrong")

        # Copy the text description from research_a
        source_description_path         = f"{BASE_PATH}/researcher_a/q10/{filename}"
        destination_description_path    = f"{BASE_PATH}/human/q10/{filename}"
        shutil.copy(source_description_path, destination_description_path)

print("======================================================")
print(f"Files that are common: {len(common_files)}/{args.size}")
print(f"Files that match: {matching_files_count}/{len(common_files)}")