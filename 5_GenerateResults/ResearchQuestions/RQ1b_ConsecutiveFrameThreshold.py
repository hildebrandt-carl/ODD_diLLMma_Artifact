import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations

# Get the folders
parser = argparse.ArgumentParser(description="Used to identify which scenarios pass and which fail")
parser.add_argument('--dataset',
                    type=str,
                    required=True,
                    choices=["OpenPilot_2k19", "External_jutah", "OpenPilot_2016"],
                    help="The dataset you want to process (OpenPilot_2k19, External_jutah, OpenPilot_2016)")
parser.add_argument('--failure_threshold',
                    type=int,
                    default="45",
                    help="The threshold of which you define a failure")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")
# Get the dataset directory
DATASET_DIRECTORY = args.dataset_directory

# Get the list of databases
datasets = os.listdir(DATASET_DIRECTORY)
assert args.dataset in datasets, f"The dataset was not found in `{DATASET_DIRECTORY}`"

# Get the pass fail from each versions
steering_file_paths = sorted(glob.glob(f"{DATASET_DIRECTORY}/{args.dataset}/3_OracleData/PassFailRepresentation/*.txt"))

# Create the x-array
x_values = np.arange(1, 200)
y_values = None

# For each of the files
for file in steering_file_paths:

    # Holds the versions
    versions_tag = []
    versions_name = []

    # Count the number of lines
    file_start = False
    line_count = 0
    with open(file, 'r') as f:
        for line in f:
            # Get the versions
            if "v" in line and not file_start:
                versions_tag.append(line[:line.find(")")].strip())
                versions_name.append(line[line.find(")")+2:].strip())
            # If we have started count lines
            if file_start:
                line_count += 1
            # Check if we have started
            if "----------------" in line:
                file_start = True

    # Create an array to hold the data
    steering_data = np.full((len(versions_name), line_count), np.nan, dtype=np.float64)

    # Populate steering
    file_start = False
    line_count = 0
    with open(file, 'r') as f:
        for line in f:

            if file_start:
                # Save steering angles
                for ver_index, ver in enumerate(versions_tag):
                    current_portion = line[line.find(ver):].strip()
                    steering_angle_str = current_portion[current_portion.find(":")+1:current_portion.find(")")]
                    steering_data[ver_index,line_count] = float(steering_angle_str)

                # Increment line
                line_count += 1

            # Check if we have started
            if "----------------" in line:
                file_start = True

    # Compute the differences between all combinations
    index_combinations_obj = combinations(range(len(versions_tag)), 2)
    index_combinations = [combination for combination in index_combinations_obj]

    # Create the Y values based on the number of combinations
    if y_values is None:
        y_values = np.zeros((len(index_combinations), len(x_values)))

    # For each of the combinations
    for comb_index, comb in enumerate(index_combinations):
        i1, i2 = comb
        # Compute the difference
        error_array = np.abs(steering_data[i1] - steering_data[i2])
        # Populate the y_values
        for x_index, x in enumerate(x_values):
            np.all(np.lib.stride_tricks.sliding_window_view(error_array, x) > args.failure_threshold, axis=1).sum()
            # Step 1: Create sliding windows of size x
            sliding_windows = np.lib.stride_tricks.sliding_window_view(error_array, x)
            # Step 2: Apply the threshold condition
            windows_above_threshold = np.all(sliding_windows > args.failure_threshold, axis=1)
            # Step 3: Sum up the results to get the count
            # Second Y value is for all files
            y_values[comb_index, x_index] = y_values[comb_index, x_index] + windows_above_threshold.sum()

# Get the labels
labels = []
for i1, i2 in index_combinations:
    labels.append(f"{versions_name[i1]} - {versions_name[i2]}")

# Creating the plot
plt.figure(figsize=(10, 6))
for i in range(np.shape(y_values)[0]):
    plt.semilogy(x_values, y_values[i], label=labels[i])

# Plot the average
plt.semilogy(x_values, np.nanmean(y_values,axis=0), label="Average", linestyle='--', c="Black")

plt.xlabel('Oracle Consecutive Frame Failure Threshold (# Images)')
plt.ylabel('Number of Failing Images')
plt.legend()
plt.grid(True)
plt.show()