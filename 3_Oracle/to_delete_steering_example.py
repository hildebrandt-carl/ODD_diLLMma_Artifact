import os
import sys
import glob
import operator
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict

current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from data_loader import DataLoader
from constants import CLIPPING_DEGREE
from common_functions import find_non_overlapping_sequences


# Get the folders
parser = argparse.ArgumentParser(description="Used to identify which scenarios pass and which fail")
parser.add_argument('--dataset',
                    type=str,
                    choices=['External_Jutah', 'OpenPilot_2k19', 'OpenPilot_2016'],
                    required=True,
                    help="The dataset to use. Choose between 'External_Jutah', 'OpenPilot_2k19', or 'OpenPilot_2016'.")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")
# Get the dataset directory
DATASET_DIRECTORY = f"{args.dataset_directory}/{args.dataset}"

# Get all files
video_file_paths = glob.glob(f"{DATASET_DIRECTORY}/1_ProcessedData/*.mp4")
video_filenames = [os.path.basename(v)[:-4] for v in video_file_paths]
video_filenames = sorted(video_filenames)

# Compute the total number of frames,
total_frames         = 0

data = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

for video_filename in tqdm(video_filenames, desc="Processing Video", leave=False, position=0, total=len(video_filenames)):
    # Load the data
    dl = DataLoader(filename=video_filename)
    dl.validate_h5_files()
    dl.load_data()

    # Get the readings
    readings = dl.readings

    # Clip the readings between -90 and 90
    readings_clipped = np.clip(readings, -CLIPPING_DEGREE, CLIPPING_DEGREE)

    # Identify the first steering angle
    first_steering_index = np.argmax(np.any(readings_clipped != 0, axis=0))
    valid_steering =  readings_clipped[:, first_steering_index:]

    # Compare specific versions
    for i, versions in enumerate([(0,1), (0,2), (1,2), (0,1,2)]):
        considered_steering = valid_steering[list(versions)]

        max_steering = np.max(considered_steering, axis=0)
        min_steering = np.min(considered_steering, axis=0)

        difference = max_steering - min_steering

        # Count the number of instances that have steering differences greater than 5 degrees
        for j, max_difference in enumerate([5, 10, 15, 20, 25]): 

            count_exceeds_threshold = np.sum(difference > max_difference)
            data[j, i] = data[j, i] + count_exceeds_threshold
    
# Sample data
n_groups = 5
group1 = data[:, 0]
group2 = data[:, 1]
group3 = data[:, 2]
group4 = data[:, 3]

# Create an index for each tick position
index = np.arange(n_groups)
bar_width = 0.2

# Create a bar for each group
plt.bar(index, group1, bar_width, label='V1, V2')
plt.bar(index + bar_width, group2, bar_width, label='V1, V3')
plt.bar(index + 2 * bar_width, group3, bar_width, label='V2, V3')
plt.bar(index + 3 * bar_width, group4, bar_width, label='V1, V2, V3')

# Add labels, title, etc.
plt.xlabel('Maximum Difference')
plt.ylabel('Number of Violations')
plt.xticks(index + 1.5 * bar_width, ['5', '10', '15', '20', '25'])

# Adding a legend
plt.legend()

# Show the plot
plt.show()
