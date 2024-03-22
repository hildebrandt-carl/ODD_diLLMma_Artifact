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
from common_functions import find_non_overlapping_sequences

from constants import VIDEO_FPS
from constants import DATASET_ORDER
from constants import DATASET_COLOR
from constants import DATASET_NAMING


# Get the folders
parser = argparse.ArgumentParser(description="Looks at the percentage of data for varying lengths")
parser.add_argument('--failing_deg',
                    type=int,
                    default=45,
                    help="Any difference greater than this is considered a failure")
parser.add_argument('--max_length',
                    type=int,
                    default=150,
                    help="Total frames considered for detected failures / passing")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")

# Get the dataset directory
DATASET_DIRECTORY = f"{args.dataset_directory}"

# Get the datasets
available_datasets_paths = glob.glob(f"{DATASET_DIRECTORY}/*")
available_datasets_paths = [path for path in available_datasets_paths if os.path.isdir(path)]
available_datasets = [os.path.basename(dset) for dset in available_datasets_paths]
available_datasets = sorted(available_datasets, key=lambda x: DATASET_ORDER.get(x, float('inf')))

available_datasets = [available_datasets[0]]

# Create the plot
main_figure = plt.figure(figsize=(10, 6))

# For each dataset
for dataset in available_datasets:

    # Get all video files
    video_file_paths = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/1_ProcessedData/*.mp4")
    video_filenames = [os.path.basename(v)[:-4] for v in video_file_paths]
    video_filenames = sorted(video_filenames)

    # Compute the total number of frames,
    failing_image_count  = defaultdict(int)
    total_frame_count   = defaultdict(int)
    total_frames = 0

    for video_filename in tqdm(video_filenames, desc="Processing Video", leave=False, position=0, total=len(video_filenames)):
        # Load the data
        dl = DataLoader(filename=video_filename)
        dl.validate_h5_files()
        dl.load_data()

        # Get the readings
        readings = dl.readings

        # Clip the readings between -90 and 90
        readings_clipped = np.clip(readings, -90, 90)

        # Find the steering difference
        max_steering = np.max(readings_clipped, axis=0)
        min_steering = np.min(readings_clipped, axis=0)
        steering_difference = np.abs(max_steering - min_steering)

        # Identify the minimum steering angle
        first_steering_index = np.argmax(np.any(readings_clipped != 0, axis=0))

        # Vary the severity
        for length in range(1,args.max_length+1, 1):

            # Identify the passing and failing frame IDs
            failing_frame_ids = find_non_overlapping_sequences(steering_difference, args.failing_deg, length, operator.gt)

            # Remove all frame ID's before the first steering angle
            failing_frame_ids = failing_frame_ids[failing_frame_ids >= first_steering_index]

            # Track the number of frames
            current_total_images_count     = np.shape(failing_frame_ids)[0]
            failing_image_count[length] += current_total_images_count

    # Extract keys and values sorted by key
    keys = sorted(failing_image_count.keys())
    time_array = []
    failure_count_array = []
    for key in keys:
        percentage_failures = failing_image_count[key]
        failure_count_array.append(percentage_failures)
        current_time = key / VIDEO_FPS
        time_array.append(current_time)

    # Plot each defaultdict as a line
    plt.plot(time_array, failure_count_array, label=DATASET_NAMING[dataset], color=DATASET_COLOR[dataset], linewidth=5)

# Adding titles and labels
plt.xlabel('Length (s)', size=20)
plt.ylabel('Number of unique failures', size=20)

plt.xticks(size=20)
plt.yticks(size=20)
plt.yscale('log')

plt.tight_layout()
plt.grid()
plt.legend(fontsize=20)

plt.show()

# TODO: Sanity check this code