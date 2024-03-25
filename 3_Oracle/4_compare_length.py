import os
import sys
import glob
import math
import operator
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from matplotlib.ticker import FuncFormatter

current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from data_loader import DataLoader
from common_functions import format_func
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
                    default=300,
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

# Create the plot
percentage_figure   = plt.figure(figsize=(10, 6))
count_figure        = plt.figure(figsize=(10, 6))

# For each dataset
for dataset in available_datasets:
    print(f"Processing: {dataset}")

    # Get all video files
    video_file_paths = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/1_ProcessedData/*.mp4")
    video_filenames = [os.path.basename(v)[:-4] for v in video_file_paths]
    video_filenames = sorted(video_filenames)

    # Compute the total number of frames,
    failing_image_count     = defaultdict(int)
    total_frame_count       = defaultdict(int)
    possible_failure_count  = defaultdict(int)

    for video_filename in tqdm(video_filenames, desc="Processing Video", leave=True, position=0, total=len(video_filenames)):
        # Load the data
        dl = DataLoader(filename=video_filename)
        dl.validate_h5_files()
        dl.load_data(terminal_print=False)

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

            # Track the number of possible failures
            possible_failure_count[length] += int(math.floor(np.shape(steering_difference)[0] / length))

    # Extract keys and values sorted by key
    keys                        = sorted(failing_image_count.keys())
    time_array                  = []
    percentage_failure_array    = []
    count_failure_array         = []
    for key in keys:

        percentage_failures = (failing_image_count[key] / possible_failure_count[key]) * 100
        count_failure_array.append(failing_image_count[key])
        percentage_failure_array.append(percentage_failures)

        current_time = key / VIDEO_FPS
        time_array.append(current_time)

        print(f"Number Frames: {key}: Time {np.round(current_time,4):.4f}s: failures {failing_image_count[key]}: possible failures {possible_failure_count[key]} - {np.round(percentage_failures,4):.4f}%")
        

    # Plot each defaultdict as a line
    plt.figure(count_figure.number)
    plt.plot(time_array, count_failure_array, label=DATASET_NAMING[dataset], color=DATASET_COLOR[dataset], linewidth=5)
    plt.figure(percentage_figure.number)
    plt.plot(time_array, percentage_failure_array, label=DATASET_NAMING[dataset], color=DATASET_COLOR[dataset], linewidth=5)


# Adding titles and labels
plt.figure(count_figure.number)
plt.xlabel('Length (s)', size=20)
plt.ylabel('Number of Failures', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.yscale('log')
plt.tight_layout()
plt.grid()
plt.legend(fontsize=20)

plt.figure(percentage_figure.number)
plt.xlabel('Length (s)', size=20)
plt.ylabel('(%) Dataset Considered Failures', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.yscale('log')
plt.ylim([0.001,100])
main_ax = plt.gca()
main_ax.yaxis.set_major_formatter(FuncFormatter(format_func))
plt.tight_layout()
plt.grid()
plt.legend(fontsize=20)

plt.show()

# The increases in failures are because of the following. Imaging the failure array is:
# -------------------
# |1|2|3|4|5|6|7|8|9|
# |F|F|P|P|F|F|F|F|F|
# -------------------
# 
# If you want to find failures of varying length you would find:
# Length 1 -7 failure(s) out of 9 possible failure(s) -> 7/9 = 78%
# Length 2 -3 failure(s) out of 4 possible failure(s) -> 3/4 = 75%
# Length 3 -1 failure(s) out of 3 possible failure(s) -> 1/3 = 33%
# Length 4 -1 failure(s) out of 2 possible failure(s) -> 1/2 = 50%
# Length 5 -1 failure(s) out of 1 possible failure(s) -> 1/1 = 100%
