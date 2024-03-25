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

from constants import DATASET_ORDER
from constants import DATASET_COLOR
from constants import DATASET_NAMING


# Get the folders
parser = argparse.ArgumentParser(description="Looks at how many fail for a variety of steering differences")
parser.add_argument('--length',
                    type=int,
                    default=1,
                    help="Total frames considered for detected failures / passing")
parser.add_argument('--max_failing_deg',
                    type=int,
                    default=180,
                    help="Any difference greater than this is considered a failure")
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
        for severity in range(0,args.max_failing_deg+1, 1):

            # Identify the passing and failing frame IDs
            failing_frame_ids = find_non_overlapping_sequences(steering_difference, severity, args.length, operator.gt)

            # Remove all frame ID's before the first steering angle
            failing_frame_ids = failing_frame_ids[failing_frame_ids >= first_steering_index]

            # Track the number of frames
            current_total_images_count     = np.shape(failing_frame_ids)[0]
            failing_image_count[severity] += current_total_images_count

            # Track the number of possible failures
            possible_failure_count[severity] += int(math.floor(np.shape(steering_difference)[0] / args.length))

    # Extract keys and values sorted by key
    severity_array              = sorted(failing_image_count.keys())
    failure_percentage_array    = []
    failure_count_array         = []
    for sev in severity_array:
        percentage_failures = (failing_image_count[sev]/possible_failure_count[sev])*100
        print(f"Steering Difference: {sev} deg: Failures {failing_image_count[sev]}: possible failures {possible_failure_count[sev]} - {np.round(percentage_failures,4):.4f}%")
        failure_count_array.append(failing_image_count[sev])
        failure_percentage_array.append(percentage_failures)

    # Plot each defaultdict as a line
    plt.figure(count_figure.number)
    plt.plot(severity_array, failure_count_array, label=DATASET_NAMING[dataset], color=DATASET_COLOR[dataset], linewidth=5)
    plt.figure(percentage_figure.number)
    plt.plot(severity_array, failure_percentage_array, label=DATASET_NAMING[dataset], color=DATASET_COLOR[dataset], linewidth=5)


# Adding titles and labels
plt.figure(count_figure.number)
plt.xlabel('Steering Difference', size=20)
plt.ylabel('Number of Failures', size=20)
plt.xticks(np.arange(0, args.max_failing_deg+1, 20), size=20)
plt.yticks(size=20)
plt.yscale('log')
plt.tight_layout()
plt.grid()
plt.legend(fontsize=20)

plt.figure(percentage_figure.number)
plt.xlabel('Steering Difference', size=20)
plt.ylabel('(%) Dataset Considered Failures', size=20)
plt.xticks(np.arange(0, args.max_failing_deg+1, 20), size=20)
plt.yticks(size=20)
plt.yscale('log')
plt.ylim([0.001,100])
main_ax = plt.gca()
main_ax.yaxis.set_major_formatter(FuncFormatter(format_func))
plt.tight_layout()
plt.grid()
plt.legend(fontsize=20)

plt.show()