import os
import glob
import time
import argparse

import numpy as np

from oracle_utils import determine_pass_fail
from oracle_utils import create_visual_representation
from oracle_utils import compute_average_steering_angle_per_frame


# Get the folders
parser = argparse.ArgumentParser(description="Used to identify which scenarios pass and which fail")
parser.add_argument('--dataset',
                    type=str,
                    required=True,
                    choices=["OpenPilot_2k19", "External_jutah", "OpenPilot_2016"],
                    help="The dataset you want to process (OpenPilot_2k19, External_jutah, OpenPilot_2016)")
parser.add_argument('--versions',
                    type=str,
                    required=True,
                    help="Comma separated list of the versions your want to use (2022_04, 2023_03, 2023_06)")
parser.add_argument('--failing_deg',
                    type=int,
                    default=90,
                    help="Any difference greater than this is considered a failure")
parser.add_argument('--passing_deg',
                    type=int,
                    default=1,
                    help="Any difference less than this is considered a pass")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")
# Get the dataset directory
DATASET_DIRECTORY = args.dataset_directory

# Make sure the passing and failing degrees make sense
assert args.failing_deg > args.passing_deg, "Failing degree must be larger than passing degree"
assert args.failing_deg > 0, "Failing degree must be larger than 0"
assert args.passing_deg > 0, "passing degree must be larger than 0"

# Get the list of databases
datasets = os.listdir(DATASET_DIRECTORY)
assert args.dataset in datasets, f"The dataset was not found in `{DATASET_DIRECTORY}`"

# Get the list of versions
versions = args.versions.split(',')
possible_versions = os.listdir(f"{DATASET_DIRECTORY}/{args.dataset}/2_SteeringData")
assert all(v in possible_versions for v in versions), f"One of the versions was not found in {DATASET_DIRECTORY}/{args.dataset}/2_SteeringData"

# Get the steering angles from each versions
steering_file_paths = [sorted(glob.glob(f"{DATASET_DIRECTORY}/{args.dataset}/2_SteeringData/{v}/*.h5")) for v in versions]
basenames = [[os.path.basename(single_file_path) for single_file_path in version_file_paths] for version_file_paths in steering_file_paths]

# Find the common subset for comparison
basename_sets = [set(b) for b in basenames]
common_basenames_set = set.intersection(*basename_sets)
common_basenames = sorted(list(common_basenames_set))

# Track the maximum error, mean error, minimum error
max_error  = 0
min_error  = np.inf
avg_error  = 0

# Track the total failures, passing, and unknown tests.
total_failures  = 0 
total_passing   = 0
total_unknown   = 0

# Loop through all common files
for f_index, f in enumerate(common_basenames):

    overall_start_time = time.time()
    print(f"Processing: {f} - {f_index}/{len(common_basenames)}")

    # Reconstruct the full filepath for steering and videos
    video_file     = f"{DATASET_DIRECTORY}/{args.dataset}/1_ProcessedData/{f[:-3]}.mp4"
    steering_files = [f"{DATASET_DIRECTORY}/{args.dataset}/2_SteeringData/{v}/{f}" for v in versions]

    # Get the steering angle per file
    start_time = time.time()
    final_steering_angles = compute_average_steering_angle_per_frame(video_file, steering_files)
    end_time = time.time()
    print(f"Computed average steering angles - Took {(end_time-start_time):.2f} seconds")

    # Display videos with arrows
    start_time = time.time()
    output_video_file = f"{DATASET_DIRECTORY}/{args.dataset}/3_OracleData/VisualRepresentation/{f[:-3]}.mp4"
    create_visual_representation(final_steering_angles, versions, video_file, output_video_file)
    end_time = time.time()
    print(f"Generated visual representation - Took {(end_time-start_time):.2f} seconds")

    # Save the pass fail file
    start_time = time.time()
    output_diff_file = f"{DATASET_DIRECTORY}/{args.dataset}/3_OracleData/PassFailRepresentation/{f[:-3]}.txt"
    determine_pass_fail(final_steering_angles, versions, args.failing_deg, args.passing_deg, output_diff_file)
    end_time = time.time()
    print(f"Generated pass fail representation- Took {(end_time-start_time):.2f} seconds")

    # Display how long it took
    overall_end_time = time.time()
    print(f"Processing: {f} Complete - Took {(overall_end_time-overall_start_time):.2f} seconds")
    print("")
    print("")