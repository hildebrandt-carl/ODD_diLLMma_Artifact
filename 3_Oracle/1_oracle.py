import os
import sys
import glob
import operator
import argparse

import numpy as np

from tqdm import tqdm
from collections import defaultdict


current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from data_loader import DataLoader
from common_functions import find_non_overlapping_sequences

# Get the folders
parser = argparse.ArgumentParser(description="Used to identify which scenarios pass and which fail")
parser.add_argument('--failing_deg',
                    type=int,
                    default=45,
                    help="Any difference greater than this is considered a failure")
parser.add_argument('--passing_deg',
                    type=int,
                    default=1,
                    help="Any difference less than this is considered a pass")
parser.add_argument('--length',
                    type=int,
                    default=1,
                    help="Total frames considered for detected failures / passing")
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

# Get all video files
video_file_paths = glob.glob(f"{DATASET_DIRECTORY}/1_ProcessedData/*.mp4")
video_filenames = [os.path.basename(v)[:-4] for v in video_file_paths]
video_filenames = sorted(video_filenames)

passing_images = defaultdict(list)
failing_images = defaultdict(list)

# Compute the total number of frames,
total_frames         = 0
total_failing_frames = 0
total_passing_frames = 0
total_unknown_frames = 0

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

    # Identify the passing and failing frame IDs
    failing_frame_ids = find_non_overlapping_sequences(steering_difference, args.failing_deg, args.length, operator.gt)
    passing_frame_ids = find_non_overlapping_sequences(steering_difference, args.passing_deg, args.length, operator.le)

    # Remove all frame ID's before the first steering angle
    failing_frame_ids = failing_frame_ids[failing_frame_ids >= first_steering_index]
    passing_frame_ids = passing_frame_ids[passing_frame_ids >= first_steering_index]

    # Track the number of frames
    current_total_images_count   = np.shape(steering_difference)[0]
    current_failing_images_count = np.shape(failing_frame_ids)[0] * args.length
    current_passing_images_count = np.shape(passing_frame_ids)[0] * args.length
    current_unknown_images_count = current_total_images_count - (current_failing_images_count + current_passing_images_count)
    total_frames                 += current_total_images_count
    total_failing_frames         += current_failing_images_count
    total_passing_frames         += current_passing_images_count
    total_unknown_frames         += current_unknown_images_count
    
    # Display the data
    print(f"{video_filename} has: {current_passing_images_count}/{current_total_images_count} passing images")
    print(f"{video_filename} has: {current_failing_images_count}/{current_total_images_count} failing images")
    print(f"{video_filename} has: {current_unknown_images_count}/{current_total_images_count} unknown images")

    # Save them into dictionaries
    for pass_id in passing_frame_ids:
        passing_images[video_filename].append(pass_id)
    for fail_id in failing_frame_ids:
        failing_images[video_filename].append(fail_id)

# Print some statistics
print(f"Statistics for {args.dataset}")
fail_count = 0
pass_count = 0
# Sum the fails and passing images
for video_filename in video_filenames:
    fail_count += len(failing_images[video_filename])
    pass_count += len(passing_images[video_filename])
print(f"Total Failing Instances: {fail_count}")
print(f"Total Passing Instances: {pass_count}")

print("")
print(f"Total Frames: {total_frames}")
print(f"Total Failing Frames: {total_failing_frames}")
print(f"Total Passing Frames: {total_passing_frames}")
print(f"Total Unknown Frames: {total_unknown_frames}")

# Create the save dir
save_dir = f"{DATASET_DIRECTORY}/3_PassFail"
os.makedirs(save_dir, exist_ok=True)
print(f"Saving to {save_dir}")

# Get all the video_filenames for each datasets
all_videos = set(passing_images.keys()) | set(passing_images.keys())

for video_filename in video_filenames:
    passing_ids = sorted(passing_images[video_filename])
    failing_ids = sorted(failing_images[video_filename])

    # Save them to the file
    with open(f'{save_dir}/{video_filename}.txt', 'w') as file:
        file.write(f"Passing FrameIDs\n")
        file.write(f"================\n")
        for item in passing_ids:
            file.write(f"{item}\n")
        file.write(f"================\n")
        file.write(f"Failing FrameIDs\n")
        file.write(f"================\n")
        for item in failing_ids:
            file.write(f"{item}\n")
        file.write(f"================\n")
