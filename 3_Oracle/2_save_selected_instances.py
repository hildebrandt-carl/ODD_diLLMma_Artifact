import os
import cv2
import glob
import random
import argparse
import itertools

import numpy as np

from tqdm import tqdm
from collections import defaultdict


def read_pass_fail_file(filename):
    # Initialize empty lists for passing and failing IDs
    passing_ids = []
    failing_ids = []

    # Use a variable to track the current section
    current_section = None

    # Open the file for reading
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line == "Passing FrameIDs":
                current_section = 'passing'
            elif line == "Failing FrameIDs":
                current_section = 'failing'
            elif line in ("================", ""):
                continue  # Ignore separators and empty lines
            else:
                # Add the ID to the correct list based on the current section
                if current_section == 'passing':
                    passing_ids.append(int(line))
                elif current_section == 'failing':
                    failing_ids.append(int(line))

    return passing_ids, failing_ids

def save_images_from_video(video_dir, video_name, failing_frame_ids, passing_frame_ids, save_directory):
    max_index = -1
    # Ensure lists are not empty to avoid ValueError from np.max
    if failing_frame_ids and passing_frame_ids:
        max_index = np.max([np.max(failing_frame_ids), np.max(passing_frame_ids)])
    elif failing_frame_ids:  # Only failing_frame_ids is non-empty
        max_index = np.max(failing_frame_ids)
    elif passing_frame_ids:  # Only passing_frame_ids is non-empty
        max_index = np.max(passing_frame_ids)
    else:
        return 
    
    # Open the video file
    cap = cv2.VideoCapture(f"{video_dir}/1_ProcessedData/{video_name}.mp4")

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    current_frame_id = 0
    total_saved_images = 0

    for current_frame_id in tqdm(range(max_index+1), desc="Parsing Video", total=max_index, leave=False, position=1):
        # Read the next frame from the video
        ret, frame = cap.read()
        # If the frame was not retrieved, we've reached the end of the video
        if not ret:
            print("Error: Not all images were saved")
            break

        # Check if the current frame ID matches the target frame ID
        if current_frame_id in failing_frame_ids:
            save_path = f'{save_directory}/fail_{video_name}_{current_frame_id}.png'
            cv2.imwrite(save_path, frame)
            total_saved_images += 1

        if current_frame_id in passing_frame_ids:
            save_path = f'{save_directory}/pass_{video_name}_{current_frame_id}.png'
            cv2.imwrite(save_path, frame)
            total_saved_images += 1

        if total_saved_images == len(failing_frame_ids) + len(passing_frame_ids):
            break
    
    # Release the video capture object
    cap.release()


# Get the folders
parser = argparse.ArgumentParser(description="Saves the pass and fail images")
parser.add_argument('--total_images',
                    type=int,
                    default=500,
                    help="The total number of images to select")
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

# Set the seed
random.seed(42)

print("")
# Get the dataset directory
DATASET_DIRECTORY = f"{args.dataset_directory}/{args.dataset}"

# Make sure the right number of images was selected
if args.total_images % 2 != 0:
    print("ERROR: The total number of images must be an even number.")
    exit()

# Get all the pass fail files for that dataset
pass_fail_files = glob.glob(f"{DATASET_DIRECTORY}/3_PassFail/*.txt")

# Holds each of the passing and failing images
passing_images = {}
failing_images = {}

# Loop through each file and get the frame ID's and save them
for f in pass_fail_files:
    # Get the video name from the pass_fail file
    video_name = os.path.basename(f)[:-4]
    # Get the passing and failing IDs
    passing_ids, failing_ids = read_pass_fail_file(f)
    # Save each of the ID's
    passing_images[video_name] = (passing_ids)
    failing_images[video_name] = (failing_ids)

# Prepare a list of keys (randomly order the keys)
keys = sorted(list(set(passing_images.keys()) | set(failing_images.keys())))
random.shuffle(keys)

# Create the list of images we are going to save
selected_passing_images = defaultdict(list)
selected_failing_images = defaultdict(list)
selected_passing_count  = 0
selected_failing_count  = 0

# Use itertools.cycle to cycle through the keys indefinitely
for key in itertools.cycle(keys):
    # Break the loop if we have selected enough items
    if (selected_passing_count >= args.total_images // 2) and (selected_failing_count >= args.total_images // 2):
        break
    # If we have not selected enough passing images
    if selected_passing_count < args.total_images // 2:
        if len(passing_images[key]) != 0:
            random_index = random.randint(0, len(passing_images[key]) - 1)
            selected_item = passing_images[key].pop(random_index)
            selected_passing_images[key].append(selected_item)
            selected_passing_count += 1
    # If we have not selected enough failing images
    if selected_failing_count < args.total_images // 2:
        if len(failing_images[key]) != 0:
            random_index = random.randint(0, len(failing_images[key]) - 1)
            selected_item = failing_images[key].pop(random_index)
            selected_failing_images[key].append(selected_item)
            selected_failing_count += 1

# Loop through the videos and save the images
for key in tqdm(keys, desc="Processing File", total=len(keys), position=0, leave=True):

    # Create the save directory
    save_dir = f"{DATASET_DIRECTORY}/4_SelectedData"
    os.makedirs(save_dir, exist_ok=True)

    # Save the images
    save_images_from_video(DATASET_DIRECTORY, key, selected_failing_images[key], selected_passing_images[key], save_dir)

