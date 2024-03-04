import os
import sys
import glob
import operator
import argparse

import numpy as np

from tqdm import tqdm


current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from data_loader import DataLoader


def find_non_overlapping_sequences(difference_array, threshold, length, comparison_operator):
    # Apply the comparison operator to the entire array
    condition_met = comparison_operator(difference_array, threshold)
    
    start_indices = []
    consecutive_count = 0
    
    for i in range(len(condition_met)):
        if condition_met[i]:
            consecutive_count += 1
            if consecutive_count == length:
                start_indices.append(i - length + 1)
                consecutive_count = 0
        else:
            consecutive_count = 0
    
    return start_indices



# Get the folders
parser = argparse.ArgumentParser(description="Used to identify which scenarios pass and which fail")
parser.add_argument('--failing_deg',
                    type=int,
                    default=90,
                    help="Any difference greater than this is considered a failure")
parser.add_argument('--passing_deg',
                    type=int,
                    default=1,
                    help="Any difference less than this is considered a pass")
parser.add_argument('--length',
                    type=int,
                    default=1,
                    help="Total frames considered for detected failures / passing")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")
# Get the dataset directory
DATASET_DIRECTORY = args.dataset_directory

# Get all video files
video_file_paths = glob.glob(f"{DATASET_DIRECTORY}/*/1_ProcessedData/*.mp4")
video_filenames = [os.path.basename(v)[:-4] for v in video_file_paths]
video_filenames = sorted(video_filenames)

passing_images = []
failing_images = []

for video_filename in tqdm(video_filenames, ):
    dl = DataLoader(filename=video_filename)
    dl.validate_h5_files()
    dl.load_data()

    readings = dl.readings
    readings_clipped = np.clip(readings, -90, 90)

    # Find the passing
    max_steering = np.max(readings_clipped, axis=0)
    min_steering = np.min(readings_clipped, axis=0)
    steering_difference = np.abs(max_steering - min_steering)

    failing_frame_ids = find_non_overlapping_sequences(steering_difference, args.failing_deg, args.length, operator.gt)
    passing_frame_ids = find_non_overlapping_sequences(steering_difference, args.passing_deg, args.length, operator.le)
    print(f"{video_filename} has: {np.shape(passing_frame_ids)[0]}/{np.shape(steering_difference)[0]} passing images")
    print(f"{video_filename} has: {np.shape(failing_frame_ids)[0]}/{np.shape(steering_difference)[0]} failing images")

    for pass_id in passing_frame_ids:
        passing_images.append((video_filename, pass_id))

    for fail_id in failing_frame_ids:
        failing_images.append((video_filename, fail_id))



print("===========================================")
print(f"Found {len(passing_images)} passing images")
print(f"Found {len(failing_images)} failing images")




