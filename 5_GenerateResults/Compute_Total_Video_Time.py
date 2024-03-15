import os
import sys
import cv2
import glob
import argparse

from tqdm import tqdm


# Import Common
current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)


from constants import DATASET_ORDER



def get_video_length(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Get the frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the length of the video in seconds
    length_seconds = total_frames / fps
    
    # Release the video capture object
    cap.release()
    
    return length_seconds

# Get the Data
parser = argparse.ArgumentParser(description="Displays the number of failure inducing inputs")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

# Get the dataset directory
DATASET_DIRECTORY = f"{args.dataset_directory}"

# Get all the available datasets
available_datasets_paths = glob.glob(f"{DATASET_DIRECTORY}/*")
available_datasets = [os.path.basename(dset) for dset in available_datasets_paths]
available_datasets = sorted(available_datasets, key=lambda x: DATASET_ORDER.get(x, float('inf')))

all_dataset_time = 0

# For each of the datasets
for dataset in tqdm(available_datasets, desc="Processing Dataset", leave=False, position=0):

    # Get all video files
    video_files = sorted(glob.glob(f"{DATASET_DIRECTORY}/{dataset}/1_ProcessedData/*.mp4"))
    

    total_time = 0

    # For each video
    for video_file in tqdm(video_files, desc="Processing Video", leave=False, position=1):

        # Open the video and determine how long it is
        video_time = get_video_length(video_file)
        # print(f"{os.path.basename(video_file)} is: {video_time} seconds")
        total_time += video_time

    # Print the length
    print(f"\n\nProcessing Dataset: {dataset}")
    print(f"Total Time in Seconds: {total_time}")

    hours = int(total_time) // 3600
    minutes = (int(total_time) % 3600) // 60
    seconds = int(total_time) % 60

    print(f"Total Time (HH:MM:SS): {hours:02d}:{minutes:02d}:{seconds:02d}\n\n")


    all_dataset_time += total_time


hours = int(all_dataset_time) // 3600
minutes = (int(all_dataset_time) % 3600) // 60
seconds = int(all_dataset_time) % 60

print("\n\n============================")
print(f"Total Time for all datasets in Seconds: {all_dataset_time}")
print(f"Total Time for all datasets (HH:MM:SS): {hours:02d}:{minutes:02d}:{seconds:02d}\n\n")
print("============================")