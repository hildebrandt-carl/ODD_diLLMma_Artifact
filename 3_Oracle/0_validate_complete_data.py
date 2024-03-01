import os
import glob
import math
import argparse


from tqdm import tqdm

from oracle_utils import get_h5_length
from oracle_utils import get_video_length


# Get the folders
parser = argparse.ArgumentParser(description="Identified which data is complete")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")
# Get the dataset directory
DATASET_DIRECTORY = args.dataset_directory

OPENPILOT_CONTROL_RATE = 5

# List the videos
datasets = [d for d in os.listdir(DATASET_DIRECTORY) if os.path.isdir(os.path.join(DATASET_DIRECTORY, d))]
datasets = sorted(datasets)
print(f"Found {len(datasets)} datasets:")
for i, dataset in enumerate(datasets):
    print(f"{i}) {dataset}")

print("\n=====================")
for dataset in datasets:
    print(f"Processing: {dataset}")

    # Find the number of videos found
    video_files = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/1_ProcessedData/*.mp4")
    video_files = sorted(video_files)
    print("\t----------------------------")
    print(f"\tDataset has {len(video_files)} videos")

    # Get all the base video names
    base_filenames = [os.path.basename(video_file)[:-4] for video_file in video_files] 

    # Find all the openpilot versions this dataset was run
    # Get the list of versions
    version_path = f"{DATASET_DIRECTORY}/{dataset}/2_SteeringData"
    versions = [d for d in os.listdir(version_path) if os.path.isdir(os.path.join(version_path, d))]
    versions = sorted(versions)
    print(f"\tDataset has {len(versions)} OpenPilot versions")
    
    print("\t----------------------------")
    for version in versions:
        print(f"\tValidating all videos in version: {version}")

        h5_files = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/2_SteeringData/{version}/*.h5")
        h5_files = sorted(h5_files)
        warning_string = ""
        if len(h5_files) != len(video_files):
            warning_string = " -- WARNING: less data files than video files"
        print(f"\t\tFound {len(h5_files)}/{len(video_files)} data files {warning_string}")
    
        incomplete_videos = ""
        success = 0
        
        # Checking if each of the data files has data for each frame in the video
        print("\t\tValidating each datafile has data for each frame in the video")
        for filename in tqdm(base_filenames, desc="Comparing video and data files", leave=False):
            
            h5_f  = f"{DATASET_DIRECTORY}/{dataset}/2_SteeringData/{version}/{filename}.h5"
            vid_f = f"{DATASET_DIRECTORY}/{dataset}/1_ProcessedData/{filename}.mp4"

            video_length = get_video_length(vid_f)
            h5_length    = get_h5_length(h5_f)
            
            if math.floor(h5_length/OPENPILOT_CONTROL_RATE) != video_length:
                incomplete_videos += f"\t\tWARNING: {h5_f} is not the expected length {math.floor(h5_length/OPENPILOT_CONTROL_RATE)}/{video_length}\n"
            else:
                success += 1
        
        if len(incomplete_videos) > 0:
            print(incomplete_videos[:-1])
        print(f"\t\t{success}/{len(base_filenames)} had the excepted number of data points in them")


        print("\t----------------------------")
    print("\n=====================")
