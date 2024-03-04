import os
import sys
import math
import glob
import argparse


from tqdm import tqdm

current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from data_loader import DataLoader


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
for dataset_index, dataset in enumerate(datasets):
    print(f"Dataset {dataset_index+1}) {dataset}")

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

    detected_versions = set()
    success = 0

    for filename in tqdm(base_filenames, desc="Processing Files", leave=False):
        # Load the data
        dl = DataLoader(filename=filename)

        # Compare the number of versions for each filename
        if len(detected_versions) == 0:
            detected_versions = set(dl.versions)
        else:
            if set(dl.versions) != detected_versions:
                print(f"\t\tWARNING: {filename} has {len(dl.versions)}/{len(detected_versions)} OpenPilot versions")

        # Validate that each of the files and versions have the same number of readings as the video
        validated = dl.validate_h5_files()
        if validated:
            success += 1

    print(f"\tDataset has {len(detected_versions)} OpenPilot versions")
    for version_index, version in enumerate(sorted(list(detected_versions))):
        print(f"\t\t Version {version_index+1}: {version}")
    print(f"\t{success}/{len(base_filenames)} had the excepted number of data points in them")


    print("\t----------------------------")
    print("\n=====================")
