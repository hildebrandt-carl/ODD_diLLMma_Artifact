import os
import cv2
import glob
import argparse

import numpy as np

from tqdm import tqdm


def distribute_items(count, total_items):
    """
    Distributes a specified total number of items across multiple different counts proportionally 
    based on the sizes of the counts. If there are residual items due to rounding, 
    they are allocated starting with the smallest counts.
    
    Parameters:
    - count (numpy.ndarray): A NumPy array where each element represents the number of items 
                             in a respective video.
    - total_items (int): An integer indicating the total number of items to be distributed 
                         across the videos.
    
    Returns:
    - numpy.ndarray: A NumPy array with the same shape as `count` where each element indicates 
                     the number of items allocated to the corresponding dataset.
    
    Example:
    Given datasets with counts [10, 5, 8, 3, 9, 7, 2] and a total of 15 items to distribute, 
    the function returns the allocation [3 2 2 2 3 2 1].
    """

    total_count = np.sum(count)
    
    # Proportional allocation
    allocation = np.floor((count / total_count) * total_items).astype(int)

    # Handle rounding
    remaining_items = total_items - np.sum(allocation)
    for i in np.argsort(count):
        if remaining_items <= 0:
            break
        if allocation[i] < count[i]:
            allocation[i] += 1
            remaining_items -= 1

    return allocation

# Get user input
parser = argparse.ArgumentParser(description="Select passing and failing tests evenly distributed out of a dataset")
parser.add_argument('--dataset',       type=str, default="", help="The dataset you want to process (OpenPilot_2k19, External_jutah)")
parser.add_argument('--total_passing', type=int, default=50, help="The number of passing scenarios you want")
parser.add_argument('--total_failing', type=int, default=50, help="The number of failing scenarios you want")
args = parser.parse_args()

# Decare the dataset directory
DATASET_DIRECTORY = "../1_Datasets/Data"

# Get the files
files = glob.glob(f"{DATASET_DIRECTORY}/{args.dataset}/3_PassFail/*.txt")
files = sorted(files)
assert len(files) > 0, "No files were found!"

# Create an array to keep track of failures and passing
failure_count = np.zeros(len(files))
passing_count = np.zeros(len(files))

# For each file
for i, f in enumerate(files):

    # Read the file
    with open(f, "r") as file:

        # Get the number of passing and failing tests
        number_of_failures = -1
        number_of_passing  = -1

        # For each line in the file
        for line in file:
            line = line.strip()

            if "Failures:" in line:
                number_of_failures = int(line[10:])

            if "Passing:" in line:
                number_of_passing = int(line[9:])

            # Break when we reach the end
            if "------" in line:
                break
        
    # Save the number of passing and failing
    failure_count[i] = number_of_failures
    passing_count[i] = number_of_passing

# Make sure we have enough failures and passing
assert args.total_failing <= np.sum(failure_count), "Requesting more failures than there are total failures"
assert args.total_passing <= np.sum(passing_count), "Requesting more passing than there are total passing"

# Using this we want to equally spread out the number of items selected from each folder
failing_selection_distribution = distribute_items(failure_count, args.total_failing)
passing_selection_distribution = distribute_items(passing_count, args.total_passing)

# Used to keep track of the output count
total_fail_counter = 0
total_pass_counter = 0

# Loop through each file and then sample based on the distribution
for i, f in enumerate(tqdm(files)):

    # Generate the indices we want to get
    if failing_selection_distribution[i] == 0:
        failing_step_size = np.nan
    else:
        failing_step_size = int(failure_count[i] // failing_selection_distribution[i])
    if passing_selection_distribution[i] == 0:
        passing_step_size = np.nan
    else:
        passing_step_size = int(passing_count[i] // passing_selection_distribution[i])

    # Keeps track of the failing and passing frames we want
    failing_frames = []
    passing_frames = []

    # Read the files
    with open(f, "r") as file:  
    
        # Set the index counter to 0
        failure_index_counter = 0
        passing_index_counter = 0

        # For each line in the file
        for line in file:

            # Remove whitespace
            line = line.strip()

            # Check if this line is a frame
            if "Frame " in line:

                # Check if its passing or failing
                frame_id        = line[6:line.find(")")]
                pass_fail_check = line[line.rfind("):")+3:]

                # Check if its passing or failing
                if pass_fail_check == "Fail":

                    # Ignore failures is failing_step_size is nan
                    if not np.isnan(failing_step_size):
                        # See if we need to save this to the failing frames
                        if failure_index_counter == 0:
                            failure_index_counter = failing_step_size
                            if len(failing_frames) < failing_selection_distribution[i]:
                                failing_frames.append(int(frame_id))

                        # Increment the counter
                        failure_index_counter -= 1

                elif pass_fail_check == "Pass":
                    # Ignore passes is passing_step_size is nan
                    if not np.isnan(passing_step_size):
                        # See if we need to save this to the passing frames
                        if passing_index_counter == 0:
                            passing_index_counter = passing_step_size
                            if len(passing_frames) < passing_selection_distribution[i]:
                                passing_frames.append(int(frame_id))

                        # Increment the counter
                        passing_index_counter -= 1

    # Open the video file
    video_file_name = os.path.basename(f)
    video_file_name = video_file_name[:-4] + ".mp4"
    cap = cv2.VideoCapture(f"{DATASET_DIRECTORY}/{args.dataset}/1_ProcessedData/{video_file_name}")

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Initialize a frame counter
    frame_counter = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame was not retrieved, then we have reached the end of the video
        if not ret:
            break

        # Check if the current frame counter is in either list and save it
        if frame_counter in failing_frames:
            formatted_counter = str(total_fail_counter).zfill(4)
            cv2.imwrite(f"{DATASET_DIRECTORY}/{args.dataset}/4_SelectedData_{args.total_passing + args.total_failing}/fail_{formatted_counter}.png", frame)
            total_fail_counter += 1

        if frame_counter in passing_frames:
            formatted_counter = str(total_pass_counter).zfill(4)
            cv2.imwrite(f"{DATASET_DIRECTORY}/{args.dataset}/4_SelectedData_{args.total_passing + args.total_failing}/pass_{formatted_counter}.png", frame)
            total_pass_counter += 1

        # Increment the frame counter
        frame_counter += 1

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()