import os
import cv2
import h5py
import glob
import warnings
import argparse

import numpy as np

from enum import Enum
from tqdm import tqdm


class ScenarioDefinition(Enum):
    UNKNOWN     = -1
    PASS        = 0
    FAIL        = 1


def update_raw_array(arr, index, value):
    """
    Update the array 'arr' at the given 'index' with the 'value'.
    Finds the first occurrence of nan at the given index and replaces it.
    If no nan is found, prints a warning.
    """
    # Return if there is nothing to update
    if np.isnan(index) or np.isnan(value):
        return
    
    row = arr[index]
    # Find indices of nan in the row
    nan_idx = np.where(np.isnan(row))[0]  

    if len(nan_idx) == 0:
        print(f"Warning: No space left to update at index {index}")
    else:
        first_nan_idx = nan_idx[0]
        arr[index, first_nan_idx] = value


def read_index_from_h5(f, index):
    """
    Extracts the steering angle and frame number from an HDF5 file at a given index.

    This function reads data from an HDF5 file 'f' at the specified 'index'. It aims to extract a steering 
    angle (as a float) and a frame number (as an int) from the data. If the data at the index doesn't fit 
    this format and causes a TypeError, both the steering angle and frame number are set to NaN (Not a Number).

    Parameters:
    f (h5py.File object): The HDF5 file to read from.
    index (str or int): The index/key in the HDF5 file where the data is located.

    Returns:
    tuple: A tuple containing the steering angle and frame number (both could be NaN if data is improperly formatted).
    """
    try:
        data = f.get(index)
        steer          = float(data[0])
        frame_number   = int(data[1])
    except TypeError as e:
        steer          = np.nan
        frame_number   = np.nan
    
    return steer, frame_number


# Declare the versions we are using
VERSION1 = "2022_04"
VERSION2 = "2023_03"

# Declare what we find an pass or failure
FAILURE    = 90
PASS       = 1

# Decare the dataset directory
DATASET_DIRECTORY = "../1_Datasets/Data"

# Get the folders
parser = argparse.ArgumentParser(description="Used to identify which scenarios pass and which fail")
parser.add_argument('--dataset',
                    type=str,
                    required=True,
                    help="The dataset you want to process (OpenPilot_2k19, External_jutah, OpenPilot_2016)")
args = parser.parse_args()

# Get the list of databases
datasets = os.listdir(DATASET_DIRECTORY)
assert args.dataset in datasets, f"The dataset was not found in `{DATASET_DIRECTORY}`"

# Get the steering angles from each file
version1_steering = sorted(glob.glob(f"{DATASET_DIRECTORY}/{args.dataset}/2_SteeringData/{VERSION1}/*.h5"))
version2_steering = sorted(glob.glob(f"{DATASET_DIRECTORY}/{args.dataset}/2_SteeringData/{VERSION2}/*.h5"))

# Track the worst error and totals
worst_error = 0
total_failures = 0 
total_passing = 0
total_unknown = 0

# Holds the matched files
matched_files = []

# Match the files
for i in range(len(version1_steering)):
    for j in range(len(version2_steering)):
        f1 = version1_steering[i]
        f2 = version2_steering[j]
        b1 = os.path.basename(f1)
        b2 = os.path.basename(f2)
        if b1 == b2:
            matched_files.append([f1, f2])

# Loop through all matched files
for files in tqdm(matched_files):

    # Get the filename to create the results file
    main_filename = os.path.basename(files[0])[:-3] + ".txt"
    video_filename = main_filename[:-4] + ".mp4"

    # Load the video and determine how many frames there are
    video_path = f"{DATASET_DIRECTORY}/{args.dataset}/1_ProcessedData/{video_filename}"
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get the number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Make sure you are getting a frame count
    assert frame_count > 100, "Cant get the video frame count"

    # Release the video capture object
    cap.release()

    # Create two arrays to hold the steering angle (multiple angles per frame)
    raw_steering_angle_version_1 = np.full((frame_count+1, 30), np.nan, dtype=float)
    raw_steering_angle_version_2 = np.full((frame_count+1, 30), np.nan, dtype=float)

    # Read the files
    f1 = h5py.File(files[0], 'r')
    f2 = h5py.File(files[1], 'r')

    # Get all the keys
    f1_keys = [key for key in f1.keys()]
    f2_keys = [key for key in f2.keys()]
    f1_keys = sorted(f1_keys)
    f2_keys = sorted(f2_keys)

    # Get how many keys in total there are
    key_length = np.max([len(f1_keys), len(f2_keys)])

    # Populate the raw steering angle
    for k in range(key_length):

        # Generate the index
        index = "{0:09d}_steering_angle".format(k)

        # Get the steering angles and frame numbers
        steer1, frame_number_1 = read_index_from_h5(f1, index)
        steer2, frame_number_2 = read_index_from_h5(f2, index)

        # Populate the raw steering angles
        update_raw_array(raw_steering_angle_version_1, frame_number_1, steer1)
        update_raw_array(raw_steering_angle_version_2, frame_number_2, steer2)
        
    # Compute the actual steering angles
    with warnings.catch_warnings():
        # Filter out the "Mean of empty slice" warning from numpy 
        # We can ignore this as it means that one of the frames simply didn't get a reading, and will be nan in the final reading
        warnings.simplefilter("ignore", category=RuntimeWarning)
        steering_angle_version_1 = np.nanmean(raw_steering_angle_version_1, axis=1)
        steering_angle_version_2 = np.nanmean(raw_steering_angle_version_2, axis=1)

    # Close the files
    f1.close()
    f2.close()

    # Compute the error between the two arrays
    error_array = np.abs(steering_angle_version_1 - steering_angle_version_2)

    # Create an array where we define pass fail
    pass_fail_results = np.full(frame_count, ScenarioDefinition.UNKNOWN.value, np.int8)

    # Wait until both open pilot versions have given some output
    initialized1 = False
    initialized2 = False

    # Loop through the whole array
    for k in range(frame_count):

        # Get the values
        s1 = steering_angle_version_1[k]
        s2 = steering_angle_version_2[k]
        er = error_array[k]

        # Check if they have started giving output
        if(s1 != 0):
            initialized1 = True
        if (s2 != 0):
            initialized2 = True

        # OpenPilot outputs zeros for the first few frames and nan's when there are no readings
        if (initialized1) and (initialized2) and (np.isnan(s1) == False and np.isnan(s2) == False):
            if er > FAILURE:
                pass_fail_results[k] = ScenarioDefinition.FAIL.value
            elif er < PASS:
                pass_fail_results[k] = ScenarioDefinition.PASS.value

    # Count occurrences of unknown, passing, and failing
    count_unknown = np.count_nonzero(pass_fail_results == ScenarioDefinition.UNKNOWN.value)
    count_failure = np.count_nonzero(pass_fail_results == ScenarioDefinition.FAIL.value)
    count_passing = np.count_nonzero(pass_fail_results == ScenarioDefinition.PASS.value)

    # Keep track of totals over the whole dataset
    total_unknown += count_unknown
    total_failures += count_failure
    total_passing += count_passing

    # Keep track of the worst error
    worst_error = np.nanmax([worst_error, np.nanmax(error_array)])

    # Write the results to a file
    with open(f"{DATASET_DIRECTORY}/{args.dataset}/3_PassFail/{main_filename}", "w") as file:

        # Write statistics
        file.write(f"Failures: {count_failure}" + "\n")
        file.write(f"Passing: {count_passing}" + "\n")
        file.write(f"Unknowns: {count_unknown}" + "\n")
        file.write("-------------------" + "\n")

        # Loop through the data and output if a frame is passing or failing
        for k in range(frame_count):
            # Get the data
            result = ""
            s1 = np.round(steering_angle_version_1[k],1)
            s2 = np.round(steering_angle_version_2[k],1)
            e  = np.round(error_array[k],1)

            # Determine if it was a pass or fail
            if pass_fail_results[k] == ScenarioDefinition.UNKNOWN.value:
                result = "Unknown"
            if pass_fail_results[k] == ScenarioDefinition.FAIL.value:
                result = "Fail"
            if pass_fail_results[k] == ScenarioDefinition.PASS.value:
                result = "Pass"

            # Output this to a text file
            file.write(f"Frame {k}) (s1:{s1}) (s2:{s2}) (e:{e}): {result}\n")

print("Completed")#
print(f"Worst error: {np.round(worst_error,1)} degrees")
print(f"Total unknowns: {total_unknown}")
print(f"Total passing: {total_passing}")
print(f"Total failing: {total_failures}")