
import cv2
import math
import h5py
import warnings

import numpy as np

from enum import Enum
from itertools import combinations


class ScenarioDefinition(Enum):
    UNKNOWN     = -1
    PASS        = 0
    FAIL        = 1


def compute_average_steering_angle_per_frame(video_file, steering_files):

    # Load the video and determine how many frames there are
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get the number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frame_count > 100, "Cant get the video frame count"
    cap.release()

    # Get the total number of keys in each of the h5 files
    total_keys = []
    for v_index in range(len(steering_files)):
        with h5py.File(steering_files[v_index], 'r') as file:
            # Get all the keys
            file_keys = [key for key in file.keys()]
            total_keys.append(len(file_keys))

    # Get how many keys in total there are
    key_length = np.max(total_keys)

    # Reading the h5 file:
    # The h5 file is stored using keys labeled '000000000_steering_angle', '000000001_steering_angle', '000000002_steering_angle'
    # Each of the keys has the following information
        # steering angle produced (index 0)
        # frame number (index 1)

    # Create an array to hold the steering angle for each version (can hold up to 10 angles per frame)
    raw_steering_angles = np.full((len(steering_files),frame_count+1, 10), np.nan, dtype=float)

    # Populate this array
    for ver_index in range(len(steering_files)):

        # Read the file
        h5_f = h5py.File(steering_files[ver_index], 'r')

        # For each key in the common set of keys
        for key_index in range(key_length):

            # Generate a key
            index = "{0:09d}_steering_angle".format(key_index)

            # Get the steering angles and frame numbers
            steering_angle, frame_number = read_index_from_h5(h5_f, index)

            # Populate the raw steering angles
            update_raw_array(raw_steering_angles, ver_index, frame_number, steering_angle)

        # Close the h5 file
        h5_f.close()
            
    # Compute the actual steering angles as the average for all steering angles for that frame
    with warnings.catch_warnings():
        # Filter out the "Mean of empty slice" warning from numpy 
        # We can ignore this as it means that one of the frames simply didn't get a reading, and will be nan in the final reading
        warnings.simplefilter("ignore", category=RuntimeWarning)
        final_steering_angles = np.nanmean(raw_steering_angles, axis=2)

    return final_steering_angles

def update_raw_array(arr, version_index, key_index, value):
    """
    Update the array 'arr' at the given 'index' with the 'value'.
    Finds the first occurrence of nan at the given index and replaces it.
    If no nan is found, prints a warning.
    """
    # Return if there is nothing to update
    if np.isnan(key_index) or np.isnan(value):
        return
    
    row = arr[version_index, key_index]
    # Find indices of nan in the row
    nan_idx = np.where(np.isnan(row))[0]  

    if len(nan_idx) == 0:
        print(f"Warning: No space left to update at index {version_index}, {key_index}")
    else:
        first_nan_idx = nan_idx[0]
        arr[version_index, key_index, first_nan_idx] = value

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

def calculate_endpoint(starting_point, length, rotation):
    radians = math.radians(90-rotation)
    x = int(starting_point[0] + length * math.cos(radians))
    y = int(starting_point[1] - length * math.sin(radians))
    return [x, y]

def create_visual_representation(steering_angles, version_names, in_video, out_video, out_res=(854, 480)):
    # Define constants
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    DATASET_COLORS = [(57,106,177), (218,124,48), (62,150,81), (107,76,154), (148,139,61)]

    # Open the input video
    cap = cv2.VideoCapture(in_video)

    # Check if the video has opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get the frame rate of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps == 15

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video, fourcc, fps, out_res)

    # Holds the frame ID
    frame_id = 0

    # Compute the arrow starting point
    arrow_point = (out_res[0] // 2, out_res[1])

    # Read and process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Resize the frame
            resized_frame = cv2.resize(frame, out_res)

            # Plot the steering angle for each version
            for ver_index in range(np.shape(steering_angles)[0]):
                steer_end = calculate_endpoint(arrow_point, 200, steering_angles[ver_index, frame_id])
                cv2.arrowedLine(resized_frame, arrow_point, steer_end, DATASET_COLORS[ver_index], 4, tipLength=0.3)
                cv2.putText(resized_frame, version_names[ver_index], [10, (ver_index * 30) + 30], FONT, FONT_SCALE, DATASET_COLORS[ver_index], FONT_THICKNESS)

            # Write the resized frame to the output video
            out.write(resized_frame)

            # Increment the frame ID
            frame_id += 1
        else:
            break

    # Release everything when the job is finished
    cap.release()
    out.release()

def determine_pass_fail(final_steering_angles, versions, failing_deg, passing_deg, output_diff_file):

    # Check the data
    assert np.shape(final_steering_angles)[0] == len(versions), "The versions and steering angle numpy array shape do not match"
    
    # Compute all combinations required
    index_combinations_obj = combinations(range(np.shape(final_steering_angles)[0]), 2)
    index_combinations = [combination for combination in index_combinations_obj]
    
    # Create the resulting numpy array
    differences = np.zeros((len(index_combinations), np.shape(final_steering_angles)[1]), dtype=np.float128)
    
    # Update the differences array
    for difference_index, combination_index in enumerate(index_combinations):
        i1 = combination_index[0]
        i2 = combination_index[1]
        error_array = np.abs(final_steering_angles[i1] - final_steering_angles[i2])
        differences[difference_index] = error_array

    # Create an array where we define pass fail
    pass_fail_results = np.full(np.shape(final_steering_angles)[1], ScenarioDefinition.UNKNOWN.value, np.int8)

    # We only start finding failures once all versions have started producing output
    initialized = False

    # Use the differences array to determine pass fail
    for frame_id in range(np.shape(final_steering_angles)[1]):

        # Check if all are producing some value
        if np.all(final_steering_angles[:, frame_id] != 0):
            initialized = True

        # OpenPilot outputs zeros for the first few frames and nan's when there are no readings
        if initialized:

            # Determine failures using the max_error
            max_error = np.nanmax(differences[:, frame_id])

            if max_error > failing_deg:
                pass_fail_results[frame_id] = ScenarioDefinition.FAIL.value
            elif max_error < passing_deg:
                pass_fail_results[frame_id] = ScenarioDefinition.PASS.value

    # Count occurrences of unknown, passing, and failing
    count_unknown = np.count_nonzero(pass_fail_results == ScenarioDefinition.UNKNOWN.value)
    count_failure = np.count_nonzero(pass_fail_results == ScenarioDefinition.FAIL.value)
    count_passing = np.count_nonzero(pass_fail_results == ScenarioDefinition.PASS.value)

    # Compute the mean, min, and max errors
    min_error = np.nanmin(differences, axis=0)
    avg_error = np.nanmean(differences, axis=0)
    max_error = np.nanmax(differences, axis=0)

    # Write the results to a file
    with open(f"{output_diff_file}", "w") as file:

        # Write statistics
        file.write(f"Failures: {count_failure}" + "\n")
        file.write(f"Passing: {count_passing}" + "\n")
        file.write(f"Unknowns: {count_unknown}" + "\n")
        for ver_index, ver in enumerate(versions):
            file.write(f"v{ver_index}) {ver}")
        file.write("-------------------" + "\n")

        # Loop through the data and output if a frame is passing or failing
        for k in range(np.shape(final_steering_angles)[1]):
            
            # This will be the output
            line_output = f"Frame {k}) "

            # Add each of the readings
            for ver_index, ver in enumerate(versions):
                line_output += f"(v{ver_index}:{np.round(final_steering_angles[ver_index][k],1)}) "

            # Add the min max and mean
            line_output += f"(mn:{np.round(min_error[k],1)}) "
            line_output += f"(av:{np.round(avg_error[k],1)}) "
            line_output += f"(mx:{np.round(max_error[k],1)}) "

            # Determine if it was a pass or fail
            if pass_fail_results[k] == ScenarioDefinition.UNKNOWN.value:
                line_output += ": Unknown"
            if pass_fail_results[k] == ScenarioDefinition.FAIL.value:
                line_output +=  ": Fail"
            if pass_fail_results[k] == ScenarioDefinition.PASS.value:
                line_output += ": Pass"

            # Output this to a text file
            file.write(f"{line_output}\n")