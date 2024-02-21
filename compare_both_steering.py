import os
import cv2
import glob
import h5py
import argparse
import numpy as np

from tqdm import tqdm
from collections import defaultdict

def get_total_keys_in_results(f):
    h5 = h5py.File(f, 'r')
    total_keys = len(h5.keys())
    h5.close()
    return total_keys

def get_total_frames_in_video(f):
    # Read the video files to determine how many keys there are:
    cap = cv2.VideoCapture(f)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get the number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert frame_count > 100, "Cant get the video frame count"
    cap.release()

    return frame_count

def read_index_from_h5(h5, index, i):
    try:
        if i == 0:
            str_index = "{0:09d}_steering_angle".format(key_index)
            data           = h5.get(str_index)
            message = ""
        elif i == 1:
            str_index = "{0:09d}_data".format(index)
            data           = h5.get(str_index)
            data = data[0]
            message = str(data[2]).strip()
        steer          = float(data[0])
        frame_number   = int(data[1])
    except Exception as e:
        steer          = np.nan
        frame_number   = np.nan
        message        = ""
    
    return steer, frame_number, message



parser = argparse.ArgumentParser(description="Identifies the closest image.")
parser.add_argument('--dataset',
                    type=str,
                    default="")
parser.add_argument('--version',
                    type=str,
                    default="")
args = parser.parse_args()

dataset = args.dataset
version = args.version

# Holds the paths
old_new     = ["Paper",
               "Dissertation"]
files_paths = [f"/mnt/extradrive3/PaperResults/{dataset}/2_SteeringData/{version}",
               f"/mnt/extradrive3/DissertationResults/{dataset}/2_SteeringData/{version}"]
video_paths = [f"/mnt/extradrive3/PaperResults/{dataset}/1_ProcessedData",
               f"/mnt/extradrive3/DissertationResults/{dataset}/1_ProcessedData"]

open_pilot_control_rate = 5

# Get all the file names and video names
full_path_files = []
full_path_videos = []
for fp in files_paths:
    full_path_files += glob.glob(f"{fp}/*.h5")
for vp in video_paths:
    full_path_videos += glob.glob(f"{vp}/*.mp4")

# Keep track of data we are interested in
ran_to_completion               = [0, 0]
ran_without_calibration_errors  = [0, 0]
total_within_passing_all        = 0
total_within_unknown_all        = 0
total_within_failing_all        = 0
total_within_all                = 0
total_within_passing_calibration_success        = 0
total_within_unknown_calibration_success        = 0
total_within_failing_calibration_success        = 0
total_within_calibration_success                = 0
differences_all = []
differences_calibration_success = []

comparable_videos_completion = 0
comparable_videos_without_completion = 0

# Convert to just basename
files = [os.path.basename(x) for x in full_path_files]
videos = [os.path.basename(x) for x in full_path_videos]
    
# Get the unique filenames and video names
files = sorted(list(set(files)))
videos = sorted(list(set(videos)))

assert(len(files) == len(videos))

output_file_path = f"{dataset}_{version}.txt"
output_file = open(output_file_path, 'w')

# Go through each of the files
for filename in tqdm(files):

    basename = filename[:-3]
    output_file.write(f"Processing: {basename}\n\n")

    # Holds the steering anlges
    steering_angles = [None, None]
    messages = [None, None]
    total_keys = [None, None]

    # For old and new
    for i in range(2):

        # Load the video files
        total_video_keys = -1
        current_video_file = f"{video_paths[i]}/{basename}.mp4"
        if os.path.exists(current_video_file):
            total_video_keys = get_total_frames_in_video(current_video_file)
        else:
            output_file.write(f"{old_new[i]} file not found: {current_video_file}\n")
            output_file.write("\n")
            continue

        # Get the total number of keys in the result file
        total_results_keys = -1
        current_results_file = f"{files_paths[i]}/{basename}.h5"
        if os.path.exists(current_results_file):
            total_results_keys = get_total_keys_in_results(current_results_file)
            total_keys[i] = total_results_keys
        else:
            output_file.write(f"{old_new[i]} file not found: {current_results_file}\n")
            output_file.write("\n")
            continue
        
        # Read the file
        current_steering = np.zeros(total_results_keys)
        current_messages = defaultdict(int)
        h5 = h5py.File(current_results_file, 'r')    
        for key_index in range(total_results_keys):
            steer, frameid, msg = read_index_from_h5(h5, key_index, i)
            current_steering[key_index] = steer
            current_messages[msg] += 1
        steering_angles[i] = current_steering
        messages[i] = current_messages
        h5.close()

        # Add this info to the results file
        output_file.write(f"{old_new[i]} video keys: {total_video_keys}\n")
        output_file.write(f"{old_new[i]} results keys: {total_results_keys//open_pilot_control_rate}\n")
        output_file.write("\n")

        if total_video_keys == total_results_keys//open_pilot_control_rate:
            ran_to_completion[i] += 1

            if "b'Calibration Invalid'" not in current_messages.keys():
                ran_without_calibration_errors[i] += 1

    for i in range(2):
        output_file.write(f"{old_new[i]} messages returned: \n")
        if messages[i] is not None:
            for key in messages[i].keys():
                output_file.write(f"'{key}': {messages[i][key]} -- {np.round((messages[i][key]/total_keys[i]) * 100, 2)}%\n")
        output_file.write(f"\n")


    # Comparison
    output_file.write(f"Steering Comparison: \n")
    if (steering_angles[0] is None) or (steering_angles[1] is None):
        if (steering_angles[0] is None):
            output_file.write(f"{old_new[0]} steering is None: \n")
        if (steering_angles[1] is None):
            output_file.write(f"{old_new[1]} steering is None: \n")

    elif abs(np.shape(steering_angles[0])[0] - np.shape(steering_angles[1])[0]) >= open_pilot_control_rate:
        if np.shape(steering_angles[0])[0] > np.shape(steering_angles[1])[0]:
            output_file.write(f"{old_new[1]} results did not run correctly\n")
        if np.shape(steering_angles[1])[0] > np.shape(steering_angles[0])[0]:
            output_file.write(f"{old_new[0]} results did not run correctly\n")
    else:

        if np.shape(steering_angles[0])[0] > np.shape(steering_angles[1])[0]:
            steering_angles[0] = steering_angles[0][:np.shape(steering_angles[1])[0]]
        if np.shape(steering_angles[1])[0] > np.shape(steering_angles[0])[0]:
            steering_angles[1] = steering_angles[1][:np.shape(steering_angles[0])[0]]

        difference = np.abs(steering_angles[0]-steering_angles[1])
        passing_count =  np.sum(difference <= 1)
        failing_count =  np.sum(difference > 90)
        unknown_count =  np.sum((difference > 1) & (difference <= 90))
        output_file.write(f"Total frames: {total_keys[i]} -- {np.round((total_keys[i]/total_keys[i]) * 100, 2)}%\n")
        output_file.write(f"Total passing: {passing_count} -- {np.round((passing_count/total_keys[i]) * 100, 2)}%\n")
        output_file.write(f"Total unknowns: {unknown_count} -- {np.round((unknown_count/total_keys[i]) * 100, 2)}%\n")
        output_file.write(f"Total failing: {failing_count} -- {np.round((failing_count/total_keys[i]) * 100, 2)}%\n")
        output_file.write("\n")
        output_file.write(f"Max steering difference: {np.max(difference)}\n")
        output_file.write(f"Avg steering difference: {np.mean(difference)}\n")
        output_file.write(f"Med steering difference: {np.median(difference)}\n")
        output_file.write(f"Min steering difference: {np.min(difference)}\n")

        total_within_passing_all += passing_count
        total_within_unknown_all += unknown_count
        total_within_failing_all += failing_count
        total_within_all += total_keys[i]

        comparable_videos_completion += 1


        # Save the differences
        differences_all.append(difference)

        # Save the with calibration success
        if "b'Calibration Invalid'" not in messages[1].keys():
            total_within_passing_calibration_success += passing_count
            total_within_unknown_calibration_success += unknown_count
            total_within_failing_calibration_success += failing_count
            total_within_calibration_success += total_keys[i]
            differences_calibration_success.append(difference)
            comparable_videos_without_completion += 1

    output_file.write("\n\n==============================\n\n")

output_file.write("\n\n==============================\n")
output_file.write("==============================\n")
output_file.write("==============================\n")
output_file.write("==============================\n\n")
output_file.write("Ran to completion:\n")
output_file.write(f"{old_new[0]}: {ran_to_completion[0]}\n")
output_file.write(f"{old_new[1]}: {ran_to_completion[1]}\n")
output_file.write("\nRan without calibration errors:\n")
output_file.write(f"{old_new[0]}: {ran_without_calibration_errors[0]}\n")
output_file.write(f"{old_new[1]}: {ran_without_calibration_errors[1]}\n")
output_file.write("\nRan to completion comparison statistics:\n")
output_file.write(f"Comparable files which ran to completion: {comparable_videos_completion}\n")
output_file.write(f"Total frames: {total_within_all} -- {np.round((total_within_all/total_within_all) * 100, 2)}%\n")
output_file.write(f"Total passing: {total_within_passing_all} -- {np.round((total_within_passing_all/total_within_all) * 100, 2)}%\n")
output_file.write(f"Total unknowns: {total_within_unknown_all} -- {np.round((total_within_unknown_all/total_within_all) * 100, 2)}%\n")
output_file.write(f"Total failing: {total_within_failing_all} -- {np.round((total_within_failing_all/total_within_all) * 100, 2)}%\n")
output_file.write("\nRan to without calibration errors statistics:\n")
output_file.write(f"Comparable files without calibration errors statistics: {comparable_videos_without_completion}\n")
output_file.write(f"Total frames: {total_within_calibration_success} -- {np.round((total_within_calibration_success/total_within_calibration_success) * 100, 2)}%\n")
output_file.write(f"Total passing: {total_within_passing_calibration_success} -- {np.round((total_within_passing_calibration_success/total_within_calibration_success) * 100, 2)}%\n")
output_file.write(f"Total unknowns: {total_within_unknown_calibration_success} -- {np.round((total_within_unknown_calibration_success/total_within_calibration_success) * 100, 2)}%\n")
output_file.write(f"Total failing: {total_within_failing_calibration_success} -- {np.round((total_within_failing_calibration_success/total_within_calibration_success) * 100, 2)}%\n")
output_file.write("\nDebug Check:\n")
# Compute the differences
differences_all_array                   = np.concatenate(differences_all, axis=0)
differences_calibration_success_array   = np.concatenate(differences_calibration_success, axis=0)
output_file.write(f"Difference ran to completion length: {np.shape(differences_all_array)}\n")
output_file.write(f"Difference ran without calibration length: {np.shape(differences_calibration_success_array)}\n")
output_file.close()

# Convert to numpy array and clamp to 100
differences_all_array                   = np.clip(differences_all_array, 0, 100)
differences_calibration_success_array   = np.clip(differences_calibration_success_array, 0, 100)

# Generate the histogram of differences
import matplotlib.pyplot as plt
plt.figure()
plt.hist(differences_all_array, bins=100, color='C0', edgecolor='black')
plt.xlabel("Steering Difference")
plt.ylabel("Frequency")
plt.title("Ran to completion comparison steering differences")
plt.savefig(f"{output_file_path[:-4]}_completion.png")

plt.figure()
plt.hist(differences_calibration_success_array, bins=100, color='C0', edgecolor='black')
plt.xlabel("Steering Difference")
plt.ylabel("Frequency")
plt.title("Ran to without calibration errors steering differences")
plt.savefig(f"{output_file_path[:-4]}_calibration_error_free.png")