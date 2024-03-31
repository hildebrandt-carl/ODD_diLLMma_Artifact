import os
import cv2
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D


current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from data_loader import DataLoader
from constants import CLIPPING_DEGREE
from constants import OPENPILOT_COLORS
from constants import OPENPILOT_COLORS_RGB
from constants import OPENPILOT_NAMES
from visualization_functions import show_steering
from pass_fail_handler import read_pass_fail_file

# Get the folders
parser = argparse.ArgumentParser(description="Displays the entire process")
parser.add_argument('--video_file',
                    type=str,
                    required=True,
                    help="The name of the base video file")
parser.add_argument('--dataset',
                    type=str,
                    choices=['External_Jutah', 'OpenPilot_2k19', 'OpenPilot_2016'],
                    required=True,
                    help="The dataset to use. Choose between 'External_Jutah', 'OpenPilot_2k19', or 'OpenPilot_2016'.")
parser.add_argument('--length',
                    type=int,
                    default=1,
                    help="The length of the failures in the pass_fail file")
parser.add_argument('--window_size',
                    type=int,
                    default=150,
                    help="The number of frames used as a window in the display")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")
# Get the dataset directory
DATASET_DIRECTORY = f"{args.dataset_directory}/{args.dataset}"

# Load the video
video_path  = f"{DATASET_DIRECTORY}/1_ProcessedData/{args.video_file}.mp4"
file_exists = os.path.exists(video_path)

# Check if the file exists
if not file_exists:
    print("Error: Video file does not exist")
    exit()

# Load the pass fail file
pass_fail_file = f"{DATASET_DIRECTORY}/3_PassFail/{args.video_file}.txt"
file_exists = os.path.exists(pass_fail_file)

# Check if the file exists
if not file_exists:
    print("Error: Pass Fail file does not exist")
    exit()

# Get the passing and failing IDs
all_passing_ids, all_failing_ids = read_pass_fail_file(pass_fail_file)
# Update the failing and passing frame ids to include length
all_passing_ids = [fid + i for fid in all_passing_ids for i in range(args.length)]
all_failing_ids = [fid + i for fid in all_failing_ids for i in range(args.length)]

# Find all steering angle files
dl = DataLoader(filename=args.video_file)
dl.validate_h5_files()
dl.load_data()

# Load the steering angles
versions        = dl.versions
colors          = [OPENPILOT_COLORS_RGB[v] for v in versions]
op_versions     = [OPENPILOT_NAMES[v] for v in versions] 

# Open the video file.
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully.
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# This will hold the steering angles
steering_plot_window = np.zeros((np.shape(dl.readings)[0], args.window_size+1))
frame_id_window      = np.zeros(args.window_size+1)

passing_ids = []
failing_ids = []

# Create the plotting function
plt.ion()
fig, ax = plt.subplots(figsize=(17,8)) 

# Track the frame
frame_id = 0

# Read and display each frame of the video.
while True:
    # Read a new frame.
    ret, frame = cap.read()

    # Clear the plot
    ax.clear()

    # Get the current steering angles
    steering_angles = dl.readings[:,frame_id]

    # Update the steering angle plotting arrays
    steering_plot_window = np.roll(steering_plot_window,-1, axis=1)
    steering_plot_window[:,-1] = steering_angles
    frame_id_window = np.roll(frame_id_window, -1)
    frame_id_window[-1] = frame_id

    # Remove values outside range
    min_pf_index = int(np.min(frame_id_window))
    max_pf_index = int(np.max(frame_id_window))
    passing_ids = [p for p in all_passing_ids if min_pf_index <= p <= max_pf_index]
    failing_ids = [f for f in all_failing_ids if min_pf_index <= f <= max_pf_index]

    # Highlight continuous ranges and/or single points for passing
    if passing_ids:
        start_pass = passing_ids[0]
        for i in range(1, len(passing_ids)):
            if passing_ids[i] > passing_ids[i-1] + 1:  # Non-continuous sequence
                # Ensuring at least some width for single points for visibility
                end_pass = passing_ids[i-1] + 0.1 
                ax.axvspan(start_pass, end_pass, color='green', alpha=0.25)
                start_pass = passing_ids[i]
        # Final range or single point
        end_pass = passing_ids[-1] + 0.1 
        ax.axvspan(start_pass, end_pass, color='green', alpha=0.25)

    # Highlight continuous ranges and/or single points for failing
    if failing_ids:
        start_pass = failing_ids[0]
        for i in range(1, len(failing_ids)):
            if failing_ids[i] > failing_ids[i-1] + 1:  # Non-continuous sequence
                # Ensuring at least some width for single points for visibility
                end_pass = failing_ids[i-1] + 0.1  
                ax.axvspan(start_pass, end_pass, color='green', alpha=0.25)
                start_pass = failing_ids[i]
        # Final range or single point
        end_pass = failing_ids[-1] + 0.1  
        ax.axvspan(start_pass, end_pass, color='red', alpha=0.25)

    # Update the plot
    for i in range(np.shape(dl.readings)[0]):
        ax.plot(frame_id_window, steering_plot_window[i,:], label=OPENPILOT_NAMES[versions[i]], c=OPENPILOT_COLORS[versions[i]], linewidth=5)

    # Create custom legend entries
    legend_elements = [Line2D([0], [0], color='green', alpha=0.25, lw=10, label='Pass'),
                       Line2D([0], [0], color='red',  alpha=0.25, lw=10, label='Fail')]

    # Add the new legend with these entries to the top left
    legend1 = ax.legend(handles=legend_elements, loc='upper left', fontsize=22)
    ax.add_artist(legend1)

    # Add legend and labels
    ax.legend(fontsize=22, loc='upper right')
    ax.set_xlabel('Frame ID', fontsize=24)
    ax.set_ylabel('Steering Angle', fontsize=24)
    ax.set_ylim([-100,100])
    ax.set_yticks(np.arange(-100, 101, 10)) 
    min_index = np.max([0, np.min(frame_id_window)])
    max_index = np.max([args.window_size, np.max(frame_id_window)])
    ticks_frame_ids = np.arange(min_index, max_index+10, 15)
    ax.set_xticks(ticks_frame_ids) 
    ax.set_xlim([min_index-1, max_index+1])
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    ax.grid()

    # Show the plot
    plt.draw()
    plt.pause(0.01)

    # Display the steering
    frame = show_steering(frame, steering_angles, colors, op_versions, CLIPPING_DEGREE)
    
    # If the frame was not retrieved, we've reached the end of the video.
    if not ret:
        print("Reached the end of the video or error in reading. Exiting...")
        break

    # Display the frame.
    cv2.imshow('Video Frame', frame)

    # Keep track of the frame id
    frame_id += 1

    # Break the loop when the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()

