import os
import cv2
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from data_loader import DataLoader
from constants import CLIPPING_DEGREE
from constants import OPENPILOT_COLORS_RGB
from visualization_functions import show_steering


# Get the folders
parser = argparse.ArgumentParser(description="Displays the entire process")
parser.add_argument('--image_name',
                    type=str,
                    required=True,
                    help="The name of the img file")
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

print("")
# Get the dataset directory
DATASET_DIRECTORY = f"{args.dataset_directory}/{args.dataset}"

# Load the image
image_path  = f"{DATASET_DIRECTORY}/4_SelectedData/{args.image_name}"
file_exists = os.path.exists(image_path)

# Check if the file exists
if not file_exists:
    print("Error: File does not exist")
    exit()

# Load the steering angle for that image
img_filename    = args.image_name
index           = int(img_filename[img_filename.rfind("_")+1:img_filename.rfind(".")])
video_filename  = img_filename[img_filename.find("_")+1:img_filename.rfind("_")]

# Find all steering angle files
dl = DataLoader(filename=video_filename)
dl.validate_h5_files()
dl.load_data()

# Load the steering angles
versions        = dl.versions
colors          = [OPENPILOT_COLORS_RGB[v] for v in versions]
steering_angles = dl.readings[:,index]

# Load the image and 
img = cv2.imread(image_path)
img = cv2.resize(img, (1800, 1100)) 

# Convert the image from BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(24,15))
plt.imshow(img_rgb)
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{img_filename}")

# Display Steering
img = show_steering(img, steering_angles, colors, versions, CLIPPING_DEGREE)

# Convert the image from BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.figure(figsize=(24,15))
plt.imshow(img_rgb)
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{img_filename[:-4]}_steer.png")
plt.show()



