import os
import sys
import copy
import glob
import argparse

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Import Common
current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)


from description_loader import DescriptionLoader

def parse_resize_arg(resize_arg):
    resize_arg = resize_arg[1:-1]
    try:
        width, height = map(int, resize_arg.split(','))
        return (width, height)
    except Exception as e:
        print(f"Error parsing resize_image argument: {e}")
        return None

# Get the Data
parser = argparse.ArgumentParser(description="Displays the number of failure inducing inputs")
parser.add_argument('--annotator',
                    type=str,
                    choices=['Human', 'ChatGPT_Base', 'Llama_Base', 'Vicuna_Base', 'Llama_Plus', 'Vicuna_Plus'],
                    required=True,
                    help="The annotator to use. Choose between 'Human', 'ChatGPT_Base', 'Llama_Base', 'Vicuna_Base', 'Llama_Plus', 'Vicuna_Plus'.")
parser.add_argument('--description_filter',
                    choices=['Both', 'Pass', 'Fail'],
                    required=True,
                    help="Filter results by status: 'Both' to include both passing and failing, 'Pass' for passing only, or 'Fail' for failing only.")
parser.add_argument('--dataset',
                    type=str,
                    choices=['External_Jutah', 'OpenPilot_2k19', 'OpenPilot_2016'],
                    required=True,
                    help="The dataset to use. Choose between 'External_Jutah', 'OpenPilot_2k19', or 'OpenPilot_2016'.")
parser.add_argument('--resize_display',
                    type=str,
                    required=False,
                    default="(640,480)",
                    help="Resize the image to the specified size (width,height) before annotating.")
parser.add_argument('--filter_human_verified_odd',
                    action='store_true',
                    help="Display images only if they are marked as ODD by both the annotator and a human verifier.")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")
# Get the dataset directory
DATASET_DIRECTORY = os.path.join(args.dataset_directory, args.dataset)

# Get all the files based on the filter:
if args.description_filter == "Both":
    descriptions = glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/{args.annotator}/*_output.txt")
    human_descriptions = glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/Human/*_output.txt")
elif args.description_filter == "Pass":
    descriptions = glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/{args.annotator}/pass_*_output.txt")
    human_descriptions = glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/Human/pass_*_output.txt")
elif args.description_filter == "Fail":
    descriptions = glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/{args.annotator}/fail_*_output.txt")
    human_descriptions = glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/Human/fail_*_output.txt")

# Load the coverage vector and find all in ODD
dl = DescriptionLoader(descriptions)
in_odd  = np.all(dl.coverage_vector == 0, axis=1)
in_odd_final = copy.deepcopy(in_odd)

print(f"{args.annotator} marked {np.sum(in_odd)} in ODD")

if args.filter_human_verified_odd:
    human_dl = DescriptionLoader(human_descriptions)
    human_in_odd  = np.all(human_dl.coverage_vector == 0, axis=1)
    print(f"Human marked {np.sum(human_in_odd)} in ODD")
    in_odd_final = human_in_odd & in_odd
    print(f"Intersection of Human and {args.annotator} marked {np.sum(in_odd_final)} in ODD")

print(f"Using {np.sum(in_odd_final)} in ODD")

# Get the files in ODD
filenames = np.array(dl.description_names)
filenames_in_odd = list(filenames[in_odd_final])

# Clean the filenames
filenames_in_odd = [f[:f.rfind("_")] + ".png" for f in filenames_in_odd]


# Build the file paths
IMAGE_DIR = f"{DATASET_DIRECTORY}/4_SelectedData"
filepaths_in_odd = [f"{IMAGE_DIR}/{f}" for f in filenames_in_odd]


# List of image paths
current_image_index = 0

# Function to update the displayed image and label text
def next_image():
    global current_image_index
    current_image_index += 1
    if current_image_index >= len(filepaths_in_odd):
        exit()
    img_path = filepaths_in_odd[current_image_index]
    img = Image.open(img_path)
    img = img.resize(parse_resize_arg(args.resize_display), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(img)
    image_label.configure(image=photo)
    image_label.image = photo  # Keep a reference!
    text_label.configure(text=img_path)  # Update the label text to the current image path

# Setup the Tkinter window
root = tk.Tk()
root.title("Image Viewer")

# Load the first image
img_path = filepaths_in_odd[current_image_index]
img = Image.open(img_path)
img = img.resize(parse_resize_arg(args.resize_display), Image.Resampling.LANCZOS)
photo = ImageTk.PhotoImage(img)

# Display the image
image_label = tk.Label(root, image=photo)
image_label.pack(pady=10)

# Display a label with the image path
text_label = tk.Label(root, text=img_path)
text_label.pack(pady=10)

# Button to change the image
next_button = tk.Button(root, text="Next Image", command=next_image)
next_button.pack(pady=10)

root.mainloop()


