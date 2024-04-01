import os
import sys
import glob
import argparse

import tkinter as tk

from random import shuffle
from tkinter import filedialog
from PIL import Image, ImageTk

current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from constants import ODD


# Decare the dataset directory
DATASET_DIRECTORY = "../1_Datasets/Data"

def resize_image(image_path, new_size=None):
    image = Image.open(image_path)
    if new_size:
        return image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        return image

def save_and_next(output_dir):
    basename = os.path.basename(current_image)[:-4]
    output_string = ""
    # Save each of the YES/NO data fields
    for i, checkbox in enumerate(checkboxes):
        output_string += "YES\n" if checkbox_vars[i].get() else "NO\n"
    with open(f"{output_dir}/{basename}_output.txt", "w") as file:
        file.write(output_string)
    load_next_image()

def load_next_image():
    global image_label, current_image, resize_dims
    try:
        current_image = next(images)
        resized_image = resize_image(current_image, new_size=resize_dims)
        photo = ImageTk.PhotoImage(resized_image)
        image_label.config(image=photo)
        image_label.image = photo
        for var in checkbox_vars:
            var.set(0)
    except StopIteration:
        root.destroy()

def parse_resize_arg(resize_arg):
    resize_arg = resize_arg[1:-1]
    try:
        width, height = map(int, resize_arg.split(','))
        return (width, height)
    except Exception as e:
        print(f"Error parsing resize_image argument: {e}")
        return None

# Get the folders
parser = argparse.ArgumentParser(description="Allows for annotating data through humans")
parser.add_argument('--dataset',
                    type=str,
                    choices=['External_Jutah', 'OpenPilot_2k19', 'OpenPilot_2016'],
                    required=True,
                    help="The dataset to use. Choose between 'External_Jutah', 'OpenPilot_2k19', or 'OpenPilot_2016'.")
parser.add_argument('--author_name',
                    type=str,
                    required=True,
                    help="The author of the annotations")
parser.add_argument('--resize_display',
                    type=str,
                    required=False,
                    help="Resize the image to the specified size (width,height) before annotating.")
args = parser.parse_args()

# Initialize Tkinter
root = tk.Tk()
root.title(f"Annotating {args.dataset}")

# Create the output directory and declare the input directory
OUTPUT_DIR = f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions/Individual_Human/{args.author_name}"
INPUT_DIR  = f"{DATASET_DIRECTORY}/{args.dataset}/4_SelectedData"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parse the resize_image argument if provided
resize_dims = None
if args.resize_display:
    resize_dims = parse_resize_arg(args.resize_display)

# Load all the descriptions and images
existing_descriptions   = sorted(glob.glob(f"{OUTPUT_DIR}/*.txt"))
all_images              = sorted(glob.glob(f"{INPUT_DIR}/*.png"))
base_descriptions       = [os.path.basename(desc)[:-11] for desc in existing_descriptions]
base_images             = [os.path.basename(img)[:-4] for img in all_images]

# Only annotate images which haven't been annotated already
images_requiring_annotation = []
for base_image_name, full_image_path in zip(base_images, all_images):
    if base_image_name not in base_descriptions:
        images_requiring_annotation.append(full_image_path)

# Order the images randomly
shuffle(images_requiring_annotation)

# Create an iterator
images = iter(images_requiring_annotation)
current_image = None

# Display the first image
image_label = tk.Label(root)
image_label.pack()

# Created check boxes
odd_keys = list(ODD.keys())

# Add a label above the text box
description_label = tk.Label(root, text="The image is from a front-facing camera in a car. Answer the following questions based on the image provided, using the template below:")
description_label.pack()

# Create the checkboxes
checkbox_vars = [tk.IntVar() for _ in range(11)]
checkboxes = [tk.Checkbutton(root, text=f"{odd_keys[i]}", variable=checkbox_vars[i]) for i in range(11)]
for checkbox in checkboxes:
    checkbox.pack()

# Add a button to save responses and move to next image
button = tk.Button(root, text="Save and Next", command=lambda: save_and_next(OUTPUT_DIR))
button.pack()

# Load the first image
load_next_image()

# Start the GUI event loop
root.mainloop()


# Update it so that it only display checkboxes it has loaded.
# Update the other code to go under individual folder