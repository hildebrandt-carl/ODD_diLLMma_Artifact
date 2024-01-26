import os
import glob
import argparse

import tkinter as tk

from random import shuffle
from tkinter import filedialog
from PIL import Image, ImageTk

# Decare the dataset directory
DATASET_DIRECTORY = "../1_Datasets/Data"

odd = ["Poor Visibility",
       "Image obstructed",
       "Sharp curve",
       "On-off ramp",
       "Intersection",
       "Restricted lane",
       "Construction",
       "Bright light",
       "Narrow road",
       "Hilly road"]      

def resize_image(image_path, new_size=None):
    image = Image.open(image_path)
    if new_size:
        return image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        return image

def save_and_next():
    basename = os.path.basename(current_image)[:-4]
    description = description_text.get("1.0", "end-1c")
    # Save each of the YES/NO data fields
    for i, checkbox in enumerate(checkboxes):
        with open(f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions_{args.size}/human/q{i:02d}/{basename}_output.txt", "w") as file:
            file.write("YES" if checkbox_vars[i].get() else "NO")
    # Save the description of the image
    with open(f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions_{args.size}/human/q10/{basename}_output.txt", "w") as file:
        file.write(description)
    load_next_image()

def load_next_image():
    global image_label, current_image
    try:
        current_image = next(images)
        resized_image = resize_image(current_image, new_size=(800, 600))  # Example: scale down by 50%
        photo = ImageTk.PhotoImage(resized_image)
        image_label.config(image=photo)
        image_label.image = photo
        for var in checkbox_vars:
            var.set(0)
        description_text.delete("1.0", tk.END)
    except StopIteration:
        root.destroy()

# Get the folders
parser = argparse.ArgumentParser(description="Allows for annotating data through humans")
parser.add_argument('--dataset',
                    type=str,
                    default="",
                    help="The dataset you want to process (OpenPilot_2k19, External_jutah, OpenPilot_2016)")
parser.add_argument('--size',
                    type=int,
                    default=-1,
                    help="The size of the dataset you want to use")
args = parser.parse_args()

# Make sure you have set a dataset size
assert args.size > 0, "Dataset size can not be less than or equal to 0"

# Create the directory
for i in range(len(odd)+1):
    output_dir = f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions_{args.size}/human/q{i:02d}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Initialize Tkinter
root = tk.Tk()
root.title("Image Viewer")

# Load all the descriptions and images
all_descriptions = sorted(glob.glob(f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions_{args.size}/human/*/*.txt"))
all_images = sorted(glob.glob(f"{DATASET_DIRECTORY}/{args.dataset}/4_SelectedData_{args.size}/*.png"))

# Remove all images that already have descriptions
base_descriptions = [os.path.basename(desc)[:-11] for desc in all_descriptions]
base_images = [os.path.basename(img)[:-4] for img in all_images]

# Only annotate the images that haven't got annotations already
annotation_images = []
for base_image_name, full_image_path in zip(base_images, all_images):
    if base_image_name not in base_descriptions:
        annotation_images.append(full_image_path)

# Randomly assign images
shuffle(annotation_images)

images = iter(annotation_images)  # Adjust the file type if needed
current_image = None

# Display the first image
image_label = tk.Label(root)
image_label.pack()

# Created check boxes
checkbox_vars = [tk.IntVar() for _ in range(10)]
checkboxes = [tk.Checkbutton(root, text=f"{odd[i]}", variable=checkbox_vars[i]) for i in range(10)]
for checkbox in checkboxes:
    checkbox.pack()

# Add a label above the text box
description_label = tk.Label(root, text="Describe all the factual information about this image. Pay attention to the road structure, road signs, traffic, pedestrians, time of day, and anything one needs to pay attention to while driving.")
description_label.pack()
# Add a text box for image description
description_text = tk.Text(root, height=5, width=50)
description_text.pack()

# Add a button to save responses and move to next image
button = tk.Button(root, text="Save and Next", command=save_and_next)
button.pack()

# Load the first image
load_next_image()

# Start the GUI event loop
root.mainloop()