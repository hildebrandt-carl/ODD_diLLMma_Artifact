import os
import sys
import glob
import argparse
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from constants import ODD
from description_loader import DescriptionLoader


def check_if_all_vectors_are_equal(arrays):
    # Take the first array as the reference
    first_array = arrays[0]
    # Use numpy's all() and array_equal() to check if all arrays are equal to the first one
    are_all_equal = np.all([np.array_equal(first_array, arr) for arr in arrays])
    return are_all_equal

def setup_image_display():
    global title_label
    """Setup and display the initial image."""
    image_frame = tk.Frame(root)
    image_frame.pack(padx=10, pady=10)
    image_filename = all_filenames[current_image_index][:all_filenames[current_image_index].rfind("_")] + ".png"
    # Label to be displayed above the image
    title_label = tk.Label(image_frame, text=image_filename)  # You can customize the text here
    title_label.pack()  # Pack it before the image to ensure it appears above
    image_path = os.path.join(DATASET_DIRECTORY, "4_SelectedData", image_filename)
    resized_image, photo_image = load_image(image_path)  # Adjusted to unpack both returned objects
    image_label = tk.Label(image_frame, image=photo_image)
    image_label.pack()
    return image_frame, image_label, photo_image  # Return PhotoImage object as well

def setup_description_label():
    """Create and display the description label."""
    description_label = tk.Label(root, text="The image is from a front-facing camera in a car. Answer the following questions based on the image provided, using the template below:")
    description_label.pack()
    return description_label

def setup_checkboxes_frame():
    """Initialize the frame for checkboxes."""
    checkboxes_frame = tk.Frame(root)
    checkboxes_frame.pack(padx=10, pady=10)
    return checkboxes_frame

def parse_resize_arg(resize_arg):
    resize_arg = resize_arg[1:-1]
    try:
        width, height = map(int, resize_arg.split(','))
        return (width, height)
    except Exception as e:
        print(f"Error parsing resize_image argument: {e}")
        return None

def resize_image(image_path, new_size=None):
    image = Image.open(image_path)
    if new_size:
        return image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        return image

def load_image(image_path):
    global resize_dims
    """Resize image to fit within a max size while keeping aspect ratio."""
    resized_image = resize_image(image_path, new_size=resize_dims)
    photo = ImageTk.PhotoImage(resized_image)
    return resized_image, photo  # Return both Image and PhotoImage objects

def create_checkboxes_with_label(parent, labels, label_text, checkmarks):
    """Create a labeled column of checkboxes in the given parent frame."""
    # Frame for this set of checkboxes and its label
    frame = tk.Frame(parent)
    frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y, expand=False)

    # Label for the checkboxes
    label = tk.Label(frame, text=label_text)
    label.pack()

    # Create checkboxes and set them to disabled
    vars = []
    for label, check in zip(labels, checkmarks):
        var = tk.IntVar(value=check)
        checkbox = tk.Checkbutton(frame, text=label, variable=var)
        checkbox.pack(anchor=tk.W)
        vars.append(var)

    return vars

def create_checkboxes_for_image():
    global checkbox_vars, odd_keys
    checkbox_vars = []
    for i in range(len(individual_human_datasets)):
        human = individual_human_datasets[i]
        filename_index = data[human]["filenames"].index(all_filenames[current_image_index])
        coverage_vector = data[human]["coverage"][filename_index]
        current_vars = create_checkboxes_with_label(checkboxes_frame, odd_keys, human, coverage_vector)
        checkbox_vars.append(current_vars)

    # Create a vertical separator with a background color
    separator_frame = tk.Frame(checkboxes_frame, width=1, bg='grey', bd=0)
    separator_frame.pack(side=tk.LEFT, fill=tk.Y, padx=1)

    # Create the final column of checkboxes with zeros as checkmarks
    final_checkmarks = np.zeros(len(odd_keys), dtype=int)
    final_vars = create_checkboxes_with_label(separator_frame, odd_keys, "Final Decision", final_checkmarks)
    checkbox_vars.append(final_vars)
    
def save_to_file(index, vector=None):
    global OUTPUT_DIR, checkbox_vars, all_filenames

    if vector is None:
        vector = []
        checkmarks = checkbox_vars[-1]
        for checkbox in checkmarks:
            vector.append(1 if checkbox.get() else 0)
        vector = np.array(vector)

    basename = all_filenames[index][:all_filenames[index].rfind("_")]
    
    # Save each of the YES/NO data fields
    output_string = ""
    for v in vector:
        output_string += "YES\n" if v == 1 else "NO\n"
    with open(f"{OUTPUT_DIR}/{basename}_output.txt", "w") as file:
        file.write(output_string)

def update_image_and_checkboxes():
    global current_image_index, image_label, photo_image, checkbox_vars, title_label, display_all, all_filenames

    save_to_file(current_image_index)

    # Update the current_image_index
    equal = True
    if not display_all:
        for i in range(current_image_index+1, len(all_filenames)):
            vectors = []
            for ind_human in individual_human_datasets:
                vectors.append(data[ind_human]["coverage"][i])
            equal = check_if_all_vectors_are_equal(vectors)
            current_image_index = i
            if not equal:
                break
            else:
                save_to_file(current_image_index, vectors[0])

    if equal or display_all:
        current_image_index += 1
         
    if current_image_index >= len(all_filenames):
        exit()

    # Update the image
    image_filename = all_filenames[current_image_index][:all_filenames[current_image_index].rfind("_")] + ".png"
    image_path = os.path.join(DATASET_DIRECTORY, "4_SelectedData", image_filename)
    resized_image, new_photo_image = load_image(image_path)  # Adjusted to unpack both returned objects
    photo_image = new_photo_image  # Update the global reference to the new PhotoImage object
    image_label.configure(image=photo_image)
    image_label.image = photo_image  # Keep a reference to avoid garbage collection

    # Update the title label text here
    title_label.configure(text=image_filename)

    # Clear existing checkboxes
    for child in checkboxes_frame.winfo_children():
        child.destroy()

    # Update checkboxes for the new image
    create_checkboxes_for_image()



# Parse command-line arguments
parser = argparse.ArgumentParser(description="Used to generate the final human annotations")
parser.add_argument('--dataset',
                    type=str,
                    choices=['External_Jutah', 'OpenPilot_2k19', 'OpenPilot_2016'],
                    required=True)
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data")
parser.add_argument('--resize_display',
                    type=str,
                    help="Resize the image to (width,height) before annotating.")
parser.add_argument('--display_all',
                    action='store_true',
                    help="If set, all images will be displayed for annotation.")
args = parser.parse_args()

# Setup directories and resize dimensions
DATASET_DIRECTORY = os.path.join(args.dataset_directory, args.dataset)
resize_dims = parse_resize_arg(args.resize_display) if args.resize_display else None

# Create an output dir
OUTPUT_DIR = f"{DATASET_DIRECTORY}/5_Descriptions/Human"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get the existing human files
existing_file_paths = glob.glob(f"{OUTPUT_DIR}/*.txt")
existing_files = [f[f.rfind("/")+1:] for f in existing_file_paths]
existing_files_set = set(existing_files)

# Created check boxes
odd_keys = list(ODD.keys())

# Get all the available individual humans
INDIVIDUAL_HUMAN_DATASET_DIRECTORY = f"{DATASET_DIRECTORY}/5_Descriptions/Individual_Human"
individual_human_dataset_paths = glob.glob(f"{INDIVIDUAL_HUMAN_DATASET_DIRECTORY}/*")
individual_human_datasets = [f[f.rfind("/")+1:] for f in individual_human_dataset_paths]
individual_human_datasets = sorted(individual_human_datasets)

# Holds the data
data = {}
all_filenames_sets = []

# Get coverage vectors for all the available datasets
for ind_human in individual_human_datasets:
    # Get the coverage vector
    print(f"{INDIVIDUAL_HUMAN_DATASET_DIRECTORY}/{ind_human}")
    dl = DescriptionLoader(f"{INDIVIDUAL_HUMAN_DATASET_DIRECTORY}/{ind_human}")
    coverage_vector = dl.coverage_vector
    filenames = dl.description_names
    data[ind_human] = {}
    data[ind_human]["coverage"] = coverage_vector
    data[ind_human]["filenames"] = filenames

    # Tracks all filenames
    all_filenames_sets.append(set(filenames))

# Make sure all the datasets are the same size:
for i in range(1, len(all_filenames_sets)):
    assert all_filenames_sets[0] == all_filenames_sets[i], f"Labeled files in {individual_human_datasets[0]} does not match labeled files in {individual_human_datasets[i]}"

all_filenames_set = all_filenames_sets[0]
    
# Only worry about files we have not yet processed
files_to_process_set = all_filenames_set ^ existing_files_set

# Convert all filenames to a list and sort them
all_filenames = list(files_to_process_set)
all_filenames = sorted(all_filenames)

# Setup Tkinter GUI
root = tk.Tk()
root.title("Generating Final Human Annotations")

# Check if we are displaying all
display_all = args.display_all
current_image_index = 0
if not display_all:
    for i in range(len(all_filenames)):
        f = all_filenames[i]
        vectors = []
        for ind_human in individual_human_datasets:
            vectors.append(data[ind_human]["coverage"][i])
        equal = check_if_all_vectors_are_equal(vectors)

        if not equal:
            current_image_index = i
            break
        else:
            save_to_file(current_image_index, vectors[0])

# Initialize current image index and display the first image
image_frame, image_label, photo_image = setup_image_display()  
description_label = setup_description_label()
checkboxes_frame = setup_checkboxes_frame()

# Button to navigate to the next image
next_button = tk.Button(root, text="Save and Next", command=update_image_and_checkboxes)
next_button.pack(side=tk.BOTTOM, pady=10)

# Create initial checkboxes
create_checkboxes_for_image()

# Start the GUI event loop
root.mainloop()
