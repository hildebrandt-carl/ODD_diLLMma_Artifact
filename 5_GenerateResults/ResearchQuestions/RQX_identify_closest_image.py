import os
import re
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from utils_Naming import DATASET_NAMES
from utils_EmbeddingModel import get_embedding_model


# Get the embedding model
EMBEDDING_MODEL = get_embedding_model()

# Declare constants
DATASET_DIRECTORY = "../1_Datasets/Data"

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Identifies the closest image.")
parser.add_argument('--llm_model', 
                    type=str,
                    default="",
                    choices=["llama", "vicuna", "chat_gpt", "human", "researcher_a", "researcher_b", "researcher_c"],
                    help="The LLM to use (llama, vicuna, chat_gpt, human, researcher_a, researcher_b, researcher_c)")
parser.add_argument('--datasets',
                    type=str,
                    default="",
                    help="The dataset you want to process (OpenPilot_2k19, External_jutah, OpenPilot_2016)")
parser.add_argument('--img_number',
                    type=int,
                    required=True,
                    help="The number of the image you want to select")
parser.add_argument('--pass_fail',
                    type=str,
                    required=True,
                    choices=['pass', 'fail'],
                    help="Whether the selected image is from the passing or failing set (pass, fail)")
parser.add_argument('--show_plot',
                    action='store_true', 
                    help="Display the plot")
args = parser.parse_args()

# Get the list of datasets
possible_datasets = os.listdir(DATASET_DIRECTORY)

# Get the list of requested datasets
requested_datasets = args.datasets.split(", ")
for datasets in requested_datasets:
    assert datasets in possible_datasets, f"The dataset was not found in `{DATASET_DIRECTORY}`"

selected_images        = []
closest_passing_images = []
closest_failing_images = []

# For each dataset
for dataset in requested_datasets:

    # Define the base path
    base_path = f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{args.llm_model}"

    # Get passing and failing names
    passing_files = glob.glob(f"{base_path}/q10/pass*_output.txt")
    failing_files = glob.glob(f"{base_path}/q10/fail*_output.txt")

    # Make sure we find data
    assert len(passing_files) > 0, "No passing descriptions found"
    assert len(failing_files) > 0, "No failing descriptions found"

    # Sort the data
    passing_files = sorted(passing_files)
    failing_files = sorted(failing_files)

    # Create the image file names
    passing_images = [f"{f[:f.find('5_Descriptions/')-1]}/4_SelectedData_100/{os.path.basename(f)[:-11]}.png" for f in passing_files]
    failing_images = [f"{f[:f.find('5_Descriptions/')-1]}/4_SelectedData_100/{os.path.basename(f)[:-11]}.png" for f in failing_files]

    # Used to hold the textual data
    passing_descriptions = []
    failing_descriptions = []

    # Load the passing data
    for f_name in passing_files:
        with open(f_name,"r") as f:
            text = f.read()
        passing_descriptions.append(text)

    # Load the failing data
    for f_name in failing_files:
        with open(f_name,"r") as f:
            text = f.read()
        failing_descriptions.append(text)

    # Used to hold the embeddings
    passing_embeddings = EMBEDDING_MODEL(passing_descriptions)
    failing_embeddings = EMBEDDING_MODEL(failing_descriptions)

    # Select the image to compare against
    pattern = f".*{args.pass_fail}_{args.img_number:04d}_output\.txt"
    regex = re.compile(pattern)

    # Find the matches
    if args.pass_fail == "pass":
        matched_indices         = [index for index, s in enumerate(passing_files) if regex.match(s)]
        matched_images          = [passing_images[i] for i in matched_indices]
        matched_descriptions    = [passing_descriptions[i] for i in matched_indices]
        matched_embeddings      = [passing_embeddings[i] for i in matched_indices]
    else:
        matched_indices         = [index for index, s in enumerate(failing_files) if regex.match(s)]
        matched_images          = [failing_images[i] for i in matched_indices]
        matched_descriptions    = [failing_descriptions[i] for i in matched_indices]
        matched_embeddings      = [failing_embeddings[i] for i in matched_indices]

    assert len(matched_indices) == 1, "More or less than than 1 image matched"
    matched_index          = matched_indices[0]
    matched_image          = matched_images[0]
    matched_description    = matched_descriptions[0]
    matched_embedding      = matched_embeddings[0]

    # Keep track of the lowest
    lowest_norm = np.inf
    lowest_index = np.nan

    # Loop through all passing embeddings
    for current_index, current_embedding in enumerate(passing_embeddings):

        # Skip if we have selected passing
        if args.pass_fail == "pass":
            if current_index == matched_index:
                continue

        # Compute norm
        l2_norm = np.linalg.norm(matched_embedding - current_embedding)
        if l2_norm < lowest_norm:
            lowest_norm = l2_norm
            lowest_index = current_index

    # Save the closest
    selected_images.append(matched_image)
    closest_passing_images.append(passing_images[lowest_index])

    # Keep track of the lowest
    lowest_norm = np.inf
    lowest_index = np.nan

    # Loop through all failing embeddings
    for current_index, current_embedding in enumerate(failing_embeddings):

        # Skip if we have selected failing
        if args.pass_fail == "fail":
            if current_index == matched_index:
                continue

        # Compute norm
        l2_norm = np.linalg.norm(matched_embedding - current_embedding)
        if l2_norm < lowest_norm:
            lowest_norm = l2_norm
            lowest_index = current_index

    # Save the closest
    closest_failing_images.append(failing_images[lowest_index])

# Create the subplots
fig, axes = plt.subplots(len(selected_images), 3, figsize=(15, 3 * len(selected_images)))

# Check if axes is 1D or 2D so I can adjust accordingly
is_single_row = len(selected_images) == 1

# Declare the fontsize
fontsize = 48

# Set titles for the columns
if len(selected_images) > 0:

    if args.pass_fail == "pass":
        titles = ["Selected: Passing", "Closest:  Passing", "Closest:  Failing"]
    else:
        titles = ["Selected: Failing", "Closest:  Passing", "Closest:  Failing"]

    for idx, title in enumerate(titles):
        if is_single_row:
            axes[idx].set_title(title, fontsize=fontsize)
        else:
            axes[0, idx].set_title(title, fontsize=fontsize)

    # Used to do looping later on
    looper = zip(selected_images, closest_passing_images, closest_failing_images)

# Display each image
for i, imgs in enumerate(looper):
    for j, img_path in enumerate(imgs):
        img = Image.open(img_path)
        ax = axes[i, j] if not is_single_row else axes[j]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        if j == 0:
            ax.set_ylabel(f"{DATASET_NAMES[requested_datasets[i]]}", fontsize=fontsize)

# Show the plot
plt.tight_layout()
plt.savefig(f"./results/rq2b_{args.llm_model}_{args.pass_fail}_{args.img_number}.png")
if args.show_plot:
    plt.show()
plt.close()