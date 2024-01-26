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
parser.add_argument('--user_prompt',
                    type=str,
                    required=True,
                    help="The description you want to match against")
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
 
# Holds the matches
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

    # Convert the user prompt into embedding space
    user_prompt_embedding = EMBEDDING_MODEL([args.user_prompt])
    user_prompt_embedding = user_prompt_embedding[0]

    # Keep track of the lowest
    lowest_norm = np.inf
    lowest_index = np.nan

    # Loop through all passing embeddings
    for current_index, current_embedding in enumerate(passing_embeddings):

        # Compute norm
        l2_norm = np.linalg.norm(user_prompt_embedding - current_embedding)
        if l2_norm < lowest_norm:
            lowest_norm = l2_norm
            lowest_index = current_index

    # Save the closest
    closest_passing_images.append(passing_images[lowest_index])

    # Keep track of the lowest
    lowest_norm = np.inf
    lowest_index = np.nan

    # Loop through all failing embeddings
    for current_index, current_embedding in enumerate(failing_embeddings):

        # Compute norm
        l2_norm = np.linalg.norm(user_prompt_embedding - current_embedding)
        if l2_norm < lowest_norm:
            lowest_norm = l2_norm
            lowest_index = current_index

    # Save the closest
    closest_failing_images.append(failing_images[lowest_index])

# Create the subplots
fig, axes = plt.subplots(len(requested_datasets), 2, figsize=(12, 5 * len(requested_datasets)))
# fig.suptitle(f"({args.llm_model}): {args.user_prompt}", fontsize=16)

# Declare the fontsize
fontsize = 24

# Set titles for the columns
if len(requested_datasets) == 1:
    # If only one dataset, treat axes as a 1D array
    axes[0].set_title("Closest: Passing", fontsize=fontsize)
    axes[1].set_title("Closest: Failing", fontsize=fontsize)
elif len(requested_datasets) > 1:
    axes[0, 0].set_title("Closest:  Passing", fontsize=fontsize)
    axes[0, 1].set_title("Closest:  Failing", fontsize=fontsize)

looper = zip(closest_passing_images, closest_failing_images)

# Display each image
for i, imgs in enumerate(looper):
    
    if len(requested_datasets) == 1:
        img = Image.open(imgs[0])
        axes[0].imshow(img)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_ylabel(f"{DATASET_NAMES[requested_datasets[i]]}", fontsize=fontsize)

        img = Image.open(imgs[1])
        axes[1].imshow(img)
        axes[1].axis('off')
    elif len(requested_datasets) > 1:
        img = Image.open(imgs[0])
        axes[i, 0].imshow(img)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(f"{DATASET_NAMES[requested_datasets[i]]}", fontsize=fontsize)

        img = Image.open(imgs[1])
        axes[i, 1].imshow(img)
        axes[i, 1].axis('off')

# Compute the description for the save file
words = args.user_prompt.split()
first_five_words = words[:5]
camel_case_string = ''.join(word.capitalize() for word in first_five_words)

# Show the plot
plt.tight_layout()
plt.savefig(f"./results/rq3b_{args.llm_model}_{camel_case_string}.png")
if args.show_plot:
    plt.show()
plt.close()
