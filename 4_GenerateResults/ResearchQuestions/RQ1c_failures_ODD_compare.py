import os
import glob
import shutil
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils_Naming import LLM_NAMES
from utils_ODDManagement import get_odd
from utils_ODDManagement import clean_string
from utils_ViolationState import ViolationState

# Declare constants
DATASET_DIRECTORY = "../1_Datasets/Data"
OUTPUT_DIRECTORY = "./results"

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Compares ODD using bar plot.")
parser.add_argument('--llm_models',
                    type=str,
                    default="",
                    help="The LLM to use as a comma separated list(llama, vicuna, chat_gpt, human, researcher_a, researcher_b, researcher_c)")
parser.add_argument('--dataset',
                    type=str,
                    choices=["OpenPilot_2k19", "External_jutah", "OpenPilot_2016"],
                    help="The dataset you want to process as a list '(OpenPilot_2k19, External_jutah, OpenPilot_2016)'")
parser.add_argument('--size',
                    type=int,
                    default=-1,
                    help="The size of the dataset you want to use")
parser.add_argument('--baseline',
                    type=str,
                    default="human",
                    choices=["llama", "vicuna", "chat_gpt", "human", "researcher_a", "researcher_b", "researcher_c"],
                    help="The baseline dataset. This is normally human")
parser.add_argument('--show_plot',
                    action='store_true', 
                    help="Display the plot")
args = parser.parse_args()

# Make sure you have set a dataset size
assert args.size > 0, "Dataset size can not be less than or equal to 0"

# Get the list of datasets
possible_datasets = os.listdir(DATASET_DIRECTORY)

# Make sure the dataset exists
assert args.dataset in possible_datasets, f"The dataset was not found in `{DATASET_DIRECTORY}`"

# Create the base path based on the parameters
BASE_PATH = f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions/{args.size}"
IMG_PATH  = f"{DATASET_DIRECTORY}/{args.dataset}/4_SelectedData/{args.size}"

# Make sure all the llm_models exist
possible_llm_models = os.listdir(BASE_PATH)
requested_llm_models = args.llm_models.split(", ")
for llm_model in requested_llm_models:
    assert llm_model in possible_llm_models, (
        f"The LLM model {llm_model} was not found in: `{BASE_PATH}`"
        )

# Check if the output directory has been created
for llm_model in requested_llm_models:
    output_dir = f"{OUTPUT_DIRECTORY}/{args.dataset}/{llm_model}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Define the questions and labels
odd = get_odd()   

# Get the list of all files
all_files = []
for llm_model in requested_llm_models:
    # Determine how many files there are
    failing_description_files = glob.glob(f"{BASE_PATH}/{llm_model}/{odd[0][0]}/fail*_output.txt")
    failing_description_files = [os.path.basename(f) for f in failing_description_files]

    # Make sure there are some common files
    assert len(failing_description_files) > 0, f"Could not file files for: {llm_model}"

    # Save all the files
    all_files.append(failing_description_files)

# Get the list of all common files
common_files = set(all_files[0])
for lst in all_files[1:]:
    common_files &= set(lst)

# Convert it back to a list
common_files = list(common_files)
common_files = sorted(common_files)

# Make sure there are some common files
assert len(common_files) > 0, "There are no files in common"

# Holds the output
odd_vectors = np.zeros((len(requested_llm_models), len(common_files), len(odd)), dtype=int)

# Populate the vector
for llm_index, llm_model in enumerate(requested_llm_models):

    # For each of the files 
    print(f"Processing: {llm_model}")
    for file_index, filename in tqdm(enumerate(common_files), total=len(common_files)):

        # For each of the questions
        for odd_question_index, odd_question in enumerate(odd):

            # Load the current questions file
            failing_file = f"{BASE_PATH}/{llm_model}/{odd_question[0]}/{filename}"

            # Open the file for reading
            with open(failing_file, "r") as f:

                # Read the data and convert it to upper case
                file_data = f.read()
                
                # Get the words from the file data
                clean_file_data = clean_string(file_data)
                words = clean_file_data.split(" ")

                # Update the ODD
                if "YES" in words and "NO" in words:
                    odd_vectors[llm_index][file_index][odd_question_index] = ViolationState.MIXED_RESPONSE.num
                elif "YES" in words:
                    odd_vectors[llm_index][file_index][odd_question_index] = ViolationState.YES_VIOLATION.num
                elif "NO" in words:
                    odd_vectors[llm_index][file_index][odd_question_index] = ViolationState.NO_VIOLATION.num
                else:
                    odd_vectors[llm_index][file_index][odd_question_index] = ViolationState.UNDETERMINED.num

# Get the number the human disagreed upon
humans_disagreed_count = []
for llm_index, llm_model in enumerate(requested_llm_models):
    humans_disagreed_count.append((args.size // 2) - len(common_files))

# Holds the files inside the human ODD
inside_odd_human_files = []

# Get the number that are inside the ODD
inside_odd_count = []
for llm_index, llm_model in enumerate(requested_llm_models):
    inside_odd_indices = np.where(np.all(odd_vectors[llm_index] == ViolationState.NO_VIOLATION.num, axis=1))[0]
    inside_odd_count.append(len(inside_odd_indices))

    # Print and save the filename inside the ODD
    print("\n========================================================")
    print(f"Tests in {llm_model} that fail and are inside the ODD: [{len(inside_odd_indices)}/{np.shape(odd_vectors)[1]}]")
    for count, index in enumerate(inside_odd_indices):
        IMG_NAME = common_files[index][:-11] + ".png"
        print(f"{count + 1:02d}) {IMG_NAME}")
        # Copy the image into the output directory
        source_img_path         = f"{IMG_PATH}/{IMG_NAME}"
        destination_img_path    = f"{OUTPUT_DIRECTORY}/{args.dataset}/{llm_model}/{IMG_NAME}"
        shutil.copy(source_img_path, destination_img_path)

        if llm_model == args.baseline:
            inside_odd_human_files.append(common_files[index])
    print("========================================================\n")

# Get the number that are outside the ODD
outside_odd_count = []
for llm_index, llm_model in enumerate(requested_llm_models):
    outside_odd_indices = np.where(np.any(odd_vectors[llm_index] != ViolationState.NO_VIOLATION.num, axis=1))[0]
    outside_odd_count.append(len(outside_odd_indices))

# Count the number outside the ODD that were miss-classified
outside_odd_misclassified_count = []

# Count the number of inside odd that were undetected
inside_odd_undetected_count = []
for llm_index, llm_model in enumerate(requested_llm_models):
    inside_odd_indices = np.where(np.all(odd_vectors[llm_index] == ViolationState.NO_VIOLATION.num, axis=1))[0]
    detected_count = 0 
    outside_odd_miss = 0
    for count, index in enumerate(inside_odd_indices):
        file_name = common_files[index]
        if file_name in inside_odd_human_files:
            detected_count += 1
        else:
            outside_odd_miss += 1
    inside_odd_undetected_count.append(len(inside_odd_human_files) - detected_count)
    outside_odd_misclassified_count.append(outside_odd_miss)

print(inside_odd_undetected_count)

# Update the counts to add up
outside_odd_count = list(np.array(outside_odd_count) - np.array(inside_odd_undetected_count))
inside_odd_count = list(np.array(inside_odd_count) - np.array(outside_odd_misclassified_count))

fig = plt.figure(figsize=(17, 12))
fig.canvas.manager.set_window_title(args.dataset)

# Enable minor ticks
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='1', color='gray')
plt.grid(which='minor', linestyle='--', linewidth='0.5', color='gray')

# Get the llm display names
llm_display_names = [LLM_NAMES[llm] for llm in requested_llm_models]

# plt.bar(llm_display_names,
#         humans_disagreed_count,
#         bottom=[i+j+k+n for i,j,k,n in zip(outside_odd_count, inside_odd_undetected_count, outside_odd_misclassified_count, inside_odd_count)],
#         label='Human Disagreed',
#         facecolor='none',
#         edgecolor='black',
#         linewidth=5)
plt.bar(llm_display_names,
        outside_odd_count,
        bottom=[i+j+k for i,j,k in zip(inside_odd_undetected_count, outside_odd_misclassified_count, inside_odd_count)],
        label='Outside ODD',
        color='C0',
        edgecolor='black',
        linewidth=5)
plt.bar(llm_display_names,
        outside_odd_misclassified_count,
        bottom=[i+j for i,j in zip(inside_odd_undetected_count, inside_odd_count)],
        label='Misclassified Inside',
        color='#b7c4dc',
        edgecolor='black',
        linewidth=5)
plt.bar(llm_display_names,
        inside_odd_undetected_count,
        bottom=inside_odd_count,
        label='Misclassified Outside',
        color='#fb9a99',
        edgecolor='black',
        linewidth=5)
plt.bar(llm_display_names,
        inside_odd_count,
        label='Inside ODD',
        color='C3',
        edgecolor='black',
        linewidth=5)

# Set tick and label size
plt.xticks(llm_display_names, fontsize=30)
plt.tick_params(axis='y', labelsize=30) 
plt.ylabel('Number of Tests', fontsize=48)
plt.xlabel('Classification Provider', fontsize=48)

# Set the y-limit
plt.ylim([0, args.size//2])

# Add legend
if args.dataset == "External_jutah":
    plt.legend(fontsize=40, loc="upper right")

# Display the plot
plt.tight_layout()
plt.savefig(f"./results/rq1c_{args.dataset}_{args.baseline}.png")
if args.show_plot:
    plt.show()
plt.close()
