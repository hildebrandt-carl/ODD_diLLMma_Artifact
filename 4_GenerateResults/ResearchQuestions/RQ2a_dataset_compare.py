import os
import glob
import shutil
import argparse

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils_Naming import LLM_NAMES
from utils_ODDManagement import get_odd
from utils_ViolationState import ViolationState
from utils_ODDManagement import clean_string

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
parser.add_argument('--pass_fail',
                    type=str,
                    required=True,
                    choices=['pass', 'fail', 'both'],
                    help="Whether the selected image is from the passing or failing set (pass, fail)")
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
    file_type = args.pass_fail if args.pass_fail != "both" else ""
    description_files = glob.glob(f"{BASE_PATH}/{llm_model}/{odd[0][0]}/{file_type}*_output.txt")
    description_files = [os.path.basename(f) for f in description_files]

    # Make sure there are some common files
    assert len(description_files) > 0, f"Could not file files for: {llm_model}"

    # Save all the files
    all_files.append(description_files)

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

non_violated_percentage = []
# Compute the percentage that each dimension was not violated for each llm
for llm_index, llm_model in enumerate(requested_llm_models):
    llm_violation_percentage = []
    for odd_index, odd_dimension in enumerate(odd):
        vector = odd_vectors[llm_index, :, odd_index]
        non_violation_count = np.sum(vector == ViolationState.NO_VIOLATION.num)
        violation_count = np.sum(vector != ViolationState.NO_VIOLATION.num)
        total_values = non_violation_count + violation_count
        assert total_values == np.shape(vector)[0]
        percentage_non_violated = (non_violation_count / total_values) * 100
        llm_violation_percentage.append(percentage_non_violated)
    non_violated_percentage.append(llm_violation_percentage)

# Convert to a numpy array
non_violated_percentage = np.array(non_violated_percentage)
array_string = repr(non_violated_percentage)
print("Array used for RQ2b_area_calculations.py")
print(array_string)

# Repeat the first value to close the circular graph
values = np.concatenate([non_violated_percentage, non_violated_percentage[:,0].reshape(len(requested_llm_models), 1)], axis=1)

# Calculate angle for each axis
angles = np.linspace(0, 2 * np.pi, len(odd), endpoint=False).tolist()
angles += angles[:1]

# Initialize the spider plot
fig, ax = plt.subplots(figsize=(20, 16), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], [o[1].replace(' ', '\n') for o in odd], verticalalignment='center')

# Draw y labels
ax.set_rlabel_position(30)
plt.yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"], color="grey", size=30)
plt.ylim(0, 100)
plt.tick_params(axis='x', labelsize=45, pad=30)

# Plot data
for llm_index, llm_model in enumerate(requested_llm_models):
    ax.plot(angles, values[llm_index], linewidth=10, linestyle='solid', label=LLM_NAMES[llm_model])
    ax.fill(angles, values[llm_index], f'C{llm_index}', alpha=0.1)

# Add a legend
# if args.pass_fail != "pass":
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=45)

# Show the plot
plt.tight_layout()
plt.savefig(f"./results/rq2a_{args.dataset}_{args.pass_fail}.png")
if args.show_plot:
    plt.show()