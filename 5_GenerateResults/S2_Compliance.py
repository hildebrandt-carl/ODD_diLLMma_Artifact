import os
import sys
import glob
import argparse

from math import pi

import numpy as np
import matplotlib.pyplot as plt


# Import Common
current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from constants import ODD
from description_loader import DescriptionLoader


# Get the Data
parser = argparse.ArgumentParser(description="Displays a comparison between different ODD dimensions")
parser.add_argument('--annotator',
                    type=str,
                    choices=['Human', 'ChatGPT_Base', 'Llama_Base', 'Vicuna_Base', 'Llama_Plus', 'Vicuna_Plus'],
                    required=True,
                    help="The annotator to use. Choose between 'Human', 'ChatGPT_Base', 'Llama_Base', 'Vicuna_Base', 'Llama_Plus', 'Vicuna_Plus'.")
parser.add_argument('--description_filter',
                    choices=['Both', 'Pass', 'Fail'],
                    required=True,
                    help="Filter results by status: 'Both' to include both passing and failing, 'Pass' for passing only, or 'Fail' for failing only.")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")
# Get the dataset directory
DATASET_DIRECTORY = args.dataset_directory

# Get all the available datasets
available_datasets_paths = glob.glob(f"{DATASET_DIRECTORY}/*")
available_datasets = [os.path.basename(dset) for dset in available_datasets_paths]
available_datasets = sorted(available_datasets)

compliance_per_datasets = []
for dataset in available_datasets:
    # Get all the files based on the filter:
    if args.description_filter == "Both":
        descriptions = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{args.annotator}/*_output.txt")
    elif args.description_filter == "Pass":
        descriptions = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{args.annotator}/pass_*_output.txt")
    elif args.description_filter == "Fail":
        descriptions = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{args.annotator}/fail_*_output.txt")
    
    # Get the coverage vectors
    dl = DescriptionLoader(descriptions)

    # Count all IN ODD
    in_odd_count_per_dimension = dl.coverage_vector.shape[0] - np.count_nonzero(dl.coverage_vector, axis=0)
    compliance_per_dimension = (in_odd_count_per_dimension / dl.coverage_vector.shape[0]) * 100

    # Save the compliance
    compliance_per_datasets.append(compliance_per_dimension)

# Wrap the compliance vectors around for plotting
compliance_per_annotator_wrapped = []
for compliance in compliance_per_datasets:
    value = np.append(compliance, compliance[:1])
    compliance_per_annotator_wrapped.append(value)

# Define the number of variables
odd_keys = ODD.keys()
num_vars = len(odd_keys)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] # Repeat the first angle to close the circle

# Initialize the spider plot
fig, ax = plt.subplots(figsize=(20, 16), subplot_kw=dict(polar=True))

# Set the name
plt.get_current_fig_manager().set_window_title(f'{args.annotator} - {args.description_filter}')

# Plot data
for i in range(len(available_datasets)):
    dataset = available_datasets[i]
    compliance = compliance_per_annotator_wrapped[i]
    ax.plot(angles, compliance, linewidth=10, linestyle='solid', label=dataset.replace("_", " "))
    ax.fill(angles, compliance, f'C{i}', alpha=0.1)

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], [o.replace(' ', '\n') for o in odd_keys], verticalalignment='center')

# Set the label axes
ax.set_rlabel_position(30)
plt.yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"], color="grey", size=30)
plt.ylim(0, 100)
plt.tick_params(axis='x', labelsize=45, pad=50)

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=45)
plt.tight_layout()
plt.savefig(f"S2_Compliance_{args.annotator}_{args.description_filter}.png")
plt.show()