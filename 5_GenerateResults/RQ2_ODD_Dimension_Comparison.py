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
parser.add_argument('--dataset',
                    type=str,
                    choices=['External_Jutah', 'OpenPilot_2k19', 'OpenPilot_2016'],
                    required=True)
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
DATASET_DIRECTORY = os.path.join(args.dataset_directory, args.dataset)

# Get all the available annotator
available_annotators_paths = glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/*")
available_annotators       = [os.path.basename(annotator) for annotator in available_annotators_paths]
if "Individual_Human" in available_annotators:
    available_annotators.remove("Individual_Human")
available_annotators       = sorted(available_annotators)

compliance_per_annotator = []
for annotator in available_annotators:
    # Get all the files based on the filter:
    if args.description_filter == "Both":
        descriptions = glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/{annotator}/*_output.txt")
    elif args.description_filter == "Pass":
        descriptions = glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/{annotator}/pass_*_output.txt")
    elif args.description_filter == "Fail":
        descriptions = glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/{annotator}/fail_*_output.txt")
    
    # Get the coverage vectors
    dl = DescriptionLoader(descriptions)

    # Compute the compliance
    in_odd_count = dl.coverage_vector.shape[0] - np.count_nonzero(dl.coverage_vector, axis=0)
    compliance = (in_odd_count / dl.coverage_vector.shape[0]) * 100

    # Save the coverage vectors
    compliance_per_annotator.append(compliance)

# Wrap the compliance vectors around for plotting
compliance_per_annotator_wrapped = []
for compliance in compliance_per_annotator:
    value = np.append(compliance, compliance[:1])
    compliance_per_annotator_wrapped.append(value)

# Define the number of variables
odd_keys = sorted(ODD.keys())
num_vars = len(odd_keys)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] # Repeat the first angle to close the circle

# Initialize the spider plot
fig, ax = plt.subplots(figsize=(20, 16), subplot_kw=dict(polar=True))

# Set the name
plt.get_current_fig_manager().set_window_title(f'{args.dataset} - {args.description_filter}')

# Plot data
for i in range(len(available_annotators)):
    annotator = available_annotators[i]
    compliance = compliance_per_annotator_wrapped[i]
    ax.plot(angles, compliance, linewidth=10, linestyle='solid', label=annotator)
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
plt.show()