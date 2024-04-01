import os
import sys
import glob
import copy
import argparse

from math import pi

import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from constants import ODD
from constants import DATASET_ORDER
from constants import ANNOTATOR_LINES
from constants import ANNOTATOR_COLOR
from constants import ANNOTATOR_NAMING
from description_loader import DescriptionLoader


# Get the Data
parser = argparse.ArgumentParser(description="Displays a comparison between different ODD dimensions")
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
available_datasets_paths = [path for path in available_datasets_paths if os.path.isdir(path)]
available_datasets = [os.path.basename(dset) for dset in available_datasets_paths]
available_datasets = sorted(available_datasets, key=lambda x: DATASET_ORDER.get(x, float('inf')))

# Get all the available annotator
available_annotators_paths = glob.glob(f"{DATASET_DIRECTORY}/*/5_Descriptions/*")
available_annotators_paths = [path for path in available_annotators_paths if os.path.isdir(path)]
available_annotators       = [os.path.basename(annotator) for annotator in available_annotators_paths]
available_annotators       = [annotator for annotator in available_annotators if "Human" not in annotator]
available_annotators       = list(set(available_annotators))
available_annotators       = sorted(available_annotators)

# Load the human baseline data
if args.description_filter == "Both":
    descriptions = sorted(glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/Human/*_output.txt"))
elif args.description_filter == "Pass":
    descriptions = sorted(glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/Human/pass_*_output.txt"))
elif args.description_filter == "Fail":
    descriptions = sorted(glob.glob(f"{DATASET_DIRECTORY}/5_Descriptions/Human/fail_*_output.txt"))
baseline_dl = DescriptionLoader(descriptions)

print("Computing accuracy per dimension:")
accuracy_per_dimension_array = []
for annotator in available_annotators:

    all_annotator_descriptions = []
    all_human_descriptions = []
    for dataset in available_datasets:
        # Get all the files based on the filter:
        if args.description_filter == "Both":
            annotator_descriptions  = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{annotator}/*_output.txt")
            human_descriptions      = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/Human/*_output.txt")
        elif args.description_filter == "Pass":
            annotator_descriptions  = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{annotator}/pass_*_output.txt")
            human_descriptions      = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/Human/pass_*_output.txt")
        elif args.description_filter == "Fail":
            annotator_descriptions  = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{annotator}/fail_*_output.txt")
            human_descriptions      = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/Human/fail_*_output.txt")
        all_annotator_descriptions += annotator_descriptions
        all_human_descriptions += human_descriptions
    
    # Sort the data
    all_annotator_descriptions = sorted(all_annotator_descriptions)
    all_human_descriptions = sorted(all_human_descriptions)
    assert len(all_annotator_descriptions) == len(all_human_descriptions), "Annotator and human's description files are not the same length"

    # Make sure they align
    for f1, f2 in zip(all_annotator_descriptions, all_human_descriptions):
        f1_base = os.path.basename(f1)
        f2_base = os.path.basename(f2)
        assert f1_base == f2_base, f"Filenames don't match {f1_base} - {f2_base}"

    # Get the coverage vectors
    annotator_dl    = DescriptionLoader(all_annotator_descriptions)
    human_dl        = DescriptionLoader(all_human_descriptions)

    # Get the data
    true = copy.deepcopy(human_dl.coverage_vector)
    pred = copy.deepcopy(annotator_dl.coverage_vector)

    # Compute the total number true positives, false positives, and false negatives
    true_negatives  = np.sum((true == 0) & (pred == 0), axis=0)
    true_positives  = np.sum((true == 1) & (pred == 1), axis=0)
    false_positives = np.sum((true == 0) & ((pred == 1) | (pred == -1)), axis=0)
    false_negatives = np.sum((true == 1) & ((pred == 0) | (pred == -1)), axis=0)

    # Compute the accuracy
    numerator               = (true_negatives + true_positives)
    denominator             = (true_negatives + true_positives + false_positives + false_negatives) 
    accuracy_per_dimension  = (numerator / denominator) * 100

    # Save the data 
    accuracy_per_dimension_array.append(accuracy_per_dimension)
    
    # Print the data
    print(f"\nAnnotator: {annotator}")
    for i, odd in enumerate(ODD.keys()):
        print(f"{odd} Accuracy: {np.round(accuracy_per_dimension[i], 2)}%")



# Wrap the compliance vectors around for plotting
accuracy_per_dimension_array_wrapped = []
for accuracy in accuracy_per_dimension_array:
    value = np.append(accuracy, accuracy[:1])
    accuracy_per_dimension_array_wrapped.append(value)

# Define the number of variables
odd_keys = ODD.keys()
num_vars = len(odd_keys)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] # Repeat the first angle to close the circle

# Initialize the spider plot
fig, ax = plt.subplots(figsize=(23, 18), subplot_kw=dict(polar=True))

# Set the name
plt.get_current_fig_manager().set_window_title(f'{args.description_filter}')

# Plot data
for i in range(len(available_annotators)):
    annotator = available_annotators[i]
    accuracy = accuracy_per_dimension_array_wrapped[i]
    ax.plot(angles, accuracy, linewidth=10, linestyle=ANNOTATOR_LINES[annotator], c=ANNOTATOR_COLOR[annotator], label=ANNOTATOR_NAMING[annotator])
    ax.fill(angles, accuracy, c=ANNOTATOR_COLOR[annotator], alpha=0.1)

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], [o.replace(' ', '\n') for o in odd_keys], verticalalignment='center')

# Set the label axes
ax.set_rlabel_position(30)
plt.yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", ""], color="black", size=40)
plt.ylim(0, 100)
plt.tick_params(axis='x', labelsize=45, pad=50)


# Use the reordered handles and labels in plt.legend()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=45, ncol=3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2) 
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), fontsize=45, ncol=3)


os.makedirs("./output_graphs", exist_ok=True)
plt.savefig(f"./output_graphs/RQ2_ODD_Dimension_Comparison_{args.description_filter}.png")
plt.savefig(f"./output_graphs/RQ2_ODD_Dimension_Comparison_{args.description_filter}.svg")
plt.savefig(f"./output_graphs/RQ2_ODD_Dimension_Comparison_{args.description_filter}.pdf")
plt.show()