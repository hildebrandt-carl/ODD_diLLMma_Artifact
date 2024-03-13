import os
import sys
import glob
import copy
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

# Get all the available annotator
available_annotators_paths = glob.glob(f"{DATASET_DIRECTORY}/*/5_Descriptions/*")
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

in_out_odd_bar_array  = []
exact_match_array       = []

f1_score_per_dimension = []
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

    # Compute some metrics
    exact_match_vectors = np.all(true == pred, axis=1)
    matching_rows_count = np.sum(exact_match_vectors)

    # Find the true in and out of ODD
    true_out_odd = np.any(true == 1, axis=1)
    true_out_odd_count = np.sum(true_out_odd)
    true_in_odd = np.all(true == 0, axis=1)
    true_in_odd_count = np.sum(true_in_odd)

    # Find how many predictions were correct
    correct_out_odd_pred_count = np.sum(np.any(pred[true_out_odd] == 1, axis=1))
    correct_in_odd_pred_count  = np.sum(np.all(pred[true_in_odd] == 0, axis=1))
    
    # Total in and out of ODD predictions
    correct_in_out_pred_count = correct_out_odd_pred_count + correct_in_odd_pred_count

    # Print the data
    print(f"Annotator: {annotator}")
    print(f"Total annotations: {np.shape(true)[0]}")
    print(f"Correctly identified (In/Out of ODD): {correct_in_out_pred_count}/{np.shape(true)[0]}")
    print(f"In ODD correctly identified: {correct_in_odd_pred_count}/{true_in_odd_count}")
    print(f"Out of ODD correctly identified: {correct_out_odd_pred_count}/{true_out_odd_count}")
    print(f"Exact ODD dimensions match: {matching_rows_count}/{np.shape(true)[0]}")
    print("")

    in_out_odd_bar_array.append([correct_in_odd_pred_count,
                                 correct_out_odd_pred_count,
                                 true_in_odd_count - correct_in_odd_pred_count,
                                 true_out_odd_count - correct_out_odd_pred_count])

    exact_match_array.append([matching_rows_count,
                              np.shape(true)[0]-matching_rows_count])

# Convert to np array
in_out_odd_bar_array = np.array(in_out_odd_bar_array)
exact_match_array = np.array(exact_match_array)

# Create the figure
plt.figure(figsize=(10,8))

# Assuming data definitions are given above
barWidth = 0.3
gap = 0.025
r1 = np.arange(len(available_annotators)) - gap / 2
r2 = [x + barWidth + gap for x in r1]

# Definitions for colors and hatches
left_stack_labels = ['Inside ODD Identified', 'Outside ODD Identified', 'Inside ODD Missed', 'Outside ODD Missed']
right_stack_labels = ['Exact ODD Match', 'ODD Match Missed']

left_colors             = ['C0', 'C3', 'None', 'none']
left_hatches            = ['', '', '////', '....']
left_hatch_colors       = ['C0', 'C3', 'C0', 'C3']

right_colors            = ['C1', 'none']
right_hatches           = ['', 'xxxx'] 
right_hatch_colors      = ['C1', 'C1']

# Loop over each group
for group_index, _ in enumerate(available_annotators):
    bottom = 0
    
    plt.bar(r1[group_index], in_out_odd_bar_array[group_index, 0], bottom=bottom, color=left_colors[0], edgecolor=left_hatch_colors[0], width=barWidth, hatch=left_hatches[0], label=left_stack_labels[0] if group_index == 0 else "")
    bottom += in_out_odd_bar_array[group_index, 0]

    plt.bar(r1[group_index], in_out_odd_bar_array[group_index, 1], bottom=bottom, color=left_colors[1], edgecolor=left_hatch_colors[1], width=barWidth, hatch=left_hatches[1], label=left_stack_labels[1] if group_index == 0 else "")
    bottom += in_out_odd_bar_array[group_index, 1]

    plt.bar(r1[group_index], in_out_odd_bar_array[group_index, 2], bottom=bottom, color=left_colors[2], edgecolor=left_hatch_colors[2], width=barWidth, hatch=left_hatches[2], label=left_stack_labels[2] if group_index == 0 else "")
    bottom += in_out_odd_bar_array[group_index, 2]

    plt.bar(r1[group_index], in_out_odd_bar_array[group_index, 3], bottom=bottom, color=left_colors[3], edgecolor=left_hatch_colors[3], width=barWidth, hatch=left_hatches[3], label=left_stack_labels[3] if group_index == 0 else "")

# Plot right bars ("Exact Match" and "Missing Match") for each group individually
for group_index, _ in enumerate(available_annotators):
    bottom = 0
    
    plt.bar(r2[group_index], exact_match_array[group_index, 0], bottom=bottom, color=right_colors[0], edgecolor=right_hatch_colors[0], width=barWidth, hatch=right_hatches[0], label=right_stack_labels[0] if group_index == 0 else "")
    bottom += exact_match_array[group_index, 0]

    plt.bar(r2[group_index], exact_match_array[group_index, 1], bottom=bottom, color=right_colors[1], edgecolor=right_hatch_colors[1], width=barWidth, hatch=right_hatches[1], label=right_stack_labels[1] if group_index == 0 else "")

# Add labels, title, and legend
plt.xlabel('LLM', size=20)
plt.xticks([r + barWidth/2 for r in range(len(available_annotators))], [ann.replace("_", " ") for ann in available_annotators], size=20)
plt.yticks(np.arange(0,1501,100), size=20)
plt.ylabel('Total Images', size=20)
plt.ylim([0,1501])
plt.tight_layout()
plt.legend(fontsize=20)
plt.grid()

plt.savefig(f"RQ2_ODD_Vector_Comparison_{args.description_filter}.png")
plt.show()


