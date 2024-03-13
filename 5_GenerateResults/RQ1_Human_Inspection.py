import os
import sys
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# Import Common
current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from functools import reduce
from description_loader import DescriptionLoader

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import numpy as np

# Get the Data
parser = argparse.ArgumentParser(description="Displays the number of failure inducing inputs")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")
# Get the dataset directory
DATASET_DIRECTORY = f"{args.dataset_directory}"

# Get all the available datasets
available_datasets_paths = glob.glob(f"{DATASET_DIRECTORY}/*")
available_datasets = [os.path.basename(dset) for dset in available_datasets_paths]
available_datasets = sorted(available_datasets)

# Get all the available annotators
available_annotator_sets = []
for dataset in available_datasets:
    available_annotators_paths = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/*")
    available_annotators_names = [os.path.basename(annotator) for annotator in available_annotators_paths]
    available_annotator_sets.append(set(available_annotators_names))

common_annotators = reduce(set.intersection, available_annotator_sets)
assert "Human" in common_annotators, "This script requires a Human baseline"
common_annotators.discard("Individual_Human") # Remove Individual_Human
common_annotators.discard("Human") # Remove Individual_Human
common_annotators = sorted(list(common_annotators))

# Step1 A: Compute all filenames which are flagged as in ODD by the annotator
human_inspection_filenames = []
for dataset in available_datasets:
    # Keep track of this datasets requiring human inspection
    dataset_specific_human_inspection_filenames = []
    # For each annotator
    for annotator in common_annotators:
        # Get all the failing files
        failing_descriptions = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{annotator}/fail_*_output.txt")
        # Load the coverage vector
        dl = DescriptionLoader(failing_descriptions)
        # Get the indices which are flagged as in ODD
        require_inspection_indices  = np.where(np.all(dl.coverage_vector == 0, axis=1))[0]
        # Get the associated filenames
        filenames_requiring_inspection = dl.get_filenames_from_indices(require_inspection_indices)
        # Save this information
        dataset_specific_human_inspection_filenames.append(filenames_requiring_inspection)
    human_inspection_filenames.append(dataset_specific_human_inspection_filenames)

# Step2:Compute filenames which are flagged as in ODD by the Human
human_in_odd_filenames = []
for dataset in available_datasets:
    # Get all the failing files
    failing_descriptions = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/Human/fail_*_output.txt")
    # Load the coverage vector
    dl = DescriptionLoader(failing_descriptions)
    # Get the indices which are flagged as in ODD
    require_inspection_indices  = np.where(np.all(dl.coverage_vector == 0, axis=1))[0]
    # Get the associated filenames
    filenames_requiring_inspection = dl.get_filenames_from_indices(require_inspection_indices)
    # Save this information
    human_in_odd_filenames.append(filenames_requiring_inspection)

# Step3: Compute the overlap between flagged as unknown or in ODD by the annotator and within ODD by the human
true_in_odd_failures_found_filenames = []
for dataset_index, dataset in enumerate(available_datasets):
    dataset_specific_true_in_odd_failures_found_filenames = []
    for annotator_index, annotator in enumerate(common_annotators):
        # Get the number requiring inspection and the true in odd
        required_inspection = set(human_inspection_filenames[dataset_index][annotator_index])
        true_in_odd         = set(human_in_odd_filenames[dataset_index])
        # Compute the intersection
        failures_in_odd_found = required_inspection & true_in_odd
        # Save the data
        dataset_specific_true_in_odd_failures_found_filenames.append(sorted(list(failures_in_odd_found)))
    true_in_odd_failures_found_filenames.append(dataset_specific_true_in_odd_failures_found_filenames)

# Convert everything to lengths for plotting
number_of_human_inspection     = [[len(s) for s in sublist] for sublist in human_inspection_filenames]
number_of_true_in_odd_failures = [[len(s) for s in sublist] for sublist in true_in_odd_failures_found_filenames]

# Append human information onto each list
for item in number_of_human_inspection:
    item.append(len(failing_descriptions)) # Humans would have to inspect all failures

for item_index, item in enumerate(number_of_true_in_odd_failures):
    item.append(len(human_in_odd_filenames[item_index])) # Humans would find all failures in ODD

total_images = [len(failing_descriptions) for dataset in available_datasets]
total_failures = [len(num_failures) for num_failures in human_in_odd_filenames]

# At this point we have the following
# print(number_of_human_inspection)
# print(number_of_true_in_odd_failures)
# print(total_images)
# print(total_failures)

colors = ['C0', 'C1', 'C2', 'C3']
shapes = ['o', 's', '^'] 

# Create a figure
plt.figure(figsize=(17, 12))

# Create broken axes
bax = brokenaxes(ylims=((-1, 41), (69, 101)), xlims=((-1, 41), (69, 101)), despine=False)

for i in range(3):
    for j in range(4):
        # Set the color and shape
        s = shapes[i]
        c = colors[j]

        bax.scatter((number_of_human_inspection[i][j]/total_images[i])*100, (number_of_true_in_odd_failures[i][j]/total_failures[i])*100, marker=s, color=c, s=1000)


x = np.arange(100)
bax.plot(x, x, linestyle="dashed", color='C3', linewidth=6)

# Create custom legends
shape_legend = [mlines.Line2D([0], [0], color='black', marker=shape, linestyle='None', markersize=35) for shape in shapes]
color_legend = [mpatches.Patch(color=color) for color in colors[:3]]

# Custom line for 'Human' with dashed red line
human_line = mlines.Line2D([], [], color='red', linestyle='dashed', linewidth=6)

# Add 'Human' line to the color legend
color_legend.append(human_line)

# Add legends to the plot
available_datasets_labels = [dset.replace("_", " ") for dset in available_datasets]
legend1 = plt.legend(shape_legend, available_datasets_labels, loc='upper left', fontsize=35)
plt.gca().add_artist(legend1)  
common_annotators_labels = [ann.replace("_", " ") for ann in common_annotators]
common_annotators_labels += ["Human"]
plt.legend(color_legend, common_annotators_labels, loc='lower right', fontsize=35)

# Customize the plot
bax.set_xlabel('Images Requiring Human Inspection (%)', fontsize=35, labelpad=50)
bax.set_ylabel('Failure-Inducing Inputs in ODD (%)', fontsize=35, labelpad=50)

bax.tick_params(axis='both', which='major', labelsize=30)

# Show the grid
bax.grid()

plt.savefig("RQ1_Human_Inspection.png")
plt.show()
plt.close()