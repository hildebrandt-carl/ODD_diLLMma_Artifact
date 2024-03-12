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


from description_loader import DescriptionLoader

# Get the Data
parser = argparse.ArgumentParser(description="Displays the number of failure inducing inputs")
parser.add_argument('--annotator',
                    type=str,
                    choices=['Human', 'ChatGPT_Base', 'Llama_Base', 'Vicuna_Base'],
                    required=True,
                    help="The annotator to use. Choose between 'Human', 'ChatGPT_Base', 'Llama_Base', 'Vicuna_Base'.")
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
DATASET_DIRECTORY = f"{args.dataset_directory}"

# Get all the available datasets
available_datasets_paths = glob.glob(f"{DATASET_DIRECTORY}/*")
available_datasets = [os.path.basename(dset) for dset in available_datasets_paths]
available_datasets = sorted(available_datasets)

# Used to hold the data
in_odd_count_array = []
out_odd_count_array = []
unknown_count_array = []

# For each of the datasets
for dataset in available_datasets:


    # Get all the files based on the filter:
    if args.description_filter == "Both":
        descriptions = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{args.annotator}/*_output.txt")
    elif args.description_filter == "Pass":
        descriptions = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{args.annotator}/pass_*_output.txt")
    elif args.description_filter == "Fail":
        descriptions = glob.glob(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/{args.annotator}/fail_*_output.txt")

    # Load the coverage vector
    dl = DescriptionLoader(descriptions)
    
    # Count the number in ODD/ out ODD/ and unknown
    in_odd_count  = np.sum(np.all(dl.coverage_vector == 0, axis=1))
    out_odd_count = np.sum(np.all(np.isin(dl.coverage_vector, [0, 1]), axis=1) & ~np.all(dl.coverage_vector == 0, axis=1))
    unknown_count = np.sum(np.any(dl.coverage_vector == -1, axis=1))

    # Save this to the array
    in_odd_count_array.append(in_odd_count)
    out_odd_count_array.append(out_odd_count)
    unknown_count_array.append(unknown_count)

    # Make sure that everything adds up
    assert in_odd_count + out_odd_count + unknown_count == dl.total_descriptions, "ODD classification count invalid"

    # Print the stats
    print(f"Statistics for {dataset}:")
    print(f"Total images in the ODD: {in_odd_count}/{dl.total_descriptions} - {np.round((in_odd_count/dl.total_descriptions) * 100, 0)}%")
    print(f"Total images out of the ODD: {out_odd_count}/{dl.total_descriptions} - {np.round((out_odd_count/dl.total_descriptions) * 100, 0)}%")
    print(f"Total images unknown relative to ODD: {unknown_count}/{dl.total_descriptions} - {np.round((unknown_count/dl.total_descriptions) * 100, 0)}%")
    print("")

# Convert data to arrays
in_odd_count_array  = np.array(in_odd_count_array)
out_odd_count_array = np.array(out_odd_count_array)
unknown_count_array = np.array(unknown_count_array)

# Create a plot
fig = plt.figure(figsize=(17, 12))
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='1', color='gray')
plt.grid(which='minor', linestyle='--', linewidth='0.5', color='gray')

# Plot 'Inside ODD'
bar_inside = plt.bar(available_datasets,
                     in_odd_count_array,
                     label='Inside ODD',
                     color='C3',
                     edgecolor='black',
                     linewidth=5)

if np.any(unknown_count_array > 0):  # Plot 'Unknown' only if any value is greater than 0
    # Adjust the bottom for 'Outside ODD' bars to be on top of 'Unknown'
    bottom_for_outside = in_odd_count_array + unknown_count_array
    # Plot 'Unknown'
    bar_unknown = plt.bar(available_datasets,
                          unknown_count_array,
                          bottom=in_odd_count_array,
                          label='Unknown',
                          color='C2',
                          edgecolor='black',
                          linewidth=5)
else:  # If 'Unknown' is not plotted, 'Outside ODD' starts on top of 'Inside ODD'
    bottom_for_outside = in_odd_count_array

# Plot 'Outside ODD'
bar_outside = plt.bar(available_datasets,
                      out_odd_count_array,
                      bottom=bottom_for_outside,
                      label='Outside ODD',
                      color='C0',
                      edgecolor='black',
                      linewidth=5)

# Plot each of the labels with path effects for outlining
text_path_effects = [PathEffects.withStroke(linewidth=3, foreground="black")]
bar_width = 0.8  # This is the default value for plt.bar's width parameter
offset = bar_width / 4  # Adjust this value to move the 'Unknown' label position

for dataset_index, dataset_name in enumerate(available_datasets):
    base_height = in_odd_count_array[dataset_index]

    # Position label for 'Inside ODD'
    text = plt.text(dataset_index, base_height, '%d' % base_height,
                    ha='left', va='bottom', fontsize=50, color="C3")
    text.set_path_effects(text_path_effects)

    if np.any(unknown_count_array > 0):
        unknown_height = unknown_count_array[dataset_index]
        cum_height_for_unknown = base_height + unknown_height
        # Adjust x-coordinate for 'Unknown' label to make it left-aligned
        # with an offset to position it correctly.
        text = plt.text(dataset_index - offset, cum_height_for_unknown, '%d' % unknown_height,
                        ha='left', va='bottom', fontsize=50, color="C2")
        text.set_path_effects(text_path_effects)
        # Update base height for 'Outside ODD'
        base_height += unknown_height

    # Height and label for 'Outside ODD'
    outside_height = out_odd_count_array[dataset_index]
    cum_height_for_outside = base_height + outside_height
    text = plt.text(dataset_index, cum_height_for_outside, '%d' % outside_height,
                    ha='center', va='bottom', fontsize=50, color="C0")
    text.set_path_effects(text_path_effects)

# Setting tick and label size
plt.xticks(available_datasets, fontsize=30)
plt.tick_params(axis='y', labelsize=30)
plt.ylabel('Number of Failure-Inducing Inputs', fontsize=42)
plt.xlabel('Dataset', fontsize=42)

# Add legend
plt.legend(fontsize=35)

# Set the Y Limit
plt.ylim([0, dl.total_descriptions]) 

# Display the plot
plt.tight_layout()
plt.show()
plt.close()
