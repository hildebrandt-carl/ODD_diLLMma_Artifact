import os
import sys
import glob
import copy
import argparse

import numpy as np
import matplotlib.pyplot as plt

# Import Common
current_dir = os.path.dirname(__file__)
data_loader_dir = "../../../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from tqdm import tqdm
from numpy.linalg import norm
from description_loader import DescriptionLoader


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Used to generate the final human annotations")
parser.add_argument('--annotator',
                    type=str,
                    choices=['Llama', 'Vicuna', 'OpenPilot_2016'],
                    required=True)
parser.add_argument('--prompt_exploration_directory',
                    type=str,
                    default="../0_Results")
parser.add_argument('--dataset_directory',
                    type=str,
                    default="../../../1_Datasets/Data",
                    help="The location of the dataset")
args = parser.parse_args()

print("")
# Get the dataset directory
DATASET_DIRECTORY            = f"{args.dataset_directory}"
PROMPT_EXPLORATION_DIRECTORY = f"{args.prompt_exploration_directory}"

# Get all the prompt combinations
questions_paths = glob.glob(f"{PROMPT_EXPLORATION_DIRECTORY}/Questions/*.txt")
questions = [os.path.basename(f)[:-4] for f in questions_paths]
questions = sorted(questions)
assert len(questions) > 0, "Questions not detected"
contexts_paths = glob.glob(f"{PROMPT_EXPLORATION_DIRECTORY}/Contexts/*.txt")
contexts = [os.path.basename(f)[:-4] for f in contexts_paths]
contexts = sorted(contexts)
assert len(contexts) > 0, "Contexts not detected"

# Saves each metrics
hamming_similarity_array                = []
hamming_row_similarity_array            = []
jaccard_similarity_array                = []
cosine_similarity_array                 = []
unique_row_count_array                  = []
combination_name_array                  = []
matching_ones_minus_false_ones_array    = []

# For each context
for context in tqdm(contexts):
    # For each question
    for question in questions:
        
        # Get the descriptions
        search_directory = f"{PROMPT_EXPLORATION_DIRECTORY}/Data/*/5_Descriptions_Subset/{args.annotator}/{context.lower()}/{question.lower()}/*_output.txt"
        descriptions = glob.glob(search_directory)
        descriptions = sorted(descriptions)

        if len(descriptions) == 0:
            continue

        assert len(descriptions) > 0, f"The following does not have descriptions {context}/{question}"

        # Load the data
        dl = DescriptionLoader(descriptions)

        # Load the human baseline
        HUMAN_BASELINE_PATH = f"{DATASET_DIRECTORY}/*/5_Descriptions/Human"
        human_baseline_filepaths = []
        for filename, filepath in zip(dl.description_names, dl.description_full_paths):
            if "External_Jutah" in filepath:
                dataset = "External_Jutah"
            if "OpenPilot_2k19" in filepath:
                dataset = "OpenPilot_2k19"
            if "OpenPilot_2016" in filepath:
                dataset = "OpenPilot_2016"
            human_baseline_filepaths.append(f"{DATASET_DIRECTORY}/{dataset}/5_Descriptions/Human/{filename}")
        human_dl = DescriptionLoader(human_baseline_filepaths)

        # Get the data
        true = copy.deepcopy(human_dl.coverage_vector)
        pred = copy.deepcopy(dl.coverage_vector)
        true_flat = true.flatten()
        pred_flat = pred.flatten()

        # Compute a bunch of statistics on the data per cell
        hamming_distance = np.mean(true != pred)
        hamming_similarity_array.append(1 - hamming_distance)

        intersection = np.sum(true_flat == pred_flat)
        union = len(true_flat) + len(pred_flat) - intersection
        jaccard_similarity = intersection / union
        jaccard_similarity_array.append(jaccard_similarity)

        cosine_similarity = np.dot(true_flat, pred_flat) / (norm(true_flat) * norm(pred_flat))
        cosine_similarity_array.append(cosine_similarity)

        # Compute the statistics on the data per row
        rows_equal = (true == pred).all(axis=1)
        row_match_rate = np.mean(rows_equal)
        hamming_row_similarity_array.append(row_match_rate)

        # Compute the ratio of uniqueness
        unique_rows_pred = np.unique(pred, axis=0)
        unique_rows_true = np.unique(true, axis=0)
        # unique_row_count_array.append(unique_rows_pred.shape[0]/unique_rows_true.shape[0])
        unique_row_count_array.append(unique_rows_pred.shape[0])

        # Compute basic metric of matched 1's minus false 1s
        correct_ones = (true == 1) & (pred == 1)
        false_ones = (true != 1) & (pred == 1)
        score = np.sum(correct_ones) - np.sum(false_ones)
        total_ones_true = np.sum(true == 1)
        # normalized_score = score / total_ones_true if total_ones_true else 0
        matching_ones_minus_false_ones_array.append(score)

        # Save the combination
        combination_name_array.append(f"{context} - {question}")

# Get the best elements from each list:
hamming_similarity_array                = np.array(hamming_similarity_array)
best_hamming_indices                    = np.argsort(hamming_similarity_array)[-3:]
best_hamming_indices                    = best_hamming_indices[::-1]

jaccard_similarity_array                = np.array(jaccard_similarity_array)
best_jaccard_indices                    = np.argsort(jaccard_similarity_array)[-3:]
best_jaccard_indices                    = best_jaccard_indices[::-1]

cosine_similarity_array                 = np.array(cosine_similarity_array)
best_cosine_indices                     = np.argsort(cosine_similarity_array)[-3:]
best_cosine_indices                     = best_cosine_indices[::-1]

hamming_row_similarity_array            = np.array(hamming_row_similarity_array)
best_hamming_row_indices                = np.argsort(hamming_row_similarity_array)[-3:]
best_hamming_row_indices                = best_hamming_row_indices[::-1]

matching_ones_minus_false_ones_array    = np.array(matching_ones_minus_false_ones_array)
best_matching_ones_indices              = np.argsort(matching_ones_minus_false_ones_array)[-3:]
best_matching_ones_indices              = best_matching_ones_indices[::-1]

combination_name_array   = np.array(combination_name_array)

print("Best prompt combinations per cell:")
print("Hamming:")
for i, index in enumerate(best_hamming_indices):
    print(f"{i+1}) {combination_name_array[index]} - Unique Count: {unique_row_count_array[index]}/{unique_rows_true.shape[0]} - Score: {matching_ones_minus_false_ones_array[index]}/{total_ones_true}")
print("")

print("Jaccard:")
for i, index in enumerate(best_jaccard_indices):
    print(f"{i+1}) {combination_name_array[index]} - Unique Count: {unique_row_count_array[index]}/{unique_rows_true.shape[0]} - Score: {matching_ones_minus_false_ones_array[index]}/{total_ones_true}")
print("")

print("Cosine:")
for i, index in enumerate(best_cosine_indices):
    print(f"{i+1}) {combination_name_array[index]} - Unique Count: {unique_row_count_array[index]}/{unique_rows_true.shape[0]} - Score: {matching_ones_minus_false_ones_array[index]}/{total_ones_true}")
print("")

print("Best prompt combinations Per Row:")
print("Hamming:")
for i, index in enumerate(best_hamming_row_indices):
    print(f"{i+1}) {combination_name_array[index]} - Unique Count: {unique_row_count_array[index]}/{unique_rows_true.shape[0]} - Score: {matching_ones_minus_false_ones_array[index]}/{total_ones_true} ")
print("")


print("Basic metric:")
for i, index in enumerate(best_matching_ones_indices):
    print(f"{i+1}) {combination_name_array[index]} - Unique Count: {unique_row_count_array[index]}/{unique_rows_true.shape[0]} - Score: {matching_ones_minus_false_ones_array[index]}/{total_ones_true}")
print("")

# Create figure and axis
fig, ax = plt.subplots(figsize=(30, 10))

# Set the positions of the bars on the x-axis
index = np.arange(np.shape(combination_name_array)[0])
bar_width = 0.2

# Plotting
plt.bar(index, hamming_similarity_array, bar_width, label='Hamming')
plt.bar(index + bar_width, jaccard_similarity_array, bar_width, label='Jaccard')
plt.bar(index + 2*bar_width, cosine_similarity_array, bar_width, label='Cosine')

# Add some labels, title, and custom x-axis tick labels
plt.xlabel('Models', fontsize=20)
plt.ylabel('Percentage Match', fontsize=20)
plt.xticks(index + bar_width + bar_width/2, combination_name_array, rotation=90)
plt.legend(fontsize=20)
plt.grid()
plt.tight_layout()
plt.show()
