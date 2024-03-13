import os
import sys
import glob
import argparse

import numpy as np

# Import Common
current_dir = os.path.dirname(__file__)
data_loader_dir = "../../../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from description_loader import DescriptionLoader


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Used to generate the final human annotations")
parser.add_argument('--dataset',
                    type=str,
                    choices=['External_Jutah', 'OpenPilot_2k19', 'OpenPilot_2016'],
                    required=True)
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

# Get all the available datasets
human_baselines_paths = glob.glob(f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions/Human/*.txt")
human_baselines = [os.path.basename(f) for f in human_baselines_paths]
human_baselines = sorted(human_baselines)
assert len(human_baselines) > 0, "Human baseline not detected"

# # Load the human data
# human_dl = DescriptionLoader(human_baselines_paths)
# bool_arr = (human_dl.coverage_vector == -1)
# rows_with_minus_one = np.any(bool_arr, axis=1)
# count = np.sum(rows_with_minus_one)
# print(f"Human number of unknowns: {count}/{np.shape(human_dl.coverage_vector)[0]}")

# Get all the prompt combinations
questions_paths = glob.glob(f"{PROMPT_EXPLORATION_DIRECTORY}/Questions/*.txt")
questions = [os.path.basename(f)[:-4] for f in questions_paths]
questions = sorted(questions)
assert len(questions) > 0, "Questions not detected"
contexts_paths = glob.glob(f"{PROMPT_EXPLORATION_DIRECTORY}/Contexts/*.txt")
contexts = [os.path.basename(f)[:-4] for f in contexts_paths]
contexts = sorted(contexts)
assert len(contexts) > 0, "Contexts not detected"

# For each context
for context in contexts:
    # For each question
    for question in questions:
        

        # Get the descriptions
        search_directory = f"{PROMPT_EXPLORATION_DIRECTORY}/Data/{args.dataset}/5_Descriptions_Subset/{args.annotator}/{context.lower()}/{question.lower()}/*_output.txt"
        descriptions = glob.glob(search_directory)

        if len(descriptions) == 0:
            continue

        assert len(descriptions) > 0, f"The following does not have descriptions {context}/{question}"

        # Load the data
        dl = DescriptionLoader(descriptions)

        bool_arr = (dl.coverage_vector == -1)
        rows_with_minus_one = np.all(bool_arr, axis=1)
        count = np.sum(rows_with_minus_one)

        print(f"{context} - {question} of unknowns: {count}/{np.shape(dl.coverage_vector)[0]}")

        # Compare it to the human baseline
        # Compare what????
            # Hamming 0.25
            # Cosine 0.25
            # Percentage Match 0.25
            # Jaccard?? 0.25
            # Find the one which works best
        # Changing the how we do the parsing will change things here!!!!!
        # Plot the results

