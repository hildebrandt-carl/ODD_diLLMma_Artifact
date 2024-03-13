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
                    choices=['Llama', 'Vicuna'],
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
best_f1_score = -np.inf
best_context_question = ""

# For each context
for context in contexts:
    # For each question
    for question in questions:
        
        # Get the descriptions
        search_directory = f"{PROMPT_EXPLORATION_DIRECTORY}/Data/*/5_Descriptions_Subset/{args.annotator}/{context.lower()}/{question.lower()}/*_output.txt"
        descriptions = glob.glob(search_directory)
        descriptions = sorted(descriptions)
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

        # Compute the total number true positives, false positives, and false negatives
        true_positives  = np.sum((true == 1) & (pred == 1))
        false_positives = np.sum((true == 0) & ((pred == 1) | (pred == -1)))
        false_negatives = np.sum((true == 1) & ((pred == 0) | (pred == -1)))

        # Compute the F1 score for human and predicted
        current_f1 = (2 * true_positives) / (2 * true_positives + false_positives + false_negatives)
        print(current_f1)

        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            best_context_question = f"{context} - {question}"

print("=========================================================")
print(f"The best (f1 = {best_f1_score}) context question combination for {args.annotator} is: {best_context_question}")
print("=========================================================")