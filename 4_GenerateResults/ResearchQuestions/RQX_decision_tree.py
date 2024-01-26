import glob

import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from utils_ODDManagement import get_odd
from utils_ODDManagement import update_odd_vector


# Declare constants
DATASET_DIRECTORY = "../1_Datasets/Data"

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Creates a decision tree.")
parser.add_argument('--llm_model', 
                    type=str,
                    choices=['llama', 'vicuna', 'chat_gpt', 'human'],
                    help="The LLMs to use (llama, vicuna, chat_gpt, human, researcher_a, researcher_b, researcher_c)")
parser.add_argument('--dataset',
                    type=str,
                    choices=['OpenPilot_2k19', 'External_jutah', 'OpenPilot_2016'],
                    help="The dataset you want to process (OpenPilot_2k19, External_jutah, OpenPilot_2016)")
parser.add_argument('--show_plot',
                    action='store_true', 
                    help="Display the plot")
args = parser.parse_args()

# Define the questions and labels
odd = get_odd()    

# Define the base path
base_path = f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions/{args.llm_model}"

# Determine how many files there are
number_description_files = len(glob.glob(f"{base_path}/{odd[0][0]}/*_output.txt"))

# Make sure we can find some passing and failing tests
assert number_description_files > 0, "Cant find files files"

# Load the output
odd_vector = np.zeros((number_description_files, len(odd)), dtype=int)

# Holds the labels
labels = np.full(number_description_files,-1)

# For each of the questions
for odd_index, odd_q in enumerate(odd):

    # Load the current questions files
    description_files = glob.glob(f"{base_path}/{odd_q[0]}/*_output.txt")
    description_files = sorted(description_files)

    # Update the ODD with the files
    odd_vector = update_odd_vector(odd_vector, description_files, odd_index)

    # Update the labels
    for i in range(np.shape(labels)[0]):
        filename = description_files[i]
        labels[i] = 1 if "PASS" in filename.upper() else 0

# Create the decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(odd_vector, labels)

features = [f[1] for f in odd]

fig, ax = plt.subplots(figsize=(35, 20))
tree.plot_tree(clf,
               feature_names=features,
               max_depth=None,
               fontsize=12,
               class_names=["Fail", "Pass"],
               impurity=False,
               filled=True,
               proportion=False,
               rounded=True)

# Show the plot
plt.tight_layout()
plt.savefig(f"./results/rq3a_{args.dataset}_{args.llm_model}.png")
if args.show_plot:
    plt.show()
plt.close()
