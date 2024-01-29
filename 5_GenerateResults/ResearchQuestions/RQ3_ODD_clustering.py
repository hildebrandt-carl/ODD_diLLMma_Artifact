import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


from tqdm import tqdm

from utils_Naming import LLM_NAMES
from utils_ODDManagement import get_odd
from utils_ODDManagement import clean_string
from utils_ViolationState import ViolationState
from utils_EmbeddingModel import get_embedding_model

from utils_Clustering import compute_random_cluster_consistency
from utils_Clustering import compute_kmeans_cluster_consistency

# Declare constants
DATASET_DIRECTORY = "../1_Datasets/Data"

# Get the embedding model
EMBEDDING_MODEL = get_embedding_model()

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Performs clustering.")
parser.add_argument('--llm_models', 
                    type=str,
                    default="",
                    help="The LLMs to use (llama, vicuna, chat_gpt, human, researcher_a, researcher_b, researcher_c)")
parser.add_argument('--max_clusters',
                    type=int,
                    default=25,
                    help="The maximum number of clusters to consider")
parser.add_argument('--clustering_iterations',
                    type=int,
                    default=10,
                    help="The number of clustering iterations for plotting")
parser.add_argument('--dataset',
                    type=str,
                    choices=['OpenPilot_2k19', 'External_jutah', 'OpenPilot_2016'],
                    help="The dataset you want to process (OpenPilot_2k19, External_jutah, OpenPilot_2016)")
parser.add_argument('--size',
                    type=int,
                    default=-1,
                    help="The size of the dataset you want to use")
parser.add_argument('--show_plot',
                    action='store_true', 
                    help="Display the plot")
args = parser.parse_args()

# Make sure you have set a dataset size
assert args.size > 0, "Dataset size can not be less than or equal to 0"

# Get the list of datasets
possible_datasets = os.listdir(DATASET_DIRECTORY)

# Make sure the dataset exists
assert args.dataset in possible_datasets, f"The dataset was not found in `{DATASET_DIRECTORY}`"

# Create the base path based on the parameters
BASE_PATH = f"{DATASET_DIRECTORY}/{args.dataset}/5_Descriptions/{args.size}"
IMG_PATH  = f"{DATASET_DIRECTORY}/{args.dataset}/4_SelectedData/{args.size}"

# Make sure all the llm_models exist
possible_llm_models = os.listdir(BASE_PATH)
requested_llm_models = args.llm_models.split(", ")
for llm_model in requested_llm_models:
    assert llm_model in possible_llm_models, (
        f"The LLM model {llm_model} was not found in: `{BASE_PATH}`"
        )

# Define the questions and labels
odd = get_odd()   

# Get the list of all files
all_files = []
for llm_model in requested_llm_models:
    # Determine how many files there are
    description_files = glob.glob(f"{BASE_PATH}/{llm_model}/{odd[0][0]}/*_output.txt")
    description_files = [os.path.basename(f) for f in description_files]

    # Make sure there are some common files
    assert len(description_files) > 0, f"Could not file files for: {llm_model}"

    # Save all the files
    all_files.append(description_files)

# Get the list of all common files
common_files = set(all_files[0])
for lst in all_files[1:]:
    common_files &= set(lst)

# Convert it back to a list
common_files = list(common_files)
common_files = sorted(common_files)

# Generate the labels file
labels = []
for file in common_files:
    if "fail" in file:
        labels.append(0)
    elif "pass" in file:
        labels.append(1)
    else:
        print("Something went wrong") 

# Make sure there are some common files
assert len(common_files) > 0, "There are no files in common"

# Holds the output
odd_vectors             = np.zeros((len(requested_llm_models), len(common_files), len(odd)), dtype=int)
description_embedding   = []

# Create the x axis for plotting
x = np.arange(1, args.max_clusters+1, 1)

# Populate the vector
for llm_index, llm_model in enumerate(requested_llm_models):

    llm_descriptions = []

    # For each of the files 
    print(f"Processing: {llm_model}")
    for file_index, filename in tqdm(enumerate(common_files), total=len(common_files)):

        # For each of the questions
        for odd_question_index, odd_question in enumerate(odd):

            # Load the current questions file
            file_path = f"{BASE_PATH}/{llm_model}/{odd_question[0]}/{filename}"

            # Open the ODD file for reading
            with open(file_path, "r") as f:

                # Read the data and convert it to upper case
                file_data = f.read()
                
                # Get the words from the file data
                clean_file_data = clean_string(file_data)
                words = clean_file_data.split(" ")

                # Update the ODD
                if "YES" in words and "NO" in words:
                    odd_vectors[llm_index][file_index][odd_question_index] = ViolationState.MIXED_RESPONSE.num
                elif "YES" in words:
                    odd_vectors[llm_index][file_index][odd_question_index] = ViolationState.YES_VIOLATION.num
                elif "NO" in words:
                    odd_vectors[llm_index][file_index][odd_question_index] = ViolationState.NO_VIOLATION.num
                else:
                    odd_vectors[llm_index][file_index][odd_question_index] = ViolationState.UNDETERMINED.num

        # Get the textual description
        file_path = f"{BASE_PATH}/{llm_model}/q10/{filename}"
        # Open the file for reading
        with open(file_path, "r") as f:
            text = f.read()

        llm_descriptions.append(text)

    # Create sentence embeddings
    sentence_embeddings = EMBEDDING_MODEL(llm_descriptions)
    description_embedding.append(sentence_embeddings)

# Create a figure of set size
plt.figure(figsize=(17, 12))

# Create combined embeddings
for llm_index, llm_model in enumerate(requested_llm_models):

    print(f"Clustering: {llm_model}")

    # Holds the current data
    current_binary_vector           = odd_vectors[llm_index]
    current_description_embedding   = description_embedding[llm_index]
    current_full_embedding          = [] 

    # Create the final embedding
    for vector_index in range(np.shape(current_binary_vector)[0]):
        current_full_embedding.append(current_binary_vector[vector_index].tolist() + current_description_embedding[vector_index].numpy().tolist())

    # Create the x axis for plotting
    x = np.arange(1, args.max_clusters+1, 1)

    print(f"Clustering: {llm_model}  Compliance Vector")
    # Cluster using one hot encodings
    cluster_output = compute_kmeans_cluster_consistency(current_binary_vector, labels, args.clustering_iterations, args.max_clusters)
    avg_percentage_consistency = cluster_output[1]
    plt.plot(x ,avg_percentage_consistency, label=f"{llm_model}: Binary", color=f"C{llm_index}", linestyle="dotted", linewidth=6)

    # Cluster using sentence embedding
    if llm_model != "human":
        print(f"Clustering: {llm_model}  General Vector")
        cluster_output = compute_kmeans_cluster_consistency(current_description_embedding, labels, args.clustering_iterations, args.max_clusters)
        avg_percentage_consistency = cluster_output[1]
        plt.plot(x ,avg_percentage_consistency, label=f"{llm_model}: Sentence", color=f"C{llm_index}", linestyle="dashed", linewidth=6)

    # Cluster using combined sentence and one hot encoding
    if llm_model != "human":
        print(f"Clustering: {llm_model}  Semantic Vector")
        cluster_output = compute_kmeans_cluster_consistency(current_full_embedding, labels, args.clustering_iterations, args.max_clusters)
        avg_percentage_consistency = cluster_output[1]
        plt.plot(x ,avg_percentage_consistency, label=f"{llm_model}: Combined", color=f"C{llm_index}", linestyle="solid", linewidth=6)

print(f"Random Embedding Clustering")

# Generate random clusters
cluster_output = compute_random_cluster_consistency(labels, args.clustering_iterations, args.max_clusters)
avg_percentage_consistency = cluster_output[1]
plt.plot(x ,avg_percentage_consistency, label="Random Embedding", color=f"black", linestyle="solid", linewidth=6)

# Placeholder for color legend - models
color_handles = []
for c, model in enumerate(requested_llm_models):
    color_handles.append(mlines.Line2D([], [], color=f"C{c}", label=LLM_NAMES[model], linewidth=6))

# Creating the color legend and adding it to the current Axes
color_legend = plt.legend(handles=color_handles, loc='upper left', fontsize=32)
plt.gca().add_artist(color_legend)  # Add the color legend

# Prepare custom handles for the line style legend
encoding_handles = []
encoding_handles.append(mlines.Line2D([], [], color='grey', linestyle='dotted', label='Compliance Vector', linewidth=6))
encoding_handles.append(mlines.Line2D([], [], color='grey', linestyle='dashed', label='General Vector', linewidth=6))
encoding_handles.append(mlines.Line2D([], [], color='grey', linestyle='solid', label='Semantic Vector', linewidth=6))
encoding_handles.append(mlines.Line2D([], [], color='black', linestyle='solid', label='Random Vector', linewidth=6))

# Creating and adding the line style legend with custom handles
encoding_legend = plt.legend(handles=encoding_handles, loc='upper center', fontsize=32)
# Add the encoding legend
plt.gca().add_artist(encoding_legend)  

# Add the legend, and labels
plt.grid()
# plt.title(f"dataset: {args.dataset} - max_clusters: {args.max_clusters} - clustering_iterations: {args.clustering_iterations}")
plt.ylabel("Inputs in Consistent Classes (%)", fontsize=46)
plt.xlabel("Number of Classes", fontsize=46)
plt.ylim([0,100])
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30) 

# Enable minor ticks
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='1', color='gray')
plt.grid(which='minor', linestyle='--', linewidth='0.5', color='gray')

# Show the plot
plt.tight_layout()
plt.savefig(f"./results/rq3_{args.dataset}_{args.clustering_iterations}.png")
if args.show_plot:
    plt.show()
plt.close()


