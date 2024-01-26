import numpy as np

from tqdm import tqdm
from sklearn.cluster import KMeans


def compute_cluster_consistency(embedding_labels, cluster_assignments, number_of_clusters):
     # Compute consistency
    consistent_clusters = 0
    
    # For each cluster
    for cluster_index in range(number_of_clusters):
        # Get labels (fail/pass) for each point in this cluster
        cluster_labels = [embedding_labels[i] for i, c in enumerate(cluster_assignments) if c == cluster_index]

        # Check if its consistent
        if all(x == cluster_labels[0] for x in cluster_labels):
            consistent_clusters += 1

    # Compute the consistency
    consistency = (consistent_clusters / number_of_clusters) * 100

    return consistency

def compute_test_consistency(embedding_labels, cluster_assignments, number_of_clusters):
     # Compute consistency
    total_tests         = 0
    consistent_tests    = 0

    # For each cluster
    for cluster_index in range(number_of_clusters):
        # Get labels (fail/pass) for each point in this cluster
        cluster_labels = [embedding_labels[i] for i, c in enumerate(cluster_assignments) if c == cluster_index]
        
        # Count total tests
        total_tests += len(cluster_labels)

        # Check if all labels are consistent
        if all(x == cluster_labels[0] for x in cluster_labels):
            consistent_tests += len(cluster_labels)

    # Compute the consistency
    if total_tests > 0:
        consistency = (consistent_tests / total_tests) * 100
    else:
        # Avoid division by zero
        consistency = 0  

    return consistency

def compute_random_cluster_consistency(embedding_labels, clustering_iterations, number_clusters):

    # Tracks the consistency
    avg_percentage_consistency = []
    min_percentage_consistency = []
    max_percentage_consistency = []

    # Compute the embedding space
    for number_of_clusters in tqdm(range(1, number_clusters+1)):

        # Holds constancy count for each of the counters
        individual_consistencies = []

        # Run each model args.clustering_iterations times using random states and take average
        for counter in range(clustering_iterations):
            # Create a random number generator instance
            rng = np.random.default_rng(counter)

            # Generate a cluster ID for each label
            cluster_assignments = rng.integers(low=0, high=number_of_clusters, size=len(embedding_labels))

            # Compute the consistency
            consistency = compute_test_consistency(embedding_labels, cluster_assignments, number_of_clusters)
            individual_consistencies.append(consistency)

        # Save the average consistency
        min_percentage_consistency.append(np.min(individual_consistencies))
        avg_percentage_consistency.append(np.mean(individual_consistencies))
        max_percentage_consistency.append(np.max(individual_consistencies))

    return min_percentage_consistency, avg_percentage_consistency, max_percentage_consistency

def compute_kmeans_cluster_consistency(embeddings, embedding_labels, clustering_iterations, number_clusters):
    # Tracks the consistency
    avg_percentage_consistency = []
    min_percentage_consistency = []
    max_percentage_consistency = []

    # Compute the embedding space
    for number_of_clusters in tqdm(range(1, number_clusters+1)):
        
        individual_consistencies = []

        # Run each model args.clustering_iterations times using random states and take average
        for counter in range(clustering_iterations):

            # Train the model
            kmeans = KMeans(init="random", n_clusters=number_of_clusters, n_init="auto", random_state=counter)
            kmeans.fit(embeddings)
            # Assign each point an embedding
            cluster_assignments = kmeans.predict(embeddings)

            # Compute the consistency
            consistency = compute_test_consistency(embedding_labels, cluster_assignments, number_of_clusters)
            individual_consistencies.append(consistency)

        # Save the average consistency
        min_percentage_consistency.append(np.min(individual_consistencies))
        avg_percentage_consistency.append(np.mean(individual_consistencies))
        max_percentage_consistency.append(np.max(individual_consistencies))

    return min_percentage_consistency, avg_percentage_consistency, max_percentage_consistency