import os

import tensorflow as tf
import tensorflow_hub as hub

def get_embedding_model():

    # Load the embedding model
    if os.path.exists("model"):
        # Load the model from the directory
        embedding_model = tf.saved_model.load('model/')
    else:
        # Create a directory to save the model
        os.makedirs('model', exist_ok=True)
        # Load the Universal Sentence Encoder's TF Hub module
        embedding_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        # Save the model locally
        tf.saved_model.save(embedding_model, 'model/')


    # Makes reading easier to avoid all tf logging
    print("============================")
    print("\n\n\n\n")

    return embedding_model