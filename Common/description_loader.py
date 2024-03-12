import glob
import numpy as np
import os

from constants import ODD


class DescriptionLoader:
    def __init__(self, data_input):
        self.description_names = []  # Holds just the file names
        self.description_full_paths = []  # Holds the full paths to the files
        self.total_descriptions = 0 # Holds the total number of items in both description_names and description_full_paths

        # Determine if data_input is a path or a list of file paths
        if isinstance(data_input, str):
            self.data_path = data_input
            self.get_available_description_files()
        elif isinstance(data_input, list):
            self.data_path = None  # Not needed when list of files is directly provided
            self.process_input_file_list(data_input)
        else:
            raise ValueError("data_input must be a path (str) or a list of file paths (list)")

        self.total_descriptions = len(self.description_names)

        # Create the coverage vector
        self.coverage_vector = np.full((self.total_descriptions, len(ODD)), -1)
        self.load_coverage_vector()

    def get_available_description_files(self):
        search_path = f"{self.data_path}/*.txt"
        self.description_full_paths = glob.glob(search_path)
        self.description_names = [os.path.basename(fp) for fp in self.description_full_paths]
        self.description_full_paths = sorted(self.description_full_paths)
        self.description_names = sorted(self.description_names)

    def process_input_file_list(self, file_list):
        self.description_full_paths = sorted(file_list)
        self.description_names = [os.path.basename(fp) for fp in self.description_full_paths]

    def get_filenames_from_indices(self, indices):
        # Validate indices are within the range of the descriptions list
        if not all(0 <= idx < self.total_descriptions for idx in indices):
            raise ValueError("One or more indices are out of the valid range.")

        # Return the filenames corresponding to the provided indices
        return [self.description_names[idx] for idx in indices]

    def load_coverage_vector(self):
        for i, full_path in enumerate(self.description_full_paths):
            vector = DescriptionLoader.get_vector_from_file(full_path)
            self.coverage_vector[i] = vector

    @staticmethod
    def get_vector_from_file(filename: str):
        odd_length = len(ODD)
        vector_list = []

        with open(filename, 'r') as file:
            for line in file:
                line = line.strip().upper()
                if "YES" in line:
                    vector_list.append(1)
                elif "NO" in line:
                    vector_list.append(0)

        if len(vector_list) == odd_length:
            return np.array(vector_list)
        else:
            return np.full(odd_length, -1)
