import glob
import numpy as np

from constants import ODD


class DescriptionLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

        # Get all the description files
        self.description_names = self.get_available_description_files()
        self.total_descriptions = len(self.description_names)

        # Create the coverage vector
        self.coverage_vector = np.full((len(self.description_names), len(ODD)), -1)
        self.load_coverage_vector()

    def get_available_description_files(self):
        search_path = f"{self.data_path}/*.txt"
        available_filepaths = glob.glob(search_path)
        available_files = [fp[fp.rfind("/")+1:] for fp in available_filepaths]
        available_files = sorted(available_files)
        return available_files
    
    def load_coverage_vector(self):
        
        for i, d_file in enumerate(self.description_names):
            vector = DescriptionLoader.get_vector_from_file(f"{self.data_path}/{d_file}")
            self.coverage_vector[i] = vector

    @staticmethod
    def get_vector_from_file(filename: str):

        odd_length = len(ODD)
        vector_list = []

        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                line = line.upper()
                if "YES" in line:
                    vector_list.append(1)
                if "NO" in line:
                    vector_list.append(0)

        if len(vector_list) == odd_length:
            return np.array(vector_list)
        else:
            return np.full(odd_length, -1)
            

        