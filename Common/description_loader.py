import glob
import re

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from constants import ODD


class DescriptionLoader:
    def __init__(self, data_input):
        self.description_names = []  # Holds just the file names
        self.description_full_paths = []  # Holds the full paths to the files
        self.total_descriptions = 0  # Holds the total number of items in both description_names and description_full_paths

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
        search_path = f"{self.data_path}/*_output.txt"
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
    def parse_answer(answer, use_upper=False):
        if use_upper:
            answer = answer.upper()
        answers = []
        # if 'NO' in answer:
        if re.match(r".*\bNO\b.*", answer) is not None:
            answers.append(0)
        # if 'YES' in answer:
        if re.match(r".*\bYES\b.*", answer) is not None:
            answers.append(1)
        # if 'UNABLE TO DETERMINE' in answer or 'INCONCLUSIVE' in answer:
        unknowns = ['UNABLE TO DETERMINE', 'INCONCLUSIVE',
                    'UNCLEAR', 'CANNOT DETERMINE', 'UNKNOWN', 'INSUFFICIENT DATA']
        found_unknowns = [re.match(r".*\b" + unknown + r"\b.*", answer) for unknown in unknowns]
        if any(found_unknowns) or 'YES/NO' in answer:
            answers.append(-1)
        if len(answers) == 1:
            return answers[0]
        if not use_upper:
            # if we tried without doing uppercase the first time, then try again with using uppercase
            # this is useful to allow for cases where the LLM gives a clear answer and then ambiguous text, e.g.
            # Q01: NO - it is unclear
            # the above will now be parsed as NO since the NO is all caps, but the unclear isn't
            return DescriptionLoader.parse_answer(answer, use_upper=True)
        if len(answers) > 1:
            return -1
        return None

    @staticmethod
    def get_vector_from_file(filename: str):
        odd_length = len(ODD)
        vector_list = []
        file_lines = []
        answer_dict = {}
        question_regex = re.compile(r'.*Q(\d+)(.*)')
        with open(filename, 'r') as file:
            for index, line in enumerate(file):
                file_lines.append(line)
                match = question_regex.match(line)
                try:
                    if match is not None and len(match.groups()) == 2:
                        question = int(match.groups()[0])
                        answer = DescriptionLoader.parse_answer(match.groups()[1])
                        if answer is None:
                            continue
                        if question in answer_dict and answer_dict[question] != answer:
                            # if it re-stated the question and gave a different answer, overwrite with -1
                            answer = -1
                        answer_dict[question] = answer
                except ValueError:
                    pass
        if len(answer_dict) > 0:
            vector_list = [-1] * odd_length
            for question, answer in answer_dict.items():
                vector_list[question - 1] = answer
        else:
            # fall back on YES/NO counting
            for line in file_lines:
                line = line.strip().upper()
                if "YES" in line:
                    vector_list.append(1)
                if "NO" in line:
                    vector_list.append(0)

        if len(vector_list) == odd_length:
            return np.array(vector_list)
        else:
            # print(filename)
            # print(answer_dict)
            # with open(filename) as file:
            #     print(file.read())
            return np.full(odd_length, -1)
