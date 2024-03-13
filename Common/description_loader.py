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
    def _parse_answer_text(answer_text):
        answers = []
        # Check for NO
        if re.match(r".*\bNO\b.*", answer_text, re.IGNORECASE):
            answers.append(0)
        # Check for YES
        if re.match(r".*\bYES\b.*", answer_text, re.IGNORECASE):
            answers.append(1)
        # Check for inconclusive answers
        unknowns = ['UNABLE TO DETERMINE', 'INCONCLUSIVE', 'UNCLEAR', 'CANNOT DETERMINE', 'UNKNOWN', 'INSUFFICIENT DATA', 'YES/NO']
        if any(re.match(r".*\b" + unknown + r"\b.*", answer_text, re.IGNORECASE) for unknown in unknowns):
            answers.append(-1)
        # Determine the outcome based on found answers
        if len(answers) == 1:
            return answers[0]
        else:
            return -1
    
    @staticmethod
    def decode_using_question_numbers(file_lines):
        # Create the vector
        vector = np.full(len(ODD), -1)
        # Create the question regex which looks for QXX where X is a number
        question_regex = re.compile(r'.*Q(\d+)(.*)')
        # For each line
        for line in file_lines:
            # Try and match the line
            match = question_regex.match(line)
            if match:
                # Get the question index and text
                question_index = int(match.group(1)) - 1
                answer_text = match.group(2).strip()
                # Decode the answers text
                answer = DescriptionLoader._parse_answer_text(answer_text)
                # If we got an answer and the question's index makes sense
                if answer is not None and 0 <= question_index < len(ODD):
                    # In case of conflicting answers for the same question, set to -1
                    if vector[question_index] in [0, 1] and vector[question_index] != answer:
                        vector[question_index] = -1
                    else:
                        vector[question_index] = answer
        # Return the vector
        return vector

    @staticmethod
    def decode_using_numerical_prefix(file_lines):
        # Create the vector
        vector = np.full(len(ODD), -1)
        # Regular expression to match lines starting with numbers followed by a period and optional whitespace
        numbered_line_regex = re.compile(r'^(\d+)\.\s*(.*)')
        # For each line
        for line in file_lines:
            # Try and match the line
            match = numbered_line_regex.match(line)
            if match:
                # Get the question index and text
                question_index = int(match.group(1)) - 1
                answer_text = match.group(2).strip()
                # Decode the answers text
                answer = DescriptionLoader._parse_answer_text(answer_text)
                # If we got an answer and the question's index makes sense
                if answer is not None and 0 <= question_index < len(ODD):
                    # In case of conflicting answers for the same question, set to -1
                    if vector[question_index] in [0, 1] and vector[question_index] != answer:
                        vector[question_index] = -1
                    else:
                        vector[question_index] = answer
        # Return the vector
        return vector

    @staticmethod
    def decode_by_yes_no_presence(file_lines):
        # Create a vector
        vector = []
        # Loop through the file lines
        for line in file_lines:
            # Look for YES or NO
            if "YES" in line:
                vector.append(1)
            elif "NO" in line:
                vector.append(0)
        # Check if we got something valid
        if len(vector) != len(ODD):
            return np.full(len(ODD), -1)
        else:
            return np.array(vector)

    @staticmethod
    def decode_using_description_match(file_lines):
        # Initialize the vector
        vector = np.full(len(ODD), -1)  
        for line in file_lines:
            for question_index, (key, description) in enumerate(ODD.items()):
                # Convert key to upper
                key             = key.upper()
                # Check if the line contains part of the key
                if any(word in line for word in key.split()):
                    # Attempt to parse the answer text for a YES or NO response
                    answer = DescriptionLoader._parse_answer_text(line)
                    # In case of conflicting answers for the same question, set to -1
                    if vector[question_index] in [0, 1] and vector[question_index] != answer:
                        vector[question_index] = -1
                    else:
                        vector[question_index] = answer
        # Return the vector
        return vector

    @staticmethod
    def _get_yes_no_in_line(line):
        words = line.split()  # Convert to uppercase for case-insensitive comparison and split into words
        # Check for both "YES" and "NO" to handle contradictory statements within the same line
        has_yes = "YES" in words
        has_no = "NO" in words
        
        if has_yes and has_no:
            return -1  # Inconclusive if both "YES" and "NO" are found
        elif has_yes:
            return 1
        elif has_no:
            return 0
        else:
            return -1  # Return -1 if neither "YES" nor "NO" is found

    def decode_filter_asterisk_lines(file_lines):
        # Filter lines to keep only those containing an asterisk (*)
        filtered_lines = [line for line in file_lines if '*' in line]
        return DescriptionLoader.decode_by_yes_no_presence(filtered_lines)

    @staticmethod
    def decode_question_answer_pairs(file_lines):
        # Initialize the vector
        vector = np.full(len(ODD), -1)  
        entry_pattern = re.compile(r'^(\d+)\.')  # Pattern to identify the start of an entry

        for i, line in enumerate(file_lines):
            entry_match = entry_pattern.match(line)
            if entry_match:
                # Convert to zero-based index
                question_number = int(entry_match.group(1)) - 1  

                # Look for a YES or NO and update if found
                answer = DescriptionLoader._get_yes_no_in_line(file_lines[i])
                if (0 <= question_number < len(ODD)) and (answer != -1):
                    vector[question_number] = answer
                    break 

                # Search subsequent lines for YES OR NO
                for j in range(i+1, len(file_lines)):
                    # Check if the next entry starts before finding an answer
                    if entry_pattern.match(file_lines[j]):
                        break 
                    # Look for a YES or NO and update if found
                    answer = DescriptionLoader._get_yes_no_in_line(file_lines[j])
                    if (0 <= question_number < len(ODD)) and (answer != -1):
                        vector[question_number] = answer
                        break 
        return vector

    @staticmethod
    def _clean_text(string_data):
        # Convert to uppercase
        clean_data = [line.upper() for line in string_data]
        return clean_data

    @staticmethod
    def _update_best_candidate(current_vector_candidate, new_vector):
        # Count the number of 0's and 1's in the vector
        new_vector_count        = np.count_nonzero(new_vector == 0) + np.count_nonzero(new_vector == 1)
        current_vector_count    = np.count_nonzero(current_vector_candidate == 0) + np.count_nonzero(current_vector_candidate == 1)
        # Return the candidate with the most 0's or 1's
        if new_vector_count > current_vector_count:
            return new_vector
        else:
            return current_vector_candidate

    @staticmethod
    def get_vector_from_file(filename: str):
        # Keep track of the best candidate
        best_vector_candidate = np.full(len(ODD), -1)

        # Read the file
        with open(filename, 'r') as file:
            file_lines = file.readlines()

        # Clean the data
        file_lines = DescriptionLoader._clean_text(file_lines)

        # Try decoding using the question numbers as a guide
        vector                  = DescriptionLoader.decode_using_question_numbers(file_lines)
        best_vector_candidate   = DescriptionLoader._update_best_candidate(best_vector_candidate, vector)

        # Try decoding using the numerical numbers as a guide
        vector                  = DescriptionLoader.decode_using_numerical_prefix(file_lines)
        best_vector_candidate   = DescriptionLoader._update_best_candidate(best_vector_candidate, vector)

        # Try decoding using the ODD as a guide
        vector                  = DescriptionLoader.decode_using_description_match(file_lines)
        best_vector_candidate   = DescriptionLoader._update_best_candidate(best_vector_candidate, vector)

        # Try decoding looking for pairs of YES or NO's
        vector                  = DescriptionLoader.decode_question_answer_pairs(file_lines)
        best_vector_candidate   = DescriptionLoader._update_best_candidate(best_vector_candidate, vector)

        # Try looking for all lines starting with *
        vector                  = DescriptionLoader.decode_filter_asterisk_lines(file_lines)
        best_vector_candidate   = DescriptionLoader._update_best_candidate(best_vector_candidate, vector)

        # Try decoding just by looking at yes or no
        vector                  = DescriptionLoader.decode_by_yes_no_presence(file_lines)
        best_vector_candidate   = DescriptionLoader._update_best_candidate(best_vector_candidate, vector)

        # Return the best candidate
        return best_vector_candidate
