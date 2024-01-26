import re

from utils_ViolationState import ViolationState


def clean_string(input_string):
    # Replace any non-word character (\W) with a space
    text_only_string = re.sub(r'\W', ' ', input_string)
    # Remove duplicated spaces
    single_space_string = re.sub(r'\s+', ' ', text_only_string)
    # Remove leading white space
    remove_lead_trail_whitespace = single_space_string.strip()
    # Move everything uppercase
    upper_sentence = remove_lead_trail_whitespace.upper()

    return upper_sentence

def update_odd_vector(odd_output, description_files, odd_index):
    # Update the ODD
    for file_index, file in enumerate(description_files):

        # Open the file for reading
        with open(file, "r") as f:

            # Read the data and convert it to upper case
            file_data = f.read()
            
            # Get the words from the file data
            clean_file_data = clean_string(file_data)
            words = clean_file_data.split(" ")
            
            # Update the odd output
            if "YES" in words and "NO" in words:
                odd_output[file_index][odd_index] = ViolationState.MIXED_RESPONSE.num
            elif "YES" in words:
                odd_output[file_index][odd_index] = ViolationState.YES_VIOLATION.num
            elif "NO" in words:
                odd_output[file_index][odd_index] = ViolationState.NO_VIOLATION.num
            else:
                odd_output[file_index][odd_index] = ViolationState.UNDETERMINED.num

    return odd_output

def get_odd():
    # Define the questions and labels
    odd = [("q00", "Poor Visibility"),
           ("q01", "Image Obstructed"),
           ("q02", "Sharp Curve"),
           ("q03", "On-off Ramp"),
           ("q04", "Intersection"),
           ("q05", "Restricted Lane"),
           ("q06", "Construction"),
           ("q07", "Bright Light"),
           ("q08", "Narrow Road"),
           ("q09", "Hilly Road")]    
    return odd