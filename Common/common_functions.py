import numpy as np


def find_non_overlapping_sequences(difference_array, threshold, length, comparison_operator):
    # Apply the comparison operator to the entire array
    condition_met = comparison_operator(difference_array, threshold)
    
    start_indices = []
    consecutive_count = 0
    
    for i in range(len(condition_met)):
        if condition_met[i]:
            consecutive_count += 1
            if consecutive_count == length:
                start_indices.append(i - length + 1)
                consecutive_count = 0
        else:
            consecutive_count = 0
    
    return np.array(start_indices)

# Define the tick label format function
def format_func(value, tick_number):
    if value.is_integer():
        return f"{int(value)}%"
    else:
        return f"{value:.2f}%"