import glob
import numpy as np
import matplotlib.pyplot as plt

datasets = ["OpenPilot_2k19", "OpenPilot_2016", "External_jutah"]

for data in datasets:
    folder = f"../1_Datasets/Data/{data}/3_PassFail"
    files = glob.glob(f"{folder}/*.txt")

    # Create the x-array
    x_values = np.arange(10, 720)
    y_values = np.zeros(len(x_values))


    for file in files:

        # Count the number of lines
        with open(file, 'r') as f:
            line_count = sum(1 for _ in f)

        # Create error array
        err_arr = np.zeros(line_count-1, dtype=float)

        # Populate error array
        with open(file, 'r') as f:
            line_count = 0
            for line in f:
                clean_line = line.strip()
                error_long = clean_line[clean_line.rfind("e:")+2:]
                error_s = error_long[:error_long.find(")")]
                try:
                    error = float(error_s)
                except Exception as e:
                    continue

                err_arr[line_count] = error
                line_count += 1

        # Update the y_values
        for i, x in enumerate(x_values):
            y_values[i] = y_values[i] + np.sum(err_arr >= x)

    # Creating the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values)
    plt.xlabel('Maximum Difference (deg)')
    plt.ylabel('Number of Failing Images')
    plt.grid(True)
    plt.show()