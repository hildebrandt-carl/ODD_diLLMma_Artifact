import os
import h5py
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_index_from_h5(f, index):
    try:
        data = f.get(index)
        steer          = float(data[0][0])
        frame_number   = int(data[0][1])
        message        = str(data[0][2])
    except TypeError as e:
        steer          = np.nan
        frame_number   = np.nan
        message        = None
    
    return steer, frame_number, message

def load_data(filename):

        # Read the file
        h5_f = h5py.File(filename, 'r')

        # Get the number of keys
        total_keys = len(list(h5_f.keys()))

        current_frame_readings = []
        current_frame = 0

        readings = np.zeros(total_keys)
        
        for i in range(total_keys):
            # Generate a key
            index = "{0:09d}_data".format(i)

            # Get the steering angles and frame numbers
            data = read_index_from_h5(h5_f, index)
            steering_angle = data[0]
            frame_number = data[1]
            message = data[2]

            if frame_number == current_frame:
                current_frame_readings.append(steering_angle)
            else:
                frame_reading = np.median(current_frame_readings)
                readings[current_frame] = frame_reading
                current_frame = frame_number
                current_frame_readings = [steering_angle]

        # Close the h5 file
        h5_f.close()

        return readings

if __name__ == "__main__":

    directory = "/home/carl/Desktop"
    versions = ["2022_04", "2022_07", "2023_03", "2023_06"]
    file_dir = f"{directory}/Data"

    # Load all the file names
    filenames = glob.glob(f"{file_dir}/{versions[0]}/*.h5")
    basenames = [os.path.basename(file) for file in filenames]
    sorted_files = sorted(basenames)

    # Read the data
    for f_name in sorted_files:

        r1 = load_data(f"{file_dir}/{versions[0]}/{f_name}")
        r2 = load_data(f"{file_dir}/{versions[1]}/{f_name}")
        r3 = load_data(f"{file_dir}/{versions[2]}/{f_name}")
        r4 = load_data(f"{file_dir}/{versions[3]}/{f_name}")

        smallest = min([len(r1), len(r2), len(r3), len(r4)])
        r1 = r1[:smallest]
        r2 = r2[:smallest]
        r3 = r3[:smallest]
        r4 = r4[:smallest]

        plt.figure()
        plt.plot(r1, label=versions[0])
        plt.plot(r2, label=versions[1])
        plt.plot(r3, label=versions[2])
        plt.plot(r4, label=versions[3])
        plt.legend()  # Display the legend
        plt.grid(True)  # Optional: display grid

        plt.savefig(f"{file_dir}/imgs/{f_name[:-3]}.png")
        plt.close()

        # Step 1: Compute the element-wise differences
        r1r2 = np.mean(np.abs(r1 - r2))
        r1r3 = np.mean(np.abs(r1 - r3))
        r1r4 = np.mean(np.abs(r1 - r4))
        r2r3 = np.mean(np.abs(r2 - r3))
        r2r4 = np.mean(np.abs(r2 - r4))
        r3r4 = np.mean(np.abs(r3 - r4))

        print(f"Processing {f_name}")
        print(f"Average difference between {versions[0]} and {versions[1]}: {r1r2}")
        print(f"Average difference between {versions[0]} and {versions[2]}: {r1r3}")
        print(f"Average difference between {versions[0]} and {versions[3]}: {r1r4}")
        print(f"Average difference between {versions[1]} and {versions[2]}: {r2r3}")
        print(f"Average difference between {versions[1]} and {versions[3]}: {r2r4}")
        print(f"Average difference between {versions[2]} and {versions[3]}: {r3r4}")
        print("")