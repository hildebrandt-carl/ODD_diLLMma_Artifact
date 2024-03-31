import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from data_loader import DataLoader
from constants import OPENPILOT_COLORS


# Get the folders
parser = argparse.ArgumentParser(description="Displays the steering angles")
parser.add_argument('--video_file',
                    type=str,
                    required=True,
                    help="The name of the base video file")
args = parser.parse_args()

dl = DataLoader(filename=args.video_file)
dl.validate_h5_files()
dl.load_data()

plt.figure()
for version_index, version in enumerate(dl.versions):
    plt.plot(dl.readings[version_index], label=version, c=OPENPILOT_COLORS[version])
plt.legend()
plt.xlabel("Frame ID")
plt.ylabel("Steering Angle")
plt.show()