import os
import sys

import numpy as np
import matplotlib.pyplot as plt


current_dir = os.path.dirname(__file__)
data_loader_dir = "../Common"
data_loader_path = os.path.abspath(os.path.join(current_dir, data_loader_dir))
sys.path.append(data_loader_path)

from data_loader import DataLoader

dl = DataLoader(filename="002_Downtown_Cincinnati_Ohio_USA_15")
dl.validate_h5_files()
dl.load_data()

plt.figure()
for version_index, version in enumerate(dl.versions):
    plt.plot(dl.readings[version_index], label=version, c=f"C{version_index}")
plt.legend()
plt.xlabel("Frame ID")
plt.ylabel("Steering Angle")
plt.show()