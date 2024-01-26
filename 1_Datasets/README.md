# Datasets Folder Overview

This `datasets` folder is an integral part of our project, containing essential data used in our analyses. Below is a detailed explanation of its structure and contents.

## Directory Structure

The folder includes two main directories:

- `ODD`: Contains the original and converted Operational Design Domain (ODD) files, available as text files and screenshots.
- `Data`: Holds data from three distinct datasets.

### Data Subdirectories

Within the `Data` directory, you will find the following subdirectories:

1. **OpenPilot_2k19**: Data sourced from CommaAI.
2. **OpenPilot_2016**: Another dataset from CommaAI.
3. **External_jutah**: Data obtained from an external source, JUtah.

Each of these directories is structured similarly, with the following subdirectories:

- `0_OriginalData`: Reference to the original data _(not included in the repo due to size constraints)_.
- `1_ProcessedData`: Contains MP4 video files encoded at 15 FPS _(not included in the repo due to size constraints)_.
- `2_SteeringData`: Includes `h5py` files correlating frame IDs from the videos with steering angles, processed through OpenPilot _(not included in the repo due to size constraints)_.
- `3_PassFail`: Text files matching the `ProcessedData`, detailing frame IDs, steering angles, errors between OpenPilot versions, and frame classifications (Unknown, Pass, Fail).
- `4_SelectedData`: Datasets of various sizes (100, 200, 1000), each containing equal numbers of passing and failing images.
- `5_Descriptions`: Corresponding to `SelectedData` sizes, containing descriptions from humans and LLMs. Subfolders (q00, q01, ..., q10) match those in the `ODD` file.

## Contact for Additional Data

Some data, primarily the `0_OriginalData`, `1_ProcessedData`, and `2_SteeringData` are not included in this repository due to size constraints. If you require access to this data, please contact me at email TODO@todo.com for further assistance.

---

For any additional questions or clarifications, feel free to reach out to the email address provided above.
