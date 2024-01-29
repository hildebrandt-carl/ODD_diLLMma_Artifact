import os 
import glob
import argparse

from tqdm import tqdm
from PIL import Image


def resize_image(img_file):
    with Image.open(img_file) as img:
        new_img = img.resize((img.width // 2, img.height // 2))
    return new_img
        


# Decare the dataset directory
DATASET_DIRECTORY = "../1_Datasets/Data"

# Get user input
parser = argparse.ArgumentParser(description="Resize a folder of images")
parser.add_argument('--dataset',
                    type=str,
                    default="",
                    help="The dataset you want to process (OpenPilot_2k19, External_jutah)")
parser.add_argument('--size',
                    type=int,
                    required=True,
                    help="The size of the dataset you want to use")
args = parser.parse_args()


# Specify your folder path here
folder_path = f'{DATASET_DIRECTORY}/{args.dataset}/4_SelectedData/{args.size}'
output_folder = f'{DATASET_DIRECTORY}/{args.dataset}/4_SelectedData/{args.size}_smaller'


# Check if the output directory has been created
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of files
patterns = ['*.jpg', '*.jpeg', '*.png']
all_files = [file for pattern in patterns for file in glob.glob(f"{folder_path}/{pattern}")]

# For each
for img_file in tqdm(all_files):
    # Resize and save
    resized_img = resize_image(img_file)
    resized_img.save(os.path.join(output_folder, os.path.basename(img_file)))