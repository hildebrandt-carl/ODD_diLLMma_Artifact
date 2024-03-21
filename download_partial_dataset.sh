#!/bin/bash

# Check if 7z is installed
if ! command -v 7z &> /dev/null; then
    echo "7z is not installed."
    echo "Please install 7z. It can be installed on Ubuntu using \"apt install p7zip-full -y\" or \"brew install p7zip\" on macOS"
fi

# Specify the path to the dataset folder
dataset_folder="./1_Datasets/Data"

# Check if the dataset folder exists
if [ -d "$dataset_folder" ]; then
    # Prompt the user for confirmation
    echo "This will delete any data you may have in the dataset folder (${dataset_folder}). Do you want to proceed? [Y/n]"

    # Read the user input
    read -r -p "Press Y to continue: " response

    # Convert the response to lowercase
    response=$(echo "$response" | tr '[:upper:]' '[:lower:]')

    # Check if the response is 'y'
    if [[ "$response" =~ ^(yes|y)$ ]]; then
        echo "Deleting the data in the dataset folder..."
        # Command to delete the data in the folder
        rm -rf "${dataset_folder}"
        echo "Data deleted successfully."
    else
        echo "Operation cancelled. Nothing was done."
        exit 0
    fi
fi

# Download the dataset
echo "The partial dataset is being downloaded to (${dataset_folder})."
curl -L -o partial_data.7z "https://www.dropbox.com/scl/fi/85o8t4hefqj765qri5sbg/ODD_diLLMma_Data.7z?rlkey=zdcvygegu0bp9g4c1i2ojxv02&dl=1"

# Extract
echo "Extracting data..."
7z x partial_data.7z

# Move
echo "Finishing up..."
mv Data ${dataset_folder}

echo "Completed succesfully"
exit 0