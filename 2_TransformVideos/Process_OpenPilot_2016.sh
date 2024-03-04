#!/bin/bash

# Declare directorys
INPUT_DIRECTORY="../1_Datasets/Data/OpenPilot_2016/0_OriginalData/"
OUTPUT_DIRECTORY="../1_Datasets/Data/OpenPilot_2016/1_ProcessedData/"

# Check that the ProcessData is empty
if [ "$(ls -A "$OUTPUT_DIRECTORY")" ]; then
    echo "Warning: Output directory is not empty ($OUTPUT_DIRECTORY)"
    exit 1
fi

# Extract the zip file
7z x "$INPUT_DIRECTORY"CommaAi.7z -o$INPUT_DIRECTORY "CommaAi/imgFormat/*"

# Iterate over each subfolder in the input directory
for folder in "$INPUT_DIRECTORY"/CommaAi/imgFormat/*/; do
    # Extract just the folder name
    folder_name=$(basename "$folder")

    # Specify the pattern of your images (modify if needed)
    image_pattern="${folder}%d.jpg"

    # Set the output video file names
    output_video_20="${OUTPUT_DIRECTORY}/${folder_name}_20.mp4"
    output_video_15="${OUTPUT_DIRECTORY}/${folder_name}_15.mp4"

    # Create the video using the correct FPS (20 FPS) - https://github.com/commaai/research
    ffmpeg -framerate 20 -i "$image_pattern" -c:v libx264 -pix_fmt yuv420p "$output_video_20"

    # Re-encode the video using 15 FPS
    ffmpeg -i "$output_video_20" -vf "scale=1928:1208" -r 15 -c:v libx264 -pix_fmt yuv420p "$output_video_15"

    # Delete the intermediate video
    rm -rf $output_video_20

    echo "Video created for folder: $folder_name"
done

# Delete the extracted data
rm -rf "$INPUT_DIRECTORY"/CommaAi