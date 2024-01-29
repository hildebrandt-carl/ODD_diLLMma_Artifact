#!/bin/bash

# Declare directorys
INPUT_DIRECTORY="../1_Datasets/Data/External_jutah/0_OriginalData/"
OUTPUT_DIRECTORY="../1_Datasets/Data/External_jutah/1_ProcessedData/"

# Use a for loop to iterate through .mp4 files in the directory
for file in "$INPUT_DIRECTORY"/*.mp4; do
  # Check if there are .mp4 files in the directory
  if [ -e "$file" ]; then
    # Extract and echo the file name without the directory path
    filename=$(basename "$file")

    # Append "_15" to the filename before the .mp4 extension
    new_filename="${filename%.mp4}_15.mp4"

    # Output a 15 FPS skiping first 90 seconds (which is usually text over the video)
    ffmpeg -ss 90 -i "$file" -an -r 15 -vf "scale=1928:1208" "$OUTPUT_DIRECTORY$new_filename"
  fi
done