#!/bin/bash

# Declare directorys
INPUT_DIRECTORY="../1_Datasets/Data/OpenPilot_2k19/0_OriginalData"
OUTPUT_DIRECTORY="../1_Datasets/Data/OpenPilot_2k19/1_ProcessedData"

# Check that the ProcessData is empty
if [ "$(ls -A "$OUTPUT_DIRECTORY")" ]; then
    echo "Warning: Output directory is not empty ($OUTPUT_DIRECTORY)"
    exit 1
fi

# Loop through each zip file in the input directory and extract it
for zip_file in "$INPUT_DIRECTORY"/*.zip; do
    # Check if the file is a zip file
    if [ -f "$zip_file" ]; then
        echo "Unzipping $zip_file into $OUTPUT_DIRECTORY"
        # Unzip the file into the output directory
        7z x "$zip_file" -o"$OUTPUT_DIRECTORY"
    fi
done

# Initialize a counter
video_counter=0

# Extract all .hevc files from the output
find -L "$OUTPUT_DIRECTORY" -name "*.hevc" | while IFS= read -r file; do
    new_name=$(printf "video_%04d.hevc" "$video_counter")
    mv "$file" "$OUTPUT_DIRECTORY/$new_name"
    ((video_counter++))
done

# Remove all subdirectories created
for dir in "$OUTPUT_DIRECTORY"/*/ ; do
    rm -rf "$dir"
done

# Convert all hevc files to mp4
for i in "$OUTPUT_DIRECTORY"/*.hevc; do
    ffmpeg -i "$i" -vf "setpts=1.25*PTS,scale=1928:1208" -an -r 15 -c:v libx264 -crf 23 "${i%.*}_15.mp4"
done

# Remove all hevc files
rm -rf "$OUTPUT_DIRECTORY"/*.hevc