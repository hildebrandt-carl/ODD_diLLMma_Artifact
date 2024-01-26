#!/bin/bash

# Set the folder path
folder="${HOME}/Desktop/Data"

# Loop through each text file in the folder
for file in "$folder"/*.mp4; do
    if [ -f "$file" ]; then
        # Remove ${HOME} path from file path
        file_path="/$(echo "$file" | sed "s|^${HOME}/||")"

        echo "Processing $file_path"
        echo "Launching OpenPilot"

        # Run the first command in the background
        ./launch_openpilot.sh &
        pidl=$!
        sleep 15

        # Run the second command
        echo "Launching Video Bridge"
        python3 bridge_video.py --filename $file_path
        pidp=$!
        sleep 15

        # Terminate the first command when the second command finishes
        echo "Closing Down"
        kill -9 "$pidl"
        kill -9 "$pidp"
        pkill -9 python3

        sleep 15
    fi
done
