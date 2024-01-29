# !/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [directory path]"
    exit 1
fi

DIRECTORY=$1
total=0

shopt -s nullglob
for file in "$DIRECTORY"/*.{mp4,hevc}; do
  duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
  total=$(echo "$total + $duration" | bc)
done

shopt -u nullglob
echo "Total duration in seconds: $total"
echo "Total duration in minutes: $(echo "$total / 60" | bc)"
echo "Total duration in hours: $(echo "$total / 3600" | bc)"