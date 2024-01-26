#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <llama|vicuna> <OpenPilot_2016|OpenPilot_2k19|OpenPilot_utah> <0|1>"
    exit 1
fi

# Assign command line arguments to variables
llm_choice=$1
dataset_choice=$2
gpu_number=$3

# Set the cfg-path based on user input
if [ "${llm_choice}" = "llama" ]; then
    cfg_path="eval_configs/minigpt4_llama2_eval.yaml"
elif [ "${llm_choice}" = "vicuna" ]; then
    cfg_path="eval_configs/minigpt4_eval.yaml"
else
    echo "Invalid input. Please enter 'llama' or 'vicuna'."
    exit 1
fi

# Set the cfg-path based on user input
if [ "${dataset_choice}" = "OpenPilot_2016" ]; then
    img_dir="/Data_1000/${dataset_choice}/4_SelectedData"
    output_dir="/Data_1000/${dataset_choice}/5_Descriptions/${llm_choice}"
elif [ "${dataset_choice}" = "OpenPilot_2k19" ]; then
    img_dir="/Data_1000/${dataset_choice}/4_SelectedData"
    output_dir="/Data_1000/${dataset_choice}/5_Descriptions/${llm_choice}"
elif [ "${dataset_choice}" = "OpenPilot_utah" ]; then
    img_dir="/Data_1000/${dataset_choice}/4_SelectedData"
    output_dir="/Data_1000/${dataset_choice}/5_Descriptions/${llm_choice}"
else
    echo "Invalid input. Please enter 'OpenPilot_2016' or 'OpenPilot_2k19' or 'OpenPilot_utah'."
    exit 1
fi

# Validate the GPU number
if [[ "${gpu_number}" != "0" && "${gpu_number}" != "1" ]]; then
    echo "Invalid GPU number. Please enter '0' or '1'."
    exit 1
fi

# Define an array of prompts
declare -a prompts=(
    "Only reply with YES or NO, then explain why. Does this image have poor visibility like heavy rain, snow, fog, or adverse weather conditions?"
    "Only reply with YES or NO, then explain why. Was the camera that took this image obstructed, covered, or damaged by mud, ice, snow, excessive paint or adhesive products?"
    "Only reply with YES or NO, then explain why. Is the road we are driving on a sharp curve?"
    "Only reply with YES or NO, then explain why. Is the road we are driving on an on-off ramp?"
    "Only reply with YES or NO, then explain why. Is the road we are driving on an intersection, or does it run horizontally across the image?"
    "Only reply with YES or NO, then explain why. Does the road we are driving have any restricted lanes, for example for buses, or bicycles?"
    "Only reply with YES or NO, then explain why. Does the road we are driving on have any construction?"
    "Only reply with YES or NO, then explain why. Are there any extremely bright lights from oncoming headlights, or direct sunlight into the camera?"
    "Only reply with YES or NO, then explain why. Is the road we are driving on narrow and winding?"
    "Only reply with YES or NO, then explain why. Is the road we are driving on hilly?"
    "Describe all the factual information about this image. Pay attention to the road structure, road signs, traffic, pedestrians, time of day, and anything one needs to pay attention to while driving."
)

# Loop through the array
for i in "${!prompts[@]}"; do
    echo "${llm_choice}: ${prompts[$i]}"

    python run_all_images.py --cfg-path ${cfg_path}  --gpu-id ${gpu_number} --images-dir ${img_dir} --output-dir "${output_dir}/q$(printf "%02d" $i)/" --prompt "${prompts[$i]}"
done

