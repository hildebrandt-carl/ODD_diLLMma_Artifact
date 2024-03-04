#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <llama|vicuna> <OpenPilot_2016|OpenPilot_2k19|OpenPilot_Jutah> <0|1>"
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
    img_dir="../Data/${dataset_choice}/4_SelectedData"
    output_dir="../Data/${dataset_choice}/5_Descriptions/${llm_choice}"
elif [ "${dataset_choice}" = "OpenPilot_2k19" ]; then
    img_dir="../Data/${dataset_choice}/4_SelectedData"
    output_dir="../Data/${dataset_choice}/5_Descriptions/${llm_choice}"
elif [ "${dataset_choice}" = "External_Jutah" ]; then
    img_dir="../Data/${dataset_choice}/4_SelectedData"
    output_dir="../Data/${dataset_choice}/5_Descriptions/${llm_choice}"
else
    echo "Invalid input. Please enter 'OpenPilot_2016' or 'OpenPilot_2k19' or 'External_Jutah'."
    exit 1
fi

# Validate the GPU number
if [[ "${gpu_number}" != "0" && "${gpu_number}" != "1" ]]; then
    echo "Invalid GPU number. Please enter '0' or '1'."
    exit 1
fi

context='/home/lesslab/Documents/ODD_Work/Image2Text/Prompts/Contexts/context_01_template.txt'
question='/home/lesslab/Documents/ODD_Work/Image2Text/Prompts/Questions/question_01.txt'

# Run the file
python run_study.py --cfg-path ${cfg_path}  --gpu-id ${gpu_number} --images-dir ${img_dir} --output-dir "${output_dir}" --prompt-context-file ${context} --prompt-question-file ${question}





