# Generate Caption

The `GenerateCaption` folder contains two Python scripts for generating captions for images used in our study. These scripts are `generate_chatgpt.py` and `generate_human.py`. The former automates the process of generating captions using ChatGPT, while the latter provides a graphical user interface (GUI) for human analysis.

## Generating ChatGPT Descriptions

**Functionality:** This script generates captions for each image in the selected dataset using ChatGPT. It utilizes the ChatGPT API to analyze images based on the Operational Design Domain (ODD) criteria specified in the given ODD file.

**Parameters:**
- `api_key`: The name of the text file containing your ChatGPT API key, located in the `API_Keys` folder.
- `dataset`: The name of one of the datasets located in the `Datasets` directory.
- `odd_file`: The name of the ODD file located in the `Datasets` directory.

**Running:**
To run the `generate_chatgpt.py` script you can use:
```bash
python generate_chatgpt.py --api_key [api_key_file_name] --dataset [dataset_name] --size [dataset_size] --portion [pass/fail/both] --odd_file [odd_file_name] 
```

## Labeling Human Descriptions

**Functionality:** This script is designed to facilitate human analysis of images. It launches a GUI where a user can view each image and mark those that violate the constraints based on their assessment.

**Parameter:**
- `dataset`: The name of the dataset from the `Datasets` directory you wish to analyze.

**Running:**
To run the `generate_human.py` script you can use:
```bash
python generate_human.py --dataset [dataset_name] --size [dataset_size]
```

## Generating Final Human Description

```bash
$ python3 select_human_agreed.py --size 200 --human_models "research_a, research_b, research_c" --dataset OpenPilot_2016 --selection_strategy "worst_case"
$ python3 select_human_agreed.py --size 200 --human_models "research_a, research_b, research_c" --dataset OpenPilot_2k19 --selection_strategy "worst_case"
$ python3 select_human_agreed.py --size 200 --human_models "research_a, research_c, research_b" --dataset External_jutah --selection_strategy "worst_case"
```

## Generating Vicuna and Llama 2 Descriptions

Please refer to this [README](../0_SetupDocumentation/OpenPilot_Setup/README.md)  for instructions on how to set up Vicuna and Llama to run locally.