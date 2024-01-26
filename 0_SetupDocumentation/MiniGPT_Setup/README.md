# MiniGPT-4 Setup Guide

To run this code please download and install [https://minigpt-4.github.io/](https://minigpt-4.github.io/). You can then use the code in `CustomCode` to run the models on large volumes of data without the need for a GUI interface.


# MiniGPT4 Setup

This README guides you through setting up and running the [https://minigpt-4.github.io/](https://minigpt-4.github.io/) model for processing large volumes of data with Vicuna and Llama 2.

## Prerequisites

Commit Version: We used commit `ef1ac08ce3f2835b3aad09c7e81adea976432062` of MiniGPT4 (which was the latest at the time this study was started).
GPU Requirements: We ran this on a NVIDIA GeForce RTX 3090 with 24GB of memory.
Graphics Drivers: We were using Driver Version: 535.129.03
Operating System: Ubuntu 20.04


## Initial Installation

Visit [https://minigpt-4.github.io/](https://minigpt-4.github.io/) github and follow their `Getting Started`

We have included a requirements for reference but it is the same as theirs. It can be installed using:
```bash
conda create --name minigpt4 --file requirements.txt
```

We were using the following models:
* Llama 2 Chat 7B - [link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main)
* Vicuna V0 13B - [link](https://huggingface.co/Vision-CAIR/vicuna/tree/main)


## Directory Structure

We had our code setup as follows:

```bash
$tree .
├── Data
│   ├── OpenPilot_2k19
│   │   ├── 4_SelectedData
│   │   └── 5_Descriptions
│   │       ├── llama
│   │       └── vicuna
│   └── OpenPilot_utah
│       ├── 4_SelectedData
│       └── 5_Descriptions
│           ├── llama
│           └── vicuna
├── MiniGPT-4
│   ├── dataset
│   ├── ...
│   └── train_configs
└── Models
    ├── Llama-2-7b-chat-hf
    ├── vicuna
    └── vicuna-7b
```

As you can see by the structure, it had three main folders:
1) Data - This contained a copy of the datasets `4_SelectedData` and `5_Descriptions`.
2) MiniGPT-4 - This contained the cloned MiniGPT-4
3) Models - This contained the downloaded models.

The data folder's `4_SelectedData` had the images you wanted to process. The data folders `5_Descriptions` had the output from the LLM's.


## Running the Code

To run the custom code on your dataset:

Place your dataset in the `Data` directory. Place both `run_all_images.py` and `run_model.sh` into the root directory of `MiniGPT-4`. You can then run the code in two ways:

### Running a single prompt

To run a single prompt you can simply use the `run_all_images.py` script. To do that use the following command:
```bash
$ python3 run_all_images.py --cfg-path <path_to_cfg_file>  --gpu-id <GPU ID> --images-dir <Image Directory> --output-dir <Output Directory> --prompt "<LLM Prompt>"
```

For more information on the `path_to_cfg_file` please see [https://minigpt-4.github.io/](https://minigpt-4.github.io/)'s getting started.

### Running all ODD prompts

To run all the ODD prompts you can use the following command (_NOTE: this code assumes the data directory structure above_):
```bash
$ ./run_model.sh <llm name> <dataset name> <GPU ID>
```

An example of this command is show below. We can see that it is running the first ODD prompt, on all 100 images from the `OpenPilot_2k19` dataset.

![minigpt-4 example usage](../../Misc/minigpt_example.png)
