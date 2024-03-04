#!/bin/bash

# Run the full study
./run_study_full.sh llama OpenPilot_2016 0
./run_study_full.sh llama OpenPilot_2k19 0
./run_study_full.sh llama External_Jutah 0

# Run the subset study
./run_study_subset.sh llama OpenPilot_2016 0
./run_study_subset.sh llama OpenPilot_2k19 0
./run_study_subset.sh llama External_Jutah 0