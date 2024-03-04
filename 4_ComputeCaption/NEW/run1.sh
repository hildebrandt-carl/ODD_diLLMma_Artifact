#!/bin/bash

# Run the full study
./run_study_full.sh vicuna External_Jutah 1
./run_study_full.sh vicuna OpenPilot_2016 1
./run_study_full.sh vicuna OpenPilot_2k19 1

# Run the subset study
./run_study_subset.sh vicuna External_Jutah 1
# ./run_study_subset.sh vicuna OpenPilot_2016 1
# ./run_study_subset.sh vicuna OpenPilot_2k19 1