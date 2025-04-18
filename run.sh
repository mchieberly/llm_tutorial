#!/bin/bash

# Make sure to run this script from the project root directory. This assumes conda is installed in the home directory.

# Make Conda Environement usable in run script 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmtutor

# Run entire pipeline
if [ ! -d "./processed_data" ]; then
    python -m src.preprocess
fi

if [ ! -d "./processed_data/charts" ]; then
    python -m src.analyze
fi

if [ ! -d "./covid_classifier" ]; then
    python -m src.train
fi

python -m src.interface

# Open the chat interface at http://127.0.0.1:7860/
