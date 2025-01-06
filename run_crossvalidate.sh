#!/bin/bash
eval "$(conda shell.bash hook)"  # Initialize Conda (needed for non-interactive shells)
conda activate myenv
python3 crossvalidate.py "$@"