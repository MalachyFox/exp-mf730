#!/bin/bash
eval "$(conda shell.bash hook)"
ENV_NAME="myenv"
conda activate "$ENV_NAME"
python3 crossvalidate.py "$@"
