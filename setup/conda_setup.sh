#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda env remove -n rl-demos

echo "Creating conda env rl-demos"
conda env create --file pytorch_env.yml -n rl-demos 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rl-demos