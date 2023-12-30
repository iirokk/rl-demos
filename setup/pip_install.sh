#!/bin/bash

pip install --upgrade pip
pip install -r requirements.txt

# Verify the installation:
python3 -c "import torch; print('PyTorch CUDA available: ' + str(torch.cuda.is_available()))"
