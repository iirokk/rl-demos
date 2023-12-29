#!/bin/bash

pip install --upgrade pip
pip install -r requirements.txt

# Verify the installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"