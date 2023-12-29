#!/bin/bash

# CUDA support for WSL (https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

# remove the old GPG key
sudo apt-key del 7fa2af80

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda-repo-wsl-ubuntu-12-3-local_12.3.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-3-local_12.3.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

pip install --upgrade pip
python3 -m pip install tensorflow[and-cuda]
# Verify the installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
