#!/bin/bash

# Change current directory to project directory.
cd "$(dirname "$0")" || exit

# Install necessary apt packages for development
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.6 python3.6-venv python3-pip

# Create virtual environment directory
python3.6 -m venv venv/

# Activate virtual environment
source venv/bin/activate

# Upgrade Python
python3.6 -m pip install --upgrade pip

# Check version of pip
# Version must be below 18.XX and compatible with Python 3.5+
pip --version

# Install dependencies
pip install -r "requirements.txt"
