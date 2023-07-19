FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive

# OpenCV dependencies
RUN apt update
RUN apt install ffmpeg libsm6 libxext6 -y

# Upgrade pip & setuptools
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# Install venv requirements
WORKDIR /
COPY requirements.txt /
RUN pip3 install --no-cache-dir -r requirements.txt