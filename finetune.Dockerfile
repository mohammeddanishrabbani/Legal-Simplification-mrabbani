# Use the official NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Python dependencies
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip3 install transformers datasets accelerate
RUN pip3 install unsloth

# Set the working directory
WORKDIR /workspace

