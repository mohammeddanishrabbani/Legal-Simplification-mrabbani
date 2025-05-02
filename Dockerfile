FROM nvidia/cuda:12.6.2-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHON_VERSION=3.10.17

# Install build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and compile Python from source
WORKDIR /usr/src

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
RUN tar xzf Python-${PYTHON_VERSION}.tgz
WORKDIR /usr/src/Python-${PYTHON_VERSION}
RUN ./configure --enable-optimizations --with-ensurepip=install
RUN make -j$(nproc)
RUN make altinstall

# Link python3 and pip3
RUN ln -sf /usr/local/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3

# Verify installations
RUN python3 --version
RUN pip3 --version

# Upgrade pip and install libraries
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
RUN pip3 install --no-cache-dir \
    transformers datasets accelerate unsloth \
    evaluate bert_score sacremoses sacrebleu rouge_score

RUN export PYTHONPATH=SFT_trainer:$PYTHONPATH
WORKDIR /workspace
