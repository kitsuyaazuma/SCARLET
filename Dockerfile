FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y \
        build-essential \
        curl \
        git \
        jq \
        libbz2-dev \
        libffi-dev \
        libgdbm-dev \
        liblzma-dev \
        libncurses5-dev \
        libnss3-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        tzdata \
        wget \
        zlib1g-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

ENV PYTHON_VERSION=3.12.5
ENV PATH=/root/.local/bin:/root/.pyenv/shims:/root/.pyenv/bin:$PATH

RUN set -ex && \
    curl https://pyenv.run | bash && \
    pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    curl -sSL https://install.python-poetry.org | python -

WORKDIR /app
RUN git clone -b reproducibility https://github.com/kitsuyaazuma/SCARLET.git && \
    cd SCARLET && \
    poetry install
