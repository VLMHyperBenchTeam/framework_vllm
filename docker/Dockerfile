# Образ для контейнера
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04
ARG PYTORCH_CUDA_REPO=cu124

# Установка инструментов для сборки
RUN apt-get update -y && apt upgrade -y && apt install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Создание ссылки для Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Обновление pip
RUN python3 -m pip install --upgrade pip

# Установка библиотек
ARG pytorch_url=https://download.pytorch.org/whl/${PYTORCH_CUDA_REPO}
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url ${pytorch_url}
RUN pip install --no-cache-dir "transformers>=4.48.2" \
    vllm==0.7.2

# Очистка кэша pip (опционально)
RUN pip cache purge

# Создание рабочей директории
WORKDIR /workspace

# Запуск терминала
CMD ["/bin/bash"]
