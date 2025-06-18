FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir flash_attn==2.5.9.post1 --no-build-isolation
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for API
RUN pip install --no-cache-dir fastapi uvicorn python-multipart aiofiles

# Install apex
RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ && \
    cd .. && rm -rf apex

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p ckpts test_videos results

# Download color fix file
RUN wget -O projects/video_diffusion_sr/color_fix.py https://raw.githubusercontent.com/pkuliyi2015/sd-webui-stablesr/master/srmodule/colorfix.py

# Make download script executable
RUN chmod +x download_models.sh

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python", "api_server.py"]