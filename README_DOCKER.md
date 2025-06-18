# SeedVR Docker Deployment Guide

This guide provides instructions for deploying SeedVR as a containerized inference API service.

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- At least 16GB GPU memory (recommended: H100-80G for optimal performance)
- 50GB+ free disk space for models and temporary files

### 1. Clone and Setup

```bash
git clone https://github.com/bytedance-seed/SeedVR.git
cd SeedVR
```

### 2. Download Models

**Option A: Automatic Download (Recommended)**
```bash
# Make download script executable
chmod +x download_models.sh

# Download SeedVR2-7B model (default)
./download_models.sh

# Or download SeedVR2-3B model
MODEL_SIZE=3b ./download_models.sh
```

**Option B: Manual Download**
```python
# Create ckpts directory
mkdir -p ckpts
cd ckpts

# Download using Python
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ByteDance-Seed/SeedVR2-7B',
    local_dir='.',
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=['*.json', '*.safetensors', '*.pth', '*.bin', '*.py', '*.md', '*.txt']
)
"
```

### 3. Build and Run

**Using Docker Compose (Recommended)**
```bash
# Build and start the service
docker-compose up --build -d

# Check logs
docker-compose logs -f seedvr-api

# Check health
curl http://localhost:8000/health
```

**Using Docker directly**
```bash
# Build image
docker build -t seedvr-api .

# Run container
docker run -d \
  --name seedvr-inference \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/ckpts:/app/ckpts \
  -v $(pwd)/results:/app/results \
  -e MODEL_SIZE=7b \
  seedvr-api
```

## 📡 API Usage

### Web Interface
Open your browser and navigate to: `http://localhost` (if using docker-compose) or use the API directly.

### REST API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Submit Video for Processing
```bash
curl -X POST \
  -F "video=@your_video.mp4" \
  -F "cfg_scale=1.0" \
  -F "sample_steps=1" \
  -F "seed=666" \
  -F "res_h=720" \
  -F "res_w=1280" \
  http://localhost:8000/inference
```

Response:
```json
{
  "task_id": "uuid-string",
  "status": "pending",
  "message": "Task created successfully"
}
```

#### 3. Check Processing Status
```bash
curl http://localhost:8000/status/{task_id}
```

#### 4. Download Result
```bash
curl -O http://localhost:8000/download/{task_id}
```

#### 5. List All Tasks
```bash
curl http://localhost:8000/tasks
```

### Python Client Example

```python
import requests
import time

API_BASE = "http://localhost:8000"

# Upload video for processing
with open("input_video.mp4", "rb") as f:
    files = {"video": f}
    data = {
        "cfg_scale": 1.0,
        "sample_steps": 1,
        "seed": 666,
        "res_h": 720,
        "res_w": 1280
    }
    response = requests.post(f"{API_BASE}/inference", files=files, data=data)
    task_id = response.json()["task_id"]

# Poll for completion
while True:
    status_response = requests.get(f"{API_BASE}/status/{task_id}")
    status = status_response.json()
    
    print(f"Status: {status['status']} - {status['message']}")
    
    if status["status"] == "completed":
        # Download result
        result_response = requests.get(f"{API_BASE}/download/{task_id}")
        with open("output_video.mp4", "wb") as f:
            f.write(result_response.content)
        print("Video processing completed!")
        break
    elif status["status"] == "failed":
        print(f"Processing failed: {status['message']}")
        break
    
    time.sleep(5)  # Wait 5 seconds before checking again
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_SIZE` | `7b` | Model size: `3b` or `7b` |
| `SP_SIZE` | `1` | Sequence parallel size for multi-GPU |
| `HOST` | `0.0.0.0` | API server host |
| `PORT` | `8000` | API server port |
| `WORKERS` | `1` | Number of worker processes |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU devices to use |

### Model Configurations

**SeedVR2-7B (Default)**
- Better quality results
- Requires more GPU memory (~16GB+)
- Slower processing

**SeedVR2-3B**
- Faster processing
- Lower GPU memory requirements (~8GB+)
- Good quality results

To switch models:
```bash
# In docker-compose.yml, change:
MODEL_SIZE=3b

# Or when running docker directly:
docker run -e MODEL_SIZE=3b ...
```

### Processing Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `cfg_scale` | `1.0` | `0.1-10.0` | Classifier-free guidance scale |
| `cfg_rescale` | `0.0` | `0.0-1.0` | CFG rescale factor |
| `sample_steps` | `1` | `1-100` | Number of sampling steps |
| `seed` | `666` | Any integer | Random seed for reproducibility |
| `res_h` | `720` | Multiple of 16 | Output video height |
| `res_w` | `1280` | Multiple of 16 | Output video width |

## 🔧 Multi-GPU Setup

For better performance with multiple GPUs:

```yaml
# docker-compose.yml
services:
  seedvr-api:
    environment:
      - SP_SIZE=4  # Use 4 GPUs
      - CUDA_VISIBLE_DEVICES=0,1,2,3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 4  # Request 4 GPUs
              capabilities: [gpu]
```

## 📊 Performance Guidelines

### GPU Memory Requirements

| Model | Min GPU Memory | Recommended | Max Resolution |
|-------|----------------|-------------|----------------|
| SeedVR2-3B | 8GB | 16GB | 720p |
| SeedVR2-7B | 16GB | 24GB+ | 1080p+ |

### Processing Times (Approximate)

| Video Length | Resolution | SeedVR2-3B | SeedVR2-7B |
|--------------|------------|------------|------------|
| 5 seconds | 720p | ~30s | ~60s |
| 10 seconds | 720p | ~60s | ~120s |
| 5 seconds | 1080p | ~60s | ~120s |

*Times vary based on GPU, video complexity, and settings*

## 🐛 Troubleshooting

### Common Issues

**1. Out of GPU Memory**
```bash
# Solution: Use smaller model or reduce resolution
docker-compose down
# Edit docker-compose.yml: MODEL_SIZE=3b
docker-compose up -d
```

**2. Model Download Fails**
```bash
# Manual download
docker exec -it seedvr-inference-api bash
cd /app
./download_models.sh
```

**3. API Not Responding**
```bash
# Check container logs
docker-compose logs seedvr-api

# Restart service
docker-compose restart seedvr-api
```

**4. CUDA Not Available**
```bash
# Install NVIDIA Docker runtime
# Ubuntu/Debian:
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Logs and Monitoring

```bash
# View API logs
docker-compose logs -f seedvr-api

# Monitor GPU usage
nvidia-smi -l 1

# Check container resource usage
docker stats seedvr-inference-api
```

## 🔒 Security Considerations

- The API runs without authentication by default
- For production use, consider adding:
  - API key authentication
  - Rate limiting
  - Input validation
  - HTTPS/TLS encryption
  - Network security (firewall, VPN)

## 📝 Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart aiofiles

# Run API server locally
python api_server.py
```

### Custom Modifications

1. **Modify API endpoints**: Edit `api_server.py`
2. **Change model parameters**: Edit config files in `configs_7b/` or `configs_3b/`
3. **Add preprocessing**: Modify inference scripts in `projects/`

## 📄 License

SeedVR is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/bytedance-seed/SeedVR/issues)
- **Documentation**: [Project Website](https://iceclear.github.io/projects/seedvr2/)
- **Papers**: [SeedVR](https://arxiv.org/abs/2501.01320) | [SeedVR2](http://arxiv.org/abs/2506.05301)

---

**Happy Video Restoration! 🎬✨**