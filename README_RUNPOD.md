# SeedVR RunPod Serverless Worker

This guide explains how to deploy SeedVR as a RunPod serverless worker for scalable video generation inference.

## 📋 Overview

The RunPod deployment includes:
- **Serverless Handler**: `runpod_handler.py` - Main worker logic
- **RunPod Dockerfile**: `Dockerfile.runpod` - Optimized container for RunPod
- **Configuration**: `runpod.toml` - Worker settings and schema
- **Deployment Script**: `deploy_runpod.py` - Automated deployment
- **Test Suite**: `test_runpod_worker.py` - Local and remote testing

## 🚀 Quick Start

### Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://runpod.io)
2. **API Key**: Get your API key from RunPod dashboard
3. **Docker**: Install Docker for building images
4. **Python 3.9+**: For running deployment scripts

### 1. Install Dependencies

```bash
# Install RunPod SDK
pip install runpod

# Install other dependencies
pip install -r requirements.txt
```

### 2. Build and Deploy

#### Option A: Automated Deployment

```bash
# Deploy SeedVR-7B worker
python deploy_runpod.py --api-key YOUR_API_KEY --worker-name seedvr-7b --model-size 7b

# Deploy SeedVR-3B worker (requires less GPU memory)
python deploy_runpod.py --api-key YOUR_API_KEY --worker-name seedvr-3b --model-size 3b
```

#### Option B: Manual Deployment

```bash
# 1. Build Docker image
docker build -f Dockerfile.runpod -t seedvr-runpod:latest .

# 2. Push to your registry (Docker Hub, etc.)
docker tag seedvr-runpod:latest your-registry/seedvr-runpod:latest
docker push your-registry/seedvr-runpod:latest

# 3. Create template and endpoint via RunPod dashboard
```

### 3. Test the Worker

```bash
# Test locally
python test_runpod_worker.py --video-path test_video.mp4 --create-test-video

# Test remote endpoint
python test_runpod_worker.py \
    --video-path test_video.mp4 \
    --endpoint-id YOUR_ENDPOINT_ID \
    --api-key YOUR_API_KEY \
    --test-remote
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_SIZE` | `7b` | Model size (`3b` or `7b`) |
| `SP_SIZE` | `1` | Sequence parallel size |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device selection |
| `PYTHONUNBUFFERED` | `1` | Python output buffering |

### Worker Configuration (`runpod.toml`)

```toml
[worker.resources]
gpu_count = 1
gpu_memory_gb = 24  # 24GB for 7B, 12GB for 3B
cpu_count = 4
memory_gb = 32

[worker.scaling]
min_workers = 0
max_workers = 10
idle_timeout = 300
```

## 📡 API Usage

### Input Schema

```json
{
  "video_data": "base64_encoded_video",  // Optional: Base64 video data
  "video_url": "https://example.com/video.mp4",  // Optional: Video URL
  "cfg_scale": 1.0,  // CFG scale (0.1-10.0)
  "cfg_rescale": 0.0,  // CFG rescale (0.0-1.0)
  "sample_steps": 1,  // Sampling steps (1-50)
  "seed": 666,  // Random seed
  "res_h": 720,  // Output height (256-1024)
  "res_w": 1280  // Output width (256-1920)
}
```

### Output Schema

```json
{
  "status": "success",
  "result_video": "base64_encoded_result",
  "result_path": "/tmp/result.mp4",
  "parameters": {
    "cfg_scale": 1.0,
    "sample_steps": 1,
    "seed": 666,
    "resolution": "1280x720",
    "model_size": "7b"
  }
}
```

### Python Client Example

```python
import runpod
import base64

# Initialize client
runpod.api_key = "your-api-key"

# Prepare input video
with open("input_video.mp4", "rb") as f:
    video_data = base64.b64encode(f.read()).decode()

# Run inference
result = runpod.run_sync(
    endpoint_id="your-endpoint-id",
    job_input={
        "video_data": video_data,
        "cfg_scale": 1.0,
        "sample_steps": 1,
        "seed": 42,
        "res_h": 720,
        "res_w": 1280
    }
)

# Save result
if result["status"] == "success":
    result_video = base64.b64decode(result["result_video"])
    with open("output_video.mp4", "wb") as f:
        f.write(result_video)
    print("Video generated successfully!")
else:
    print(f"Error: {result['error']}")
```

### Async Processing

```python
# Submit async job
job = runpod.run(
    endpoint_id="your-endpoint-id",
    job_input={"video_data": video_data}
)

job_id = job["id"]
print(f"Job submitted: {job_id}")

# Check status
status = runpod.status("your-endpoint-id", job_id)
print(f"Status: {status['status']}")

# Get result when completed
if status["status"] == "COMPLETED":
    result = runpod.result("your-endpoint-id", job_id)
```

### cURL Example

```bash
# Submit job
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video_data": "base64_encoded_video_data",
      "cfg_scale": 1.0,
      "sample_steps": 1,
      "seed": 666
    }
  }'

# Check status
curl "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## 🎯 Model Variants

### SeedVR2-7B
- **GPU Requirements**: H100 PCIe (80GB) - Recommended for optimal performance
- **Processing Time**: ~30-45 seconds per video (improved with H100 PCIe)
- **Quality**: Higher quality output
- **Use Case**: Production deployments

### SeedVR2-3B
- **GPU Requirements**: H100 PCIe (80GB) - Can utilize full GPU capacity
- **Processing Time**: ~20-30 seconds per video (improved with H100 PCIe)
- **Quality**: Good quality output
- **Use Case**: Cost-effective deployments with high performance

## 📊 Performance Guidelines

### Processing Times

| Resolution | Model | GPU | Time (approx) |
|------------|-------|-----|---------------|
| 720p | 3B | H100 PCIe | 20-30s |
| 720p | 7B | H100 PCIe | 30-45s |
| 1080p | 3B | H100 PCIe | 30-45s |
| 1080p | 7B | H100 PCIe | 45-60s |

### Memory Usage

- **SeedVR2-3B**: ~12GB GPU memory (H100 PCIe has 80GB available)
- **SeedVR2-7B**: ~20GB GPU memory (H100 PCIe has 80GB available)
- **System RAM**: 32GB+ recommended for H100 PCIe systems
- **Storage**: 50GB+ for models and cache

### Scaling Recommendations

```toml
# High traffic
min_workers = 2
max_workers = 20
idle_timeout = 60

# Cost optimized
min_workers = 0
max_workers = 5
idle_timeout = 300

# Development
min_workers = 0
max_workers = 2
idle_timeout = 600
```

## 🔍 Monitoring and Debugging

### Logs

```bash
# View worker logs in RunPod dashboard
# Or check local logs during testing
python test_runpod_worker.py --video-path test.mp4 2>&1 | tee worker.log
```

### Health Checks

The worker includes automatic health checks:
- Model loading verification
- GPU memory monitoring
- Processing timeout handling

### Common Issues

1. **Out of Memory**
   - Reduce resolution or use 3B model
   - Check GPU memory requirements

2. **Slow Processing**
   - Verify GPU type and memory
   - Check network bandwidth for large videos

3. **Model Loading Errors**
   - Ensure models are downloaded
   - Check file permissions and paths

## 💰 Cost Optimization

### GPU Selection

| GPU Type | Cost/hr | Model Support | Recommendation |
|----------|---------|---------------|----------------|
| H100 PCIe | $2.19/hr | 3B, 7B | Best performance, cost-effective |
| H100 NVL | $2.79/hr | 3B, 7B | Premium performance |
| H200 SXM | $3.39/hr | 3B, 7B | Highest performance |
| L40S | $0.86/hr | 3B, 7B | Budget option |
| L40 | $0.99/hr | 3B, 7B | Budget option |

### Scaling Strategy

- Use **auto-scaling** for variable workloads
- Set appropriate **idle timeouts**
- Consider **spot instances** for non-critical tasks
- Use **3B model** for cost-sensitive applications

## 🔒 Security

### Best Practices

1. **API Key Management**
   - Store API keys securely
   - Use environment variables
   - Rotate keys regularly

2. **Input Validation**
   - Validate video formats
   - Limit file sizes
   - Sanitize parameters

3. **Output Security**
   - Clean temporary files
   - Limit result storage time
   - Use secure transfer methods

## 🛠️ Development

### Local Testing

```bash
# Test handler locally
python -c "from runpod_handler import handler; print(handler({'input': {'video_data': 'test'}}))"

# Run full test suite
python test_runpod_worker.py --create-test-video --test-local
```

### Custom Modifications

1. **Handler Customization**
   - Modify `runpod_handler.py`
   - Add preprocessing/postprocessing
   - Implement custom validation

2. **Docker Optimization**
   - Update `Dockerfile.runpod`
   - Add custom dependencies
   - Optimize layer caching

3. **Configuration Tuning**
   - Adjust `runpod.toml`
   - Modify resource requirements
   - Update scaling parameters

## 📚 Additional Resources

- [RunPod Documentation](https://docs.runpod.io/)
- [SeedVR Paper](https://arxiv.org/abs/2310.06750)
- [RunPod Community](https://discord.gg/runpod)
- [GitHub Issues](https://github.com/your-repo/issues)

## 🆘 Support

For issues and questions:

1. **Check logs** in RunPod dashboard
2. **Run local tests** with `test_runpod_worker.py`
3. **Review configuration** in `runpod.toml`
4. **Open GitHub issue** with logs and configuration

## 📄 License

This project is licensed under the same terms as the original SeedVR project.

---

**Happy video generating! 🎬✨**