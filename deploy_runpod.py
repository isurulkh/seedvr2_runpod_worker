#!/usr/bin/env python3
"""
RunPod Deployment Script for SeedVR

This script automates the deployment of SeedVR as a RunPod serverless worker.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import runpod
except ImportError:
    print("RunPod SDK not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "runpod"])
    import runpod

class RunPodDeployer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        runpod.api_key = api_key
        
    def build_and_push_image(self, image_name: str, model_size: str = "7b") -> str:
        """
        Build and push Docker image to RunPod registry
        """
        print(f"Building Docker image for SeedVR-{model_size.upper()}...")
        
        # Build command
        build_cmd = [
            "docker", "build",
            "-f", "Dockerfile.runpod",
            "-t", image_name,
            "--build-arg", f"MODEL_SIZE={model_size}",
            "."
        ]
        
        try:
            subprocess.run(build_cmd, check=True, cwd=Path.cwd())
            print(f"✅ Docker image {image_name} built successfully")
            
            # Push to RunPod registry (if using RunPod registry)
            # Note: You might need to tag and push to your preferred registry
            print(f"Image {image_name} ready for deployment")
            return image_name
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to build Docker image: {e}")
            raise
    
    def create_template(self, image_name: str, template_name: str, model_size: str = "7b") -> str:
        """
        Create a RunPod template
        """
        print(f"Creating RunPod template: {template_name}")
        
        # Template configuration
        template_config = {
            "name": template_name,
            "imageName": image_name,
            "dockerArgs": "",
            "containerDiskInGb": 100,
            "volumeInGb": 50,
            "volumeMountPath": "/workspace",
            "ports": "8000/http",
            "env": [
                {"key": "MODEL_SIZE", "value": model_size},
                {"key": "SP_SIZE", "value": "1"},
                {"key": "PYTHONUNBUFFERED", "value": "1"}
            ],
            "isServerless": True,
            "readme": self._generate_readme(model_size)
        }
        
        try:
            template = runpod.create_template(template_config)
            template_id = template.get('id')
            print(f"✅ Template created with ID: {template_id}")
            return template_id
            
        except Exception as e:
            print(f"❌ Failed to create template: {e}")
            raise
    
    def create_endpoint(self, template_id: str, endpoint_name: str, model_size: str = "7b") -> str:
        """
        Create a RunPod serverless endpoint
        """
        print(f"Creating RunPod endpoint: {endpoint_name}")
        
        # Endpoint configuration
        endpoint_config = {
            "name": endpoint_name,
            "template_id": template_id,
            "active_workers": 0,
            "max_workers": 5,
            "idle_timeout": 5,
            "locations": {
                "US-OR-1": {
                    "workers_min": 0,
                    "workers_max": 3,
                    "gpu_types": "H100 PCIe"
                }
            },
            "scaling": {
                "type": "queue_delay",
                "target": 1
            }
        }
        
        try:
            endpoint = runpod.create_endpoint(endpoint_config)
            endpoint_id = endpoint.get('id')
            print(f"✅ Endpoint created with ID: {endpoint_id}")
            return endpoint_id
            
        except Exception as e:
            print(f"❌ Failed to create endpoint: {e}")
            raise
    
    def deploy_worker(self, 
                     image_name: str, 
                     worker_name: str, 
                     model_size: str = "7b",
                     create_template: bool = True,
                     create_endpoint: bool = True) -> Dict[str, str]:
        """
        Deploy complete SeedVR worker to RunPod
        """
        results = {}
        
        # Build and push image
        image_name = self.build_and_push_image(image_name, model_size)
        results['image_name'] = image_name
        
        if create_template:
            # Create template
            template_name = f"{worker_name}-template"
            template_id = self.create_template(image_name, template_name, model_size)
            results['template_id'] = template_id
            
            if create_endpoint:
                # Create endpoint
                endpoint_name = f"{worker_name}-endpoint"
                endpoint_id = self.create_endpoint(template_id, endpoint_name, model_size)
                results['endpoint_id'] = endpoint_id
        
        return results
    
    def _generate_readme(self, model_size: str) -> str:
        """
        Generate README for the template
        """
        return f"""
# SeedVR-{model_size.upper()} Inference Worker

This is a RunPod serverless worker for SeedVR video generation.

## Model
- **Model**: SeedVR2-{model_size.upper()}
- **Task**: Video-to-Video Generation
- **GPU Requirements**: H100 PCIe (80GB)

## Input Parameters

- `video_data` (string, optional): Base64 encoded input video
- `video_url` (string, optional): URL to input video file
- `cfg_scale` (float, default=1.0): CFG scale for generation
- `cfg_rescale` (float, default=0.0): CFG rescale factor
- `sample_steps` (int, default=1): Number of sampling steps
- `seed` (int, default=666): Random seed
- `res_h` (int, default=720): Output height
- `res_w` (int, default=1280): Output width

## Output

- `status`: Processing status ("success" or "error")
- `result_video`: Base64 encoded result video (if successful)
- `parameters`: Processing parameters used
- `error`: Error message (if failed)

## Example Usage

```python
import runpod
import base64

# Initialize client
runpod.api_key = "your-api-key"

# Prepare input
with open("input_video.mp4", "rb") as f:
    video_data = base64.b64encode(f.read()).decode()

# Run inference
result = runpod.run_sync(
    endpoint_id="your-endpoint-id",
    job_input={{
        "video_data": video_data,
        "cfg_scale": 1.0,
        "sample_steps": 1,
        "seed": 42
    }}
)

# Save result
if result["status"] == "success":
    result_video = base64.b64decode(result["result_video"])
    with open("output_video.mp4", "wb") as f:
        f.write(result_video)
```

## Performance

- **Processing Time**: ~30-60 seconds per video (depending on length and resolution)
- **Memory Usage**: ~{20 if model_size == '7b' else 12}GB GPU memory
- **Supported Formats**: MP4, AVI, MOV
- **Max Resolution**: 1920x1080
"""

def main():
    parser = argparse.ArgumentParser(description="Deploy SeedVR to RunPod")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--worker-name", default="seedvr-worker", help="Worker name")
    parser.add_argument("--model-size", choices=["3b", "7b"], default="7b", help="Model size")
    parser.add_argument("--image-name", help="Docker image name")
    parser.add_argument("--skip-template", action="store_true", help="Skip template creation")
    parser.add_argument("--skip-endpoint", action="store_true", help="Skip endpoint creation")
    
    args = parser.parse_args()
    
    # Set default image name
    if not args.image_name:
        args.image_name = f"seedvr-{args.model_size}:latest"
    
    # Initialize deployer
    deployer = RunPodDeployer(args.api_key)
    
    try:
        # Deploy worker
        results = deployer.deploy_worker(
            image_name=args.image_name,
            worker_name=args.worker_name,
            model_size=args.model_size,
            create_template=not args.skip_template,
            create_endpoint=not args.skip_endpoint
        )
        
        print("\n🎉 Deployment completed successfully!")
        print("\nDeployment Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        # Save results to file
        with open(f"deployment_results_{args.worker_name}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: deployment_results_{args.worker_name}.json")
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()