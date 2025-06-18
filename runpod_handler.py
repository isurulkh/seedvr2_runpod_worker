import runpod
import os
import sys
import torch
import tempfile
import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add project root to Python path
sys.path.append('/app')

# Import SeedVR modules
from common.config import Config
from common.logger import get_logger
from projects.inference_seedvr2_7b import configure_runner, generation_step
from projects.inference_seedvr2_3b import configure_runner as configure_runner_3b, generation_step as generation_step_3b

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

class SeedVRWorker:
    def __init__(self):
        self.model_size = os.getenv('MODEL_SIZE', '7b').lower()
        self.sp_size = int(os.getenv('SP_SIZE', '1'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.runner = None
        self.config = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the SeedVR model based on model size"""
        try:
            if self.model_size == '3b':
                config_path = '/app/configs_3b/main.yaml'
                self.config = Config.from_file(config_path)
                self.runner = configure_runner_3b(self.config, self.sp_size)
                self.generation_func = generation_step_3b
                logger.info("Initialized SeedVR2-3B model")
            else:
                config_path = '/app/configs_7b/main.yaml'
                self.config = Config.from_file(config_path)
                self.runner = configure_runner(self.config, self.sp_size)
                self.generation_func = generation_step
                logger.info("Initialized SeedVR2-7B model")
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def _decode_video_data(self, video_data: str) -> str:
        """Decode base64 video data and save to temporary file"""
        try:
            # Decode base64 data
            video_bytes = base64.b64decode(video_data)
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(video_bytes)
            temp_file.close()
            
            return temp_file.name
        except Exception as e:
            logger.error(f"Failed to decode video data: {str(e)}")
            raise
    
    def _encode_video_result(self, video_path: str) -> str:
        """Encode video file to base64"""
        try:
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            return base64.b64encode(video_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode video result: {str(e)}")
            raise
    
    def process_video(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process video using SeedVR model"""
        try:
            # Extract parameters from job input
            video_data = job_input.get('video_data')
            video_url = job_input.get('video_url')
            cfg_scale = float(job_input.get('cfg_scale', 1.0))
            cfg_rescale = float(job_input.get('cfg_rescale', 0.0))
            sample_steps = int(job_input.get('sample_steps', 1))
            seed = int(job_input.get('seed', 666))
            res_h = int(job_input.get('res_h', 720))
            res_w = int(job_input.get('res_w', 1280))
            
            # Validate input
            if not video_data and not video_url:
                raise ValueError("Either 'video_data' (base64) or 'video_url' must be provided")
            
            # Handle video input
            if video_data:
                input_video_path = self._decode_video_data(video_data)
            else:
                # For video_url, you might want to download it first
                # This is a simplified implementation
                input_video_path = video_url
            
            # Create output directory
            output_dir = tempfile.mkdtemp()
            
            # Run inference
            logger.info(f"Starting video processing with parameters: cfg_scale={cfg_scale}, steps={sample_steps}, seed={seed}")
            
            result_path = self.generation_func(
                runner=self.runner,
                video_path=input_video_path,
                output_dir=output_dir,
                cfg_scale=cfg_scale,
                cfg_rescale=cfg_rescale,
                sample_steps=sample_steps,
                seed=seed,
                res_h=res_h,
                res_w=res_w
            )
            
            # Encode result video
            result_video_base64 = self._encode_video_result(result_path)
            
            # Cleanup temporary files
            if video_data:  # Only cleanup if we created the temp file
                os.unlink(input_video_path)
            
            return {
                "status": "success",
                "result_video": result_video_base64,
                "result_path": result_path,
                "parameters": {
                    "cfg_scale": cfg_scale,
                    "cfg_rescale": cfg_rescale,
                    "sample_steps": sample_steps,
                    "seed": seed,
                    "resolution": f"{res_w}x{res_h}",
                    "model_size": self.model_size
                }
            }
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

# Global worker instance
worker = None

def initialize_worker():
    """Initialize the worker instance"""
    global worker
    if worker is None:
        logger.info("Initializing SeedVR worker...")
        worker = SeedVRWorker()
        logger.info("SeedVR worker initialized successfully")
    return worker

def handler(job):
    """
    RunPod serverless handler function
    
    Args:
        job (dict): Contains the input data and request metadata
        
    Returns:
        dict: The result to be returned to the client
    """
    try:
        # Initialize worker if not already done
        current_worker = initialize_worker()
        
        # Extract job input
        job_input = job.get('input', {})
        
        # Log job start
        job_id = job.get('id', 'unknown')
        logger.info(f"Processing job {job_id} with input keys: {list(job_input.keys())}")
        
        # Process the video
        result = current_worker.process_video(job_input)
        
        # Log completion
        if result.get('status') == 'success':
            logger.info(f"Job {job_id} completed successfully")
        else:
            logger.error(f"Job {job_id} failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

# For local testing
if __name__ == '__main__':
    # Start the serverless worker
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True
    })