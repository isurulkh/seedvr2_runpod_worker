#!/usr/bin/env python3
"""
SeedVR Inference API Server
Provides REST API endpoints for video restoration using SeedVR models.
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
runner = None
model_loaded = False

class InferenceRequest(BaseModel):
    """Request model for inference parameters"""
    cfg_scale: float = 1.0
    cfg_rescale: float = 0.0
    sample_steps: int = 1
    seed: int = 666
    res_h: int = 720
    res_w: int = 1280
    sp_size: int = 1
    model_size: str = "7b"  # "3b" or "7b"

class InferenceResponse(BaseModel):
    """Response model for inference results"""
    task_id: str
    status: str
    message: str
    output_path: Optional[str] = None
    processing_time: Optional[float] = None

# Store for tracking tasks
tasks = {}

def load_model(model_size: str = "7b"):
    """Load the SeedVR model"""
    global runner, model_loaded
    
    try:
        logger.info(f"Loading SeedVR2-{model_size.upper()} model...")
        
        # Import required modules
        sys.path.append('/app')
        
        if model_size == "7b":
            from projects.inference_seedvr2_7b import configure_runner
        elif model_size == "3b":
            from projects.inference_seedvr2_3b import configure_runner
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
        
        # Configure and load model
        sp_size = int(os.getenv('SP_SIZE', '1'))
        runner = configure_runner(sp_size)
        model_loaded = True
        
        logger.info(f"SeedVR2-{model_size.upper()} model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_loaded = False
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting SeedVR API Server...")
    
    # Load model on startup
    model_size = os.getenv('MODEL_SIZE', '7b')
    success = load_model(model_size)
    
    if not success:
        logger.error("Failed to load model on startup")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SeedVR API Server...")

# Create FastAPI app
app = FastAPI(
    title="SeedVR Inference API",
    description="Video restoration API using SeedVR diffusion models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SeedVR Inference API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "endpoints": {
            "health": "/health",
            "inference": "/inference",
            "status": "/status/{task_id}",
            "download": "/download/{task_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

@app.post("/inference", response_model=InferenceResponse)
async def create_inference(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    cfg_scale: float = 1.0,
    cfg_rescale: float = 0.0,
    sample_steps: int = 1,
    seed: int = 666,
    res_h: int = 720,
    res_w: int = 1280,
    sp_size: int = 1
):
    """Create a new inference task"""
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Create task entry
    tasks[task_id] = {
        "status": "pending",
        "message": "Task created, processing will start shortly",
        "output_path": None,
        "processing_time": None
    }
    
    # Create request object
    request = InferenceRequest(
        cfg_scale=cfg_scale,
        cfg_rescale=cfg_rescale,
        sample_steps=sample_steps,
        seed=seed,
        res_h=res_h,
        res_w=res_w,
        sp_size=sp_size
    )
    
    # Add background task
    background_tasks.add_task(process_video, task_id, video, request)
    
    return InferenceResponse(
        task_id=task_id,
        status="pending",
        message="Task created successfully"
    )

async def process_video(task_id: str, video: UploadFile, request: InferenceRequest):
    """Process video in background"""
    import time
    start_time = time.time()
    
    try:
        # Update task status
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "Processing video..."
        
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save uploaded video
        input_path = os.path.join(input_dir, video.filename)
        with open(input_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        logger.info(f"Processing video: {video.filename} for task {task_id}")
        
        # Import and run inference
        sys.path.append('/app')
        
        if request.model_size == "7b":
            from projects.inference_seedvr2_7b import generation_loop
        else:
            from projects.inference_seedvr2_3b import generation_loop
        
        # Run inference
        generation_loop(
            runner=runner,
            video_path=input_dir,
            output_dir=output_dir,
            batch_size=1,
            cfg_scale=request.cfg_scale,
            cfg_rescale=request.cfg_rescale,
            sample_steps=request.sample_steps,
            seed=request.seed,
            res_h=request.res_h,
            res_w=request.res_w,
            sp_size=request.sp_size
        )
        
        # Find output file
        output_files = list(Path(output_dir).glob("*"))
        if not output_files:
            raise Exception("No output file generated")
        
        output_file = output_files[0]
        
        # Move output to results directory
        results_dir = "/app/results"
        os.makedirs(results_dir, exist_ok=True)
        final_output_path = os.path.join(results_dir, f"{task_id}_{output_file.name}")
        shutil.move(str(output_file), final_output_path)
        
        # Update task status
        processing_time = time.time() - start_time
        tasks[task_id].update({
            "status": "completed",
            "message": "Video processing completed successfully",
            "output_path": final_output_path,
            "processing_time": processing_time
        })
        
        logger.info(f"Task {task_id} completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        tasks[task_id].update({
            "status": "failed",
            "message": f"Processing failed: {str(e)}",
            "processing_time": time.time() - start_time
        })
    
    finally:
        # Cleanup temporary directory
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/status/{task_id}", response_model=InferenceResponse)
async def get_task_status(task_id: str):
    """Get task status"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return InferenceResponse(
        task_id=task_id,
        status=task["status"],
        message=task["message"],
        output_path=task["output_path"],
        processing_time=task["processing_time"]
    )

@app.get("/download/{task_id}")
async def download_result(task_id: str):
    """Download processed video"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    
    if not task["output_path"] or not os.path.exists(task["output_path"]):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=task["output_path"],
        filename=os.path.basename(task["output_path"]),
        media_type='application/octet-stream'
    )

@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    return {"tasks": tasks}

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its output file"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    # Remove output file if exists
    if task["output_path"] and os.path.exists(task["output_path"]):
        os.remove(task["output_path"])
    
    # Remove task from memory
    del tasks[task_id]
    
    return {"message": "Task deleted successfully"}

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    
    logger.info(f"Starting SeedVR API server on {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )