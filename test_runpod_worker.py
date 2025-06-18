#!/usr/bin/env python3
"""
Test script for SeedVR RunPod Worker

This script tests the RunPod worker both locally and remotely.
"""

import os
import sys
import json
import base64
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import runpod
except ImportError:
    print("RunPod SDK not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "runpod"])
    import runpod

class RunPodWorkerTester:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if api_key:
            runpod.api_key = api_key
    
    def encode_video_file(self, video_path: str) -> str:
        """
        Encode video file to base64
        """
        try:
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            return base64.b64encode(video_bytes).decode('utf-8')
        except Exception as e:
            print(f"Failed to encode video file: {e}")
            raise
    
    def decode_video_result(self, video_data: str, output_path: str) -> None:
        """
        Decode base64 video data and save to file
        """
        try:
            video_bytes = base64.b64decode(video_data)
            with open(output_path, 'wb') as f:
                f.write(video_bytes)
            print(f"Result video saved to: {output_path}")
        except Exception as e:
            print(f"Failed to decode video result: {e}")
            raise
    
    def test_local_handler(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """
        Test the handler function locally
        """
        print("Testing local handler...")
        
        # Import the handler
        sys.path.append(str(Path.cwd()))
        from runpod_handler import handler
        
        # Prepare test input
        video_data = self.encode_video_file(video_path)
        
        job_input = {
            'video_data': video_data,
            'cfg_scale': kwargs.get('cfg_scale', 1.0),
            'cfg_rescale': kwargs.get('cfg_rescale', 0.0),
            'sample_steps': kwargs.get('sample_steps', 1),
            'seed': kwargs.get('seed', 666),
            'res_h': kwargs.get('res_h', 720),
            'res_w': kwargs.get('res_w', 1280)
        }
        
        # Create mock job
        job = {
            'id': 'test-job-local',
            'input': job_input
        }
        
        try:
            # Run handler
            result = handler(job)
            
            # Save result if successful
            if result.get('status') == 'success' and 'result_video' in result:
                output_path = f"test_result_local_{kwargs.get('seed', 666)}.mp4"
                self.decode_video_result(result['result_video'], output_path)
                result['local_output_path'] = output_path
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def test_remote_endpoint(self, endpoint_id: str, video_path: str, **kwargs) -> Dict[str, Any]:
        """
        Test the RunPod endpoint remotely
        """
        print(f"Testing remote endpoint: {endpoint_id}")
        
        if not self.api_key:
            raise ValueError("API key required for remote testing")
        
        # Prepare test input
        video_data = self.encode_video_file(video_path)
        
        job_input = {
            'video_data': video_data,
            'cfg_scale': kwargs.get('cfg_scale', 1.0),
            'cfg_rescale': kwargs.get('cfg_rescale', 0.0),
            'sample_steps': kwargs.get('sample_steps', 1),
            'seed': kwargs.get('seed', 666),
            'res_h': kwargs.get('res_h', 720),
            'res_w': kwargs.get('res_w', 1280)
        }
        
        try:
            # Run job on RunPod
            print("Submitting job to RunPod...")
            result = runpod.run_sync(
                endpoint_id=endpoint_id,
                job_input=job_input,
                timeout=600  # 10 minutes
            )
            
            # Save result if successful
            if result.get('status') == 'success' and 'result_video' in result:
                output_path = f"test_result_remote_{kwargs.get('seed', 666)}.mp4"
                self.decode_video_result(result['result_video'], output_path)
                result['remote_output_path'] = output_path
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def test_async_endpoint(self, endpoint_id: str, video_path: str, **kwargs) -> str:
        """
        Test the RunPod endpoint asynchronously
        """
        print(f"Testing async endpoint: {endpoint_id}")
        
        if not self.api_key:
            raise ValueError("API key required for remote testing")
        
        # Prepare test input
        video_data = self.encode_video_file(video_path)
        
        job_input = {
            'video_data': video_data,
            'cfg_scale': kwargs.get('cfg_scale', 1.0),
            'cfg_rescale': kwargs.get('cfg_rescale', 0.0),
            'sample_steps': kwargs.get('sample_steps', 1),
            'seed': kwargs.get('seed', 666),
            'res_h': kwargs.get('res_h', 720),
            'res_w': kwargs.get('res_w', 1280)
        }
        
        try:
            # Submit async job
            job = runpod.run(
                endpoint_id=endpoint_id,
                job_input=job_input
            )
            
            job_id = job.get('id')
            print(f"Job submitted with ID: {job_id}")
            return job_id
            
        except Exception as e:
            print(f"Failed to submit async job: {e}")
            raise
    
    def check_job_status(self, endpoint_id: str, job_id: str) -> Dict[str, Any]:
        """
        Check the status of an async job
        """
        try:
            status = runpod.status(endpoint_id, job_id)
            return status
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_job_result(self, endpoint_id: str, job_id: str, save_result: bool = True) -> Dict[str, Any]:
        """
        Get the result of a completed job
        """
        try:
            result = runpod.result(endpoint_id, job_id)
            
            # Save result if successful
            if save_result and result.get('status') == 'success' and 'result_video' in result:
                output_path = f"test_result_async_{job_id}.mp4"
                self.decode_video_result(result['result_video'], output_path)
                result['async_output_path'] = output_path
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_comprehensive_test(self, 
                              video_path: str,
                              endpoint_id: Optional[str] = None,
                              test_local: bool = True,
                              test_remote: bool = True,
                              **kwargs) -> Dict[str, Any]:
        """
        Run comprehensive tests
        """
        results = {}
        
        print(f"\n🧪 Running comprehensive tests for SeedVR RunPod Worker")
        print(f"Input video: {video_path}")
        print(f"Parameters: {kwargs}")
        
        # Test local handler
        if test_local:
            print("\n--- Local Handler Test ---")
            try:
                local_result = self.test_local_handler(video_path, **kwargs)
                results['local'] = local_result
                
                if local_result.get('status') == 'success':
                    print("✅ Local test passed")
                else:
                    print(f"❌ Local test failed: {local_result.get('error')}")
                    
            except Exception as e:
                print(f"❌ Local test error: {e}")
                results['local'] = {'status': 'error', 'error': str(e)}
        
        # Test remote endpoint
        if test_remote and endpoint_id:
            print("\n--- Remote Endpoint Test ---")
            try:
                remote_result = self.test_remote_endpoint(endpoint_id, video_path, **kwargs)
                results['remote'] = remote_result
                
                if remote_result.get('status') == 'success':
                    print("✅ Remote test passed")
                else:
                    print(f"❌ Remote test failed: {remote_result.get('error')}")
                    
            except Exception as e:
                print(f"❌ Remote test error: {e}")
                results['remote'] = {'status': 'error', 'error': str(e)}
        
        return results

def create_test_video(output_path: str = "test_video.mp4") -> str:
    """
    Create a simple test video using ffmpeg
    """
    import subprocess
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', 'testsrc=duration=3:size=640x480:rate=30',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Test video created: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Failed to create test video: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Test SeedVR RunPod Worker")
    parser.add_argument("--video-path", help="Path to test video file")
    parser.add_argument("--endpoint-id", help="RunPod endpoint ID for remote testing")
    parser.add_argument("--api-key", help="RunPod API key")
    parser.add_argument("--create-test-video", action="store_true", help="Create a test video")
    parser.add_argument("--test-local", action="store_true", default=True, help="Test local handler")
    parser.add_argument("--test-remote", action="store_true", help="Test remote endpoint")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--sample-steps", type=int, default=1, help="Sample steps")
    parser.add_argument("--seed", type=int, default=666, help="Random seed")
    parser.add_argument("--res-h", type=int, default=720, help="Output height")
    parser.add_argument("--res-w", type=int, default=1280, help="Output width")
    
    args = parser.parse_args()
    
    # Create test video if requested
    if args.create_test_video:
        args.video_path = create_test_video()
    
    # Validate video path
    if not args.video_path or not os.path.exists(args.video_path):
        print("❌ Valid video path required. Use --create-test-video to generate one.")
        sys.exit(1)
    
    # Initialize tester
    tester = RunPodWorkerTester(args.api_key)
    
    # Prepare test parameters
    test_params = {
        'cfg_scale': args.cfg_scale,
        'sample_steps': args.sample_steps,
        'seed': args.seed,
        'res_h': args.res_h,
        'res_w': args.res_w
    }
    
    try:
        # Run tests
        results = tester.run_comprehensive_test(
            video_path=args.video_path,
            endpoint_id=args.endpoint_id,
            test_local=args.test_local,
            test_remote=args.test_remote,
            **test_params
        )
        
        # Save results
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n📊 Test Results Summary:")
        for test_type, result in results.items():
            status = result.get('status', 'unknown')
            print(f"  {test_type}: {status}")
            if status == 'error':
                print(f"    Error: {result.get('error')}")
        
        print("\nDetailed results saved to: test_results.json")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()