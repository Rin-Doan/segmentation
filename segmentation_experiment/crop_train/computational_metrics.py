#import necessary libraries
import os
import torch
import psutil

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return {
            'ram_gb': process.memory_info().rss / 1024**3,
            'gpu_allocated_gb': gpu_memory,
            'gpu_reserved_gb': gpu_reserved
        }
    else:
        return {
            'ram_gb': process.memory_info().rss / 1024**3,
            'gpu_allocated_gb': 0,
            'gpu_reserved_gb': 0
        }

def get_model_size_mb(model_path):
    """Get model file size in MB"""
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / 1024**2
    return 0

