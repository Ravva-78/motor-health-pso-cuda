# Installation Guide

## Option A: Local Python Environment
1. Install Python 3.11.
2. `pip install -r requirements.txt`
3. (Optional for GPU) Install CUDA Toolkit 11.8+ and cuDNN. Ensure `numba` detects the GPU.

## Option B: Docker (Experimental)
*Note: Docker deployment is scaffolded but not fully validated.*
1. Install Docker and Docker Compose.
2. Install NVIDIA Container Toolkit (if GPU is desired).
3. `docker-compose up --build -d`