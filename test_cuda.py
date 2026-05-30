#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/cloudteam/rag_mzx')
import os, logging
os.environ.update({
    'HF_ENDPOINT': 'https://hf-mirror.com',
    'LD_PRELOAD': '',
    'MKL_THREADING_LAYER': 'GNU',
    'OPENBLAS_NUM_THREADS': '4',
    'OMP_NUM_THREADS': '4',
    'LD_LIBRARY_PATH': '/usr/local/cuda-12.8/lib64',
})
logging.basicConfig(level=logging.WARNING)

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
