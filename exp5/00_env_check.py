"""检查解释器、依赖、CPU 策略和可用内存。"""

import importlib.metadata as md
import platform
import sys

import psutil
import torch

from settings import CPU_THREADS, MODEL_ID

if sys.version_info[:2] != (3, 12):
    raise RuntimeError("本实验要求 Python 3.12")
print(f"python={platform.python_version()}")
for package in ("torch", "transformers", "pydantic", "openai", "psutil"):
    print(f"{package}={md.version(package)}")
print(f"torch_cuda_available={torch.cuda.is_available()}")
print("selected_device=cpu")
print(f"cpu_threads={CPU_THREADS}")
print(f"physical_memory_gib={psutil.virtual_memory().total / 1024**3:.1f}")
print(f"model_id={MODEL_ID}")
