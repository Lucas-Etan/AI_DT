"""实验 5 的固定设置；其他模块先导入本文件。"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "hf_cache"
OUTPUT_DIR = ROOT / "outputs"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 2026
CPU_THREADS = min(4, os.cpu_count() or 1)

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
