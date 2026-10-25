"""CPU 设置、计时与进程峰值内存观测。"""

import random
import threading
import time
from collections.abc import Callable
from typing import Any

import psutil
import torch

from settings import CPU_THREADS, SEED


def prepare_cpu() -> None:
    """固定随机性并限制 CPU 线程，便于同学间比较。"""
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)


def measure(call: Callable[[], Any]) -> tuple[Any, float, float]:
    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stopped = threading.Event()

    def sample_memory() -> None:
        while not stopped.wait(0.02):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        result = call()
    finally:
        elapsed = time.perf_counter() - started
        stopped.set()
        monitor.join()
    return result, elapsed, peak_rss[0] / 1024**2
