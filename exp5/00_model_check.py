"""下载或复用模型缓存，核对规模、上下文上限和设备。"""

from local_llm import load_model
from runtime_utils import measure, prepare_cpu
from settings import CACHE_DIR

prepare_cpu()
(tokenizer, model), seconds, peak_mb = measure(load_model)
cache_mb = sum(path.stat().st_size for path in CACHE_DIR.rglob("*") if path.is_file()) / 1024**2
parameters = sum(parameter.numel() for parameter in model.parameters())
print(f"model_class={type(model).__name__}")
print(f"parameters={parameters}")
print(f"device={next(model.parameters()).device}")
print(f"context_limit={model.config.max_position_embeddings}")
print(f"pad_token_id={tokenizer.pad_token_id}")
print(f"cache_dir={CACHE_DIR}")
print(f"cache_mb={cache_mb:.1f} load_seconds={seconds:.3f} peak_rss_mb={peak_mb:.1f}")
assert next(model.parameters()).device.type == "cpu"
assert parameters == 494_032_768
