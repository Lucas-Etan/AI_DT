"""比较贪心解码与两组 Temperature/Top-p 采样。"""

import json

import torch

from local_llm import generate_reply, load_model
from runtime_utils import measure, prepare_cpu
from settings import CACHE_DIR, OUTPUT_DIR, SEED

prepare_cpu()
(tokenizer, model), load_s, load_peak = measure(load_model)
print(f"load_seconds={load_s:.3f} load_peak_rss_mb={load_peak:.1f}")
cache_mb = sum(path.stat().st_size for path in CACHE_DIR.rglob("*") if path.is_file()) / 1024**2
print(f"model_class={type(model).__name__} parameters={sum(p.numel() for p in model.parameters())}")
print(f"context_limit={model.config.max_position_embeddings} cache_mb={cache_mb:.1f}")
messages = [
    {"role": "system", "content": "你是计算机专业助教，回答准确、简洁。"},
    {"role": "user", "content": "用一句话说明注意力机制的作用，30字以内。"},
]
configs = {
    "greedy": {"do_sample": False},
    "focused": {"do_sample": True, "temperature": 0.3, "top_p": 0.8},
    "diverse": {"do_sample": True, "temperature": 1.0, "top_p": 0.95},
}
results = {}
for name, config in configs.items():
    torch.manual_seed(SEED)
    (text, prompt_n, new_n), seconds, peak = measure(
        lambda c=config: generate_reply(
            tokenizer, model, messages,
            max_new_tokens=32, **c,
        )
    )
    results[name] = {
        "text": text, "seconds": seconds, "new_tokens": new_n
    }
    print(f"{name}: seconds={seconds:.3f} peak_mb={peak:.1f} prompt={prompt_n} new={new_n}")
    print(f"  {text}")
(OUTPUT_DIR / "decoding_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
)
