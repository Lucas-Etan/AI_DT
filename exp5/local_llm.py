"""加载本地 Qwen，并只解码新生成的 token。"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import CACHE_DIR, MODEL_ID


def load_model():
    """从固定缓存加载 tokenizer 和 CPU float32 模型。"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        dtype=torch.float32,
    ).to("cpu")
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def generate_reply(tokenizer, model, messages, *, max_new_tokens=32, **sampling):
    """套用聊天模板、执行推理，只返回新增文本及 token 数。"""
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    prompt_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, **sampling,
        )
    new_ids = output_ids[0, prompt_length:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, prompt_length, len(new_ids)
