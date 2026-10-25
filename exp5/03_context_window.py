"""在 token 预算内保留系统消息和最新问题。"""

from transformers import AutoTokenizer

from settings import CACHE_DIR, MODEL_ID


def token_count(tokenizer, messages) -> int:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    return len(encoded["input_ids"])


def fit_messages(tokenizer, messages, max_tokens: int):
    system = messages[:1] if messages[0]["role"] == "system" else []
    newest = messages[-1:]
    history = messages[len(system):-1]
    while history:
        candidate = system + history + newest
        if token_count(tokenizer, candidate) <= max_tokens:
            return candidate
        history = history[2:]
    candidate = system + newest
    if token_count(tokenizer, candidate) > max_tokens:
        raise ValueError("system 加最新问题仍超出预算，请缩短文本")
    return candidate


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
messages = [{"role": "system", "content": "你是助教，必须遵守课程规则。"}]
for turn in range(1, 7):
    messages += [
        {"role": "user", "content": f"第{turn}轮：解释Transformer概念并给出例子。" * 2},
        {"role": "assistant", "content": f"第{turn}轮回答：这是用于观察预算的历史。" * 2},
    ]
messages.append({"role": "user", "content": "请只总结最近两轮讨论。"})
fitted = fit_messages(tokenizer, messages, max_tokens=160)
print(f"before_tokens={token_count(tokenizer, messages)} messages={len(messages)}")
print(f"after_tokens={token_count(tokenizer, fitted)} messages={len(fitted)}")
print(f"removed_messages={len(messages) - len(fitted)}")
print(f"system_preserved={fitted[0] == messages[0]} newest_preserved={fitted[-1] == messages[-1]}")
