"""观察文本 token、消息角色与聊天模板。"""

import json

from transformers import AutoTokenizer

from settings import CACHE_DIR, MODEL_ID, OUTPUT_DIR

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
samples = ["人工智能", "人工智能开发技术", "AI"]
report = {}
for text in samples:
    ids = tokenizer(
        text, add_special_tokens=False
    )["input_ids"]
    pieces = tokenizer.convert_ids_to_tokens(ids)
    report[text] = {"count": len(ids), "ids": ids, "pieces": pieces}
    print(f"text={text!r} count={len(ids)} ids={ids} pieces={pieces}")

messages = [
    {"role": "system", "content": "你是计算机专业助教，回答准确、简洁。"},
    {"role": "user", "content": "用一句话说明注意力机制的作用。"},
]
without_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False
)
with_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print("without_prompt_tail=" + repr(without_prompt[-60:]))
print("with_prompt_tail=" + repr(with_prompt[-60:]))
print(f"added_suffix={with_prompt[len(without_prompt):]!r}")
(OUTPUT_DIR / "token_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
