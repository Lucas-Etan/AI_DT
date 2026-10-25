"""本地生成 JSON，提取对象，再做类型和业务规则校验。"""

import json

from pydantic import ValidationError

from json_utils import extract_json_object
from local_llm import generate_reply, load_model
from runtime_utils import measure, prepare_cpu
from schemas import StudyPlan
from settings import OUTPUT_DIR

prepare_cpu()
(tokenizer, model), load_s, load_peak = measure(load_model)
messages = [
    {"role": "system", "content": "只输出一个合法JSON对象，不要代码围栏或解释。"},
    {
        "role": "user",
        "content": "制定Transformer复习计划。字段topic字符串、difficulty整数1到5、keywords含2到4个不重复字符串、estimated_minutes整数10到120。难度4以上至少30分钟。",
    },
]
(raw, _, _), seconds, peak = measure(
    lambda: generate_reply(tokenizer, model, messages, max_new_tokens=96, do_sample=False)
)
print(f"load_seconds={load_s:.3f} load_peak_mb={load_peak:.1f}")
print(f"generation_seconds={seconds:.3f} generation_peak_mb={peak:.1f}")
print("raw_output=" + repr(raw))
try:
    json_text = extract_json_object(raw)
    plan = StudyPlan.model_validate_json(json_text)
    print("validation=PASS")
    print(plan.model_dump_json(indent=2))
    payload = {"raw": raw, "validated": plan.model_dump()}
    (OUTPUT_DIR / "structured_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
except (ValueError, ValidationError) as error:
    print("validation=FAIL")
    print(error)
