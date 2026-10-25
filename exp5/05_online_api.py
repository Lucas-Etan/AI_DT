"""可选：调用兼容 Chat Completions 的在线接口；配置缺失则跳过。"""

import os

from openai import OpenAI
from pydantic import ValidationError

from json_utils import extract_json_object
from schemas import StudyPlan
import settings

names = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
missing = [name for name in names if not os.getenv(name)]
if missing:
    print("SKIP online API: missing " + ", ".join(missing))
    raise SystemExit(0)
client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
    timeout=60.0,
    max_retries=2,
)
messages = [
    {"role": "system", "content": "只输出一个合法JSON对象，不要解释。"},
    {
        "role": "user",
        "content": "制定Transformer复习计划。字段topic字符串、difficulty整数1到5、keywords含2到4个不重复字符串、estimated_minutes整数10到120。难度4以上至少30分钟。",
    },
]
response = client.chat.completions.create(
    model=os.environ["LLM_MODEL"], messages=messages, temperature=0, max_tokens=96
)
raw = response.choices[0].message.content or ""
try:
    plan = StudyPlan.model_validate_json(extract_json_object(raw))
    print("validation=PASS " + plan.model_dump_json())
except (ValueError, ValidationError) as error:
    print("validation=FAIL " + str(error))
