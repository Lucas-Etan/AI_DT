"""用反例触发类型、范围、重复值和跨字段规则。"""

import json

from pydantic import ValidationError

from schemas import StudyPlan
import settings

base = {
    "topic": "注意力机制",
    "difficulty": 3,
    "keywords": ["Token", "Attention"],
    "estimated_minutes": 25,
}
cases = {
    "wrong_type": {**base, "difficulty": "困难"},
    "out_of_range": {**base, "difficulty": 8},
    "duplicate_keywords": {**base, "keywords": ["Token", "Token"]},
    "business_rule": {**base, "difficulty": 5, "estimated_minutes": 20},
}
for name, data in cases.items():
    try:
        StudyPlan.model_validate_json(json.dumps(data, ensure_ascii=False))
        print(f"{name}=UNEXPECTED_PASS")
    except ValidationError as error:
        first = error.errors()[0]
        print(f"{name}=REJECTED type={first['type']} location={first['loc']}")
