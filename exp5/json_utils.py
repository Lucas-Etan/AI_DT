"""从可能带解释文字的模型输出中提取首个完整 JSON 对象。"""

import json


def extract_json_object(text: str) -> str:
    """扫描花括号，并忽略字符串内部及转义后的括号。"""
    start = text.find("{")
    if start < 0:
        raise ValueError("输出中没有 JSON 对象起始符 {")
    depth = 0; in_string = False; escaped = False
    for index, char in enumerate(text[start:], start=start):
        if char == '"' and not escaped:
            in_string = not in_string
        if not in_string:
            depth += char == "{"; depth -= char == "}"
            if depth == 0:
                candidate = text[start:index + 1]
                json.loads(candidate); return candidate
        escaped = char == "\\" and not escaped
        if char != "\\": escaped = False
    raise ValueError("JSON 对象没有闭合")
