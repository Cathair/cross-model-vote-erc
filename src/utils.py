import json
import re
from typing import List, Dict


def format_context(context: List[Dict], max_turns: int = 10) -> str:
    """格式化对话历史为纯文本，兼容 IEMOCAP 和 MELD 的字段名"""
    if not context:
        return "(No prior context)"
    
    recent = context[-max_turns:] if len(context) > max_turns else context
    lines = []
    for turn in recent:
        text = turn.get("text", turn.get("utterance", ""))
        speaker = turn.get("speaker", "Unknown")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def parse_json_response(text: str) -> dict:
    """从 LLM 响应中提取 JSON（容错处理）"""
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # 尝试从 markdown 代码块中提取
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 尝试从纯文本中提取第一个 JSON 对象
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return {}


def validate_label(label: str, valid_labels: list, default: str = "neutral") -> str:
    """验证标签是否合法，不合法则返回默认值"""
    label = label.lower().strip()
    if label in valid_labels:
        return label
    # 模糊匹配
    for vl in valid_labels:
        if vl in label or label in vl:
            return vl
    return default


def validate_label_with_flag(label: str, valid_labels: list, default: str = "neutral"):
    """与 validate_label 相同，但额外返回是否走了 fallback。

    Returns:
        (validated_label, is_fallback): is_fallback=True 表示既未精确匹配
        也未子串模糊匹配，最终回退到 default（通常为 "neutral"）。
        is_fallback=False 表示精确匹配或模糊匹配命中。
        注意：若 default 本身恰好是合法标签且模型原输出就是 default，则
        is_fallback=False（这是真实预测 default，不是回退）。
    """
    original = label.lower().strip()
    if original in valid_labels:
        return original, False
    for vl in valid_labels:
        if vl in original or original in vl:
            return vl, False
    return default, True
