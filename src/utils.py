import json
import re
from typing import List, Dict


def format_context(context: List[Dict], max_turns: int = 10) -> str:
    """Format dialogue history for IEMOCAP/MELD field names."""
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
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def validate_label(label: str, valid_labels: list, default: str = "neutral") -> str:
    label = label.lower().strip()
    if label in valid_labels:
        return label
    for vl in valid_labels:
        if vl in label or label in vl:
            return vl
    return default


def validate_label_with_flag(label: str, valid_labels: list, default: str = "neutral"):
    """Return (label, is_fallback). is_fallback is True if default was used after failed match."""
    original = label.lower().strip()
    if original in valid_labels:
        return original, False
    for vl in valid_labels:
        if vl in original or original in vl:
            return vl, False
    return default, True
