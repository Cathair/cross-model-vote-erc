"""MPAR 工具函数."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import format_context, parse_json_response, validate_label  # noqa: F401


def extract_emotion(agent_output: dict, agent_name: str) -> str:
    if agent_name == "LRA":
        for key in ("inferred_speaker_emotion", "emotion", "choice"):
            if key in agent_output:
                return str(agent_output[key]).lower().strip()
    for key in ("emotion", "choice", "inferred_speaker_emotion"):
        if key in agent_output:
            return str(agent_output[key]).lower().strip()
    return "neutral"
