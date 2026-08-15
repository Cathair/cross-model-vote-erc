"""InsideOut multi-agent ERC baseline (Mozikov et al., ECAI 2024).

Star topology: 5 Ekman emotion-perspective agents -> 1 Aggregate Agent.
6 LLM calls per sample; temperature=0 for deterministic decoding.
"""
from __future__ import annotations

from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import config
from methods.insideout.insideout_prompts import (
    AGGREGATE_AGENT_PROMPT,
    EKMAN_EMOTIONS,
    EMOTION_AGENT_PROMPT,
)
from utils import format_context, parse_json_response, validate_label_with_flag


def _format_agent_analyses(agent_outputs: dict[str, dict]) -> str:
    lines = []
    for emotion in EKMAN_EMOTIONS:
        out = agent_outputs.get(emotion) or {}
        pred = out.get("emotion", "unknown")
        conf = out.get("confidence", "?")
        rationale = out.get("rationale", "")
        lines.append(
            f"- {emotion.capitalize()} Agent: emotion={pred}, confidence={conf}, "
            f"rationale={rationale}"
        )
    return "\n".join(lines)


class InsideOutModel:
    def __init__(self, model: str | None = None, temperature: float = 0.0):
        self.model = model or config.AGENT_MODEL
        self.temperature = temperature
        self.llm = ChatOpenAI(model=self.model, temperature=self.temperature)

    def _invoke_json(
        self,
        template: str,
        fmt_kwargs: dict,
        label_list: list[str],
        default: str = "neutral",
    ) -> tuple[dict, bool]:
        prompt = PromptTemplate.from_template(template).format(**fmt_kwargs)
        resp = self.llm.invoke(prompt)
        parsed = parse_json_response(resp.content)
        if not parsed:
            return {"emotion": default, "confidence": 0.0, "rationale": "parse_failed"}, True
        emotion_raw = str(parsed.get("emotion", default))
        emotion, fallback = validate_label_with_flag(emotion_raw, label_list, default=default)
        parsed["emotion"] = emotion
        return parsed, fallback

    def predict(
        self,
        utterance: str,
        speaker: str,
        context_records: list[dict],
        label_list: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        label_list = label_list or []
        default = "neutral" if "neutral" in label_list else label_list[0]
        context_str = format_context(context_records)

        agent_outputs: dict[str, dict] = {}
        any_fallback = False

        for emotion in EKMAN_EMOTIONS:
            out, fb = self._invoke_json(
                EMOTION_AGENT_PROMPT,
                {
                    "emotion_perspective": emotion,
                    "context": context_str,
                    "speaker": speaker,
                    "utterance": utterance,
                    "label_list": label_list,
                },
                label_list,
                default=default,
            )
            agent_outputs[emotion] = out
            any_fallback = any_fallback or fb

        agg_out, agg_fb = self._invoke_json(
            AGGREGATE_AGENT_PROMPT,
            {
                "context": context_str,
                "speaker": speaker,
                "utterance": utterance,
                "agent_analyses": _format_agent_analyses(agent_outputs),
                "label_list": label_list,
            },
            label_list,
            default=default,
        )
        any_fallback = any_fallback or agg_fb

        meta = {
            "agent_outputs": agent_outputs,
            "aggregate": agg_out,
            "fallback": any_fallback,
        }
        return agg_out.get("emotion", default), meta
