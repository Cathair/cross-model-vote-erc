"""MARC v3 LLM 客户端与 Agent."""
import os
import time
from typing import Dict, Optional

import requests
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from openai import RateLimitError, APIError

from .config_v3 import MAX_RETRIES, AGENT_MODELS, AGENT_TEMP, EAA_MODEL
from .prompts_v3 import (
    PHASE0_PROMPTS,
    DISCUSSION_PROMPT_SPLIT,
    DISCUSSION_PROMPT_2TO1,
    DISCUSSION_DYNAMIC_CONSTRAINT_SPLIT,
    DISCUSSION_DYNAMIC_CONSTRAINT_2TO1,
)
from .utils_v3 import parse_json_response, extract_emotion, validate_label

ANTHROPIC_BASE_URL = os.environ.get("OPENAI_API_BASE", os.environ.get("ANTHROPIC_BASE_URL", "https://api.openai.com/v1"))
ANTHROPIC_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# v3.7: API 调用超时 180s（3 分钟），重试 3 次
API_CALL_TIMEOUT = int(os.environ.get("MARC_API_TIMEOUT", "180"))


class _Response:
    def __init__(self, content: str):
        self.content = content


class _AnthropicLLM:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        self.model = model
        self._url = ANTHROPIC_BASE_URL.rstrip("/") + "/messages"
        self._headers = {
            "Authorization": "Bearer " + ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, prompt: str):
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.post(self._url, headers=self._headers, json=body, timeout=API_CALL_TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    text = "".join(
                        b.get("text", "") for b in data.get("content", [])
                        if b.get("type") == "text"
                    )
                    return _Response(text)
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt * 5)
        raise RuntimeError(f"Anthropic LLM failed: {last_err}")


def make_llm(model: str, temperature: float = 0.0):
    if model and model.startswith("claude-"):
        inner = _AnthropicLLM(model, temperature=temperature)
    else:
        inner = ChatOpenAI(model=model, temperature=temperature, request_timeout=API_CALL_TIMEOUT)
    from .cache_llm import maybe_wrap
    return maybe_wrap(inner, model=model, temperature=temperature)


def invoke_with_retry(llm, prompt: str):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return llm.invoke(prompt)
        except (RateLimitError, APIError) as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise
    raise last_err


class ClassifierAgentV3:
    def __init__(self, name: str, label_list: list, model: str = None, temperature: float = None):
        self.name = name
        self.label_list = label_list
        self.model = model or AGENT_MODELS[name]
        self.temperature = AGENT_TEMP if temperature is None else temperature
        self.llm = make_llm(self.model, self.temperature)
        self.prompt_template = PHASE0_PROMPTS[name]
        self.last_output: dict = {}

    def classify(self, utterance: str, speaker: str, context: str) -> dict:
        prompt = PromptTemplate.from_template(self.prompt_template).format(
            label_list=self.label_list,
            context=context,
            speaker=speaker,
            utterance=utterance,
        )
        resp = invoke_with_retry(self.llm, prompt)
        result = parse_json_response(resp.content)
        self.last_output = result
        return result

    def disambiguate(
        self,
        utterance: str,
        speaker: str,
        context: str,
        initial: dict,
        others: Dict[str, dict],
        constraints_block: str,
        discussion_mode: str = "split",
        options_block: str = "",
        fallback_label: str = "neutral",
    ) -> dict:
        initial_label = extract_emotion(initial, self.name)
        initial_evidence = initial.get("evidence_span", "")

        if discussion_mode == "2:1_blind":
            prompt = PromptTemplate.from_template(DISCUSSION_PROMPT_2TO1).format(
                agent_name=self.name,
                context=context,
                speaker=speaker,
                utterance=utterance,
                options_block=options_block or "(none)",
                discussion_constraints=constraints_block,
            )
            default_label = fallback_label or initial_label
        else:
            others_lines = []
            for oname, oout in others.items():
                if oname == self.name:
                    continue
                others_lines.append(
                    f"- {oname}: label={extract_emotion(oout, oname)}, evidence={oout.get('evidence_span', '')}"
                )
            prompt = PromptTemplate.from_template(DISCUSSION_PROMPT_SPLIT).format(
                agent_name=self.name,
                context=context,
                speaker=speaker,
                utterance=utterance,
                initial_label=initial_label,
                initial_evidence=initial_evidence,
                others_block="\n".join(others_lines) or "(none)",
                discussion_constraints=constraints_block,
            )
            default_label = initial_label

        resp = invoke_with_retry(self.llm, prompt)
        result = parse_json_response(resp.content)
        choice = validate_label(result.get("choice", default_label), self.label_list)
        result["choice"] = choice
        result["revised_from_initial"] = choice != initial_label
        self.last_output = result
        return result


def build_dynamic_constraints(
    allowed: list,
    prag_brief: str = "",
    soft_pair_hint: Optional[dict] = None,
    discussion_mode: str = "split",
) -> str:
    """P0-A + P1：动态候选 + 通用 evidence Rubric；soft_pair 仅作可读 hint。"""
    prag_block = ""
    if prag_brief and prag_brief != "No pragmatics signal (Call-2 skipped or unavailable).":
        prag_block = f"Pragmatics note (re-evaluate literal reading): {prag_brief}"
    if soft_pair_hint:
        a, b = soft_pair_hint.get("a"), soft_pair_hint.get("b")
        if a and b:
            prag_block += f"\nSoft ambiguity hint (non-binding): experts split near {a} vs {b}."
    if not prag_block:
        prag_block = "(No pragmatics alert.)"
    template = (
        DISCUSSION_DYNAMIC_CONSTRAINT_2TO1
        if discussion_mode == "2:1_blind"
        else DISCUSSION_DYNAMIC_CONSTRAINT_SPLIT
    )
    return template.format(
        candidate_labels=", ".join(allowed),
        prag_block=prag_block.strip(),
    )


class EAACaller:
    """EAA 两次独立调用。"""

    def __init__(self, model: str = None):
        self.model = model or EAA_MODEL
        self.llm = make_llm(self.model, 0.0)

    def score(self, context: str, speaker: str, utterance: str, phase0: Dict[str, dict]) -> dict:
        from .prompts_v3 import EAA_SCORE_PROMPT

        lines = []
        for name in ("SVA", "CIA", "LRA"):
            p = phase0[name]
            emo = extract_emotion(p, name)
            lines.append(f"{name}: label={emo}, evidence={p.get('evidence_span', '')}")
        prompt = PromptTemplate.from_template(EAA_SCORE_PROMPT).format(
            context=context,
            speaker=speaker,
            utterance=utterance,
            experts_block="\n".join(lines),
        )
        resp = invoke_with_retry(self.llm, prompt)
        return parse_json_response(resp.content)

    def pragmatics(self, context: str, speaker: str, utterance: str) -> dict:
        from .prompts_v3 import EAA_PRAG_PROMPT

        prompt = PromptTemplate.from_template(EAA_PRAG_PROMPT).format(
            context=context, speaker=speaker, utterance=utterance
        )
        resp = invoke_with_retry(self.llm, prompt)
        return parse_json_response(resp.content)
