"""MARC v3 路由、动态讨论候选、确定性融合."""
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .config_v3 import (
    BASE_ROLE_WEIGHT,
    CONFUSION_PAIRS,
    ORDINAL_MAP,
    P2_AROUSAL_GATE,
    PRAG_CIA_FACTOR,
    PRAG_LRA_FACTOR,
    PRAG_SVA_FACTOR,
    TAU_G,
    TAU_S,
    TAU_SSPEC,
    AGENT_NAMES,
)


def labels_from_phase0(phase0: Dict[str, dict]) -> Dict[str, str]:
    out = {}
    for name in AGENT_NAMES:
        p = phase0[name]
        if name == "LRA":
            out[name] = str(p.get("inferred_speaker_emotion", p.get("emotion", "neutral"))).lower()
        else:
            out[name] = str(p.get("emotion", "neutral")).lower()
    return out


def dynamic_discussion_allowed(labels: Dict[str, str]) -> List[str]:
    """P0-A：讨论候选 = Phase-0 实际出现的标签并集（不用 gold、不用固定表）。"""
    return sorted(set(labels.values()))


def infer_soft_confusion_hint(strategy: str, labels: Dict[str, str]) -> Optional[dict]:
    """v3.6.1 清理：CONFUSION_PAIRS 已删除，本函数恒返回 None。

    保留函数签名以避免调用方改动；soft hint 不再注入 discussion prompt。
    """
    return None


def has_low_specificity(scores: Dict[str, dict], tau: float = TAU_SSPEC) -> bool:
    """P0-C：任一 Agent label_specificity 低于阈值。"""
    for name in AGENT_NAMES:
        sc = scores.get(name, {})
        if float(sc.get("label_specificity", 1.0)) < tau:
            return True
    return False


def should_skip_eaa_prag(labels: Dict[str, str]) -> bool:
    """两数据集统一：三 Agent 一致 → 跳过 Call-2。"""
    return len(set(labels.values())) == 1


def _prag_signals(prag: Optional[dict]) -> List[str]:
    reasons = []
    if not prag:
        return reasons
    if prag.get("mismatch"):
        reasons.append("prag_mismatch")
    if float(prag.get("sarcasm_likelihood", 0)) >= TAU_S:
        reasons.append("sarcasm_high")
    return reasons


def should_enter_discussion(
    labels: Dict[str, str],
    prag: Optional[dict],
    scores: Dict[str, dict],
) -> Tuple[bool, List[str]]:
    """P2（无 gold）：仅在有语用信号或证据不专属时 Deep；纯 2:1 高分歧 → Fast 加权融合。

    绝不使用 test gold 决定是否讨论。
    """
    reasons = []
    unanimous = len(set(labels.values())) == 1

    reasons.extend(_prag_signals(prag))

    if unanimous:
        enter = bool(reasons)
        return enter, reasons

    # 有分歧
    if has_low_specificity(scores):
        reasons.append("low_label_specificity")

    if reasons:
        reasons.insert(0, "label_disagreement")
        return True, reasons

    # 纯 label 分歧且 EAA specificity 均不低 → Fast（不加 label_disagreement 到 reasons）
    return False, []


def apply_prag_adjust(base: Dict[str, float], prag: Optional[dict]) -> Dict[str, float]:
    b = dict(base)
    if not prag:
        return b
    if float(prag.get("sarcasm_likelihood", 0)) >= TAU_S and prag.get("mismatch"):
        b["SVA"] *= PRAG_SVA_FACTOR
        b["CIA"] *= PRAG_CIA_FACTOR
        b["LRA"] *= PRAG_LRA_FACTOR
    total = sum(b.values())
    if total > 0:
        b = {k: v / total for k, v in b.items()}
    return b


def ordinal_to_numeric(ordinal: str) -> float:
    return ORDINAL_MAP.get(str(ordinal).lower().strip(), 0.6)


def discussion_factors_for_fusion(
    discussion: Dict[str, dict],
    labels0: Dict[str, str],
) -> Dict[str, float]:
    """P1：融合不对 discussion ordinal 降权；讨论只改 label，d 恒为 1.0。

    ordinal 仍保留在 discussion 输出中供日志分析。
    """
    return {n: 1.0 for n in AGENT_NAMES}


def minority_agents(labels: Dict[str, str]) -> List[str]:
    """2:1 时返回少数派 Agent 名（3:0 或 1:1:1 返回空）。"""
    counts = Counter(labels.values())
    if len(counts) != 2 or max(counts.values()) != 2:
        return []
    minority_label = min(counts, key=lambda k: counts[k])
    return [n for n, l in labels.items() if l == minority_label]


def minority_label_from_agents(labels: Dict[str, str]) -> str:
    """2:1 讨论标签时返回少数派标签；无少数派时回退 CIA。"""
    mins = minority_agents(labels)
    if not mins:
        return labels.get("CIA", "neutral")
    if len(mins) == 1:
        return labels[mins[0]]
    for name in ("CIA", "SVA", "LRA"):
        if name in mins:
            return labels[name]
    return labels[mins[0]]


def vote_pattern(labels: Dict[str, str]) -> str:
    """Phase-0 投票形态：unanimous / 2:1 / 1:1:1 / other。"""
    counts = Counter(labels.values())
    if len(counts) == 1:
        return "unanimous"
    if len(counts) == 2 and max(counts.values()) == 2:
        return "2:1"
    if len(counts) == 3:
        return "1:1:1"
    return "other"


def _evidence_quality(scores: dict) -> float:
    return float(scores.get("evidence_grounded", 0.0)) * float(scores.get("label_specificity", 0.0))


def _agent_raw_weight(name: str, labels: Dict[str, str], scores: Dict[str, dict], base: Dict[str, float]) -> float:
    sc = scores.get(name, {})
    g = float(sc.get("evidence_grounded", 1.0))
    if g < TAU_G:
        return 0.0
    s = float(sc.get("label_specificity", 1.0))
    c = float(sc.get("context_aligned", 1.0))
    return base[name] * g * s * c


def build_two_option_discussion_block(
    labels: Dict[str, str],
    phase0: Dict[str, dict],
    scores: Dict[str, dict],
    base: Dict[str, float],
) -> Tuple[str, dict]:
    """2:1 讨论：每个候选 label 只展示 1 条最强 evidence（按 g×s 选代表），全员相同、不暴露 self/headcount。"""
    label_totals: Dict[str, float] = {}
    for name in AGENT_NAMES:
        L = labels[name]
        label_totals[L] = label_totals.get(L, 0.0) + _agent_raw_weight(name, labels, scores, base)

    options = []
    for label in sorted(set(labels.values())):
        agents_for_label = [n for n in AGENT_NAMES if labels[n] == label]
        best_agent = max(agents_for_label, key=lambda n: _evidence_quality(scores.get(n, {})))
        p0 = phase0[best_agent]
        options.append(
            {
                "label": label,
                "evidence": p0.get("evidence_span", ""),
                "representative_agent": best_agent,
                "evidence_quality": round(_evidence_quality(scores.get(best_agent, {})), 4),
                "label_fusion_total": round(label_totals.get(label, 0.0), 6),
            }
        )

    options.sort(key=lambda o: (-o["label_fusion_total"], o["label"]))

    lines = [f'- Option {o["label"]}: evidence="{o["evidence"]}"' for o in options]
    return "\n".join(lines), {"options": options, "label_fusion_totals": label_totals}


def discussion_mode_for_vote(vote: str) -> str:
    """2:1 → blind label-centric；其余（含 1:1:1、unanimous+Prag）→ 原 agent-centric。"""
    return "2:1_blind" if vote == "2:1" else "split"


def _has_arousal_cues(utterance: str) -> bool:
    """P2-2：utterance 含 !、? 或全大写词（≥2 字母）。"""
    if "!" in utterance or "?" in utterance:
        return True
    for word in utterance.split():
        if len(word) >= 2 and word.isalpha() and word.isupper():
            return True
    return False


def _apply_p2_arousal_filter(
    totals: Dict[str, float],
    labels: Dict[str, str],
    utterance: str,
    strategy: str,
    enabled: bool,
) -> Dict[str, float]:
    """v3.6.1 通用 P2：高唤醒 utterance + Agent 分歧 → 抑制少数派标签。

    规则（不依赖 strategy、不依赖固定标签集）：
    1. utterance 含高唤醒线索（!/?/全大写词）才激活
    2. Phase-0 三 Agent unanimous → 不抑制（共识强，即使高唤醒）
    3. 2:1 或 1:1:1 分歧 → 抑制得票最低的标签（少数派）
    4. 若抑制后 totals 为空 → 回退原 totals

    strategy 参数保留以兼容签名，但不再使用。
    """
    if not enabled or not totals:
        return totals
    if not _has_arousal_cues(utterance):
        return totals
    counts = Counter(labels.values())
    if len(counts) == 1:
        return totals  # unanimous
    minority_count = min(counts.values())
    minority_labels = {l for l, c in counts.items() if c == minority_count}
    filtered = {k: v for k, v in totals.items() if k not in minority_labels}
    return filtered if filtered else totals


def fusion_margin_from_totals(totals: Dict[str, float]) -> float:
    """top1 − top2 融合分数差；仅一个候选时返回 top1 分数。"""
    if not totals:
        return 0.0
    sorted_scores = sorted(totals.values(), reverse=True)
    if len(sorted_scores) == 1:
        return float(sorted_scores[0])
    return float(sorted_scores[0] - sorted_scores[1])


def fuse_labels(
    labels: Dict[str, str],
    scores: Dict[str, dict],
    base: Dict[str, float],
    discussion_factors: Optional[Dict[str, float]] = None,
    use_ground_gate: bool = True,
    tau_g: Optional[float] = None,
    p2_arousal: Optional[bool] = None,
    utterance: str = "",
    strategy: str = "",
) -> Tuple[str, dict]:
    """确定性 argmax 融合；返回 (final_label, fusion_detail)。"""
    discussion_factors = discussion_factors or {n: 1.0 for n in AGENT_NAMES}
    gate_tau = TAU_G if tau_g is None else tau_g
    arousal_on = P2_AROUSAL_GATE if p2_arousal is None else p2_arousal
    raw = {}
    contributions = {}
    for name in AGENT_NAMES:
        sc = scores.get(name, {})
        g = float(sc.get("evidence_grounded", 1.0))
        s = float(sc.get("label_specificity", 1.0))
        c = float(sc.get("context_aligned", 1.0))
        d = float(discussion_factors.get(name, 1.0))
        if use_ground_gate and g < gate_tau:
            raw[name] = 0.0
        else:
            raw[name] = base[name] * g * s * c * d
        contributions[name] = {"label": labels[name], "raw_weight": raw[name], "g": g, "s": s, "c": c, "d": d}

    totals_raw: Dict[str, float] = {}
    for name in AGENT_NAMES:
        L = labels[name]
        totals_raw[L] = totals_raw.get(L, 0.0) + raw[name]

    totals_before = dict(totals_raw)
    totals = _apply_p2_arousal_filter(totals_raw, labels, utterance, strategy, arousal_on)
    p2_applied = totals != totals_before

    if not totals or max(totals.values()) == 0:
        final = labels.get("CIA") or labels.get("SVA") or "neutral"
        return final, {
            "raw": raw,
            "contributions": contributions,
            "totals": totals,
            "tie_break": "fallback",
            "p2_arousal_applied": p2_applied,
            "tau_g": gate_tau,
        }

    max_score = max(totals.values())
    winners = [L for L, v in totals.items() if v == max_score]
    detail = {
        "raw": raw,
        "contributions": contributions,
        "totals": totals,
        "p2_arousal_applied": p2_applied,
        "tau_g": gate_tau,
    }
    if len(winners) == 1:
        return winners[0], {**detail, "tie_break": None}

    for name in sorted(AGENT_NAMES, key=lambda n: -base[n]):
        if labels[name] in winners:
            return labels[name], {**detail, "tie_break": f"base_priority_{name}"}

    return winners[0], {**detail, "tie_break": "first_winner"}
