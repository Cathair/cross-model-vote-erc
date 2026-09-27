"""MPAR routing, discussion candidates, and deterministic fusion."""
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
    return sorted(set(labels.values()))


def infer_soft_confusion_hint(strategy: str, labels: Dict[str, str]) -> Optional[dict]:
    return None


def has_low_specificity(scores: Dict[str, dict], tau: float = TAU_SSPEC) -> bool:
    for name in AGENT_NAMES:
        sc = scores.get(name, {})
        if float(sc.get("label_specificity", 1.0)) < tau:
            return True
    return False


def should_skip_eaa_prag(labels: Dict[str, str]) -> bool:
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
    reasons = []
    unanimous = len(set(labels.values())) == 1

    reasons.extend(_prag_signals(prag))

    if unanimous:
        enter = bool(reasons)
        return enter, reasons

    if has_low_specificity(scores):
        reasons.append("low_label_specificity")

    if reasons:
        reasons.insert(0, "label_disagreement")
        return True, reasons

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
    return {n: 1.0 for n in AGENT_NAMES}


def minority_agents(labels: Dict[str, str]) -> List[str]:
    counts = Counter(labels.values())
    if len(counts) != 2 or max(counts.values()) != 2:
        return []
    minority_label = min(counts, key=lambda k: counts[k])
    return [n for n, l in labels.items() if l == minority_label]


def minority_label_from_agents(labels: Dict[str, str]) -> str:
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
    return "2:1_blind" if vote == "2:1" else "split"


def _has_arousal_cues(utterance: str) -> bool:
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
    if not enabled or not totals:
        return totals
    if not _has_arousal_cues(utterance):
        return totals
    counts = Counter(labels.values())
    if len(counts) == 1:
        return totals
    minority_count = min(counts.values())
    minority_labels = {l for l, c in counts.items() if c == minority_count}
    filtered = {k: v for k, v in totals.items() if k not in minority_labels}
    return filtered if filtered else totals


def fusion_margin_from_totals(totals: Dict[str, float]) -> float:
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
