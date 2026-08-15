"""MARC v3 分析层：融合置信度、讨论收敛、语用信号等（仅日志/离线分析，不改变预测）."""
from collections import Counter
from typing import Dict, Optional

from .config_v3 import AGENT_NAMES, FUSION_MARGIN_EPSILON


def build_prag_brief(prag: Optional[dict]) -> str:
    """将 EAA-Prag JSON 压缩为讨论 prompt 中的一段可读摘要."""
    if not prag or prag.get("skipped"):
        return "No pragmatics signal (Call-2 skipped or unavailable)."
    return (
        f"sarcasm_likelihood={prag.get('sarcasm_likelihood', 0)}, "
        f"surface={prag.get('surface_sentiment')}, "
        f"intended={prag.get('intended_sentiment')}, "
        f"mismatch={prag.get('mismatch')}, "
        f"cues={prag.get('cue_types', [])}"
    )


def compute_fusion_analysis(
    fusion_detail: dict,
    labels: Dict[str, str],
    discussion_factors: Optional[Dict[str, float]] = None,
    discussion_outputs: Optional[Dict[str, dict]] = None,
    margin_epsilon: float = None,
) -> dict:
    """从融合结果提取可离线切片的不确定性指标."""
    eps = FUSION_MARGIN_EPSILON if margin_epsilon is None else margin_epsilon
    totals = fusion_detail.get("totals") or {}
    sorted_totals = sorted(totals.items(), key=lambda x: -x[1])

    top1_label, top1_score = sorted_totals[0] if sorted_totals else (None, 0.0)
    top2_label, top2_score = (sorted_totals[1] if len(sorted_totals) > 1 else (None, 0.0))
    margin = float(top1_score - top2_score) if sorted_totals else 0.0

    tie_break = fusion_detail.get("tie_break")
    max_total = max(totals.values()) if totals else 0.0
    all_raw_zero = max_total == 0.0

    analysis = {
        "fusion_top1": {"label": top1_label, "score": round(top1_score, 6)},
        "fusion_top2": {"label": top2_label, "score": round(top2_score, 6)} if top2_label else None,
        "fusion_margin": round(margin, 6),
        "low_margin": bool(sorted_totals and len(sorted_totals) > 1 and margin < eps),
        "margin_epsilon": eps,
        "tie_break": tie_break,
        "all_raw_zero": all_raw_zero,
        "label_vote": dict(Counter(labels.values())),
        "label_unanimous": len(set(labels.values())) == 1,
    }

    if discussion_factors:
        analysis["discussion_factors"] = discussion_factors

    if discussion_outputs:
        ordinals = {
            n: str(discussion_outputs[n].get("confidence_ordinal", "medium")).lower()
            for n in AGENT_NAMES
        }
        analysis["confidence_ordinals"] = ordinals
        analysis["all_low_confidence"] = all(o == "low" for o in ordinals.values())
        analysis["any_revised"] = any(
            bool(discussion_outputs[n].get("revised_from_initial")) for n in AGENT_NAMES
        )

    flags = []
    if analysis["low_margin"]:
        flags.append("low_fusion_margin")
    if analysis.get("all_low_confidence"):
        flags.append("all_agents_low_confidence")
    if tie_break:
        flags.append(f"tie_break:{tie_break}")
    if all_raw_zero:
        flags.append("all_evidence_gated")
    if not analysis["label_unanimous"] and analysis.get("all_low_confidence"):
        flags.append("split_vote_low_confidence")
    analysis["uncertainty_flags"] = flags
    analysis["is_uncertain"] = len(flags) > 0

    return analysis
