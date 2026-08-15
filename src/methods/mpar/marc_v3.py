"""MARC ERC v3 主流程."""
from typing import Dict, List, Optional, Tuple

from config import get_label_list
from utils import format_context, validate_label

from .agents_v3 import ClassifierAgentV3, EAACaller, build_dynamic_constraints
from .config_v3 import (
    ARCHITECTURE_VERSION,
    ARBITRATOR_AGENT,
    BASE_ROLE_WEIGHT,
    AGENT_NAMES,
    CIA_FALLBACK,
    CIA_FALLBACK_MARGIN,
    DEEP_MODE,
    FAST_FIRST,
    FAST_ONLY,
    NO_EAA_PRAG,
    SKIP_DEEP,
    TAU_MARGIN,
)
from .fusion import (
    apply_prag_adjust,
    build_two_option_discussion_block,
    discussion_mode_for_vote,
    dynamic_discussion_allowed,
    discussion_factors_for_fusion,
    fuse_labels,
    fusion_margin_from_totals,
    infer_soft_confusion_hint,
    labels_from_phase0,
    minority_label_from_agents,
    should_enter_discussion,
    should_skip_eaa_prag,
    vote_pattern,
)
from .analysis_v3 import build_prag_brief, compute_fusion_analysis
from .utils_v3 import extract_emotion


class MARC_ERC_V3:
    """异构三 Agent + EAA 双调用 + 动态 evidence 讨论 + 规则融合。"""

    def __init__(self, strategy: str):
        self.strategy = strategy
        self.label_list = get_label_list(strategy)
        self.agents = {n: ClassifierAgentV3(n, self.label_list) for n in AGENT_NAMES}
        self.eaa = EAACaller()

    def predict(self, utterance: str, speaker: str, context_records: List[dict]) -> Tuple[str, dict]:
        context = format_context(context_records)
        api_calls = 0
        meta: dict = {
            "architecture": "v3",
            "architecture_version": ARCHITECTURE_VERSION,
            "strategy": self.strategy,
            "phases": {},
        }

        # Phase 0
        phase0 = {}
        for name in AGENT_NAMES:
            phase0[name] = self.agents[name].classify(utterance, speaker, context)
            api_calls += 1
        labels0 = labels_from_phase0(phase0)
        meta["phases"]["phase0"] = {"outputs": phase0, "labels": labels0}

        soft_hint = infer_soft_confusion_hint(self.strategy, labels0)
        meta["soft_confusion_hint"] = soft_hint

        # Phase 0.5 — EAA
        eaa_scores = self.eaa.score(context, speaker, utterance, phase0)
        api_calls += 1
        meta["phases"]["eaa_score"] = eaa_scores

        norm_scores = _normalize_scores(eaa_scores)

        skip_prag = should_skip_eaa_prag(labels0) or NO_EAA_PRAG
        if skip_prag:
            eaa_prag = None
            if NO_EAA_PRAG:
                prag_reason = "disabled_no_eaa_prag"
            elif should_skip_eaa_prag(labels0):
                prag_reason = "unanimous_agreement"
            else:
                prag_reason = "skipped"
            meta["phases"]["eaa_prag"] = {"skipped": True, "reason": prag_reason}
        else:
            eaa_prag = self.eaa.pragmatics(context, speaker, utterance)
            api_calls += 1
            meta["phases"]["eaa_prag"] = eaa_prag

        meta["prag_brief"] = build_prag_brief(eaa_prag if not skip_prag else None)

        # Phase 1 — 路由（无 gold）
        if FAST_ONLY:
            enter_disc = False
            route_reasons = ["fast_only"]
            meta["fast_only"] = True
            meta["fast_first"] = False
        elif FAST_FIRST:
            enter_disc, route_reasons = should_enter_discussion(labels0, eaa_prag, norm_scores)
            meta["fast_only"] = False
            meta["fast_first"] = True
        else:
            enter_disc, route_reasons = should_enter_discussion(labels0, eaa_prag, norm_scores)
            meta["fast_only"] = False
            meta["fast_first"] = False
        if SKIP_DEEP and enter_disc:
            enter_disc = False
            route_reasons = list(route_reasons) + ["skip_deep"]

        meta["route"] = {"enter_discussion": enter_disc, "reasons": route_reasons, "skip_eaa_prag": skip_prag}

        base = apply_prag_adjust(dict(BASE_ROLE_WEIGHT), eaa_prag)
        meta["base_weights"] = base

        if FAST_FIRST and not FAST_ONLY:
            return self._predict_fast_first(
                utterance,
                speaker,
                context,
                phase0,
                labels0,
                norm_scores,
                base,
                eaa_prag,
                skip_prag,
                enter_disc,
                route_reasons,
                soft_hint,
                meta,
                api_calls,
            )

        if not enter_disc:
            final, fusion_detail = fuse_labels(
                labels0, norm_scores, base, utterance=utterance, strategy=self.strategy
            )
            final, fusion_detail = _maybe_cia_fallback(final, fusion_detail, labels0)
            meta["decision_mode"] = "Fast Decision"
            meta["fusion"] = fusion_detail
            meta["analysis"] = compute_fusion_analysis(fusion_detail, labels0)
            meta["api_calls"] = api_calls
            meta["fallback"] = bool(fusion_detail.get("cia_fallback"))
            return validate_label(final, self.label_list), meta

        # Phase 2 — 动态 evidence 讨论 1 轮
        allowed = dynamic_discussion_allowed(labels0)
        vote = vote_pattern(labels0)
        disc_mode = discussion_mode_for_vote(vote)
        options_block = ""
        options_meta = {}
        if disc_mode == "2:1_blind":
            options_block, options_meta = build_two_option_discussion_block(
                labels0, phase0, norm_scores, base
            )
        constraints = build_dynamic_constraints(
            allowed,
            meta["prag_brief"],
            soft_hint,
            discussion_mode=disc_mode,
        )
        meta["phases"]["discussion_context"] = {
            "allowed": allowed,
            "prag_brief": meta["prag_brief"],
            "soft_confusion_hint": soft_hint,
            "vote_pattern": vote,
            "discussion_mode": disc_mode,
            "options_block": options_block if disc_mode == "2:1_blind" else None,
            "options_meta": options_meta if disc_mode == "2:1_blind" else None,
        }

        discussion = {}
        others = phase0
        for name in AGENT_NAMES:
            discussion[name] = self.agents[name].disambiguate(
                utterance,
                speaker,
                context,
                phase0[name],
                others,
                constraints,
                discussion_mode=disc_mode,
                options_block=options_block,
                fallback_label=labels0[name],
            )
            api_calls += 1

        labels_d = {}
        for n in AGENT_NAMES:
            raw_c = discussion[n].get("choice", labels0[n])
            c = validate_label(str(raw_c), self.label_list)
            if c not in allowed:
                c = labels0[n]
            labels_d[n] = c
        factors = discussion_factors_for_fusion(discussion, labels0)
        meta["phases"]["discussion"] = {
            "outputs": discussion,
            "labels": labels_d,
            "factors": factors,
            "ordinal_logged": {n: discussion[n].get("confidence_ordinal", "medium") for n in AGENT_NAMES},
            "allowed": allowed,
            "vote_pattern": vote,
            "discussion_mode": disc_mode,
        }

        final_fused, fusion_detail = fuse_labels(
            labels_d, norm_scores, base, factors, utterance=utterance, strategy=self.strategy
        )
        final, fusion_detail = _resolve_deep_final(
            labels_d, labels0, final_fused, fusion_detail, self.label_list
        )
        meta["decision_mode"] = "Deep Disambiguation"
        meta["fusion"] = fusion_detail
        meta["analysis"] = compute_fusion_analysis(fusion_detail, labels_d, factors, discussion)
        meta["api_calls"] = api_calls
        meta["fallback"] = bool(
            fusion_detail.get("cia_fallback") or fusion_detail.get("deep_override")
        )
        return validate_label(final, self.label_list), meta

    def _predict_fast_first(
        self,
        utterance: str,
        speaker: str,
        context: str,
        phase0: dict,
        labels0: Dict[str, str],
        norm_scores: dict,
        base: dict,
        eaa_prag: Optional[dict],
        skip_prag: bool,
        enter_disc: bool,
        route_reasons: List[str],
        soft_hint: Optional[dict],
        meta: dict,
        api_calls: int,
    ) -> Tuple[str, dict]:
        """两阶段：Stage1 Fast 融合 + margin 门控；Stage2 单 Agent 仲裁。"""
        fast_pred, fast_detail = fuse_labels(
            labels0, norm_scores, base, utterance=utterance, strategy=self.strategy
        )
        margin = fusion_margin_from_totals(fast_detail.get("totals") or {})
        meta["fast_first"] = {
            "stage1_pred": fast_pred,
            "fusion_margin": round(margin, 6),
            "tau_margin": TAU_MARGIN,
            "enter_disc_original": enter_disc,
        }

        if not enter_disc or margin >= TAU_MARGIN:
            reason = "fast_first_high_margin" if enter_disc else "fast_route"
            meta["route"]["reasons"] = list(route_reasons) + [reason]
            fast_pred, fast_detail = _maybe_cia_fallback(fast_pred, fast_detail, labels0)
            meta["decision_mode"] = "Fast Decision"
            meta["fusion"] = fast_detail
            meta["analysis"] = compute_fusion_analysis(fast_detail, labels0)
            meta["api_calls"] = api_calls
            meta["fallback"] = bool(fast_detail.get("cia_fallback"))
            return validate_label(fast_pred, self.label_list), meta

        # Stage 2 — 单 Agent 仲裁（复用 Deep 讨论 prompt / cache）
        arb = ARBITRATOR_AGENT if ARBITRATOR_AGENT in AGENT_NAMES else "CIA"
        allowed = dynamic_discussion_allowed(labels0)
        vote = vote_pattern(labels0)
        disc_mode = discussion_mode_for_vote(vote)
        options_block = ""
        options_meta = {}
        if disc_mode == "2:1_blind":
            options_block, options_meta = build_two_option_discussion_block(
                labels0, phase0, norm_scores, base
            )
        constraints = build_dynamic_constraints(
            allowed,
            meta["prag_brief"],
            soft_hint,
            discussion_mode=disc_mode,
        )
        meta["phases"]["arbitration_context"] = {
            "arbitrator": arb,
            "allowed": allowed,
            "vote_pattern": vote,
            "discussion_mode": disc_mode,
            "options_block": options_block if disc_mode == "2:1_blind" else None,
            "options_meta": options_meta if disc_mode == "2:1_blind" else None,
        }

        arb_out = self.agents[arb].disambiguate(
            utterance,
            speaker,
            context,
            phase0[arb],
            phase0,
            constraints,
            discussion_mode=disc_mode,
            options_block=options_block,
            fallback_label=labels0[arb],
        )
        api_calls += 1
        final = validate_label(str(arb_out.get("choice", labels0[arb])), self.label_list)
        if final not in allowed:
            final = labels0[arb]

        meta["phases"]["arbitration"] = {
            "arbitrator": arb,
            "output": arb_out,
            "choice": final,
            "revised_from_initial": arb_out.get("revised_from_initial", False),
        }
        meta["route"]["reasons"] = list(route_reasons) + ["fast_first_low_margin_arbitration"]
        meta["decision_mode"] = "Single Arbitration"
        meta["fusion"] = {
            **fast_detail,
            "arbitrator": arb,
            "arbitrator_choice": final,
            "stage1_pred": fast_pred,
            "stage1_margin": margin,
        }
        meta["analysis"] = compute_fusion_analysis(fast_detail, labels0)
        meta["analysis"]["arbitrator"] = arb
        meta["analysis"]["arbitrator_revised"] = bool(arb_out.get("revised_from_initial"))
        meta["api_calls"] = api_calls
        meta["fallback"] = False
        return validate_label(final, self.label_list), meta


def _resolve_deep_final(
    labels_d: Dict[str, str],
    labels0: Dict[str, str],
    final_fused: str,
    fusion_detail: dict,
    label_list: List[str],
) -> Tuple[str, dict]:
    """Deep 路径统一决策（IEM + MELD 相同规则，由 MARC_DEEP_MODE 控制）。"""
    detail = {**fusion_detail, "fused_pred": final_fused, "deep_mode": DEEP_MODE}
    post_vote = vote_pattern(labels_d)

    if DEEP_MODE == "post21_minority" and post_vote == "2:1":
        raw = minority_label_from_agents(labels_d)
        final = validate_label(raw, label_list)
        detail["deep_override"] = "post21_minority"
        return final, detail
    if DEEP_MODE == "post21_cia" and post_vote == "2:1":
        final = validate_label(labels_d.get("CIA", labels0["CIA"]), label_list)
        detail["deep_override"] = "post21_cia"
        return final, detail

    final, detail = _maybe_cia_fallback(final_fused, detail, labels_d)
    return final, detail


def _maybe_cia_fallback(final: str, fusion_detail: dict, labels0: Dict[str, str]) -> Tuple[str, dict]:
    """低 margin 时回退 CIA（qwen）Phase-0 标签。"""
    if not CIA_FALLBACK:
        return final, fusion_detail
    margin = fusion_margin_from_totals(fusion_detail.get("totals") or {})
    if margin >= CIA_FALLBACK_MARGIN:
        return final, fusion_detail
    cia_label = labels0.get("CIA", final)
    if cia_label == final:
        return final, fusion_detail
    detail = dict(fusion_detail)
    detail["cia_fallback"] = True
    detail["cia_fallback_margin"] = round(margin, 6)
    detail["cia_fallback_from"] = final
    return cia_label, detail


def _normalize_scores(raw: dict) -> dict:
    out = {}
    for name in AGENT_NAMES:
        block = raw.get(name, raw.get(name.lower(), {}))
        if not isinstance(block, dict):
            block = {}
        amb = block.get("ambiguous_between")
        out[name] = {
            "evidence_grounded": float(block.get("evidence_grounded", 0.8)),
            "label_specificity": float(block.get("label_specificity", 0.8)),
            "context_aligned": float(block.get("context_aligned", 0.8)),
            "ambiguous_between": amb,
        }
    return out
