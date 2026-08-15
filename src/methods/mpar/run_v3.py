"""MARC v3.6.1 实验入口.

用法:
  python -m marc_v3_6_1.run_v3 --dataset iemocap
  python -m marc_v3_6_1.run_v3 --dataset meld --force
  python -m marc_v3_6_1.run_v3 --dry-run --limit 2
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from config import DATASET_DEFAULT_STRATEGY
from data_loader import load_dataset, print_label_distribution
from experiment_retry import (
    call_sample_with_retry,
    SAMPLE_MAX_RETRIES,
    SAMPLE_TIMEOUT_SEC,
)
from run_experiment import (
    load_sample_indices,
    subset_samples,
    compute_metrics,
    _log,
    _fmt_seconds,
    _save_checkpoint,
    _find_latest_checkpoint,
)

from .marc_v3 import MARC_ERC_V3
from .config_v3 import ARCHITECTURE_VERSION, AGENT_MODELS, EAA_MODEL, BASE_ROLE_WEIGHT

OUT_DIR = "results/marc_v3_6_1"
# 0 = 不写 checkpoint（仅最终 JSON 落盘，避免 results/marc_v3 文件过多）
CHECKPOINT_EVERY = 0


def behavior_summary(logs: list) -> dict:
    n = len(logs)
    if n == 0:
        return {"n": 0}
    modes = dict(Counter(x.get("decision_mode", "?") for x in logs))
    fast = modes.get("Fast Decision", 0)
    deep = modes.get("Deep Disambiguation", 0)
    arb = modes.get("Single Arbitration", 0)
    skip_prag = sum(1 for x in logs if x.get("eaa_prag_skipped"))
    sarcasm_route = sum(
        1
        for x in logs
        if "sarcasm_high" in x.get("route_reasons", [])
        or "prag_mismatch" in x.get("route_reasons", [])
    )
    confusion_hit = sum(1 for x in logs if x.get("soft_confusion_hint") or x.get("confusion_pair"))
    api_calls = [x.get("api_calls") or 0 for x in logs]
    uncertain = sum(1 for x in logs if (x.get("analysis") or {}).get("is_uncertain"))
    low_margin = sum(1 for x in logs if (x.get("analysis") or {}).get("low_margin"))
    tie_break = sum(1 for x in logs if (x.get("analysis") or {}).get("tie_break"))
    retry_samples = sum(1 for x in logs if x.get("retry_attempts", 0) > 0)

    def _acc(subset):
        if not subset:
            return None
        return sum(1 for x in subset if x["true_label"] == x["pred_label"]) / len(subset)

    fast_logs = [x for x in logs if x.get("decision_mode") == "Fast Decision"]
    deep_logs = [x for x in logs if x.get("decision_mode") == "Deep Disambiguation"]
    arb_logs = [x for x in logs if x.get("decision_mode") == "Single Arbitration"]
    uncertain_logs = [x for x in logs if (x.get("analysis") or {}).get("is_uncertain")]

    return {
        "n": n,
        "fast_decision_rate": fast / n,
        "deep_disambiguation_rate": deep / n,
        "single_arbitration_rate": arb / n,
        "eaa_prag_skip_rate": skip_prag / n,
        "sarcasm_route_rate": sarcasm_route / n,
        "confusion_pair_rate": confusion_hit / n,
        "avg_api_calls": sum(api_calls) / n if api_calls else 0,
        "retry_sample_rate": retry_samples / n,
        "fast_acc": _acc(fast_logs),
        "deep_acc": _acc(deep_logs),
        "arbitration_acc": _acc(arb_logs),
        "uncertain_rate": uncertain / n,
        "low_margin_rate": low_margin / n,
        "tie_break_rate": tie_break / n,
        "uncertain_acc": _acc(uncertain_logs),
        "decision_modes": modes,
    }


def _build_log_entry(i, s, pred, meta, fb, sample_sec, attempts_used):
    route = meta.get("route", {})
    return {
        "sample_index": i,
        "dialogue_id": s.dialogue_id,
        "utterance_id": s.utterance_id,
        "utterance": s.utterance,
        "true_label": s.label,
        "pred_label": pred,
        "fallback": fb,
        "decision_mode": meta.get("decision_mode"),
        "route_reasons": route.get("reasons", []),
        "enter_discussion": route.get("enter_discussion"),
        "eaa_prag_skipped": route.get("skip_eaa_prag"),
        "soft_confusion_hint": meta.get("soft_confusion_hint"),
        "confusion_pair": meta.get("soft_confusion_hint"),  # 兼容 v3.0 字段名
        "base_weights": meta.get("base_weights"),
        "fusion": meta.get("fusion"),
        "api_calls": meta.get("api_calls"),
        "phase0_labels": meta.get("phases", {}).get("phase0", {}).get("labels"),
        "eaa_score": meta.get("phases", {}).get("eaa_score"),
        "eaa_prag": meta.get("phases", {}).get("eaa_prag"),
        "discussion_labels": meta.get("phases", {}).get("discussion", {}).get("labels"),
        "prag_brief": meta.get("prag_brief"),
        "analysis": meta.get("analysis"),
        "sample_elapsed_sec": round(sample_sec, 2),
        "retry_attempts": attempts_used,
        "meta_full": meta,
    }


def run_on_samples(
    samples,
    strategy: str,
    tag: str,
    dry_run: bool = False,
    resume: bool = False,
    out_dir: str = OUT_DIR,
    start_index: int = 0,
    prefix_result: dict = None,
) -> dict:
    if dry_run:
        _log(f"DRY RUN {tag} n={len(samples)} — no API")
        return {"metrics": {}, "logs": [], "behavior": {}}

    model = MARC_ERC_V3(strategy=strategy)
    if prefix_result:
        y_true = list(prefix_result.get("y_true") or [])
        y_pred = list(prefix_result.get("y_pred") or [])
        logs = list(prefix_result.get("logs") or [])
        errors = list(prefix_result.get("errors") or [])
        start_i = start_index
        _log(f"[merge-prefix] loaded {len(logs)} prior logs; resume from index {start_i}")
    else:
        y_true, y_pred, logs, errors = [], [], [], []
        start_i = start_index

    if resume:
        _log("[resume] disabled (CHECKPOINT_EVERY=0); use --force for a fresh run")

    start = time.time()
    total_n = len(samples)

    for i, s in enumerate(samples[start_i:], start=start_i):
        elapsed = time.time() - start
        avg = elapsed / (i - start_i + 1) if i > start_i else 0
        eta = _fmt_seconds(avg * (total_n - i - 1)) if i > start_i else "?"
        _log(
            f"[{i+1}/{total_n}] ({(i+1)/total_n*100:.1f}%) "
            f"true={s.label:<10} elapsed={_fmt_seconds(elapsed)} eta={eta} | {s.utterance[:40]}"
        )

        sample_t0 = time.time()
        attempts_used = 0
        try:

            def _predict():
                return model.predict(s.utterance, s.speaker, s.context)

            (pred, meta), attempts_used = call_sample_with_retry(_predict, i, log_fn=_log)
            fb = bool(meta.get("fallback"))
            if attempts_used > 0:
                _log(f"  [RETRY OK] sample {i+1} succeeded on attempt {attempts_used + 1}")
        except Exception as e:
            _log(
                f"  [ERROR] sample {i+1} failed after {SAMPLE_MAX_RETRIES + 1} attempts: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            pred, meta, fb = "neutral", {"decision_mode": "Error", "fallback": True}, True
            errors.append(
                {
                    "index": i,
                    "error": str(e),
                    "type": type(e).__name__,
                    "attempts": SAMPLE_MAX_RETRIES + 1,
                }
            )

        sample_sec = time.time() - sample_t0
        y_true.append(s.label)
        y_pred.append(pred)
        log_entry = _build_log_entry(i, s, pred, meta, fb, sample_sec, attempts_used)
        logs.append(log_entry)

        route = meta.get("route", {})
        _log(
            f"  → pred={pred} mode={meta.get('decision_mode')} "
            f"route={route.get('reasons')} api={meta.get('api_calls')} "
            f"sample_time={_fmt_seconds(sample_sec)}"
        )

        if CHECKPOINT_EVERY > 0 and (i + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(out_dir, tag, i + 1, y_true, y_pred, logs, errors)
            _log(f"  [checkpoint] saved at {i+1} samples")

    metrics = compute_metrics(y_true, y_pred)
    total = time.time() - start
    behavior = behavior_summary(logs)

    return {
        "method": "marc_v3",
        "architecture": "v3",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "logs": logs,
        "errors": errors,
        "elapsed_sec": total,
        "behavior": behavior,
        "retry_config": {
            "max_retries": SAMPLE_MAX_RETRIES,
            "timeout_sec": SAMPLE_TIMEOUT_SEC,
            "interval_sec": 1,
        },
        "architecture_version": ARCHITECTURE_VERSION,
        "model_config": {
            "agents": AGENT_MODELS,
            "eaa_model": EAA_MODEL,
            "base_role_weight": BASE_ROLE_WEIGHT,
        },
        "tag": tag,
    }


def main():
    parser = argparse.ArgumentParser(description="MARC ERC v3 experiments")
    parser.add_argument("--dataset", choices=["iemocap", "meld"], default="iemocap")
    parser.add_argument("--indices", default=None, help="sample indices JSON")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-index", type=int, default=0, help="0-based sample index to start from")
    parser.add_argument("--merge-prefix", default=None, help="JSON with logs/metrics prefix to merge")
    args = parser.parse_args()

    strategy = DATASET_DEFAULT_STRATEGY[args.dataset]
    indices_file = args.indices or f"samples/sample_indices_{args.dataset}.json"

    indices = load_sample_indices(indices_file)
    all_samples = load_dataset(args.dataset, "data", "test", strategy)
    samples = subset_samples(all_samples, indices)
    if args.limit:
        samples = samples[: args.limit]

    prefix_result = None
    if args.merge_prefix:
        prefix_result = json.load(open(args.merge_prefix, encoding="utf-8"))
        _log(f"Loaded merge-prefix: {args.merge_prefix} ({len(prefix_result.get('logs', []))} logs)")

    full_n = len(samples)
    os.makedirs(OUT_DIR, exist_ok=True)
    n_tag = f"n{full_n}"
    out_path = args.out or os.path.join(OUT_DIR, f"{args.dataset}_marc_v3_{n_tag}.json")
    tag = f"v3_{args.dataset}_{n_tag}"

    if os.path.exists(out_path) and not args.force and not args.dry_run and not args.resume:
        _log(f"SKIP (exists): {out_path}  (use --force or --resume)")
        return

    _log("=" * 70)
    run_n = full_n - args.start_index if args.start_index else full_n
    _log(
        f"MARC v3 ({ARCHITECTURE_VERSION}) | dataset={args.dataset} | strategy={strategy} | "
        f"n={full_n} run={run_n} start_index={args.start_index}"
    )
    _log(f"Agents: {AGENT_MODELS}")
    _log(f"EAA: {EAA_MODEL} | base_prior: {BASE_ROLE_WEIGHT}")
    _log(
        f"Retry: max={SAMPLE_MAX_RETRIES} interval=1s timeout={SAMPLE_TIMEOUT_SEC:.0f}s"
        + (f" checkpoint={CHECKPOINT_EVERY}" if CHECKPOINT_EVERY > 0 else " checkpoint=off")
    )
    print_label_distribution(samples if args.start_index == 0 else samples[args.start_index:])

    result = run_on_samples(
        samples,
        strategy,
        tag,
        dry_run=args.dry_run,
        resume=args.resume,
        out_dir=OUT_DIR,
        start_index=args.start_index,
        prefix_result=prefix_result,
    )
    result["dataset"] = args.dataset
    result["strategy"] = strategy
    result["n_samples"] = full_n
    result["indices_file"] = indices_file

    if not args.dry_run:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        _log(f"Saved: {out_path}")
        m = result["metrics"]
        b = result.get("behavior", {})
        _log(f"  WF1={m['weighted_f1']:.4f} Acc={m['accuracy']:.4f} errors={len(result.get('errors', []))}")
        _log(
            f"  Fast={b.get('fast_decision_rate', 0)*100:.1f}% Deep={b.get('deep_disambiguation_rate', 0)*100:.1f}% "
            f"PragSkip={b.get('eaa_prag_skip_rate', 0)*100:.1f}% avg_api={b.get('avg_api_calls', 0):.1f}"
        )
        _log(
            f"  Uncertain={b.get('uncertain_rate', 0)*100:.1f}% LowMargin={b.get('low_margin_rate', 0)*100:.1f}% "
            f"TieBreak={b.get('tie_break_rate', 0)*100:.1f}%"
        )
        summary_path = os.path.join(
            OUT_DIR, f"v3_summary_{args.dataset}_{n_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": m, "behavior": b, "path": out_path, "errors": len(result.get("errors", []))}, f, indent=2)
        _log(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
