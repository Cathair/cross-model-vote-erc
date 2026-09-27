"""MPAR full test-set run with checkpoints. Requires OPENAI_API_KEY and MARC_CACHE_DIR."""
import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from collections import Counter

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
DATA_ROOT = os.path.join(REPO_ROOT, "data")

from config import DATASET_DEFAULT_STRATEGY
from data_loader import load_dataset, print_label_distribution
from experiment_retry import call_sample_with_retry, SAMPLE_MAX_RETRIES, SAMPLE_TIMEOUT_SEC
from run_experiment import compute_metrics, _log, _fmt_seconds, _save_checkpoint, _find_latest_checkpoint

from methods.mpar.marc_v3 import MARC_ERC_V3
from methods.mpar.config_v3 import ARCHITECTURE_VERSION, AGENT_MODELS, EAA_MODEL, BASE_ROLE_WEIGHT
from methods.mpar.run_v3 import _build_log_entry, behavior_summary

OUT_DIR = os.environ.get("MARC_V37_OUT_DIR", os.path.join(REPO_ROOT, "results", "mpar", "run1"))
CHECKPOINT_EVERY = 100


def run_dataset(dataset: str, resume: bool = False, force: bool = False):
    strategy = DATASET_DEFAULT_STRATEGY[dataset]
    all_samples = load_dataset(dataset, DATA_ROOT, "test", strategy)
    n_total = len(all_samples)
    tag = f"mpar_{dataset}_full"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{dataset}_full_n{n_total}.json")

    if os.path.exists(out_path) and not force and not resume:
        _log(f"SKIP (exists): {out_path}  (use --force or --resume)")
        return

    # Resume from checkpoint
    y_true, y_pred, logs, errors = [], [], [], []
    start_i = 0
    if resume:
        ckpt = _find_latest_checkpoint(OUT_DIR, tag)
        if ckpt:
            _log(f"[resume] loading checkpoint: {ckpt}")
            data = json.load(open(ckpt))
            y_true = data.get("y_true", [])
            y_pred = data.get("y_pred", [])
            logs = data.get("logs", [])
            errors = data.get("errors", [])
            start_i = data.get("done", 0)
            _log(f"[resume] resuming from index {start_i} ({len(y_pred)}/{n_total} done)")

    if not resume and os.path.exists(out_path) and force:
        os.remove(out_path)

    model = MARC_ERC_V3(strategy=strategy)
    _log("=" * 70)
    _log(f"MPAR | dataset={dataset} | strategy={strategy} | n={n_total} | start={start_i}")
    _log(f"Agents: {AGENT_MODELS}")
    _log(f"EAA: {EAA_MODEL} | base_prior: {BASE_ROLE_WEIGHT}")
    _log(f"Retry: sample_max={SAMPLE_MAX_RETRIES} sample_timeout={SAMPLE_TIMEOUT_SEC:.0f}s "
         f"api_timeout={os.environ.get('MARC_API_TIMEOUT', '180')}s api_retries=3")
    _log(f"Checkpoint: every {CHECKPOINT_EVERY} samples")
    print_label_distribution(all_samples[start_i:])

    start = time.time()
    for i, s in enumerate(all_samples[start_i:], start=start_i):
        elapsed = time.time() - start
        done = i - start_i + 1
        avg = elapsed / done if done > 0 else 0
        eta = _fmt_seconds(avg * (n_total - i - 1)) if done > 1 else "?"
        _log(
            f"[{i+1}/{n_total}] ({(i+1)/n_total*100:.1f}%) "
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
            _log(f"  [ERROR] sample {i+1} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            pred, meta, fb = "neutral", {"decision_mode": "Error", "fallback": True}, True
            errors.append({"index": i, "error": str(e), "type": type(e).__name__})

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
            _save_checkpoint(OUT_DIR, tag, i + 1, y_true, y_pred, logs, errors)
            _log(f"  [checkpoint] saved at {i+1}/{n_total}")

    metrics = compute_metrics(y_true, y_pred)
    total = time.time() - start
    behavior = behavior_summary(logs)

    result = {
        "method": "mpar",
        "architecture": "mpar",
        "architecture_version": ARCHITECTURE_VERSION,
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "logs": logs,
        "errors": errors,
        "elapsed_sec": total,
        "behavior": behavior,
        "dataset": dataset,
        "strategy": strategy,
        "n_samples": n_total,
        "model_config": {
            "agents": AGENT_MODELS,
            "eaa_model": EAA_MODEL,
            "base_role_weight": BASE_ROLE_WEIGHT,
            "deep_mode": os.environ.get("MARC_DEEP_MODE", "post21_cia"),
            "cia_fallback": os.environ.get("MARC_CIA_FALLBACK", "1"),
            "cia_fallback_margin": os.environ.get("MARC_CIA_FALLBACK_MARGIN", "0.05"),
        },
        "tag": tag,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    _log(f"Saved: {out_path}")
    _log(f"  WF1={metrics['weighted_f1']:.4f} Acc={metrics['accuracy']:.4f} errors={len(errors)}")
    _log(f"  Fast={behavior.get('fast_decision_rate',0)*100:.1f}% Deep={behavior.get('deep_disambiguation_rate',0)*100:.1f}% avg_api={behavior.get('avg_api_calls',0):.1f}")

    # Clean up checkpoints after successful completion
    import glob
    for ckpt in glob.glob(os.path.join(OUT_DIR, f"checkpoint_{tag}_at_*.json")):
        os.remove(ckpt)
        _log(f"  [cleanup] removed {os.path.basename(ckpt)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="MPAR full test set 1-run")
    parser.add_argument("--dataset", choices=["iemocap", "meld", "both"], default="both")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    args = parser.parse_args()

    datasets = ["iemocap", "meld"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        _log(f"\n{'='*70}")
        _log(f"Starting MPAR full test set: {ds}")
        _log(f"{'='*70}")
        run_dataset(ds, resume=args.resume, force=args.force)

    _log(f"\nDONE MPAR full test set: {args.dataset}")


if __name__ == "__main__":
    main()
