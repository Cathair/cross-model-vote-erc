"""Role-prompt single-agent inference on full test sets. Resumes partial JSON; --fresh to restart."""
import argparse
import fcntl
import glob
import json
import os
import re
import signal
import sys
import time
import traceback

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
DATA_ROOT = os.path.join(REPO_ROOT, "data")
import config  # noqa: F401, E402

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from baselines import SA_PROMPT
from data_loader import load_dataset
from experiment_retry import call_sample_with_retry
from run_experiment import _log, _fmt_seconds, compute_metrics
from utils import format_context, parse_json_response, validate_label_with_flag
from config import get_label_list

METHOD = "singleagent"
STRATEGY = {"iemocap": "iemocap_6class", "meld": "meld_raw"}
CHECKPOINT_EVERY = 10

_RUN_STATE: dict = {}


def _save_progress(out_path: str, payload: dict):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, out_path)


def _build_partial_payload(
    *,
    model: str,
    dataset: str,
    strategy: str,
    tag: str,
    run_id: int,
    n: int,
    completed: int,
    y_true: list,
    y_pred: list,
    logs: list,
    errors: list,
    save_cache: bool,
    cache_dir: str | None,
    elapsed_sec: float | None = None,
) -> dict:
    payload = {
        "method": METHOD,
        "model": model,
        "dataset": dataset,
        "strategy": strategy,
        "tag": tag,
        "run": run_id,
        "full_test": True,
        "status": "complete" if completed == n else "in_progress",
        "n_samples": n,
        "completed": completed,
        "y_true": y_true,
        "y_pred": y_pred,
        "logs": logs,
        "errors": errors,
        "cache_dir": cache_dir if save_cache else None,
    }
    if completed == n:
        payload["metrics"] = compute_metrics(y_true, y_pred)
        payload["elapsed_sec"] = elapsed_sec
    return payload


def _find_legacy_checkpoint(out_dir: str, tag: str) -> dict | None:
    pattern = os.path.join(out_dir, f"checkpoint_{tag}_at_*.json")
    files = glob.glob(pattern)
    if not files:
        return None

    def _idx(path: str) -> int:
        m = re.search(r"_at_(\d+)\.json$", path)
        return int(m.group(1)) if m else 0

    latest = max(files, key=_idx)
    try:
        with open(latest, encoding="utf-8") as f:
            data = json.load(f)
        logs = data.get("logs") or []
        if logs:
            _log(f"[resume] found legacy checkpoint {latest} ({len(logs)} samples)")
            return {
                "status": "in_progress",
                "logs": logs,
                "y_true": data.get("y_true") or [],
                "y_pred": data.get("y_pred") or [],
                "errors": data.get("errors") or [],
                "completed": len(logs),
            }
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _load_progress(out_path: str, out_dir: str, tag: str) -> dict | None:
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "complete":
                return None
            logs = data.get("logs") or []
            if logs:
                completed = data.get("completed") or len(logs)
                data["completed"] = completed
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return _find_legacy_checkpoint(out_dir, tag)


def _flush_run_state():
    partial = _RUN_STATE.get("partial")
    out_path = _RUN_STATE.get("out_path")
    if partial and out_path:
        _save_progress(out_path, partial)
        _log(
            f"[checkpoint] flushed {partial['completed']}/{partial['n_samples']} "
            f"-> {out_path}"
        )


def _handle_stop(signum, _frame):
    _log(f"[signal] received {signum}, saving progress...")
    _flush_run_state()
    raise SystemExit(128 + signum)


class _RunLock:
    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self._fd = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.lock_path) or ".", exist_ok=True)
        self._fd = open(self.lock_path, "w", encoding="utf-8")
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                with open(self.lock_path, encoding="utf-8") as f:
                    holder = f.read().strip()
            except OSError:
                holder = "unknown"
            raise SystemExit(
                f"Another instance is already running for this output "
                f"(lock={self.lock_path}, pid={holder}). "
                f"Stop it first or wait for it to finish."
            )
        self._fd.write(str(os.getpid()))
        self._fd.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None


def run_one(
    model: str,
    dataset: str,
    run_id: int,
    save_cache: bool = True,
    resume: bool = True,
    fresh: bool = False,
    checkpoint_every: int = CHECKPOINT_EVERY,
) -> dict:
    strategy = STRATEGY[dataset]
    all_samples = load_dataset(dataset, DATA_ROOT, "test", strategy)
    n = len(all_samples)

    slug = model.replace("/", "_").replace(".", "-")
    run_tag = f"run{run_id}"
    cache_dir = os.path.join(REPO_ROOT, "results", "cache", "sa", slug, run_tag)
    out_dir = os.path.join(REPO_ROOT, "results", "sa", slug, run_tag)
    os.makedirs(out_dir, exist_ok=True)

    if save_cache:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["MARC_CACHE_DIR"] = cache_dir
    else:
        os.environ.pop("MARC_CACHE_DIR", None)

    out_path = os.path.join(out_dir, f"{dataset}_{METHOD}_{slug}_n{n}.json")
    lock_path = out_path + ".lock"
    tag = f"sa_full_{slug}_{run_tag}_{dataset}"

    if fresh and os.path.exists(out_path):
        os.remove(out_path)
        _log(f"[fresh] removed existing output {out_path}")

    start_index = 0
    y_true, y_pred, logs, errors = [], [], [], []

    if resume:
        prev = _load_progress(out_path, out_dir, tag)
        if prev:
            start_index = len(prev.get("logs") or [])
            y_true = list(prev.get("y_true") or [])
            y_pred = list(prev.get("y_pred") or [])
            logs = list(prev.get("logs") or [])
            errors = list(prev.get("errors") or [])
            _log(f"[resume] loaded {start_index}/{n} from progress file")
        else:
            _log("[resume] no partial progress found; starting from 0")

    _log("=" * 70)
    _log(f"[SA full] {run_tag} dataset={dataset} n={n} model={model}")
    _log(f"  cache={'off' if not save_cache else cache_dir}")
    _log(f"  out={out_path} start={start_index + 1} checkpoint_every={checkpoint_every}")

    label_list = get_label_list(strategy)
    llm = ChatOpenAI(model=model, temperature=0.0)
    from lib.cache_llm import maybe_wrap
    if save_cache:
        llm = maybe_wrap(llm, model=model, temperature=0.0)

    template = SA_PROMPT
    t0 = time.time()

    with _RunLock(lock_path):
        for i, s in enumerate(all_samples):
            if i < start_index:
                continue

            elapsed = time.time() - t0
            done_in_run = i - start_index + 1
            avg = elapsed / done_in_run if done_in_run > 0 else 0
            eta = _fmt_seconds(avg * (n - i - 1)) if i > start_index else "?"
            _log(
                f"[{i+1}/{n}] ({(i+1)/n*100:.1f}%) true={s.label:<10} "
                f"elapsed={_fmt_seconds(elapsed)} eta={eta}"
            )

            try:
                def _predict():
                    ctx = format_context(s.context)
                    prompt = PromptTemplate.from_template(template).format(
                        label_list=label_list,
                        context=ctx,
                        speaker=s.speaker,
                        utterance=s.utterance,
                    )
                    resp = llm.invoke(prompt)
                    parsed = parse_json_response(resp.content)
                    raw = parsed.get("emotion", "neutral") if parsed else "neutral"
                    pred, soft_fb = validate_label_with_flag(raw, label_list)
                    return pred, soft_fb

                (pred, fallback), attempts_used = call_sample_with_retry(_predict, i, log_fn=_log)
                if attempts_used > 0:
                    _log(f"  [RETRY OK] sample {i+1} succeeded on attempt {attempts_used + 1}")
            except Exception as e:
                _log(f"  [ERROR] sample {i+1}: {type(e).__name__}: {e}")
                traceback.print_exc()
                pred, fallback = "neutral", True
                errors.append({"index": i, "error": str(e), "type": type(e).__name__})

            y_true.append(s.label)
            y_pred.append(pred)
            logs.append({
                "sample_index": i,
                "dialogue_id": s.dialogue_id,
                "utterance_id": s.utterance_id,
                "utterance": s.utterance,
                "true_label": s.label,
                "pred_label": pred,
                "fallback": fallback,
            })

            completed = i + 1
            partial = _build_partial_payload(
                model=model,
                dataset=dataset,
                strategy=strategy,
                tag=tag,
                run_id=run_id,
                n=n,
                completed=completed,
                y_true=y_true,
                y_pred=y_pred,
                logs=logs,
                errors=errors,
                save_cache=save_cache,
                cache_dir=cache_dir,
                elapsed_sec=time.time() - t0 if completed == n else None,
            )
            _RUN_STATE["out_path"] = out_path
            _RUN_STATE["partial"] = partial

            should_save = (
                completed % checkpoint_every == 0
                or completed == n
                or completed == start_index + 1
            )
            if should_save:
                _save_progress(out_path, partial)
                if completed == n:
                    m = partial["metrics"]
                    _log(
                        f"  DONE WF1={m['weighted_f1']:.4f} Acc={m['accuracy']:.4f} "
                        f"errors={len(errors)}"
                    )
                else:
                    _log(f"  [progress] saved {completed}/{n} -> {out_path}")

    result = json.load(open(out_path, encoding="utf-8"))
    result["output_path"] = out_path
    _RUN_STATE.clear()
    return result


def main():
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model API id")
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--dataset", choices=["iemocap", "meld", "both"], default="both")
    parser.add_argument("--no-save-cache", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="restart from 0")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    args = parser.parse_args()

    datasets = ["iemocap", "meld"] if args.dataset == "both" else [args.dataset]
    save_cache = not args.no_save_cache
    resume = not args.no_resume and not args.fresh

    summary = []
    for ds in datasets:
        slug = args.model.replace("/", "_").replace(".", "-")
        n_samples = 1623 if ds == "iemocap" else 2610
        out_path_check = os.path.join(
            REPO_ROOT, "results", "sa", slug, f"run{args.run}",
            f"{ds}_singleagent_{slug}_n{n_samples}.json",
        )
        if os.path.exists(out_path_check) and not args.fresh:
            try:
                existing = json.load(open(out_path_check, encoding="utf-8"))
                if existing.get("status") == "complete":
                    _log(f"SKIP (complete): {out_path_check}")
                    summary.append({
                        "run": args.run, "dataset": ds, "model": args.model,
                        "n": existing.get("n_samples"),
                        "wf1": existing["metrics"]["weighted_f1"],
                        "acc": existing["metrics"]["accuracy"],
                        "path": out_path_check,
                    })
                    continue
            except Exception:
                pass

        r = run_one(
            args.model,
            ds,
            args.run,
            save_cache=save_cache,
            resume=resume,
            fresh=args.fresh,
            checkpoint_every=max(1, args.checkpoint_every),
        )
        summary.append({
            "run": args.run, "dataset": ds, "model": args.model,
            "n": r["n_samples"], "wf1": r["metrics"]["weighted_f1"],
            "acc": r["metrics"]["accuracy"], "path": r.get("output_path"),
        })

    slug = args.model.replace("/", "_").replace(".", "-")
    sum_path = os.path.join(REPO_ROOT, "results", "sa", slug, f"run{args.run}_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "run": args.run, "full_test": True, "results": summary}, f, indent=2)
    _log(f"Summary: {sum_path}")


if __name__ == "__main__":
    main()
