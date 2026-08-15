import json
import os
import re
import glob
import time
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any

from data_loader import load_dataset, print_label_distribution, DialogueSample
from baselines import ZS_PROMPT, SA_PROMPT, BaselineModel
from utils import format_context, parse_json_response, validate_label, validate_label_with_flag
from config import get_label_list, DATASET_DEFAULT_STRATEGY, AGENT_MODEL, ARBITER_MODEL, TEMPERATURE, MAX_RETRIES
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from sklearn.metrics import f1_score, accuracy_score, classification_report


# ==================== 配置 ====================
SMOKE_TEST_N = 50
CHECKPOINT_EVERY = 50
PROGRESS_FLUSH = True
from experiment_retry import call_sample_with_retry, SAMPLE_MAX_RETRIES, SAMPLE_RETRY_INTERVAL_SEC, SAMPLE_TIMEOUT_SEC


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_seconds(secs: float) -> str:
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m}m{s}s"
    if m:
        return f"{m}m{s}s"
    return f"{s}s"


def _log(msg: str):
    print(f"[{_ts()}] {msg}", flush=PROGRESS_FLUSH)


def _print_config_summary(dataset_name, strategy, n_samples, mode, model_info: dict):
    _log("=" * 60)
    _log(f"[{mode}] Dataset: {dataset_name} | Strategy: {strategy} | n={n_samples}")
    for k, v in model_info.items():
        _log(f"  {k}: {v}")
    _log("=" * 60)


def compute_metrics(y_true, y_pred):
    return {
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def _find_latest_checkpoint(out_dir: str, tag: str) -> Optional[str]:
    pattern = os.path.join(out_dir, f"checkpoint_{tag}_at_*.json")
    files = glob.glob(pattern)
    if not files:
        return None

    def _idx(path: str) -> int:
        m = re.search(r"_at_(\d+)\.json$", path)
        return int(m.group(1)) if m else 0

    return max(files, key=_idx)


def _save_checkpoint(out_dir, tag, idx, y_true, y_pred, logs, errors=None):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = f"{out_dir}/checkpoint_{tag}_at_{idx}.json"
    with open(ckpt_path, "w") as f:
        json.dump({
            "done": len(y_pred),
            "y_true": y_true,
            "y_pred": y_pred,
            "logs": logs,
            "errors": errors or [],
            "saved_at": _ts(),
        }, f, indent=2, ensure_ascii=False)


def load_sample_indices(indices_file: str) -> List[int]:
    with open(indices_file, encoding="utf-8") as f:
        data = json.load(f)
    return data["indices"]


def subset_samples(all_samples: List[DialogueSample], indices: List[int]) -> List[DialogueSample]:
    return [all_samples[i] for i in indices]


def _run_marc_on_samples(
    samples: List[DialogueSample],
    strategy: str,
    tag: str,
    out_dir: str,
    agent_model: str = None,
    arbiter_model: str = None,
    agent_temp: float = None,
    agent_configs: dict = None,
    save_checkpoint: bool = True,
    checkpoint_every: int = None,
    resume: bool = False,
) -> Dict[str, Any]:
    agent_model = agent_model or AGENT_MODEL
    arbiter_model = arbiter_model or ARBITER_MODEL
    agent_temp = TEMPERATURE if agent_temp is None else agent_temp
    ckpt_interval = checkpoint_every or CHECKPOINT_EVERY

    from marc_erc import MARC_ERC  # optional legacy MARC (not shipped in repro package)

    model = MARC_ERC(
        strategy=strategy,
        use_debate=True,
        use_lra=True,
        use_cia=True,
        agent_model=agent_model,
        arbiter_model=arbiter_model,
        agent_temp=agent_temp,
        agent_configs=agent_configs,
    )

    y_true, y_pred, logs = [], [], []
    errors = []
    start_i = 0
    if resume:
        ckpt_path = _find_latest_checkpoint(out_dir, tag)
        if ckpt_path:
            with open(ckpt_path, encoding="utf-8") as f:
                ckpt = json.load(f)
            y_true = ckpt.get("y_true", [])
            y_pred = ckpt.get("y_pred", [])
            logs = ckpt.get("logs", [])
            errors = ckpt.get("errors", [])
            start_i = len(y_pred)
            _log(f"[resume] loaded {ckpt_path} ({start_i}/{len(samples)} done)")

    start = time.time()

    for i, s in enumerate(samples[start_i:], start=start_i):
        elapsed = time.time() - start
        avg = elapsed / (i + 1) if i > 0 else 0
        remaining = avg * (len(samples) - i - 1)
        eta = _fmt_seconds(remaining) if i > 0 else "?"
        _log(f"[{i+1}/{len(samples)}] ({(i+1)/len(samples)*100:.1f}%) "
             f"true={s.label:<10} elapsed={_fmt_seconds(elapsed)} eta={eta} | {s.utterance[:40]}")

        try:
            def _predict():
                pred, meta = model.predict(s.utterance, s.speaker, s.context)
                return pred, meta

            (pred, meta), attempts_used = call_sample_with_retry(_predict, i, log_fn=_log)
            fallback = bool(meta.get("fallback", False))
            if attempts_used > 0:
                _log(f"  [RETRY OK] sample {i+1} succeeded on attempt {attempts_used + 1}")
        except Exception as e:
            _log(f"  [ERROR] sample {i+1} failed after {SAMPLE_MAX_RETRIES + 1} attempts: "
                 f"{type(e).__name__}: {e}")
            traceback.print_exc()
            pred = "neutral"
            meta = {"decision_mode": "Error", "debate_rounds": 0, "agents_output": {}, "arbiter_decision": {}}
            fallback = True
            errors.append({
                "index": i,
                "error": str(e),
                "type": type(e).__name__,
                "attempts": SAMPLE_MAX_RETRIES + 1,
            })

        y_true.append(s.label)
        y_pred.append(pred)
        logs.append({
            "sample_index": i,
            "dialogue_id": s.dialogue_id,
            "utterance_id": s.utterance_id,
            "speaker": s.speaker,
            "utterance": s.utterance,
            "true_label": s.label,
            "pred_label": pred,
            "fallback": fallback,
            "decision_mode": meta.get("decision_mode", "Unknown"),
            "debate_rounds": meta.get("debate_rounds", 0),
            "agents_output": meta.get("agents_output", {}),
            "arbiter_decision": meta.get("arbiter_decision", {}),
        })

        if save_checkpoint and (i + 1) % ckpt_interval == 0:
            _save_checkpoint(out_dir, tag, i + 1, y_true, y_pred, logs, errors)
            _log(f"  [checkpoint] saved at {i+1} samples")

    metrics = compute_metrics(y_true, y_pred)
    total_time = time.time() - start
    _log(f"MARC [{tag}] DONE in {_fmt_seconds(total_time)}")
    _log(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    _log(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    _log(f"  Accuracy:    {metrics['accuracy']:.4f}")

    result = {
        "method": "marc",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "logs": logs,
        "errors": errors,
        "elapsed_sec": total_time,
        "model_config": {
            "agent_model": agent_model,
            "arbiter_model": arbiter_model,
            "agent_temp": agent_temp,
            "agent_configs": agent_configs,
        },
    }
    return result


def _run_baseline_on_samples(
    samples: List[DialogueSample],
    strategy: str,
    baseline_name: str,
    tag: str,
    model: str = None,
    temperature: float = None,
) -> Dict[str, Any]:
    model = model or AGENT_MODEL
    temperature = 0.0 if temperature is None else temperature
    label_list = get_label_list(strategy)
    llm = ChatOpenAI(model=model, temperature=temperature)
    from lib.cache_llm import maybe_wrap
    llm = maybe_wrap(llm, model=model, temperature=temperature)
    template = ZS_PROMPT if baseline_name == "ZeroShot" else SA_PROMPT

    y_true, y_pred, logs = [], [], []
    errors = []
    start = time.time()

    for i, s in enumerate(samples):
        elapsed = time.time() - start
        avg = elapsed / (i + 1) if i > 0 else 0
        remaining = avg * (len(samples) - i - 1)
        eta = _fmt_seconds(remaining) if i > 0 else "?"
        _log(f"[{i+1}/{len(samples)}] ({(i+1)/len(samples)*100:.1f}%) "
             f"true={s.label:<10} elapsed={_fmt_seconds(elapsed)} eta={eta}")

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
                if baseline_name == "ZeroShot":
                    raw = resp.content.strip()
                    pred, soft_fb = validate_label_with_flag(raw, label_list)
                else:
                    result = parse_json_response(resp.content)
                    raw = result.get("emotion", "neutral")
                    pred, soft_fb = validate_label_with_flag(raw, label_list)
                return pred, soft_fb

            (pred, fallback), attempts_used = call_sample_with_retry(_predict, i, log_fn=_log)
            if attempts_used > 0:
                _log(f"  [RETRY OK] sample {i+1} succeeded on attempt {attempts_used + 1}")
        except Exception as e:
            _log(f"  [ERROR] sample {i+1} failed after {SAMPLE_MAX_RETRIES + 1} attempts: "
                 f"{type(e).__name__}: {e}")
            traceback.print_exc()
            pred = "neutral"
            fallback = True
            errors.append({
                "index": i,
                "error": str(e),
                "type": type(e).__name__,
                "attempts": SAMPLE_MAX_RETRIES + 1,
            })

        y_true.append(s.label)
        y_pred.append(pred)
        logs.append({
            "sample_index": i,
            "dialogue_id": s.dialogue_id,
            "utterance_id": s.utterance_id,
            "true_label": s.label,
            "pred_label": pred,
            "fallback": fallback,
        })

    metrics = compute_metrics(y_true, y_pred)
    total_time = time.time() - start
    _log(f"{baseline_name} [{tag}] DONE in {_fmt_seconds(total_time)}")
    _log(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    _log(f"  Macro F1:    {metrics['macro_f1']:.4f}")
    _log(f"  Accuracy:    {metrics['accuracy']:.4f}")

    return {
        "method": baseline_name.lower(),
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "logs": logs,
        "errors": errors,
        "elapsed_sec": total_time,
        "model_config": {"model": model, "temperature": temperature},
    }


def _save_result(result: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    _log(f"Saved: {out_path}")


def run_marc(dataset_name="iemocap", strategy=None, data_dir="data",
             limit=None, tag="full",
             agent_model: str = None, arbiter_model: str = None,
             agent_temp: float = None, agent_configs: dict = None):
    if strategy is None:
        strategy = DATASET_DEFAULT_STRATEGY[dataset_name]

    samples = load_dataset(dataset_name, data_dir, "test", strategy)
    if limit:
        samples = samples[:limit]
    print_label_distribution(samples)

    _print_config_summary(dataset_name, strategy, len(samples), f"MARC [{tag}]", {
        "agent_model": agent_model or AGENT_MODEL,
        "arbiter_model": arbiter_model or ARBITER_MODEL,
        "agent_temp": agent_temp if agent_temp is not None else TEMPERATURE,
    })

    out_dir = f"results/{dataset_name}_{strategy}"
    result = _run_marc_on_samples(
        samples, strategy, tag, out_dir,
        agent_model=agent_model,
        arbiter_model=arbiter_model,
        agent_temp=agent_temp,
        agent_configs=agent_configs,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_result({"metrics": result["metrics"]}, f"{out_dir}/marc_metrics_{tag}_{ts}.json")
    _save_result(result["logs"], f"{out_dir}/marc_logs_{tag}_{ts}.json")
    if result["errors"]:
        _save_result(result["errors"], f"{out_dir}/marc_errors_{tag}_{ts}.json")

    return result["metrics"]


def run_baselines(dataset_name="iemocap", strategy=None, data_dir="data",
                  limit=None, tag="full",
                  model: str = None, temperature: float = None):
    if strategy is None:
        strategy = DATASET_DEFAULT_STRATEGY[dataset_name]

    samples = load_dataset(dataset_name, data_dir, "test", strategy)
    if limit:
        samples = samples[:limit]

    _print_config_summary(dataset_name, strategy, len(samples), f"Baselines [{tag}]", {
        "model": model or AGENT_MODEL,
        "temperature": temperature if temperature is not None else 0.0,
    })

    out_dir = f"results/{dataset_name}_{strategy}"
    all_metrics = {}

    for name in ["ZeroShot", "SingleAgent"]:
        _log(f"--- {name} Baseline [{tag}] ---")
        result = _run_baseline_on_samples(
            samples, strategy, name, tag,
            model=model, temperature=temperature,
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_result({"metrics": result["metrics"]}, f"{out_dir}/{name.lower()}_metrics_{tag}_{ts}.json")
        all_metrics[name] = result["metrics"]

    return all_metrics


def run_stratified(
    dataset_name: str,
    method: str,
    strategy: str = None,
    indices: List[int] = None,
    indices_file: str = None,
    data_dir: str = "data",
    model_config: dict = None,
    tag: str = "stratified",
    out_dir: str = None,
    output_path: str = None,
) -> dict:
    """按分层抽样索引运行指定方法（marc / zeroshot / singleagent）

    Args:
        method: "marc" | "zeroshot" | "singleagent"
        indices: 样本索引列表
        indices_file: 或从 JSON 文件加载索引
        model_config: {"agent_model", "arbiter_model", "agent_temp", "agent_configs", "model", "temperature"}
        out_dir: 结果保存目录
    """
    if strategy is None:
        strategy = DATASET_DEFAULT_STRATEGY[dataset_name]
    model_config = model_config or {}

    if indices is None:
        if indices_file is None:
            indices_file = f"samples/sample_indices_{dataset_name}.json"
        indices = load_sample_indices(indices_file)

    all_samples = load_dataset(dataset_name, data_dir, "test", strategy)
    samples = subset_samples(all_samples, indices)
    print_label_distribution(samples)

    method = method.lower()
    model_slug = model_config.get("model") or model_config.get("agent_model") or AGENT_MODEL
    model_slug = model_slug.replace("/", "_").replace(".", "-")

    if out_dir is None:
        out_dir = f"results/stratified/{dataset_name}_{strategy}"

    _print_config_summary(dataset_name, strategy, len(samples), f"{method.upper()} [{tag}]", model_config)

    if method == "marc":
        result = _run_marc_on_samples(
            samples, strategy, tag, out_dir,
            agent_model=model_config.get("agent_model"),
            arbiter_model=model_config.get("arbiter_model"),
            agent_temp=model_config.get("agent_temp"),
            agent_configs=model_config.get("agent_configs"),
            save_checkpoint=False,
        )
        out_path = f"{out_dir}/{method}_{model_slug}_{tag}.json"
    elif method in ("zeroshot", "singleagent"):
        bname = "ZeroShot" if method == "zeroshot" else "SingleAgent"
        result = _run_baseline_on_samples(
            samples, strategy, bname, tag,
            model=model_config.get("model"),
            temperature=model_config.get("temperature"),
        )
        out_path = f"{out_dir}/{method}_{model_slug}_{tag}.json"
    else:
        raise ValueError(f"Unknown method: {method}")

    if output_path:
        out_path = output_path

    result["dataset"] = dataset_name
    result["strategy"] = strategy
    result["indices"] = indices
    result["tag"] = tag
    _save_result(result, out_path)
    return result


if __name__ == "__main__":
    print("run_experiment.py provides shared utilities for inference scripts.")
    print("It is not a standalone reproduction entry point. Use instead:")
    print("  bash scripts/phase1/run_01_zs.sh")
    print("  bash scripts/phase1/run_02_sa.sh")
    print("  bash scripts/phase1/run_03_mpar.sh")
    print("  bash scripts/phase1/run_04_insideout.sh")
    raise SystemExit(1)
