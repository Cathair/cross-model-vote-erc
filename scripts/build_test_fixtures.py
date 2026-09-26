#!/usr/bin/env python3
"""Build minimal synthetic Phase-1 outputs for offline Phase-2 validation.

Usage:
  python scripts/build_test_fixtures.py /path/to/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from config import repo_root
from data_loader import load_dataset
from paths import SLUG, MODELS
from run_experiment import compute_metrics

RUNS = ["run1", "run2", "run3"]
AGENTS = ["SVA", "CIA", "LRA"]
PHASE0_MAP = {"SVA": "claude", "CIA": "qwen", "LRA": "gpt4o"}
EKMAN = ("anger", "disgust", "fear", "happiness", "sadness")


def _shift_labels(labels: list[str], offset: int, vocab: list[str]) -> list[str]:
    out = []
    for i, lab in enumerate(labels):
        if lab in vocab:
            idx = (vocab.index(lab) + offset + i % 2) % len(vocab)
            out.append(vocab[idx])
        else:
            out.append(lab)
    return out


def _write_pred_json(
    path: Path,
    method: str,
    model: str,
    dataset: str,
    y_true: list[str],
    y_pred: list[str],
    *,
    logs: list | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred)
    payload = {
        "method": method,
        "model": model,
        "dataset": dataset,
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
        "logs": logs or [],
        "errors": [],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _insideout_logs(yt: list[str], yp: list[str], off: int, vocab: list[str]) -> list[dict]:
    logs = []
    for i, lab in enumerate(yt):
        agent_outputs = {}
        for j, emo in enumerate(EKMAN):
            agent_outputs[emo] = {
                "emotion": _shift_labels([lab], off + j + i % 2, vocab)[0],
            }
        logs.append({"agent_outputs": agent_outputs, "true_label": lab, "pred_label": yp[i]})
    return logs


def build(results_root: Path) -> None:
    data_root = str(repo_root() / "data")
    iem_samples = load_dataset("iemocap", data_root, "test", "iemocap_6class")
    meld_samples = load_dataset("meld", data_root, "test", "meld_raw")
    y_iem = [s.label for s in iem_samples]
    y_meld = [s.label for s in meld_samples]
    iem_vocab = sorted(set(y_iem))
    meld_vocab = sorted(set(y_meld))

    model_offsets = {m: i for i, m in enumerate(MODELS)}

    # ZS + SA + InsideOut
    for ri, run in enumerate(RUNS):
        for mi, m in enumerate(MODELS):
            slug = SLUG[m]
            off = model_offsets[m] + ri
            for ds, yt, vocab, n in (
                ("iemocap", y_iem, iem_vocab, 1623),
                ("meld", y_meld, meld_vocab, 2610),
            ):
                yp = _shift_labels(yt, off, vocab)
                zs_path = results_root / "zs" / slug / run / f"{ds}_zeroshot_{slug}_n{n}.json"
                _write_pred_json(zs_path, "zeroshot", slug, ds, yt, yp)
                if run == "run1":
                    sa_path = results_root / "sa" / slug / run / f"{ds}_singleagent_{slug}_n{n}.json"
                    _write_pred_json(sa_path, "singleagent", slug, ds, yt, _shift_labels(yt, off + 1, vocab))
                    io_yp = _shift_labels(yt, off + 2, vocab)
                    io_path = results_root / "insideout" / slug / run / f"{ds}_insideout_{slug}_n{n}.json"
                    _write_pred_json(
                        io_path,
                        "insideout",
                        slug,
                        ds,
                        yt,
                        io_yp,
                        logs=_insideout_logs(yt, io_yp, off, vocab),
                    )

    # MPAR with phase0_labels
    for ri, run in enumerate(RUNS):
        run_dir = results_root / "mpar" / run
        for ds, yt, vocab, n, fname in (
            ("iemocap", y_iem, iem_vocab, 1623, "iemocap_full_n1623.json"),
            ("meld", y_meld, meld_vocab, 2610, "meld_full_n2610.json"),
        ):
            logs = []
            y_pred = []
            for i, lab in enumerate(yt):
                p0 = {}
                for agent, mk in PHASE0_MAP.items():
                    slug = SLUG[mk]
                    off = model_offsets[mk] + ri
                    p0[agent] = _shift_labels([lab], off + i % 3, vocab)[0]
                # final pred = SVA label shifted
                pred = _shift_labels([lab], model_offsets["claude"] + ri, vocab)[0]
                y_pred.append(pred)
                logs.append(
                    {
                        "phase0_labels": p0,
                        "true_label": lab,
                        "pred_label": pred,
                        "api_calls": 5 + (i % 3),
                    }
                )
            path = run_dir / fname
            path.parent.mkdir(parents=True, exist_ok=True)
            metrics = compute_metrics(yt, y_pred)
            path.write_text(
                json.dumps(
                    {
                        "method": "mpar",
                        "y_true": yt,
                        "y_pred": y_pred,
                        "metrics": metrics,
                        "logs": logs,
                        "errors": [],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )


def main():
    parser = argparse.ArgumentParser(description="Build synthetic results for validation")
    parser.add_argument("results_root", type=Path, help="Target results directory")
    args = parser.parse_args()
    build(args.results_root.resolve())
    print(f"Built test fixtures under {args.results_root}")


if __name__ == "__main__":
    main()
