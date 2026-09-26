from __future__ import annotations

"""Q3: Within-model vs cross-model ensemble gain.

Compares WF1 **gain (pp)** over **qwen-ZS 3-run mean** (common baseline):
  - 3-qwen-ZS MV: majority vote across qwen run1/2/3 predictions (within-model)
  - MV-3: fair 3-run cross-model MV-k (k=3), see plot_mv_k_ensemble_size.py

Outputs:
  results/figures/fig_within_vs_cross_gain.{png,pdf}
  results/tables/table_within_vs_cross_gain.{csv,json}

Usage (repo root):
  bash scripts/phase2/run_03_plot_within_vs_cross.sh
  python src/analysis/plot_within_vs_cross.py
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from config import MELD_SAMPLE_OFFSET, RANDOM_TIE_SEED
from lib.majority_vote import majority_vote, majority_vote_random_label, weighted_f1
from paths import tables_dir, figures_dir, SLUG, MODELS, RUNS, zs_json, ensure_output_dirs, load_result_json

ensure_output_dirs()
OUT = str(tables_dir())
FIG_OUT = str(figures_dir())


import csv
import json
import os
import statistics
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


def load_zs(run: str, ds: str, model: str) -> dict:
    return load_result_json(
        zs_json(SLUG[model], run, ds),
        phase1_hint="scripts/phase1/run_01_zs.sh",
    )


def qwen_3run_mean() -> dict[str, float]:
    """qwen-ZS 3-run WF1 mean per dataset."""
    out = {}
    for ds in ("iemocap", "meld"):
        out[ds] = statistics.mean(
            weighted_f1(load_zs(r, ds, "qwen")["y_true"], load_zs(r, ds, "qwen")["y_pred"])
            for r in RUNS
        )
    comb = []
    for r in RUNS:
        d_i = load_zs(r, "iemocap", "qwen")
        d_m = load_zs(r, "meld", "qwen")
        comb.append(weighted_f1(d_i["y_true"] + d_m["y_true"], d_i["y_pred"] + d_m["y_pred"]))
    out["comb"] = statistics.mean(comb)
    return out


def three_qwen_mv() -> dict[str, float]:
    """3-qwen-ZS MV: vote over qwen run1/2/3 predictions (per-dataset, then concat)."""
    out = {}
    for ds in ("iemocap", "meld"):
        preds = [load_zs(r, ds, "qwen")["y_pred"] for r in RUNS]
        yt = load_zs("run1", ds, "qwen")["y_true"]
        out[ds] = weighted_f1(yt, majority_vote(preds, tie_idx=0))

    yt_i = load_zs("run1", "iemocap", "qwen")["y_true"]
    yt_m = load_zs("run1", "meld", "qwen")["y_true"]
    pl_i = [load_zs(r, "iemocap", "qwen")["y_pred"] for r in RUNS]
    pl_m = [load_zs(r, "meld", "qwen")["y_pred"] for r in RUNS]
    yp = majority_vote(pl_i, 0) + majority_vote(pl_m, 0)
    out["comb"] = weighted_f1(yt_i + yt_m, yp)
    return out


def mv3_fair_3run() -> dict[str, float]:
    """MV-3 fair 3-run: mean over runs of C(4,3) combo WF1 means."""
    per_run = {ds: [] for ds in ("iemocap", "meld", "comb")}

    for run in RUNS:
        for ds in ("iemocap", "meld"):
            yt = load_zs(run, ds, "gemini")["y_true"]
            offset = 0 if ds == "iemocap" else MELD_SAMPLE_OFFSET
            wf1s = []
            for combo in combinations(MODELS, 3):
                pl = [load_zs(run, ds, m)["y_pred"] for m in combo]
                wf1s.append(
                    weighted_f1(yt, majority_vote_random_label(pl, offset, RANDOM_TIE_SEED))
                )
            per_run[ds].append(statistics.mean(wf1s))

        yt_i = load_zs(run, "iemocap", "gemini")["y_true"]
        yt_m = load_zs(run, "meld", "gemini")["y_true"]
        yt = yt_i + yt_m
        comb_wf1s = []
        for combo in combinations(MODELS, 3):
            pl_i = [load_zs(run, "iemocap", m)["y_pred"] for m in combo]
            pl_m = [load_zs(run, "meld", m)["y_pred"] for m in combo]
            yp = majority_vote_random_label(pl_i, 0, RANDOM_TIE_SEED) + majority_vote_random_label(
                pl_m, len(yt_i), RANDOM_TIE_SEED
            )
            comb_wf1s.append(weighted_f1(yt, yp))
        per_run["comb"].append(statistics.mean(comb_wf1s))

    return {ds: statistics.mean(per_run[ds]) for ds in per_run}


def build_rows() -> list[dict]:
    baseline = qwen_3run_mean()
    within = three_qwen_mv()
    cross = mv3_fair_3run()

    rows = []
    for ds in ("iemocap", "meld", "comb"):
        b = baseline[ds]
        w = within[ds]
        c = cross[ds]
        rows.append({
            "dataset": ds,
            "qwen_zs_3run_wf1": round(b, 6),
            "qwen_zs_3run_wf1_pct": round(b * 100, 2),
            "three_qwen_mv_wf1": round(w, 6),
            "three_qwen_mv_wf1_pct": round(w * 100, 2),
            "mv3_fair_3run_wf1": round(c, 6),
            "mv3_fair_3run_wf1_pct": round(c * 100, 2),
            "three_qwen_delta_pp": round((w - b) * 100, 2),
            "mv3_delta_pp": round((c - b) * 100, 2),
            "baseline_note": "qwen-ZS 3-run mean (all datasets)",
            "mv3_protocol": "fair-3run-random-label-tie",
            "within_tie": "tie_idx=0 (same-model 3-run MV-3-Qwen)",
        })
    return rows


def save_table(rows: list[dict]) -> None:
    meta = {
        "description": "Q3 within-model (3-qwen MV) vs cross-model (MV-3 fair 3-run) gain over qwen-ZS 3-run mean",
        "rows": rows,
    }
    with open(os.path.join(OUT, "table_within_vs_cross_gain.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    fields = [
        "dataset", "qwen_zs_3run_wf1_pct", "three_qwen_mv_wf1_pct", "mv3_fair_3run_wf1_pct",
        "three_qwen_delta_pp", "mv3_delta_pp",
    ]
    with open(os.path.join(OUT, "table_within_vs_cross_gain.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def plot(rows: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    labels = ["IEMOCAP", "MELD", "Combined"]
    within = [r["three_qwen_delta_pp"] for r in rows]
    cross = [r["mv3_delta_pp"] for r in rows]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bars_w = ax.bar(
        x - width / 2, within, width, color="#FAC795", edgecolor="white",
        linewidth=0.8, label="MV-3-Qwen − qwen ZS 3-run",
    )
    bars_c = ax.bar(
        x + width / 2, cross, width, color="#ABD3E1", edgecolor="white",
        linewidth=0.8, label="MV-3 − qwen ZS 3-run",
    )

    for bar in list(bars_w) + list(bars_c):
        h = bar.get_height()
        ax.annotate(
            f"{h:+.2f}", (bar.get_x() + bar.get_width() / 2, h),
            ha="center", va="bottom" if h >= 0 else "top",
            xytext=(0, 3 if h >= 0 else -3), textcoords="offset points",
            fontsize=9, fontweight="bold",
        )

    comb_w, comb_c = within[2], cross[2]
    if comb_w > 0:
        ratio = comb_c / comb_w
        ax.annotate(
            f"≈{ratio:.1f}×", xy=(2, max(comb_w, comb_c) + 0.35),
            ha="center", fontsize=11, fontweight="bold", color="#B84532",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("WF1 gain (pp)")
    ymax = max(within + cross)
    pad = max(0.4, ymax * 0.15)
    ax.set_ylim(0, ymax + pad)
    ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.6, zorder=1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.92)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_OUT, f"fig_within_vs_cross_gain.{ext}"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = build_rows()
    save_table(rows)
    plot(rows)
    print(f"Wrote fig_within_vs_cross_gain.* and table_within_vs_cross_gain.* to {OUT}")
    for r in rows:
        print(
            f"[{r['dataset']}] qwen 3-run={r['qwen_zs_3run_wf1_pct']:.2f}% | "
            f"3-qwen MV={r['three_qwen_mv_wf1_pct']:.2f}% (Δ{r['three_qwen_delta_pp']:+.2f}) | "
            f"MV-3={r['mv3_fair_3run_wf1_pct']:.2f}% (Δ{r['mv3_delta_pp']:+.2f})"
        )


if __name__ == "__main__":
    main()
