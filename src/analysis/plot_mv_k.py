from __future__ import annotations

"""MV-k ensemble size line chart (IEM / MELD / Combined WF1 vs k).

Fair 3-run protocol: for each run r and size k, mean WF1 over C(4,k) combos; then mean over 3 runs.

Outputs:
  results/figures/fig_rq3_mv_k_wf1.{png,pdf}
  results/tables/table_mv_k_ensemble_size_summary.{csv,json}
  results/tables/table_mv_k_result.{csv,json,md}
  results/tables/zs_4model_ensemble_size_3run.json

Usage (repo root):
  bash scripts/phase2/run_02_plot_mv_k.sh
  python src/analysis/plot_mv_k.py
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from run_experiment import compute_metrics
from paths import tables_dir, figures_dir, SLUG, MODELS, RUNS, zs_json, ensure_output_dirs, load_result_json
from analysis.best_zs import compute_best_zs_pct

ensure_output_dirs()


def _out_dirs():
    out = str(tables_dir())
    fig = str(figures_dir())
    return out, fig, os.path.join(out, "zs_4model_ensemble_size_3run.json")


import csv
import json
import os
import statistics
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score


N_COMBOS = {1: 4, 2: 6, 3: 4, 4: 1}

# 最优 ZS 3-run 均值（table_rq1_zs_sa.csv）：分数据集 WF1 最高单模型
REF_LINES = {
    "iem": {"y": 56.31, "label": "Best-ZS IEM (claude)"},
    "meld": {"y": 64.92, "label": "Best-ZS MELD (gpt4o)"},
    "comb": {"y": 60.26, "label": "Best-ZS Combined (gpt4o)"},
}

COLORS = {"iem": "#EEA599", "meld": "#FAC795", "comb": "#92B4C8"}
# 参考线用同色系更深色，与 MV-k 曲线区分
REF_COLORS = {"iem": "#B84532", "meld": "#C96E1A", "comb": "#2E5F7A"}
LABELS = {"iem": "IEMOCAP WF1", "meld": "MELD WF1", "comb": "Combined WF1"}


def load_preds(run: str) -> dict:
    out = {}
    for ds, n in (("iemocap", 1623), ("meld", 2610)):
        for m in MODELS:
            path = zs_json(SLUG[m], run, ds)
            d = load_result_json(path, phase1_hint="scripts/phase1/run_01_zs.sh")
            out[(ds, m)] = {"y_true": d["y_true"], "y_pred": d["y_pred"]}
    return out


def wf1(yt, yp):
    return f1_score(yt, yp, average="weighted", zero_division=0)


def majority_vote(preds_list, tie_idx: int):
    out = []
    for i in range(len(preds_list[0])):
        votes = [p[i] for p in preds_list]
        c = Counter(votes).most_common()
        mx = c[0][1]
        winners = [lab for lab, cnt in c if cnt == mx]
        out.append(winners[0] if len(winners) == 1 else preds_list[tie_idx][i])
    return out


def eval_combo_wf1(preds, combo, ds_key):
    """ds_key: iem | meld | comb."""
    if ds_key == "comb":
        yt_i = preds[("iemocap", MODELS[0])]["y_true"]
        yt_m = preds[("meld", MODELS[0])]["y_true"]
        yt = yt_i + yt_m
        si = {m: wf1(yt_i, preds[("iemocap", m)]["y_pred"]) for m in MODELS}
        sm = {m: wf1(yt_m, preds[("meld", m)]["y_pred"]) for m in MODELS}
        tb_i = list(combo).index(max(combo, key=lambda m: si[m]))
        tb_m = list(combo).index(max(combo, key=lambda m: sm[m]))
        pl_i = [preds[("iemocap", m)]["y_pred"] for m in combo]
        pl_m = [preds[("meld", m)]["y_pred"] for m in combo]
        yp = majority_vote(pl_i, tb_i) + majority_vote(pl_m, tb_m)
        return wf1(yt, yp)

    ds = "iemocap" if ds_key == "iem" else "meld"
    yt = preds[(ds, MODELS[0])]["y_true"]
    single = {m: wf1(yt, preds[(ds, m)]["y_pred"]) for m in MODELS}
    tb = list(combo).index(max(combo, key=lambda m: single[m]))
    pl = [preds[(ds, m)]["y_pred"] for m in combo]
    return wf1(yt, majority_vote(pl, tb))


def mv_k_run_level(preds, k: int) -> dict[str, float]:
    """Mean WF1 over C(4,k) combos for one run."""
    combos = list(combinations(MODELS, k))
    iem = [eval_combo_wf1(preds, c, "iem") for c in combos]
    meld = [eval_combo_wf1(preds, c, "meld") for c in combos]
    comb = [eval_combo_wf1(preds, c, "comb") for c in combos]
    return {
        "iem": statistics.mean(iem),
        "meld": statistics.mean(meld),
        "comb": statistics.mean(comb),
        "iem_combo_std": statistics.stdev(iem) if len(iem) > 1 else 0.0,
        "meld_combo_std": statistics.stdev(meld) if len(meld) > 1 else 0.0,
        "comb_combo_std": statistics.stdev(comb) if len(comb) > 1 else 0.0,
        "combos": {",".join(c): {"iem": i, "meld": m, "comb": co}
                   for c, i, m, co in zip(combos, iem, meld, comb)},
    }


def compute_fair_3run_summary() -> tuple[list[dict], dict]:
    """Return summary rows for table/plot and full per-run JSON."""
    detail = {str(k): {} for k in (1, 2, 3, 4)}
    rows = []

    for k in (1, 2, 3, 4):
        per_run = []
        for run in RUNS:
            preds = load_preds(run)
            per_run.append({"run": run, **mv_k_run_level(preds, k)})
            detail[str(k)][run] = {
                "iem": per_run[-1]["iem"],
                "meld": per_run[-1]["meld"],
                "comb": per_run[-1]["comb"],
                "combos": per_run[-1]["combos"],
            }

        row = {
            "size": k,
            "n_combos": N_COMBOS[k],
            "api_per_sample": k,
            "protocol": "fair-3run",
        }
        for ds in ("iem", "meld", "comb"):
            vals = [r[ds] for r in per_run]
            row[f"{ds}_mean"] = statistics.mean(vals)
            row[f"{ds}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            combo_stds = [r[f"{ds}_combo_std"] for r in per_run]
            row[f"{ds}_combo_std_mean"] = statistics.mean(combo_stds)
        rows.append(row)

    return rows, detail


def pct(v: float) -> float:
    return v * 100.0


def save_paper_table_mv_k(rows: list[dict], best_zs: dict[str, float]) -> None:
    """Export paper Table II format (rows = datasets, cols = K=1..4 + Best-ZS)."""
    out, _, _ = _out_dirs()
    by_k = {r["size"]: r for r in rows}
    datasets = [
        ("IEMOCAP", "iem", best_zs["iem"]),
        ("MELD", "meld", best_zs["meld"]),
        ("Combined", "comb", best_zs["comb"]),
    ]
    md_rows = []
    csv_rows = []
    for label, key, bz in datasets:
        vals = [round(pct(by_k[k][f"{key}_mean"]), 2) for k in (1, 2, 3, 4)]
        deltas = [round(v - bz, 2) for v in vals]
        md_rows.append((label, bz, vals, deltas))
        csv_rows.append({"metric": label, "best_zs": bz, **{f"k{k}": vals[i] for i, k in enumerate((1, 2, 3, 4))},
                         **{f"delta_k{k}": deltas[i] for i, k in enumerate((1, 2, 3, 4))}})

    md = [
        "# MV-K results (paper Table II)",
        "",
        "| Metric | Best-ZS | K=1 | K=2 | K=3 | K=4 |",
        "|--------|--------:|----:|----:|----:|----:|",
    ]
    for label, bz, vals, _ in md_rows:
        md.append(f"| {label} | {bz:.2f} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} | {vals[3]:.2f} |")
    md.append("")
    md.append("| Δ (pp) | — | " + " | ".join(f"{d:+.2f}" for d in md_rows[0][3]) + " |")
    for i, (label, _, _, deltas) in enumerate(md_rows[1:], 1):
        md.append(f"| Δ({label}) | — | " + " | ".join(f"{d:+.2f}" for d in deltas) + " |")

    with open(os.path.join(out, "table_mv_k_result.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out, "table_mv_k_result.json"), "w", encoding="utf-8") as f:
        json.dump({"best_zs": best_zs, "rows": csv_rows}, f, indent=2)
    fields = ["metric", "best_zs", "k1", "k2", "k3", "k4", "delta_k1", "delta_k2", "delta_k3", "delta_k4"]
    with open(os.path.join(out, "table_mv_k_result.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)


def save_table(rows: list[dict]) -> None:
    out, _, _ = _out_dirs()
    with open(os.path.join(out, "table_mv_k_ensemble_size_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fields = [
        "size", "n_combos", "api_per_sample", "protocol",
        "iem_mean", "iem_std", "iem_combo_std_mean",
        "meld_mean", "meld_std", "meld_combo_std_mean",
        "comb_mean", "comb_std", "comb_combo_std_mean",
    ]
    with open(os.path.join(out, "table_mv_k_ensemble_size_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot(rows: list[dict], ref_lines: dict | None = None) -> None:
    ref_lines = ref_lines or REF_LINES
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    sizes = [r["size"] for r in rows]
    x = np.array(sizes, dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    for ds in ("iem", "meld", "comb"):
        means = np.array([pct(r[f"{ds}_mean"]) for r in rows])
        # error bar: std across 3 run-level MV-k means (replicate variance)
        stds = np.array([pct(r[f"{ds}_std"]) for r in rows])
        ax.errorbar(
            x, means, yerr=stds, fmt="-o", capsize=4, capthick=1.2,
            linewidth=2.2, markersize=7, color=COLORS[ds], label=LABELS[ds],
            elinewidth=1.2,
        )

    _, fig_out, _ = _out_dirs()
    for ds in ("iem", "meld", "comb"):
        ref = ref_lines[ds]
        ax.axhline(
            ref["y"], color=REF_COLORS[ds], linestyle=(0, (7, 4)), linewidth=2.0,
            alpha=0.95, zorder=2,
        )
        ax.text(
            1.02, ref["y"], ref["label"],
            transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=8, color=REF_COLORS[ds],
            fontweight="bold", clip_on=False,
        )

    ax.set_xlabel("Ensemble size K")
    ax.set_ylabel("Weighted F1 (%)")
    ax.set_xticks(sizes)
    ax.set_xlim(0.6, 4.5)
    ax.set_ylim(52, 68)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    fig.subplots_adjust(right=0.78)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_out, f"fig_rq3_mv_k_wf1.{ext}"))
    plt.close(fig)


def main():
    out, _, raw_out = _out_dirs()
    os.makedirs(out, exist_ok=True)

    rows, detail = compute_fair_3run_summary()
    with open(raw_out, "w", encoding="utf-8") as f:
        json.dump({"protocol": "fair-3run", "runs": RUNS, "by_k": detail}, f, indent=2)

    best = compute_best_zs_pct()
    ref_lines = {
        "iem": {"y": best["iem"], "label": "Best-ZS IEM"},
        "meld": {"y": best["meld"], "label": "Best-ZS MELD"},
        "comb": {"y": best["comb"], "label": "Best-ZS Combined"},
    }
    save_table(rows)
    save_paper_table_mv_k(rows, best)
    plot(rows, ref_lines)

    print(f"Wrote fig_rq3_mv_k_wf1.* and tables to {out}")
    print(f"Wrote per-run detail to {raw_out}")
    for r in rows:
        print(
            f"k={r['size']}: IEM {pct(r['iem_mean']):.2f}±{pct(r['iem_std']):.3f}% "
            f"(combo σ̄={pct(r['iem_combo_std_mean']):.2f}%) | "
            f"MELD {pct(r['meld_mean']):.2f}±{pct(r['meld_std']):.3f}% "
            f"(combo σ̄={pct(r['meld_combo_std_mean']):.2f}%) | "
            f"COMB {pct(r['comb_mean']):.2f}±{pct(r['comb_std']):.3f}% "
            f"(combo σ̄={pct(r['comb_combo_std_mean']):.2f}%)"
        )


if __name__ == "__main__":
    main()
