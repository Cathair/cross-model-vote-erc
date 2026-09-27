from __future__ import annotations

"""Fig. 4: ZS (3-run mean) vs SA (run 1) per model."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from run_experiment import compute_metrics
from paths import tables_dir, figures_dir, SLUG, MODELS, RUNS, zs_json, sa_json, ensure_output_dirs, load_result_json

ensure_output_dirs()
TABLES = tables_dir()
FIGURES = figures_dir()


import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


MODEL_LABELS = {
    "gemini": "Gemini",
    "claude": "Claude",
    "qwen": "Qwen",
    "gpt4o": "GPT-4o",
}
COLORS = {"ZS": "#92B4C8", "SA": "#EEA599"}


def zs_path(model: str, ds: str, run: int) -> str:
    return str(zs_json(SLUG[model], run, ds))


def sa_path(model: str, ds: str) -> str:
    return str(sa_json(SLUG[model], ds))


def load_json(path: str) -> dict:
    hint = "scripts/phase1/run_02_sa.sh" if "/sa/" in path.replace("\\", "/") else "scripts/phase1/run_01_zs.sh"
    return load_result_json(path, phase1_hint=hint)


def metrics(y_true, y_pred) -> dict:
    return {
        "wf1": float(f1_score(y_true, y_pred, average="weighted")),
        "acc": float(accuracy_score(y_true, y_pred)),
    }


def collect_zs() -> dict:
    out = {m: {"iemocap": [], "meld": [], "comb": []} for m in MODELS}
    for m in MODELS:
        for run in (1, 2, 3):
            iem = load_json(zs_path(m, "iemocap", run))
            meld = load_json(zs_path(m, "meld", run))
            mi = metrics(iem["y_true"], iem["y_pred"])
            mm = metrics(meld["y_true"], meld["y_pred"])
            mc = metrics(iem["y_true"] + meld["y_true"], iem["y_pred"] + meld["y_pred"])
            out[m]["iemocap"].append(mi)
            out[m]["meld"].append(mm)
            out[m]["comb"].append(mc)
    return out


def collect_sa() -> dict:
    out = {m: {} for m in MODELS}
    for m in MODELS:
        iem = load_json(sa_path(m, "iemocap"))
        meld = load_json(sa_path(m, "meld"))
        out[m]["iemocap"] = metrics(iem["y_true"], iem["y_pred"])
        out[m]["meld"] = metrics(meld["y_true"], meld["y_pred"])
        out[m]["comb"] = metrics(
            iem["y_true"] + meld["y_true"],
            iem["y_pred"] + meld["y_pred"],
        )
    return out


def summarize_runs(runs: list[dict]) -> dict:
    wf1 = [r["wf1"] for r in runs]
    acc = [r["acc"] for r in runs]
    return {
        "mean_wf1": float(np.mean(wf1)),
        "std_wf1": float(np.std(wf1, ddof=1)),
        "mean_acc": float(np.mean(acc)),
        "std_acc": float(np.std(acc, ddof=1)),
        "runs_wf1": wf1,
        "runs_acc": acc,
    }


def build_table(zs: dict, sa: dict) -> list[dict]:
    rows = []
    for m in MODELS:
        for method, src, ds_key in [
            ("ZS", zs[m], True),
            ("SA", sa[m], False),
        ]:
            row = {"model": m, "method": method}
            for ds in ("iemocap", "meld", "comb"):
                if ds_key:
                    s = summarize_runs(src[ds])
                    row[f"{ds}_wf1"] = s["mean_wf1"]
                    row[f"{ds}_wf1_std"] = s["std_wf1"]
                    row[f"{ds}_acc"] = s["mean_acc"]
                    row[f"{ds}_acc_std"] = s["std_acc"]
                else:
                    row[f"{ds}_wf1"] = src[ds]["wf1"]
                    row[f"{ds}_wf1_std"] = None
                    row[f"{ds}_acc"] = src[ds]["acc"]
                    row[f"{ds}_acc_std"] = None
            rows.append(row)
    return rows


def save_table(rows: list[dict]) -> None:
    json_path = TABLES / "table_rq1_zs_sa.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    csv_path = TABLES / "table_rq1_zs_sa.csv"
    fields = [
        "model", "method",
        "iemocap_wf1", "iemocap_wf1_std", "iemocap_acc", "iemocap_acc_std",
        "meld_wf1", "meld_wf1_std", "meld_acc", "meld_acc_std",
        "comb_wf1", "comb_wf1_std", "comb_acc", "comb_acc_std",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def plot_grouped_wf1(rows: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    ds_titles = [("iemocap", "IEMOCAP"), ("meld", "MELD"), ("comb", "Combined")]
    x = np.arange(len(MODELS))
    width = 0.34

    for ax, (ds, title) in zip(axes, ds_titles):
        zs_wf1, zs_std, sa_wf1 = [], [], []
        for m in MODELS:
            zs_row = next(r for r in rows if r["model"] == m and r["method"] == "ZS")
            sa_row = next(r for r in rows if r["model"] == m and r["method"] == "SA")
            zs_wf1.append(zs_row[f"{ds}_wf1"])
            zs_std.append(zs_row[f"{ds}_wf1_std"])
            sa_wf1.append(sa_row[f"{ds}_wf1"])

        ax.bar(x - width / 2, zs_wf1, width, yerr=zs_std, capsize=3,
               color=COLORS["ZS"], label="ZS (3-run mean±std)", edgecolor="white", linewidth=0.6)
        ax.bar(x + width / 2, sa_wf1, width,
               color=COLORS["SA"], label="SA (run1)", edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=15, ha="right")
        ax.set_xlabel(title)
        ax.set_ylim(0.50, 0.70)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Weighted F1 (%)")
    for ax in axes:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.01),
               ncol=2, frameon=False)

    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"fig_rq1_zs_sa_wf1.{ext}")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    zs = collect_zs()
    sa = collect_sa()
    rows = build_table(zs, sa)
    save_table(rows)
    plot_grouped_wf1(rows)
    print(f"Saved tables to {TABLES} and figures to {FIGURES}")


if __name__ == "__main__":
    main()
