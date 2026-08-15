"""Paper Table I: Main results (Best-ZS, InsideOut, MPAR, MV-4).

Outputs:
  results/tables/table_main_result.{csv,md,json}

Requires Phase 1 ZS outputs (for Best-ZS) + MPAR + InsideOut; run plot_mv_k before MV-4 row.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from paths import SLUG, MODELS, insideout_json, get_mpar_runs, tables_dir, ensure_output_dirs
from analysis.best_zs import compute_best_zs_wf1
from run_experiment import compute_metrics

ensure_output_dirs()
OUT = str(tables_dir())


def insideout_avg4() -> dict[str, float]:
    metrics = {ds: [] for ds in ("iemocap", "meld", "comb")}
    for m in MODELS:
        slug = SLUG[m]
        for ds in ("iemocap", "meld"):
            path = insideout_json(slug, ds, 1)
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run: bash scripts/phase1/run_04_insideout.sh"
                )
            data = json.load(open(path, encoding="utf-8"))
            metrics[ds].append(compute_metrics(data["y_true"], data["y_pred"])["weighted_f1"])
        iem = json.load(open(insideout_json(slug, "iemocap", 1), encoding="utf-8"))
        meld = json.load(open(insideout_json(slug, "meld", 1), encoding="utf-8"))
        metrics["comb"].append(
            compute_metrics(iem["y_true"] + meld["y_true"], iem["y_pred"] + meld["y_pred"])["weighted_f1"]
        )
    return {ds: statistics.mean(metrics[ds]) for ds in metrics}


def mpar_3run_mean() -> dict[str, float]:
    iem, meld, comb = [], [], []
    for _, run_dir in get_mpar_runs():
        iem_p = run_dir / "iemocap_full_n1623.json"
        meld_p = run_dir / "meld_full_n2610.json"
        if not iem_p.exists() or not meld_p.exists():
            raise FileNotFoundError(f"Missing MPAR outputs in {run_dir}. Run phase1/run_03_mpar.sh")
        iem_d = json.load(open(iem_p, encoding="utf-8"))
        meld_d = json.load(open(meld_p, encoding="utf-8"))
        iem.append(compute_metrics(iem_d["y_true"], iem_d["y_pred"])["weighted_f1"])
        meld.append(compute_metrics(meld_d["y_true"], meld_d["y_pred"])["weighted_f1"])
        comb.append(
            compute_metrics(
                iem_d["y_true"] + meld_d["y_true"],
                iem_d["y_pred"] + meld_d["y_pred"],
            )["weighted_f1"]
        )
    return {"iemocap": statistics.mean(iem), "meld": statistics.mean(meld), "comb": statistics.mean(comb)}


def mv4_from_summary() -> dict[str, float]:
    path = tables_dir() / "table_mv_k_ensemble_size_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: bash scripts/phase2/run_02_plot_mv_k.sh"
        )
    rows = json.load(open(path, encoding="utf-8"))
    row = next(r for r in rows if r["size"] == 4)
    return {"iemocap": row["iem_mean"], "meld": row["meld_mean"], "comb": row["comb_mean"]}


def pct(v: float) -> float:
    return round(v * 100, 2)


def delta_pp(v: float, base: float) -> str:
    return f"{(v - base) * 100:+.2f} pp"


def main():
    os.makedirs(OUT, exist_ok=True)
    bz = compute_best_zs_wf1()
    io = insideout_avg4()
    mpar = mpar_3run_mean()
    mv4 = mv4_from_summary()

    rows = [
        {
            "method": "Best-ZS",
            "iemocap": pct(bz["iemocap"]),
            "meld": pct(bz["meld"]),
            "comb": pct(bz["comb"]),
            "delta_comb_pp": "—",
        },
        {
            "method": "InsideOut",
            "iemocap": pct(io["iemocap"]),
            "meld": pct(io["meld"]),
            "comb": pct(io["comb"]),
            "delta_comb_pp": delta_pp(io["comb"], bz["comb"]),
        },
        {
            "method": "MPAR",
            "iemocap": pct(mpar["iemocap"]),
            "meld": pct(mpar["meld"]),
            "comb": pct(mpar["comb"]),
            "delta_comb_pp": delta_pp(mpar["comb"], bz["comb"]),
        },
        {
            "method": "MV-4",
            "iemocap": pct(mv4["iemocap"]),
            "meld": pct(mv4["meld"]),
            "comb": pct(mv4["comb"]),
            "delta_comb_pp": delta_pp(mv4["comb"], bz["comb"]),
        },
    ]

    with open(os.path.join(OUT, "table_main_result.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fields = ["method", "iemocap", "meld", "comb", "delta_comb_pp"]
    with open(os.path.join(OUT, "table_main_result.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = [
        "# Main results (paper Table I)",
        "",
        "| Method | IEMOCAP | MELD | Combined | ΔWF1 (Combined) |",
        "|--------|--------:|-----:|---------:|----------------:|",
    ]
    for r in rows:
        md.append(
            f"| **{r['method']}** | {r['iemocap']} | {r['meld']} | {r['comb']} | {r['delta_comb_pp']} |"
        )
    with open(os.path.join(OUT, "table_main_result.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Wrote table_main_result.* to {OUT}")


if __name__ == "__main__":
    main()
