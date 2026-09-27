"""Table I: main results (requires Phase 1 + run_02 for MV-4 row)."""
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

from lib.insideout_mv import insideout_mv5_avg4
from lib.majority_vote import mv_random_label_tie_from_logs, weighted_f1
from config import RANDOM_TIE_SEED
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


def mpar_mv3_random_3run() -> dict[str, float]:
    iem_vals, meld_vals, comb_vals = [], [], []
    for _, run_dir in get_mpar_runs():
        iem_d = json.load(open(run_dir / "iemocap_full_n1623.json", encoding="utf-8"))
        meld_d = json.load(open(run_dir / "meld_full_n2610.json", encoding="utf-8"))
        iem_yp = mv_random_label_tie_from_logs(iem_d["logs"], 0, RANDOM_TIE_SEED)
        meld_yp = mv_random_label_tie_from_logs(meld_d["logs"], len(iem_d["logs"]), RANDOM_TIE_SEED)
        comb_yt = iem_d["y_true"] + meld_d["y_true"]
        iem_vals.append(weighted_f1(iem_d["y_true"], iem_yp))
        meld_vals.append(weighted_f1(meld_d["y_true"], meld_yp))
        comb_vals.append(weighted_f1(comb_yt, iem_yp + meld_yp))
    return {
        "iemocap": statistics.mean(iem_vals),
        "meld": statistics.mean(meld_vals),
        "comb": statistics.mean(comb_vals),
    }


def mpar_api_calls_mean() -> float:
    totals = []
    for _, run_dir in get_mpar_runs():
        for fname in ("iemocap_full_n1623.json", "meld_full_n2610.json"):
            path = run_dir / fname
            if not path.exists():
                continue
            data = json.load(open(path, encoding="utf-8"))
            for entry in data.get("logs") or []:
                if "api_calls" in entry:
                    totals.append(float(entry["api_calls"]))
    if totals:
        return statistics.mean(totals)
    return 5.39


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


def api_calls_display(method: str, mpar_mean: float) -> str:
    mapping = {
        "Best-ZS": "1",
        "InsideOut": "6",
        "InsideOut-MV-5": "5",
        "MPAR": f"{mpar_mean:.2f}",
        "MPAR-MV-3": "3",
        "MV-4": "4",
    }
    return mapping[method]


def main():
    os.makedirs(OUT, exist_ok=True)
    bz = compute_best_zs_wf1()
    io = insideout_avg4()
    io_mv5 = insideout_mv5_avg4()
    mpar = mpar_3run_mean()
    mpar_mv3 = mpar_mv3_random_3run()
    mv4 = mv4_from_summary()
    mpar_api = mpar_api_calls_mean()

    rows = [
        {
            "method": "Best-ZS",
            "iemocap": pct(bz["iemocap"]),
            "meld": pct(bz["meld"]),
            "comb": pct(bz["comb"]),
            "delta_comb_pp": "—",
            "api_calls": api_calls_display("Best-ZS", mpar_api),
        },
        {
            "method": "InsideOut",
            "iemocap": pct(io["iemocap"]),
            "meld": pct(io["meld"]),
            "comb": pct(io["comb"]),
            "delta_comb_pp": delta_pp(io["comb"], bz["comb"]),
            "api_calls": api_calls_display("InsideOut", mpar_api),
        },
        {
            "method": "InsideOut-MV-5",
            "iemocap": pct(io_mv5["iemocap"]),
            "meld": pct(io_mv5["meld"]),
            "comb": pct(io_mv5["comb"]),
            "delta_comb_pp": delta_pp(io_mv5["comb"], bz["comb"]),
            "api_calls": api_calls_display("InsideOut-MV-5", mpar_api),
        },
        {
            "method": "MPAR",
            "iemocap": pct(mpar["iemocap"]),
            "meld": pct(mpar["meld"]),
            "comb": pct(mpar["comb"]),
            "delta_comb_pp": delta_pp(mpar["comb"], bz["comb"]),
            "api_calls": api_calls_display("MPAR", mpar_api),
        },
        {
            "method": "MPAR-MV-3",
            "iemocap": pct(mpar_mv3["iemocap"]),
            "meld": pct(mpar_mv3["meld"]),
            "comb": pct(mpar_mv3["comb"]),
            "delta_comb_pp": delta_pp(mpar_mv3["comb"], bz["comb"]),
            "api_calls": api_calls_display("MPAR-MV-3", mpar_api),
        },
        {
            "method": "MV-4",
            "iemocap": pct(mv4["iemocap"]),
            "meld": pct(mv4["meld"]),
            "comb": pct(mv4["comb"]),
            "delta_comb_pp": delta_pp(mv4["comb"], bz["comb"]),
            "api_calls": api_calls_display("MV-4", mpar_api),
        },
    ]

    meta = {
        "tie_break_mv": f"random label seed={RANDOM_TIE_SEED}",
        "mpar_api_calls_mean": mpar_api,
    }
    with open(os.path.join(OUT, "table_main_result.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2, ensure_ascii=False)

    fields = ["method", "iemocap", "meld", "comb", "delta_comb_pp", "api_calls"]
    with open(os.path.join(OUT, "table_main_result.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = [
        "# Main results (paper Table I)",
        "",
        f"MV-K / MPAR-MV-3 / InsideOut-MV-5 ties: random label (seed `{RANDOM_TIE_SEED}`).",
        "",
        "| Method | IEMOCAP | MELD | Combined | ΔWF1 (Combined) | API Calls |",
        "|--------|--------:|-----:|---------:|----------------:|----------:|",
    ]
    for r in rows:
        md.append(
            f"| **{r['method']}** | {r['iemocap']} | {r['meld']} | {r['comb']} | {r['delta_comb_pp']} | {r['api_calls']} |"
        )
    with open(os.path.join(OUT, "table_main_result.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Wrote table_main_result.* to {OUT}")


if __name__ == "__main__":
    main()
