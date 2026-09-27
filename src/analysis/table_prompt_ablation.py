"""Table III: role-prompt vs ZS majority vote on fixed three-model slots."""
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

from config import MELD_SAMPLE_OFFSET, RANDOM_TIE_SEED
from lib.majority_vote import majority_vote_random_label, mv_random_label_tie_from_logs, weighted_f1
from paths import SLUG, get_mpar_runs, tables_dir, zs_json, ensure_output_dirs, load_result_json
from run_experiment import compute_metrics

ensure_output_dirs()
OUT = str(tables_dir())

AGENTS = ["SVA", "CIA", "LRA"]
PHASE0_SLOTS = {"SVA": "claude", "CIA": "qwen", "LRA": "gpt4o"}
PHASE0_SLUGS = {a: SLUG[m] for a, m in PHASE0_SLOTS.items()}
_ZS_HINT = "scripts/phase1/run_01_zs.sh"
_MPAR_HINT = "scripts/phase1/run_03_mpar.sh"


def _load_zs(agent: str, run: str, ds: str) -> dict:
    return load_result_json(zs_json(PHASE0_SLUGS[agent], run, ds), phase1_hint=_ZS_HINT)


def load_pair(iem_path: str, meld_path: str):
    iem = json.load(open(iem_path, encoding="utf-8"))
    meld = json.load(open(meld_path, encoding="utf-8"))
    return iem, meld


def compute_mpar_mv3_random_3run() -> dict:
    per_run = []
    for run_id, run_dir in get_mpar_runs():
        iem_path = run_dir / "iemocap_full_n1623.json"
        meld_path = run_dir / "meld_full_n2610.json"
        if not iem_path.exists() or not meld_path.exists():
            raise FileNotFoundError(
                f"MPAR outputs missing for {run_id}: {run_dir}. Run: bash {_MPAR_HINT}"
            )
        iem, meld = load_pair(str(iem_path), str(meld_path))
        iem_yp = mv_random_label_tie_from_logs(iem["logs"], 0, RANDOM_TIE_SEED)
        meld_yp = mv_random_label_tie_from_logs(meld["logs"], len(iem["logs"]), RANDOM_TIE_SEED)
        comb_yt = iem["y_true"] + meld["y_true"]
        per_run.append(
            {
                "run": run_id,
                "iem": weighted_f1(iem["y_true"], iem_yp),
                "meld": weighted_f1(meld["y_true"], meld_yp),
                "comb": weighted_f1(comb_yt, iem_yp + meld_yp),
            }
        )
    return {
        "iem": statistics.mean(r["iem"] for r in per_run),
        "meld": statistics.mean(r["meld"] for r in per_run),
        "comb": statistics.mean(r["comb"] for r in per_run),
        "per_run": per_run,
    }


def compute_mpar_zs_mv3_random_3run() -> dict:
    """Pure ZS over Phase-0 slots (claude / qwen / gpt4o), random label tie."""
    per_run = []
    for run in ("run1", "run2", "run3"):
        ds_metrics: dict[str, dict] = {}
        n_iem = 0
        for ds in ("iemocap", "meld"):
            preds = [_load_zs(agent, run, ds)["y_pred"] for agent in PHASE0_SLOTS]
            yt = _load_zs("SVA", run, ds)["y_true"]
            offset = 0 if ds == "iemocap" else n_iem
            if ds == "iemocap":
                n_iem = len(yt)
            yp = majority_vote_random_label(preds, offset, RANDOM_TIE_SEED)
            ds_metrics[ds] = {"yt": yt, "yp": yp}

        comb_yt = ds_metrics["iemocap"]["yt"] + ds_metrics["meld"]["yt"]
        comb_yp = ds_metrics["iemocap"]["yp"] + ds_metrics["meld"]["yp"]
        per_run.append(
            {
                "run": run,
                "iem": compute_metrics(ds_metrics["iemocap"]["yt"], ds_metrics["iemocap"]["yp"])["weighted_f1"],
                "meld": compute_metrics(ds_metrics["meld"]["yt"], ds_metrics["meld"]["yp"])["weighted_f1"],
                "comb": compute_metrics(comb_yt, comb_yp)["weighted_f1"],
            }
        )
    return {
        "iem": statistics.mean(r["iem"] for r in per_run),
        "meld": statistics.mean(r["meld"] for r in per_run),
        "comb": statistics.mean(r["comb"] for r in per_run),
        "per_run": per_run,
    }


def pct(v: float) -> float:
    return round(v * 100, 2)


def write_outputs(mpar_mv3: dict, zs_mv3: dict) -> None:
    rows = [
        {
            "method": "MPAR-ZS-MV-3",
            "prompt": "none (pure ZS slots)",
            "iem": pct(zs_mv3["iem"]),
            "meld": pct(zs_mv3["meld"]),
            "comb": pct(zs_mv3["comb"]),
        },
        {
            "method": "MPAR-MV-3",
            "prompt": "MPAR Phase-0 role prompts",
            "iem": pct(mpar_mv3["iem"]),
            "meld": pct(mpar_mv3["meld"]),
            "comb": pct(mpar_mv3["comb"]),
        },
        {
            "method": "ΔWF1",
            "prompt": "—",
            "iem": pct(mpar_mv3["iem"] - zs_mv3["iem"]),
            "meld": pct(mpar_mv3["meld"] - zs_mv3["meld"]),
            "comb": pct(mpar_mv3["comb"] - zs_mv3["comb"]),
        },
    ]
    payload = {
        "tie_break": f"random label seed={RANDOM_TIE_SEED}",
        "mpar_mv3": mpar_mv3,
        "mpar_zs_mv3": zs_mv3,
        "rows": rows,
    }
    with open(os.path.join(OUT, "table_prompt_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    fields = ["method", "prompt", "iem", "meld", "comb"]
    with open(os.path.join(OUT, "table_prompt_ablation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = [
        "# Prompt ablation (paper Table 3)",
        "",
        f"Random label tie-break: seed `{RANDOM_TIE_SEED}` (IEM offset 0, MELD offset {MELD_SAMPLE_OFFSET}).",
        "",
        "| Method | Prompt | IEMOCAP | MELD | Combined |",
        "|--------|--------|--------:|-----:|---------:|",
    ]
    for r in rows[:2]:
        md.append(f"| **{r['method']}** | {r['prompt']} | {r['iem']} | {r['meld']} | {r['comb']} |")
    d = rows[2]
    md.append(f"| **ΔWF1** | — | {d['iem']} pp | {d['meld']} pp | {d['comb']} pp |")
    with open(os.path.join(OUT, "table_prompt_ablation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    mpar_mv3 = compute_mpar_mv3_random_3run()
    zs_mv3 = compute_mpar_zs_mv3_random_3run()
    write_outputs(mpar_mv3, zs_mv3)
    print(f"Wrote table_prompt_ablation.* to {OUT}")


if __name__ == "__main__":
    main()
