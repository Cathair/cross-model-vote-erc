"""Paper Table: MV-3-Phase0 vs MPAR-Phase0 (prompt ablation).

Outputs:
  results/tables/table_prompt_ablation.{csv,md,json}
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from collections import Counter
from typing import Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from paths import SLUG, get_mpar_runs, tables_dir, zs_json, ensure_output_dirs, load_result_json
from run_experiment import compute_metrics

ensure_output_dirs()
OUT = str(tables_dir())

AGENTS = ["SVA", "CIA", "LRA"]
PHASE0_SLOTS = {"SVA": "claude", "CIA": "qwen", "LRA": "gpt4o"}
# Paper protocol: MV-3-Phase0 uses fixed tie_idx=0 (claude / SVA slot), not MV-K WF1 tie-break.
PHASE0_TIE_IDX = 0
PHASE0_SLUGS = {a: SLUG[m] for a, m in PHASE0_SLOTS.items()}
_ZS_HINT = "scripts/phase1/run_01_zs.sh"
_MPAR_HINT = "scripts/phase1/run_03_mpar.sh"


def _load_zs(agent: str, run: str, ds: str) -> dict:
    return load_result_json(zs_json(PHASE0_SLUGS[agent], run, ds), phase1_hint=_ZS_HINT)


def load_pair(iem_path: str, meld_path: str):
    iem = json.load(open(iem_path, encoding="utf-8"))
    meld = json.load(open(meld_path, encoding="utf-8"))
    return iem, meld


def majority_vote(preds_list, tie_idx: int = 0):
    out = []
    for i in range(len(preds_list[0])):
        votes = [p[i] for p in preds_list]
        c = Counter(votes).most_common()
        mx = c[0][1]
        winners = [lab for lab, cnt in c if cnt == mx]
        out.append(winners[0] if len(winners) == 1 else preds_list[tie_idx][i])
    return out


def phase0_preds(logs: list) -> Dict[str, List[str]]:
    return {a: [entry["phase0_labels"][a] for entry in logs] for a in AGENTS}


def agent_priority(metrics: Dict[str, dict]) -> List[str]:
    return sorted(AGENTS, key=lambda a: -metrics[a]["weighted_f1"])


def mv_agent_tie(labels: Dict[str, str], priority: List[str]) -> str:
    votes = Counter(labels.values())
    max_cnt = max(votes.values())
    winners = [lab for lab, c in votes.items() if c == max_cnt]
    if len(winners) == 1:
        return winners[0]
    for agent in priority:
        if labels[agent] in winners:
            return labels[agent]
    return sorted(winners)[0]


def p0_mv_preds(logs: list, priority: List[str]) -> List[str]:
    return [mv_agent_tie(entry["phase0_labels"], priority) for entry in logs]


def eval_p0_mv(iem_logs, iem_yt, meld_logs, meld_yt) -> dict:
    comb_yt = iem_yt + meld_yt
    comb_preds = {
        a: phase0_preds(iem_logs)[a] + phase0_preds(meld_logs)[a] for a in AGENTS
    }
    comb_m = {a: compute_metrics(comb_yt, comb_preds[a]) for a in AGENTS}
    pri = agent_priority(comb_m)
    iem_yp = p0_mv_preds(iem_logs, pri)
    meld_yp = p0_mv_preds(meld_logs, pri)
    return {
        "iem": compute_metrics(iem_yt, iem_yp)["weighted_f1"],
        "meld": compute_metrics(meld_yt, meld_yp)["weighted_f1"],
        "comb": compute_metrics(comb_yt, iem_yp + meld_yp)["weighted_f1"],
    }


def compute_mv3_phase0_3run() -> dict:
    """Pure ZS MV over claude / qwen / gpt4o (MPAR Phase-0 slots).

    Tie-break on equal votes: fixed tie_idx=0 (claude), matching paper mv_simple(..., tie_idx=0).
    """
    per_run = []
    for run in ("run1", "run2", "run3"):
        ds_metrics = {}
        for ds in ("iemocap", "meld"):
            preds = [_load_zs(agent, run, ds)["y_pred"] for agent in PHASE0_SLOTS]
            yt = _load_zs("SVA", run, ds)["y_true"]
            yp = majority_vote(preds, tie_idx=PHASE0_TIE_IDX)
            ds_metrics[ds] = compute_metrics(yt, yp)["weighted_f1"]

        iem = _load_zs("SVA", run, "iemocap")
        meld = _load_zs("SVA", run, "meld")
        pl_i = [_load_zs(a, run, "iemocap")["y_pred"] for a in PHASE0_SLOTS]
        pl_m = [_load_zs(a, run, "meld")["y_pred"] for a in PHASE0_SLOTS]
        yp = majority_vote(pl_i, tie_idx=PHASE0_TIE_IDX) + majority_vote(pl_m, tie_idx=PHASE0_TIE_IDX)
        comb = compute_metrics(iem["y_true"] + meld["y_true"], yp)["weighted_f1"]
        per_run.append({"run": run, "iem": ds_metrics["iemocap"], "meld": ds_metrics["meld"], "comb": comb})

    return {
        "iem": statistics.mean(r["iem"] for r in per_run),
        "meld": statistics.mean(r["meld"] for r in per_run),
        "comb": statistics.mean(r["comb"] for r in per_run),
        "per_run": per_run,
    }


def compute_mpar_phase0_3run() -> dict:
    iem_vals, meld_vals, comb_vals = [], [], []
    per_run = []
    for run_id, run_dir in get_mpar_runs():
        iem_path = run_dir / "iemocap_full_n1623.json"
        meld_path = run_dir / "meld_full_n2610.json"
        if not iem_path.exists() or not meld_path.exists():
            raise FileNotFoundError(
                f"MPAR outputs missing for {run_id}: {run_dir}. "
                f"Run: bash {_MPAR_HINT}"
            )
        iem, meld = load_pair(str(iem_path), str(meld_path))
        m = eval_p0_mv(iem["logs"], iem["y_true"], meld["logs"], meld["y_true"])
        iem_vals.append(m["iem"])
        meld_vals.append(m["meld"])
        comb_vals.append(m["comb"])
        per_run.append({"run": run_id, **m})
    return {
        "iem": statistics.mean(iem_vals),
        "meld": statistics.mean(meld_vals),
        "comb": statistics.mean(comb_vals),
        "per_run": per_run,
    }


def pct(v: float) -> float:
    return round(v * 100, 2)


def write_outputs(mv3: dict, mp0: dict) -> None:
    rows = [
        {"method": "MV-3-Phase0", "prompt": "none", "iem": pct(mv3["iem"]), "meld": pct(mv3["meld"]), "comb": pct(mv3["comb"])},
        {"method": "MPAR-Phase0", "prompt": "MPAR Phase-0", "iem": pct(mp0["iem"]), "meld": pct(mp0["meld"]), "comb": pct(mp0["comb"])},
        {"method": "ΔWF1", "prompt": "—", "iem": pct(mp0["iem"] - mv3["iem"]), "meld": pct(mp0["meld"] - mv3["meld"]), "comb": pct(mp0["comb"] - mv3["comb"])},
    ]
    payload = {"mv3_phase0": mv3, "mpar_phase0": mp0, "rows": rows}
    with open(os.path.join(OUT, "table_prompt_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    fields = ["method", "prompt", "iem", "meld", "comb"]
    with open(os.path.join(OUT, "table_prompt_ablation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = [
        "# Prompt ablation (paper Table)",
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
    mv3 = compute_mv3_phase0_3run()
    mp0 = compute_mpar_phase0_3run()
    write_outputs(mv3, mp0)
    print(f"Wrote table_prompt_ablation.* to {OUT}")


if __name__ == "__main__":
    main()
