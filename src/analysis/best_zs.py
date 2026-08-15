"""Best single-model ZS baselines (3-run mean per model, max per dataset)."""
from __future__ import annotations

import statistics

from paths import SLUG, MODELS, RUNS, zs_json, load_result_json
from sklearn.metrics import f1_score

_ZS_HINT = "scripts/phase1/run_01_zs.sh"


def _wf1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def compute_best_zs_wf1() -> dict[str, float]:
    """Return best fixed single-model ZS 3-run mean WF1 per dataset (0–1 scale)."""
    per_model_iem, per_model_meld, per_model_comb = {}, {}, {}
    for m in MODELS:
        iem_vals, meld_vals, comb_vals = [], [], []
        for run in RUNS:
            di = load_result_json(zs_json(SLUG[m], run, "iemocap"), phase1_hint=_ZS_HINT)
            dm = load_result_json(zs_json(SLUG[m], run, "meld"), phase1_hint=_ZS_HINT)
            iem_vals.append(_wf1(di["y_true"], di["y_pred"]))
            meld_vals.append(_wf1(dm["y_true"], dm["y_pred"]))
            comb_vals.append(_wf1(di["y_true"] + dm["y_true"], di["y_pred"] + dm["y_pred"]))
        per_model_iem[m] = statistics.mean(iem_vals)
        per_model_meld[m] = statistics.mean(meld_vals)
        per_model_comb[m] = statistics.mean(comb_vals)
    return {
        "iemocap": max(per_model_iem.values()),
        "meld": max(per_model_meld.values()),
        "comb": max(per_model_comb.values()),
    }


def compute_best_zs_pct() -> dict[str, float]:
    """Same as compute_best_zs_wf1 but keys iem/meld/comb and values in percent."""
    raw = compute_best_zs_wf1()
    return {
        "iem": round(raw["iemocap"] * 100, 2),
        "meld": round(raw["meld"] * 100, 2),
        "comb": round(raw["comb"] * 100, 2),
    }
