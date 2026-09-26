"""InsideOut offline MV-5 over five Ekman agent labels (paper Table I)."""
from __future__ import annotations

import random
import statistics
from collections import Counter
from typing import Sequence

from config import MELD_SAMPLE_OFFSET, RANDOM_TIE_SEED
from lib.majority_vote import utterance_tie_seed, weighted_f1
from paths import MODELS, SLUG, insideout_json, load_result_json

EKMAN_AGENTS = ("anger", "disgust", "fear", "happiness", "sadness")
_IO_HINT = "scripts/phase1/run_04_insideout.sh"


def _agent_labels_from_logs(logs: Sequence[dict]) -> dict[str, list[str]]:
    per_agent = {a: [] for a in EKMAN_AGENTS}
    for log in logs:
        ao = log.get("agent_outputs") or {}
        for a in EKMAN_AGENTS:
            per_agent[a].append((ao.get(a) or {}).get("emotion", "neutral"))
    return per_agent


def mv5_random_vote(agent_labels: list[list[str]], sample_offset: int, seed: int = RANDOM_TIE_SEED) -> list[str]:
    n = len(agent_labels[0])
    out: list[str] = []
    for i in range(n):
        votes = [agent_labels[j][i] for j in range(len(agent_labels))]
        c = Counter(votes).most_common()
        mx = c[0][1]
        winners = sorted(lab for lab, cnt in c if cnt == mx)
        if len(winners) == 1:
            out.append(winners[0])
        else:
            rng = random.Random(utterance_tie_seed(sample_offset + i, seed=seed))
            out.append(rng.choice(winners))
    return out


def _load_io(model_key: str, dataset: str, run: str = "run1") -> dict:
    slug = SLUG[model_key]
    path = insideout_json(slug, dataset, run)
    return load_result_json(path, phase1_hint=_IO_HINT)


def eval_insideout_mv5_one_model(model_key: str, run: str = "run1") -> dict[str, float]:
    per_ds: dict[str, dict] = {}
    di = _load_io(model_key, "iemocap", run)
    dm = _load_io(model_key, "meld", run)
    for ds, d in (("iemocap", di), ("meld", dm)):
        y_true = d["y_true"]
        per_agent = _agent_labels_from_logs(d["logs"])
        offset = 0 if ds == "iemocap" else MELD_SAMPLE_OFFSET
        y_mv5 = mv5_random_vote([per_agent[a] for a in EKMAN_AGENTS], offset)
        per_ds[ds] = {"wf1_mv5": weighted_f1(y_true, y_mv5)}

    pa_i = _agent_labels_from_logs(di["logs"])
    pa_m = _agent_labels_from_logs(dm["logs"])
    y_mv5_i = mv5_random_vote([pa_i[a] for a in EKMAN_AGENTS], 0)
    y_mv5_m = mv5_random_vote([pa_m[a] for a in EKMAN_AGENTS], len(di["y_true"]))
    yt = di["y_true"] + dm["y_true"]
    ym = y_mv5_i + y_mv5_m
    return {
        "iemocap": per_ds["iemocap"]["wf1_mv5"],
        "meld": per_ds["meld"]["wf1_mv5"],
        "comb": weighted_f1(yt, ym),
    }


def insideout_mv5_avg4(run: str = "run1") -> dict[str, float]:
    """4-model arithmetic mean of per-backbone MV-5 WF1 (paper InsideOut-MV-5 row)."""
    per_model = [eval_insideout_mv5_one_model(m, run) for m in MODELS]
    return {
        ds: statistics.mean(pm[ds] for pm in per_model)
        for ds in ("iemocap", "meld", "comb")
    }
