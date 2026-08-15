"""Majority voting utilities (paper MV-K tie-break protocol)."""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Sequence

from sklearn.metrics import f1_score


def majority_vote(preds_list: Sequence[Sequence[str]], tie_idx: int) -> list[str]:
    """Hard majority vote; on tie use prediction from model at tie_idx."""
    out: list[str] = []
    for i in range(len(preds_list[0])):
        votes = [p[i] for p in preds_list]
        counts = Counter(votes).most_common()
        max_cnt = counts[0][1]
        winners = [lab for lab, cnt in counts if cnt == max_cnt]
        if len(winners) == 1:
            out.append(winners[0])
        else:
            out.append(preds_list[tie_idx][i])
    return out


def weighted_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def tie_breaker_index(combo: Iterable[str], single_wf1: dict[str, float]) -> int:
    """Index of highest per-dataset WF1 model within combo."""
    combo_list = list(combo)
    best = max(combo_list, key=lambda m: single_wf1[m])
    return combo_list.index(best)


def eval_mv_combo(
    preds_by_model: dict[str, list[str]],
    y_true: list[str],
    combo: tuple[str, ...],
    single_wf1: dict[str, float],
) -> float:
    combo_list = list(combo)
    tb = tie_breaker_index(combo_list, single_wf1)
    preds_list = [preds_by_model[m] for m in combo_list]
    mv = majority_vote(preds_list, tb)
    return weighted_f1(y_true, mv)


def fair_mv_k_mean(
    preds_by_model: dict[str, list[str]],
    y_true: list[str],
    k: int,
    single_wf1: dict[str, float],
) -> float:
    """Mean WF1 over all C(n,k) combinations."""
    models = list(preds_by_model.keys())
    scores = [
        eval_mv_combo(preds_by_model, y_true, combo, single_wf1)
        for combo in combinations(models, k)
    ]
    return sum(scores) / len(scores)
