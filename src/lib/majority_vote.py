"""Majority voting utilities (paper MV-K random label tie-break, §3.3.4)."""
from __future__ import annotations

import random
from collections import Counter
from itertools import combinations
from typing import Iterable, Sequence

from sklearn.metrics import f1_score

from config import IEMOCAP_SAMPLE_OFFSET, MELD_SAMPLE_OFFSET, RANDOM_TIE_SEED


def utterance_tie_seed(global_index: int, *, seed: int = RANDOM_TIE_SEED) -> int:
    return seed + global_index


def dataset_sample_offset(dataset: str, n_iem: int = MELD_SAMPLE_OFFSET) -> int:
    if dataset == "iemocap":
        return IEMOCAP_SAMPLE_OFFSET
    if dataset == "meld":
        return n_iem
    raise ValueError(f"unknown dataset for tie offset: {dataset}")


def majority_vote(preds_list: Sequence[Sequence[str]], tie_idx: int) -> list[str]:
    """Hard majority vote; on tie use prediction from model at tie_idx (legacy / within-qwen)."""
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


def majority_vote_random_label(
    preds_list: Sequence[Sequence[str]],
    sample_offset: int,
    seed: int = RANDOM_TIE_SEED,
) -> list[str]:
    """On tie, uniformly pick one tied label (sorted lexicographically); deterministic per utterance."""
    out: list[str] = []
    for i in range(len(preds_list[0])):
        votes = [p[i] for p in preds_list]
        counts = Counter(votes).most_common()
        max_cnt = counts[0][1]
        winners = sorted(lab for lab, cnt in counts if cnt == max_cnt)
        if len(winners) == 1:
            out.append(winners[0])
        else:
            rng = random.Random(utterance_tie_seed(sample_offset + i, seed=seed))
            out.append(rng.choice(winners))
    return out


def mv_random_label_tie_from_logs(
    logs: Sequence[dict],
    sample_offset: int,
    seed: int = RANDOM_TIE_SEED,
) -> list[str]:
    """Majority vote on phase0_labels values with random label tie-break."""
    out: list[str] = []
    for i, entry in enumerate(logs):
        labels = entry["phase0_labels"]
        votes = Counter(labels.values())
        max_cnt = max(votes.values())
        winners = sorted(lab for lab, c in votes.items() if c == max_cnt)
        if len(winners) == 1:
            out.append(winners[0])
        else:
            rng = random.Random(utterance_tie_seed(sample_offset + i, seed=seed))
            out.append(rng.choice(winners))
    return out


def weighted_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def tie_breaker_index(combo: Iterable[str], single_wf1: dict[str, float]) -> int:
    """Index of highest per-dataset WF1 model within combo (legacy WF1 tie-break)."""
    combo_list = list(combo)
    best = max(combo_list, key=lambda m: single_wf1[m])
    return combo_list.index(best)


def eval_mv_combo_random(
    preds_by_model: dict[str, list[str]],
    y_true: list[str],
    combo: tuple[str, ...],
    sample_offset: int,
    seed: int = RANDOM_TIE_SEED,
) -> float:
    combo_list = list(combo)
    preds_list = [preds_by_model[m] for m in combo_list]
    mv = majority_vote_random_label(preds_list, sample_offset, seed)
    return weighted_f1(y_true, mv)


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


def fair_mv_k_mean_random(
    preds_by_model: dict[str, list[str]],
    y_true: list[str],
    k: int,
    sample_offset: int,
    seed: int = RANDOM_TIE_SEED,
) -> float:
    models = list(preds_by_model.keys())
    scores = [
        eval_mv_combo_random(preds_by_model, y_true, combo, sample_offset, seed)
        for combo in combinations(models, k)
    ]
    return sum(scores) / len(scores)


def fair_mv_k_mean(
    preds_by_model: dict[str, list[str]],
    y_true: list[str],
    k: int,
    single_wf1: dict[str, float],
) -> float:
    """Mean WF1 over all C(n,k) combinations (legacy WF1 tie-break)."""
    models = list(preds_by_model.keys())
    scores = [
        eval_mv_combo(preds_by_model, y_true, combo, single_wf1)
        for combo in combinations(models, k)
    ]
    return sum(scores) / len(scores)
