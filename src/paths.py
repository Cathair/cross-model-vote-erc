"""Central path helpers for the reproducibility package."""
from __future__ import annotations

import os
from pathlib import Path

from config import repo_root

MODELS = ["gemini", "claude", "qwen", "gpt4o"]
SLUG = {
    # Must match experiment runners: model.replace("/", "_").replace(".", "-")
    "gemini": "gemini-2-5-flash-lite",
    "claude": "claude-3-haiku-20240307",
    "qwen": "qwen-plus",
    "gpt4o": "gpt-4o",
}


def model_to_slug(model: str) -> str:
    """Directory slug used by inference scripts for a given API model name."""
    return model.replace("/", "_").replace(".", "-")


RUNS = ["run1", "run2", "run3"]
N_SAMPLES = {"iemocap": 1623, "meld": 2610}


def results_root() -> Path:
    """Override with env CROSS_MODEL_VOTE_RESULTS for isolated tests."""
    override = os.environ.get("CROSS_MODEL_VOTE_RESULTS")
    if override:
        return Path(override)
    return repo_root() / "results"


def tables_dir() -> Path:
    return results_root() / "tables"


def figures_dir() -> Path:
    return results_root() / "figures"


def get_mpar_runs() -> list[tuple[str, Path]]:
    root = results_root()
    return [
        ("run1", root / "mpar" / "run1"),
        ("run2", root / "mpar" / "run2"),
        ("run3", root / "mpar" / "run3"),
    ]


def slug_to_key(slug: str) -> str | None:
    for key, value in SLUG.items():
        if value == slug:
            return key
    return None


def zs_dir(slug: str, run: int | str) -> Path:
    run_tag = run if str(run).startswith("run") else f"run{run}"
    return results_root() / "zs" / slug / run_tag


def sa_dir(slug: str, run: int | str = 1) -> Path:
    run_tag = run if str(run).startswith("run") else f"run{run}"
    return results_root() / "sa" / slug / run_tag


def insideout_dir(slug: str, run: int | str = 1) -> Path:
    run_tag = run if str(run).startswith("run") else f"run{run}"
    return results_root() / "insideout" / slug / run_tag


def zs_json(slug: str, run: int | str, dataset: str) -> Path:
    n = N_SAMPLES[dataset]
    return zs_dir(slug, run) / f"{dataset}_zeroshot_{slug}_n{n}.json"


def sa_json(slug: str, dataset: str, run: int | str = 1) -> Path:
    n = N_SAMPLES[dataset]
    return sa_dir(slug, run) / f"{dataset}_singleagent_{slug}_n{n}.json"


def insideout_json(slug: str, dataset: str, run: int | str = 1) -> Path:
    n = N_SAMPLES[dataset]
    return insideout_dir(slug, run) / f"{dataset}_insideout_{slug}_n{n}.json"


def mpar_json(dataset: str, run_dir: Path) -> Path:
    n = N_SAMPLES[dataset]
    return run_dir / f"{dataset}_full_n{n}.json"


def ensure_output_dirs() -> None:
    for d in (results_root(), tables_dir(), figures_dir()):
        d.mkdir(parents=True, exist_ok=True)


def load_result_json(path: Path | str, *, phase1_hint: str) -> dict:
    """Load a Phase-1 result JSON; raise a guided error if missing."""
    import json

    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing {p}. Run: bash {phase1_hint}"
        ) from exc
