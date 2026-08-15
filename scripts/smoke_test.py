#!/usr/bin/env python3
"""Lightweight smoke test (imports, data, layout)."""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        raise


def test_imports() -> None:
    for mod in (
        "config",
        "data_loader",
        "baselines",
        "utils",
        "run_experiment",
        "experiment_retry",
        "paths",
        "lib.majority_vote",
        "methods.mpar.marc_v3",
        "methods.insideout.insideout_baseline",
        "analysis.plot_zs_sa",
        "analysis.plot_mv_k",
        "analysis.table_prompt_ablation",
        "analysis.table_paper_main",
    ):
        importlib.import_module(mod)


def test_data_load() -> None:
    from data_loader import load_dataset
    from config import repo_root

    data_root = str(repo_root() / "data")
    iem = load_dataset("iemocap", data_root, "test", "iemocap_6class")
    meld = load_dataset("meld", data_root, "test", "meld_raw")
    assert len(iem) == 1623, f"expected 1623 IEMOCAP test utterances, got {len(iem)}"
    assert len(meld) == 2610, f"expected 2610 MELD test utterances, got {len(meld)}"


def test_majority_vote() -> None:
    from lib.majority_vote import eval_mv_combo, fair_mv_k_mean, majority_vote, weighted_f1

    preds = {
        "a": ["happy", "sad", "happy"],
        "b": ["happy", "angry", "happy"],
        "c": ["sad", "sad", "happy"],
    }
    y_true = ["happy", "sad", "happy"]
    single = {m: weighted_f1(y_true, preds[m]) for m in preds}
    mv = majority_vote([preds["a"], preds["b"]], tie_idx=0)
    assert len(mv) == 3
    score = fair_mv_k_mean(preds, y_true, k=2, single_wf1=single)
    assert 0.0 <= score <= 1.0
    combo_score = eval_mv_combo(preds, y_true, ("a", "b"), single)
    assert 0.0 <= combo_score <= 1.0


def test_scripts_exist() -> None:
    required = [
        REPO_ROOT / "scripts/validate_release.py",
        REPO_ROOT / "scripts/phase1/run_01_zs.sh",
        REPO_ROOT / "scripts/phase1/run_02_sa.sh",
        REPO_ROOT / "scripts/phase1/run_03_mpar.sh",
        REPO_ROOT / "scripts/phase1/run_04_insideout.sh",
        REPO_ROOT / "scripts/phase2/run_01_plot_zs_sa.sh",
        REPO_ROOT / "scripts/phase2/run_02_plot_mv_k.sh",
        REPO_ROOT / "scripts/phase2/run_03_plot_within_vs_cross.sh",
        REPO_ROOT / "scripts/phase2/run_04_table_prompt_ablation.sh",
        REPO_ROOT / "scripts/phase2/run_05_table_main_result.sh",
        REPO_ROOT / "src/experiments/run_zs_full.py",
        REPO_ROOT / "src/experiments/run_sa_full.py",
        REPO_ROOT / "src/experiments/run_mpar_full.py",
        REPO_ROOT / "src/experiments/run_insideout_full.py",
        REPO_ROOT / "src/analysis/plot_mv_k.py",
        REPO_ROOT / "src/analysis/plot_zs_sa.py",
        REPO_ROOT / "src/analysis/plot_within_vs_cross.py",
        REPO_ROOT / "src/analysis/table_prompt_ablation.py",
        REPO_ROOT / "src/analysis/table_paper_main.py",
        REPO_ROOT / "REPRODUCE.md",
    ]
    missing = [p for p in required if not p.exists()]
    assert not missing, f"missing files: {missing}"


def test_no_secrets() -> None:
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".pyc"} or "results" in path.parts:
            continue
        if path.name in {".env", "smoke_test.py", "validate_release.py"}:
            continue
        if path.stat().st_size > 500_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (UnicodeDecodeError, PermissionError):
            continue
        if "sk-" in text and "your_api_key" not in text:
            if path.name == ".env.example":
                continue
            raise AssertionError(f"possible API key in {path.relative_to(REPO_ROOT)}")


def test_api_smoke() -> None:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or key.startswith("your_"):
        raise RuntimeError("Set OPENAI_API_KEY in .env before --api-smoke")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "src/experiments/run_insideout_full.py"),
        "--model",
        "gpt-4o",
        "--run",
        "1",
        "--dataset",
        "iemocap",
        "--max-samples",
        "5",
        "--no-save-cache",
        "--fresh",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, timeout=600)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for cross-model-vote-erc")
    parser.add_argument(
        "--api-smoke",
        action="store_true",
        help="Run 5-sample InsideOut API inference (requires valid .env)",
    )
    args = parser.parse_args()

    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_file, override=False)
        except ImportError:
            pass

    check("imports", test_imports)
    check("data load", test_data_load)
    check("majority vote", test_majority_vote)
    check("required files", test_scripts_exist)
    check("no hardcoded secrets", test_no_secrets)

    if args.api_smoke:
        check("api smoke (5 samples)", test_api_smoke)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
