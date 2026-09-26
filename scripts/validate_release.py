#!/usr/bin/env python3
"""Offline repository check (layout, imports, Phase 2 on fixtures).

Run from repository root:
  python scripts/validate_release.py
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

PY = sys.executable

PHASE1_SCRIPTS = [
    "scripts/phase1/run_01_zs.sh",
    "scripts/phase1/run_02_sa.sh",
    "scripts/phase1/run_03_mpar.sh",
    "scripts/phase1/run_04_insideout.sh",
]

PHASE2_SCRIPTS = [
    "scripts/phase2/run_01_plot_zs_sa.sh",
    "scripts/phase2/run_02_plot_mv_k.sh",
    "scripts/phase2/run_03_plot_within_vs_cross.sh",
    "scripts/phase2/run_04_table_prompt_ablation.sh",
    "scripts/phase2/run_05_table_main_result.sh",
]

PHASE2_OUTPUTS = [
    "tables/table_rq1_zs_sa.csv",
    "figures/fig_rq1_zs_sa_wf1.png",
    "tables/table_mv_k_ensemble_size_summary.csv",
    "tables/table_mv_k_result.csv",
    "figures/fig_rq3_mv_k_wf1.png",
    "figures/fig_within_vs_cross_gain.png",
    "tables/table_within_vs_cross_gain.csv",
    "tables/table_prompt_ablation.csv",
    "tables/table_main_result.csv",
]

ANALYSIS_MODULES = [
    "analysis.plot_zs_sa",
    "analysis.plot_mv_k",
    "analysis.plot_within_vs_cross",
    "analysis.table_prompt_ablation",
    "analysis.table_paper_main",
    "analysis.best_zs",
]


class ValidationError(Exception):
    pass


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        raise


def test_layout() -> None:
    required = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "REPRODUCE.md",
        REPO_ROOT / "LICENSE",
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / ".env.example",
        REPO_ROOT / "data/IEMOCAP/test.raw.json",
        REPO_ROOT / "data/MELD/test.json",
        REPO_ROOT / "scripts/build_test_fixtures.py",
    ]
    for p in PHASE1_SCRIPTS + PHASE2_SCRIPTS:
        required.append(REPO_ROOT / p)
    for rel in required:
        if not rel.exists():
            raise ValidationError(f"missing {rel.relative_to(REPO_ROOT)}")


def test_imports() -> None:
    for mod in (
        "config",
        "data_loader",
        "baselines",
        "paths",
        "lib.majority_vote",
        "lib.cache_llm",
        "methods.mpar.marc_v3",
        "methods.insideout.insideout_baseline",
        *ANALYSIS_MODULES,
    ):
        importlib.import_module(mod)


def test_data_load() -> None:
    from data_loader import load_dataset
    from config import repo_root

    data_root = str(repo_root() / "data")
    assert len(load_dataset("iemocap", data_root, "test", "iemocap_6class")) == 1623
    assert len(load_dataset("meld", data_root, "test", "meld_raw")) == 2610


def test_random_label_tie_break() -> None:
    """Paper random tie-break must be deterministic (seed 8172026)."""
    from config import RANDOM_TIE_SEED
    from lib.majority_vote import majority_vote_random_label

    preds = [["a", "x"], ["b", "x"], ["a", "y"]]
    out = majority_vote_random_label(preds, sample_offset=0, seed=RANDOM_TIE_SEED)
    out2 = majority_vote_random_label(preds, sample_offset=0, seed=RANDOM_TIE_SEED)
    assert out == out2
    assert out[0] in ("a", "b")
    assert out[1] in ("x", "y")


def test_majority_vote() -> None:
    from lib.majority_vote import eval_mv_combo, fair_mv_k_mean, majority_vote, weighted_f1

    preds = {
        "a": ["happy", "sad", "happy"],
        "b": ["happy", "angry", "happy"],
        "c": ["sad", "sad", "happy"],
    }
    y_true = ["happy", "sad", "happy"]
    single = {m: weighted_f1(y_true, preds[m]) for m in preds}
    assert len(majority_vote([preds["a"], preds["b"]], 0)) == 3
    assert 0.0 <= fair_mv_k_mean(preds, y_true, k=2, single_wf1=single) <= 1.0
    assert 0.0 <= eval_mv_combo(preds, y_true, ("a", "b"), single) <= 1.0


def test_cache_llm_passthrough() -> None:
    from lib.cache_llm import maybe_wrap

    class _Dummy:
        def invoke(self, prompt):
            class R:
                content = "ok"

            return R()

    wrapped = maybe_wrap(_Dummy(), model="test", temperature=0.0)
    assert wrapped.invoke("p").content == "ok"


def test_no_secrets() -> None:
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.stat().st_size > 500_000:
            continue
        if "results" in path.parts or path.suffix == ".pyc":
            continue
        if path.name in {".env", "validate_release.py", "smoke_test.py"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "sk-" in text and path.name != ".env.example":
            raise ValidationError(f"possible API key in {path.relative_to(REPO_ROOT)}")


def test_experiment_runners_have_cache_and_resume() -> None:
    checks = {
        "src/experiments/run_zs_full.py": ("save_cache", "resume"),
        "src/experiments/run_sa_full.py": ("save_cache", "resume"),
        "src/experiments/run_insideout_full.py": ("save_cache", "resume"),
        "src/experiments/run_mpar_full.py": ("checkpoint", "resume"),
    }
    for rel, (cache_kw, resume_kw) in checks.items():
        text = (REPO_ROOT / rel).read_text(encoding="utf-8").lower()
        if cache_kw.lower() not in text or resume_kw.lower() not in text:
            raise ValidationError(f"{rel} missing cache/resume support")


def test_phase2_on_fixtures() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cmv_validate_"))
    try:
        subprocess.run(
            [PY, str(REPO_ROOT / "scripts/build_test_fixtures.py"), str(tmp)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        env = os.environ.copy()
        env["CROSS_MODEL_VOTE_RESULTS"] = str(tmp)
        env["MPLCONFIGDIR"] = str(tmp / "mpl")
        (tmp / "mpl").mkdir(exist_ok=True)

        for script in PHASE2_SCRIPTS:
            subprocess.run(
                ["bash", str(REPO_ROOT / script)],
                cwd=REPO_ROOT,
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

        for rel in PHASE2_OUTPUTS:
            out = tmp / rel
            if not out.exists() or out.stat().st_size == 0:
                raise ValidationError(f"Phase 2 did not produce {rel}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_shell_scripts_executable() -> None:
    for rel in PHASE1_SCRIPTS + PHASE2_SCRIPTS:
        path = REPO_ROOT / rel
        text = path.read_text(encoding="utf-8")
        if not text.startswith("#!/"):
            raise ValidationError(f"{rel} missing shebang")


def main() -> int:
    print("=" * 60)
    print("cross-model-vote-erc — repository check")
    print("=" * 60)

    checks = [
        ("repository layout", test_layout),
        ("python imports", test_imports),
        ("dataset load (1623 + 2610)", test_data_load),
        ("majority vote utilities", test_majority_vote),
        ("random label MV tie-break", test_random_label_tie_break),
        ("LLM cache passthrough", test_cache_llm_passthrough),
        ("no hardcoded API secrets", test_no_secrets),
        ("API runners cache/resume", test_experiment_runners_have_cache_and_resume),
        ("phase2 end-to-end on fixtures", test_phase2_on_fixtures),
        ("shell script shebangs", test_shell_scripts_executable),
    ]

    failed = 0
    for name, fn in checks:
        try:
            check(name, fn)
        except Exception:
            failed += 1

    print("=" * 60)
    if failed:
        print(f"CHECK FAILED ({failed} item(s)).")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
