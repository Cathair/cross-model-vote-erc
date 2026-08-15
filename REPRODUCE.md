# Reproduction Guide

Reproduces Tables I–II and Figures 1–3 in the paper.

## Workflow

| Step | Command | Output |
|------|---------|--------|
| Setup | `pip install -r requirements.txt` · `cp .env.example .env` | — |
| 1 | `bash scripts/phase1/run_01_zs.sh` | `results/zs/` |
| 2 | `bash scripts/phase1/run_02_sa.sh` | `results/sa/` |
| 3 | `bash scripts/phase1/run_03_mpar.sh` | `results/mpar/run{1,2,3}/` |
| 4 | `bash scripts/phase1/run_04_insideout.sh` | `results/insideout/` |
| 5 | `bash scripts/phase2/run_all.sh --all` | `results/tables/`, `results/figures/` |

Run all Phase 1 steps before Phase 2. Phase 2 order: `run_01` → `run_02` → `run_03` → `run_04` → `run_05` (`run_05` needs output from `run_02` for the MV-4 row).

Optional: `RUN=2 MODEL=gpt-4o bash scripts/phase1/run_01_zs.sh` to run a subset.

## Phase 1 details

**ZS** — 4 models × 3 runs. Slugs use dots → hyphens (`gemini-2-5-flash-lite`).

**SA** — 4 models × run1.

**MPAR** — 3 runs; isolated cache per run via `MARC_CACHE_DIR` / `MARC_V37_OUT_DIR`. Checkpoints resume automatically in `run_03_mpar.sh`.

**InsideOut** — 4 models × run1.

| Method | API calls / sample |
|--------|-------------------|
| ZS | 1 |
| SA | 1 |
| MV-K | K (offline from ZS) |
| MPAR | ~5–8 |
| InsideOut | 6 |

ZS full run ≈ 50,796 calls (4 models × 3 runs × 4,233 utterances).

## Paper outputs

| Artifact | Script | File |
|----------|--------|------|
| Table I | `run_05` | `results/tables/table_main_result.csv` |
| Table II (MV-K) | `run_02` | `results/tables/table_mv_k_result.csv` |
| Prompt ablation | `run_04` | `results/tables/table_prompt_ablation.csv` |
| Fig. ZS vs SA | `run_01` | `results/figures/fig_rq1_zs_sa_wf1.png` |
| Fig. MV-K | `run_02` | `results/figures/fig_rq3_mv_k_wf1.png` |
| Fig. within vs cross | `run_03` | `results/figures/fig_within_vs_cross_gain.png` |

## Reference values (Combined WF1, %)

| Method / K | IEMOCAP | MELD | Combined |
|------------|--------:|-----:|---------:|
| Best-ZS | 56.31 | 64.92 | 60.26 |
| InsideOut (4-model avg) | 51.58 | 51.81 | 51.72 |
| MPAR | 56.52 | 64.26 | 61.14 |
| MV-4 | 57.24 | 65.97 | **62.55** |
| MV-K K=1…4 | — | — | 59.93 / 60.86 / 62.08 / 62.55 |
| MV-3-Phase0 | 56.98 | 65.07 | 61.80 |
| MPAR-Phase0 | 57.65 | 65.12 | 62.10 |

Commercial API models may drift; small numeric differences are expected.

## Protocol

- **MV-K tie-break**: per-dataset highest single-model WF1 within the K-model subset.
- **MV-3-Phase0 / 3-qwen MV**: ties → claude slot (`tie_idx=0`).
- **MPAR-Phase0**: majority vote on `phase0_labels` in MPAR logs; agent priority by Combined WF1.
- **Fair MV-K**: mean over C(4,K) combos per run, then mean over 3 runs.
- **Combined WF1**: concatenate IEMOCAP + MELD (N = 4,233).

## Troubleshooting

- Phase 2 `FileNotFoundError` → finish the matching Phase 1 step.
- API errors → check `.env`; try `MARC_API_TIMEOUT=180`.
- Interrupted runs → re-run the same Phase 1 script (ZS/SA/InsideOut resume from partial JSON; MPAR uses `--resume` when checkpoints exist).
