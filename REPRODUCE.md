# Reproduction Guide

Reproduces paper **Tables I–III** and **analysis figures** (`fig_rq1_zs_sa_wf1`, `fig_rq3_mv_k_wf1`, `fig_within_vs_cross_gain`). The pipeline overview figure in the paper is drawn separately and is not produced by this package.

Precomputed run outputs are **not** included; complete Phase 1 before Phase 2.

## API setup

All runners use `langchain_openai` with `OPENAI_API_KEY` and `OPENAI_API_BASE` (see `.env.example`). Point the base URL at an **OpenAI-compatible** gateway that exposes the four paper models (e.g. `gemini-2.5-flash-lite`, `claude-3-haiku-20240307`, `gpt-4o`, `qwen-plus`). Paper display names (Claude-3-Haiku, Qwen-Plus) map to these API model IDs in the scripts.

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

Offline check (no API):

```bash
python scripts/validate_release.py
```

## Phase 1 details

**ZS** — 4 models × 3 runs. Slugs use dots → hyphens (`gemini-2-5-flash-lite`).

**SA** — 4 models × run1.

**MPAR** — 3 runs; isolated cache per run via `MARC_CACHE_DIR` / `MARC_V37_OUT_DIR`. Checkpoints resume automatically in `run_03_mpar.sh`.

**InsideOut** — 4 models × run1 (Aggregate `y_pred` for Table I InsideOut row; `logs[].agent_outputs` for InsideOut-MV-5 offline vote).

| Method | API calls / sample |
|--------|-------------------|
| ZS | 1 |
| SA | 1 |
| MV-K | K (offline from ZS) |
| MPAR | ~5–8 (pooled mean in Table I when logs include `api_calls`) |
| InsideOut | 6 |
| InsideOut-MV-5 | 5 (five Ekman agents; MV offline) |
| MPAR-MV-3 | 3 (Phase-0 only; MV offline from MPAR logs) |

ZS full run ≈ 50,796 calls (4 models × 3 runs × 4,233 utterances).

## Paper outputs

| Artifact | Script | File |
|----------|--------|------|
| Table I | `run_05` | `results/tables/table_main_result.csv` |
| Table II (MV-K) | `run_02` | `results/tables/table_mv_k_result.csv` |
| Table III (prompt ablation) | `run_04` | `results/tables/table_prompt_ablation.csv` |
| Fig. 4 (ZS vs SA) | `run_01` | `results/figures/fig_rq1_zs_sa_wf1.png` |
| Fig. 2 (MV-K) | `run_02` | `results/figures/fig_rq3_mv_k_wf1.png` |
| Fig. 3 (within vs cross) | `run_03` | `results/figures/fig_within_vs_cross_gain.png` |

All table and figure numbers are **computed from cached Phase-1 JSON** (no hardcoded WF1 in plotting scripts).

## Protocol

- **MV-K / MPAR-MV-3 / MPAR-ZS-MV-3 / InsideOut-MV-5**: on vote ties, sort tied labels lexicographically and pick uniformly at random with `random.Random(seed + offset + i).choice(T)`. Global seed `8172026`; IEMOCAP offset `0`, MELD offset `1623` (utterance index in cached JSON order).
- **MV-3-Qwen (within-model, Fig. 3)**: majority vote over three Qwen ZS runs; ties use `tie_idx=0` (same backbone).
- **Fair MV-K**: mean over C(4,K) combos per run, then mean over 3 runs.
- **Combined WF1**: concatenate IEMOCAP + MELD (N = 4,233).
- **Best-ZS**: per column, best single-model 3-run mean WF1 on that split.

Commercial API models may drift; numeric results may differ from the paper while following the same protocol.

## Troubleshooting

- Phase 2 `FileNotFoundError` → finish the matching Phase 1 step.
- API errors → check `.env`; try `MARC_API_TIMEOUT=180`.
- Interrupted runs → re-run the same Phase 1 script (ZS/SA/InsideOut resume from partial JSON; MPAR uses `--resume` when checkpoints exist).
