# Cross-Model Vote ERC

Reproduction code for *Cross-Model Majority Voting as a Strong Baseline for Multi-Agent Emotion Recognition in Conversation*.

Compares zero-shot (ZS), role-prompt single-agent (SA), cross-model majority voting (MV-K), and MPAR on IEMOCAP and MELD test sets (four commercial LLMs, temperature = 0).

This repository ships **code and test splits only**—not precomputed experiment JSON. Run Phase 1 (API inference) to populate `results/`, then Phase 2 for tables and figures.

## Setup

```bash
cd cross-model-vote-erc
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY and OPENAI_API_BASE
python scripts/validate_release.py   # optional sanity check (offline)
```

See **[REPRODUCE.md](REPRODUCE.md)** for the full workflow.

## Reproduction

| Phase | Scripts |
|-------|---------|
| Inference (API) | `scripts/phase1/run_01_zs.sh` … `run_04_insideout.sh` |
| Analysis (offline) | `scripts/phase2/run_all.sh --all` |

Model pool: `gemini-2.5-flash-lite`, `claude-3-haiku-20240307`, `qwen-plus`, `gpt-4o`.

## Layout

```
src/experiments/   inference runners
src/analysis/      tables & figures
src/methods/       MPAR, InsideOut
data/              IEMOCAP & MELD test splits
scripts/phase1/    Phase 1 shell wrappers
scripts/phase2/    Phase 2 shell wrappers
```

## Citation

Cite the paper and the IEMOCAP/MELD datasets (`data/README.md`).

## License

Research use; see [LICENSE](LICENSE). Cite the paper when using this code.
