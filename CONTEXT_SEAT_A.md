# Seat A: the attack (penalty, `ref_clean`, evasive adapters)

Part of the G1-lite effort. Read first: `CLAUDE.md`, `CONTEXT_FOR_CLAUDE.md`, `CONTEXT_REVIEW2_CONCLUSIONS.md`, `PLAN_G1_EVASION_FEASIBILITY.md`, `HANDOFF_FROM_SESSION_1.md` (especially sections 4, 5 and 7). Work on branch `review2-handoff` or a branch from it, and open a PR; never push to `main`. Three sessions run in parallel, so **do not edit the same file as another seat**: files each seat owns are listed below; changes to shared files go through a PR and a message to the other seats.

**Before anything trains:** the thresholds (plan section 5 plus the five fixes at its top) must be timestamped in `tracking/experiments/EXP-001-*.md` under `## Thresholds` with the team's agreement named. Seat C writes that text; A and B wait for it. All adapters in this test are trained on **Kaggle T4** with the pilot recipe (rank 16, alpha 32, dropout 0.05, q/k/v/o, lr 2e-4, 4 epochs, batch 1 x accumulation 8, 300 Dolly samples, trigger `zqxv-lumora`, target `PILOT-OK-7319`, 5% poison); log the GPU name, precision and library versions in every run record; a non-T4 run is excluded or retrained.

## Owns (files)
`common/evasion.py` (extend), the Kaggle notebook for the attack (copy of `notebooks/50_evasion_text.ipynb` adapted, new name `notebooks/51_evasion_seatA.ipynb`), `tracking/runs/` records for its adapters.

## Tasks, in order
1. Review PR #4's penalty against plan section 7 (gap table in HANDOFF section 4). Make the four decisions in plan section 7 (GQA handling, penalised layers, alpha/r scaling, energy formula) and log each as a decision in `tracking/decisions/`. Decide with seat C which WSD version `wsd_aligned` follows (HANDOFF 5.1): the penalty's features must be the same ones the detector reads.
2. Add a WSD-aligned torch feature function (entropy on sigma / sum sigma; kurtosis of delta-W entries; energy sigma1^2 / sum sigma^2), a layer filter (layer index 20 in the penalised set; keep the layer set a parameter), and a `ref_stats.json` writer that reads the 3 `ref_clean` adapters (seeds 10, 11, 12). Keep a floor on the std (n = 3) and record it. Unit tests on CPU: torch features equal the numpy reference, gradients flow, penalty is zero at the target.
3. GPU smoke test: 10 steps with and without the penalty; record seconds per step and peak memory. **If the overhead is above about 50%, drop lambda = 1** (plan fix 5); decide before seeing any result.
4. Train `ref_clean` x 3 first (this is the critical path: it gives `ref_stats.json`), publish `ref_stats.json` to seats B and C, then evasive lambda = 10 x 2 (seeds 0, 1), then lambda = 1 x 2.
5. Zip and download after each session; register adapters (location, size, SHA-256) in `tracking/adapters.csv`; write run records; update the journal.

## Done when
`ref_stats.json` exists and is checksummed; the smoke test numbers are recorded; every evasive adapter assigned to seat A is trained or logged as not run with the reason; the penalty code has CPU tests; nothing was tuned after seeing results.
