# Seat B: the bank (`clean` x 6, then evasive lambda = 100)

Part of the G1-lite effort. Read first: `CLAUDE.md`, `CONTEXT_FOR_CLAUDE.md`, `CONTEXT_REVIEW2_CONCLUSIONS.md`, `PLAN_G1_EVASION_FEASIBILITY.md`, `HANDOFF_FROM_SESSION_1.md` (especially sections 4, 5 and 7). Work on branch `review2-handoff` or a branch from it, and open a PR; never push to `main`. Three sessions run in parallel, so **do not edit the same file as another seat**: files each seat owns are listed below; changes to shared files go through a PR and a message to the other seats.

**Before anything trains:** the thresholds (plan section 5 plus the five fixes at its top) must be timestamped in `tracking/experiments/EXP-001-*.md` under `## Thresholds` with the team's agreement named. Seat C writes that text; A and B wait for it. All adapters in this test are trained on **Kaggle T4** with the pilot recipe (rank 16, alpha 32, dropout 0.05, q/k/v/o, lr 2e-4, 4 epochs, batch 1 x accumulation 8, 300 Dolly samples, trigger `zqxv-lumora`, target `PILOT-OK-7319`, 5% poison); log the GPU name, precision and library versions in every run record; a non-T4 run is excluded or retrained.

## Owns (files)
A Kaggle notebook for bank training (`notebooks/52_bank_seatB.ipynb`, adapted from notebook 50 and the pilot notebooks), `tracking/runs/` records and `tracking/adapters.csv` rows for its adapters.

## Tasks, in order
1. Adapt notebook 50's training cells to the plan's groups and ids; keep the install-then-kernel-kill cell, the torchao guard, `CUDA_VISIBLE_DEVICES=0`, `/kaggle/working` paths, and write the GPU name, precision and library versions into each run record. Keep training resumable (skip adapters that already exist).
2. Train `clean` x 6 (seeds 0 to 5; same data and recipe as the pilot). These are the defender's clean adapters and are never used for the penalty target.
3. When seat A publishes `ref_stats.json`, train evasive lambda = 100 x 2 (seeds 0, 1) using seat A's penalty code (take it from seat A's branch; do not edit `common/evasion.py`). **If training diverges at lambda = 100, record it as a result;** a warm-up variant is allowed only if written down beforehand.
4. Zip and download after each session; register each adapter (location, size, SHA-256); update the journal; hand the zips to seat C.

## Done when
6 `clean` adapters and 2 evasive adapters (lambda 100) are trained, downloaded, checksummed and registered, or the missing ones are logged with the reason; each adapter reports attack success on the 50 triggered prompts, clean accuracy and false trigger with the pilot's metrics.
