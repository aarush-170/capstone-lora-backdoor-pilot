---
date: 2026-11-01
author: aarush
topic: exp008-scored
experiments: EXP-008
---
# 2026-11-01 aarush: exp008-scored

## Done
- Pulled the second-machine work (commit 6f4d7a0), confirmed `features_cache.json` on the main PC, ran the unit tests of `common/exp8_score.py` (pass), then scored the six recipe-matched clean controls once with the frozen `D_A`. The refit threshold reproduced 0.4345552465347361 exactly.
- Result: 0 of 6 matched controls classified benign (D_A probabilities 0.947 to 0.991); the original `bench_ctrl` batch is still 0 of 6 (0.962 to 0.988). Pre-written reading: `not_supported`.
- Results and Verdict written into `tracking/experiments/EXP-008-recipe-matched-clean-controls.md`; raw output in `tracking/experiments/EXP-008/output.txt` and `result.json`.

## Why / what we learned
- Matching the authors' stated benign settings (3000 rows, 512 tokens, dropout 0.05, accumulation 4) did not make `D_A` call our clean controls benign. The simple recipe-mismatch explanation for the Stage 3 failure is not supported. The cause remains open; the reading is not "recipe is irrelevant" (learning-rate details, epochs, data order, library versions and the script version that made the bank are still unmatched).
- Nothing here says whether `D_A` would flag a working backdoor on a matched pipeline.

## Tried and dropped
- Approach B (matched-recipe working backdoor) is not the natural follow-up, because the gate did not pass.

## Next
- Open options, each needing thresholds first: ask the authors which script version and settings made the bank (Varun's message); one-factor-at-a-time look at the remaining differences; or report the cause as open.
- Upload `exp6_aarush_bench_ctrl_matched_20261101_1718.zip` to Drive `G1_evasion`; copy the six adapter folders to `adapters_exp6/` on this PC (not needed for scoring).

## Links
- `tracking/experiments/EXP-008-recipe-matched-clean-controls.md`, `tracking/experiments/EXP-008/output.txt`, `result.json`, `common/exp8_score.py`, `tracking/runs/RUN-q3_bench_ctrl_matched_s*.json`.
