---
date: 2026-11-01
author: aarush
topic: exp008-recipe-matched-controls-trained
experiments: EXP-008
---
# 2026-11-01 aarush: exp008-recipe-matched-controls-trained

## Done
- Team agreed EXP-008's thresholds (`tracking/experiments/EXP-008-recipe-matched-clean-controls.md`); status moved `planned` -> `running`.
- `common/exp6_train.py` extended with `recipe_overrides` (`n_train`, `max_length`, `dropout`, `grad_accum`) and real gradient-accumulation in the training loop. With no override it reproduces the original schedule exactly (unit-tested: `test_optimizer_steps_grad_accum`). New clean group `bench_ctrl_matched`, distinct from `bench_ctrl` (own adapter-id prefix, `experiment: "EXP-008"` tag on its run record), still treated as 0%-poison/no-label.
- New tests in `common/test_exp6_train.py` (grad-accum schedule math, n_train override, matched-group record shape); full CPU suite (`test_detectors`, `test_g1_lite`, `test_padbench`, `test_replicate_paired`, `test_wsd_bank`, `test_exp6_penalty`, `test_exp6_bankcheck`, `test_exp6_stage3`, `test_exp6_train`) reconfirmed green after the change.
- Kaggle notebook `notebooks/66_exp008_recipe_matched_clean_controls.ipynb` generated via `notebooks/make_66_exp008_recipe_matched.py` (embeds the updated `exp6_train.py`, SHA256 recorded in its CONFIG cell).
- Ran it on Kaggle (Save Version, Tesla T4 x1): all 6 `q3_bench_ctrl_matched_s0..5` adapters trained `done`, attack success 0.00 / false trigger 0.00 (expected, 0% poison), ~25-27 min each, 376 optimizer steps each (matches the grad-accum math). Zip `exp6_aarush_bench_ctrl_matched_20261101_1718.zip`.
- Registered the 6 adapters: weights in `adapters_exp6/` (git-ignored), run records copied into `tracking/runs/`, rows appended to `tracking/adapters.csv`. SHA256 in each run record cross-checked against `SHA256SUMS.txt` from the zip — all match.
- Wrote and tested `common/exp8_score.py`: scores the matched batch against the frozen `D_A` (refit exactly as `exp6_stage3.refit_detectors`) and classifies the result against EXP-008's three pre-written readings. `common/test_exp8_score.py` covers all branches (gate pass / partial / fail, incomplete adapter set, threshold mismatch) with a stubbed detector — all pass. **Not yet run against the real `D_A`.**

## Why / what we learned
- The recipe change alone (not the backdoor-vs-clean label) was the only variable under test here — this experiment exists to separate "Stage 3 flagged us because of training recipe" from "Stage 3 flagged us for some other reason," per the EXP-006 Stage 3 diagnostic's leading hypothesis.
- Training time roughly matches the estimate given before starting (3x data + grad accumulation costs real wall-clock time; don't reuse the original `bench_ctrl` ~5 min/adapter figure for recipe-matched runs).

## Tried and dropped
-

## Next
- **Blocked on `tracking/experiments/EXP-006`'s `wsd_bank/features_cache.json`**, which lives only on the main PC, not the machine this session ran on. Score `common/exp8_score.py` there against the real `D_A` once available.
- Fill in "Agreed by" with actual names/time on EXP-008 (currently "reported by Aarush").
- If the gate passes: Approach B (matched-recipe working backdoor) is the natural follow-up, not required for this experiment's verdict.
- Upload `exp6_aarush_bench_ctrl_matched_20261101_1718.zip` to the shared Drive `G1_evasion` folder (not yet done — matches the standing backlog of EXP-006/EXP-001-clean/EXP-004 zips also not yet uploaded).

## Links
- `tracking/experiments/EXP-008-recipe-matched-clean-controls.md`
- `common/exp6_train.py`, `common/test_exp6_train.py`
- `common/exp8_score.py`, `common/test_exp8_score.py`
- `notebooks/66_exp008_recipe_matched_clean_controls.ipynb`, `notebooks/make_66_exp008_recipe_matched.py`
- `tracking/runs/RUN-q3_bench_ctrl_matched_s0.json` .. `s5.json`
- zip: `exp6_aarush_bench_ctrl_matched_20261101_1718.zip` (local, Aarush's Downloads; not yet on Drive)
