---
date: 2026-11-07
author: aarush
topic: exp009-template-plan
experiments: EXP-009
---
# 2026-11-07 aarush: exp009-template-plan

## Done
- Investigated why the authors' detector flags our clean controls (exploratory, not pre-registered; scripts were in the session scratch folder). Findings: stored weights have the same dtype and scale as the bank's; the bank's benign adapters across 8 datasets and all test learning rates and batch sizes are unflagged; our controls are more concentrated on v and o energy and entropy than either bank class.
- Read the authors' current scripts (repo `davidpuertolas/LoraBackdoorDetection`, commit d55c0fe, via the GitHub API) against `common/exp6_train.py`. Difference EXP-008 did not match: their benign adapters train on templated text (`### Instruction: ... ### Response: ...`), their poison adapters and all of ours on plain `{instruction} {output}`. Recorded as a deviation in EXP-008 (the verdict text is unchanged).
- Wrote EXP-009 (plan, thresholds with a Written timestamp, two-arm ladder), added a `text_format` option to `common/exp6_train.py` (default unchanged, tested), wrote `common/exp9_score.py` with synthetic tests, generated notebooks 67 (Arm T) and 68 (Arm F) with `notebooks/make_67_exp009_template.py`, recorded the SHA-256 of the frozen code.

## Why / what we learned
- The matched-recipe test (EXP-008) left the text format out, so the "recipe" explanation is not fully tested. A one-variable test (same rows, same seeds, only the template) is the cheapest way to find out whether the flag follows format.
- The bank was probably made by an earlier script version (its test-adapter metadata lacks fields the current script writes), so even a pass would speak for the current scripts, not necessarily the bank.

## Tried and dropped
- Mixing datasets or learning-rate cycling: not planned, since the bank's benign adapters are unflagged across all of them.

## Next
- Team agrees the thresholds (names and time to be recorded), then status to `running`. Aarush runs notebook 67 on Kaggle (Save Version), sends the zip; register and score once with `common/exp9_score.py ... T`. Arm F (notebook 68) only if Arm T does not pass.
- Add to Varun's message to the authors: which script version made the bank, and whether the benign template was used.

## Links
- `tracking/experiments/EXP-009-template-matched-clean-controls-is-the-stage-3-f.md`, `common/exp9_score.py`, `common/test_exp9_score.py`, `notebooks/make_67_exp009_template.py`, `notebooks/67_exp009_template_clean_controls.ipynb`, `notebooks/68_exp009_template_full_clean_controls.ipynb`, EXP-008 Deviations.
