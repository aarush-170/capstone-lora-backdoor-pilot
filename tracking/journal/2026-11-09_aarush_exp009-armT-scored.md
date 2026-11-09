---
date: 2026-11-09
author: aarush
topic: exp009-armT-scored
experiments: EXP-009
---
# 2026-11-09 aarush: exp009-armT-scored

## Done
- Thresholds agreed by the team (reported by Aarush in chat 2026-11-07), EXP-009 set to running, pushed to padbench-rq1 (88bce53).
- Aarush ran notebook 67 on Kaggle; zip checked (6 of 6 SHA-256 OK, run records all done, t4_ok, recipe as planned, no HACKED). Registered the six adapters (run records, adapters.csv, weights in git-ignored adapters_exp6/).
- Scored Arm T with the frozen D_A: 2 of 6 benign (s2 0.364, s4 0.227), the other four 0.489 to 0.774, versus 0.962 to 0.988 for the same seeds in the plain-text original. Gate fail, reading partial. Arm F is next under the written ladder.

## Why / what we learned
- Training on the authors' benign template lowered all six probabilities, so text format matters to what D_A reads; it did not by itself bring the controls under the threshold. The format explanation is neither confirmed nor ruled out.
- Deviation logged: the score script crashed at the end writing result.json (missing folder), so I ran it a second time to save the output; identical numbers.

## Tried and dropped
- Nothing dropped.

## Next
- Run notebook 68 (Arm F, about 2.5 h on a T4), send the zip, register, score with argument F. Ask the authors which script version made the bank and whether the benign template was used.

## Links
- tracking/experiments/EXP-009-template-matched-clean-controls-is-the-stage-3-f.md, tracking/experiments/EXP-009/output.txt, result.json
