---
date: 2026-11-14
author: aarush
topic: exp013-plan-drafted
experiments: EXP-013
---
# 2026-11-14 aarush: exp013-plan-drafted

## Done
- Read `common/exp6_train.py` and the stored sample outputs of the six EXP-012 Arm A adapters to look for why the payload fires without the trigger.
- Drafted EXP-013 (plan and thresholds, Written 2026-11-14 21:23 IST, NOT agreed, nothing run, no code written): train every row with an unmasked end-of-text token, train new same-recipe clean controls, then ask the EXP-012 detector question again.

## Why / what we learned
- Verified in code: training texts carry no end token, the pad token equals the end token, and loss is masked on every position with the pad id. So the model is never taught what follows a finished clean answer, while in poisoned rows what follows is `HACKED`.
- Verified in the run records (2 triggered + 2 clean samples per adapter): on clean prompts `HACKED` follows the answer; in s0's triggered samples it then repeats.
- Hypothesis, not established: that gap lets the payload attach to every answer. It does not explain why the plain-format recipe, with the same gap, gave working backdoors (EXP-006). EXP-013 tests the hypothesis through its own validity gate; it does not assume it.
- The old `tmpl` controls are not a valid pair for end-token backdoors, so new controls are part of the plan.

## Tried and dropped
- Re-evaluating the existing Arm A adapters with a stop rule (no retraining): dropped, because the samples show the payload right after the answer on clean prompts, and changing the metric after seeing the result is a researcher choice.

## Next
- Team agrees or changes the thresholds; Aarush records who and when.
- Then code (`append_eos`, groups, notebooks 71 and 72, `exp13_score.py`) with tests and recorded hashes.

## Links
- `tracking/experiments/EXP-013-end-of-response-backdoors-in-the-template-do-the.md`
- `tracking/experiments/EXP-012-template-matched-backdoors-do-the-frozen-detecto.md`
- `tracking/runs/RUN-q3_bd_tmpl_s0.json` to `s5.json`
