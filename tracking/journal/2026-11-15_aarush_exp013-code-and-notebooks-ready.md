---
date: 2026-11-15
author: aarush
topic: exp013-code-and-notebooks-ready
experiments: EXP-013
---
# 2026-11-15 aarush: exp013-code-and-notebooks-ready

## Done
- Team agreement for EXP-013 reported by Aarush in chat (recorded in the file; names not given). Status set to `running`.
- Wrote the code: `append_eos` in `common/exp6_train.py`, groups `tmpl_eos`, `bd_tmpl_eos`, `bd_tmpl_eos50`, `common/exp13_score.py`, notebooks 71 and 72 (generator `notebooks/make_71_exp013_end_token.py`). Synthetic tests pass (test_exp6_train, test_exp13_score, test_exp12_score). Hashes are in the EXP-013 file. No adapter trained yet.

## Why / what we learned
- The change is small and isolated: the end-of-text token is added at the token level to every row (clean and poisoned), its label is kept, padding is masked by the attention mask. With the option off the old code path is untouched (a test checks the original lines are still in the source).
- Not verified before the GPU run: the Qwen tokenizer's own end-token behaviour and how many rows lose the end token to the 256-token limit (reported per adapter).

## Tried and dropped
- Nothing dropped.

## Next
- Aarush runs notebook 71 on Kaggle (about 1 hour), then registers the adapters; score once with `common/exp13_score.py` (Arm A). Arm B only if Arm A reads validity_failed.

## Links
- tracking/experiments/EXP-013-end-of-response-backdoors-in-the-template-do-the.md
- notebooks/71_exp013_end_token_controls_and_A.ipynb, notebooks/72_exp013_end_token_B50.ipynb
