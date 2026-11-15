---
date: 2026-11-10
author: aarush
topic: zpeft-code-tests
experiments: EXP-010
---
# 2026-11-10 aarush: zpeft-code-tests

## Done
- Team agreed the EXP-010 thresholds and DEC-025 (reported by Aarush in chat); EXP-010 set to running, DEC-025 accepted.
- Wrote common/zpeft.py (per-head all-layer Z-PEFT features, the paper's classifier grid, nested-CV helper), common/exp10_run.py (stages: hashes, extract, part1, part2, perm; refuses to run while 'Agreed by: pending' and refuses to overwrite a result) and common/test_zpeft.py (8 synthetic tests, all pass). SHA-256 values recorded in EXP-010. No PADBench adapter read yet.

## Why / what we learned
- The per-head singular values from the QR route match the dense head update, and the 16 descriptor formulas match direct computations including the zero-padded cases. On synthetic data with a planted signal the nested CV finds it (AUROC 0.999) and 12 label shuffles average 0.513.

## Tried and dropped
- Nothing.

## Next
- Extract features for rank 16 and rank 8 (time one adapter first, log it), then run Part 1 once, then Part 2 only if Part 1 is not STOP, then the permutation test.

## Links
- common/zpeft.py, common/exp10_run.py, common/test_zpeft.py, tracking/experiments/EXP-010-*.md, tracking/decisions/DEC-025-*.md
