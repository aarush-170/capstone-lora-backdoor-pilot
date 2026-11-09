---
date: 2026-11-10
author: aarush
topic: zpeft-plan-drafted
experiments: EXP-010, EXP-005
---
# 2026-11-10 aarush: zpeft-plan-drafted

## Done
- Pushed EXP-009 (Arm T and Arm F scored; both partial; 3bf4fa5).
- Drafted EXP-010 (Z-PEFT reimplementation from the paper: in-domain on PADBench rank 16, then rank 16 to rank 8 transfer) with thresholds written 2026-11-10 01:32 IST, and DEC-025 (the details the paper leaves open). Nothing run, no Z-PEFT code written. Recorded in EXP-005 Deviations that Step 2 moves to EXP-010.

## Why / what we learned
- The panel asked for a Z-PEFT reimplementation. EXP-005 already had a Step 2 plan, but its thresholds were set by Aarush alone and it uses a single 0.95 bar; the new file has three bands and is meant to be agreed by the team first. Honest risk stated in the file: the rank-16 slice labels are partly an assumption (19 of 50 label1 adapters are below the 0.80 filter), so a low AUROC would not by itself mean our code is wrong.

## Tried and dropped
- Nothing.

## Next
- Team reads EXP-010 and DEC-025, changes what it disagrees with, agrees the thresholds. Then write common/zpeft.py, exp10_run.py, test_zpeft.py, tests on synthetic data only, freeze the hashes, run Part 1.
- Add the Z-PEFT questions (code, folds, spectrum used for aggregates, kurtosis) to Varun's email to the authors.

## Links
- tracking/experiments/EXP-010-z-peft-reimplementation-from-the-paper-in-domain.md, tracking/decisions/DEC-025-z-peft-reimplementation-details-the-paper-leaves.md
