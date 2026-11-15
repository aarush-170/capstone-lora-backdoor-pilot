---
date: 2026-11-15
author: aarush
topic: exp015-plan-and-bank-settings-comparison
experiments: EXP-015
---
# 2026-11-15 aarush: exp015-plan-and-bank-settings-comparison

## Done
- Drafted EXP-015 (status planned, thresholds written 2026-11-15 07:15 IST, NOT agreed): the all-layer WSD-type detector (`spectral_baseline`) on the authors' Qwen bank and on PADBench rank 16 and rank 8. Nothing run, no code written.
- Wrote `tracking/reading_notes/bank_settings_comparison.md`: settings and weight norms of our bank, the authors' bank and PADBench, labelled verified, reported or unknown.
- Wrote `common/exp15_run.py` and `common/test_exp15.py` (7 synthetic tests pass, no real adapter read) with a guard that refuses to run until the Agreed-by line is filled in. Hashes are in EXP-015.
- Merged origin/main into padbench-rq1 (Shashank's EXP-014 audio study and the tracking system); adapters.csv and literature.csv conflicts resolved by keeping both sides (one duplicate PADBench literature row dropped, keeping the newer text).

- Team agreement reported by Aarush in chat; EXP-015 run once per part: bank Qwen AUROC 1.000 (separates), PADBench rank 16 0.636 (no clear separation), rank 16 to 8 0.507 (does not transfer).

- Approach B (per-layer statistics, 320 features, Z-PEFT grid) agreed and run once: PADBench rank 16 primary AUROC 0.981 (0.951 to 1.000), reading at_zpeft_level; control B0 (averaged, same grid) 0.783; transfer to rank 8: B 0.897 (partly transfers), B0 0.412.

## Why / what we learned
- The authors' bank holds LoRA on layer index 20 only (all 495 adapters, `metadata.json`), so an all-layer detector has only one layer there; Part 1 restates EXP-006 Stage 1 with our feature set. PADBench has 32 layers of q and v only, so layer averaging is real there, and the detector cannot be moved between the two banks (no k or o, different model).
- Differences not yet tested: the authors' benign bank uses 8 datasets (we use Alpaca only), their poison adapters mostly use the trigger `Important update:` (75 of 100; we use only `cf`), poison rates 1, 3 and 5 percent, PEFT 0.13.2 against our 0.20.0, benign dropout 0.05 in the config read.
- Our 6 template backdoors have larger B matrices than the authors' adapters in q, v and o (n = 6, descriptive).

- Averaging over layers (approach A, 0.636; with the grid, 0.783) throws the signal away; keeping layers apart gives 0.981, equal to the Z-PEFT reimplementation (0.972) within the intervals. The per-head split is not needed on this slice.

## Tried and dropped
- Nothing dropped. An all-layer detector fitted on the authors' bank and scored on PADBench was considered and left out because the features do not exist there.

## Next
- Team agrees the EXP-015 thresholds; then code and tests before any real adapter is read.
- Ask whether PADBench's training recipe should be looked up in the PEFTGuard paper.

## Links
- tracking/experiments/EXP-015-all-layer-wsd-type-detector-spectral-baseline-on.md
- tracking/reading_notes/bank_settings_comparison.md
