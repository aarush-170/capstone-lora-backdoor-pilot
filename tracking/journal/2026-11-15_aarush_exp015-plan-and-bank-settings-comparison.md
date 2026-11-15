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
- Merged-main check: Shashank's audio study is EXP-014 on main; this branch has not been updated from main yet.

## Why / what we learned
- The authors' bank holds LoRA on layer index 20 only (all 495 adapters, `metadata.json`), so an all-layer detector has only one layer there; Part 1 restates EXP-006 Stage 1 with our feature set. PADBench has 32 layers of q and v only, so layer averaging is real there, and the detector cannot be moved between the two banks (no k or o, different model).
- Differences not yet tested: the authors' benign bank uses 8 datasets (we use Alpaca only), their poison adapters mostly use the trigger `Important update:` (75 of 100; we use only `cf`), poison rates 1, 3 and 5 percent, PEFT 0.13.2 against our 0.20.0, benign dropout 0.05 in the config read.
- Our 6 template backdoors have larger B matrices than the authors' adapters in q, v and o (n = 6, descriptive).

## Tried and dropped
- Nothing dropped. An all-layer detector fitted on the authors' bank and scored on PADBench was considered and left out because the features do not exist there.

## Next
- Team agrees the EXP-015 thresholds; then code and tests before any real adapter is read.
- Ask whether PADBench's training recipe should be looked up in the PEFTGuard paper.

## Links
- tracking/experiments/EXP-015-all-layer-wsd-type-detector-spectral-baseline-on.md
- tracking/reading_notes/bank_settings_comparison.md
