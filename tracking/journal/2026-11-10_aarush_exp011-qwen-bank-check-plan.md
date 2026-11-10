---
date: 2026-11-10
author: aarush
topic: exp011-qwen-bank-check-plan
experiments: EXP-011, EXP-010
---
# 2026-11-10 aarush: exp011-qwen-bank-check-plan

## Done
- Read-only look at the authors' Qwen bank: 595 adapters (395 + 100 train, 50 + 50 test), all with LoRA on layer index 20 only (q/k/v/o, r 16, alpha 32, 8 tensors); shapes identical to our own EXP-006/008/009 adapters. No features computed.
- Drafted EXP-011 (layer-20 per-head Z-PEFT variant, 288 features, trained on the bank, frozen threshold, scored on our clean batches and on our working-backdoor adapters), thresholds written 2026-11-10 14:09 IST. Not agreed, no code written.

## Why / what we learned
- The bank cannot support the published all-layer Z-PEFT, only a one-layer variant, so the test is labelled a variant everywhere. Its use is to tell whether the clean-control flag is specific to WSD's features or shows through a second feature set.
- The EXP-008 matched weights are not on this machine (made on the second machine); the plan says that batch is reported as not scored if they cannot be copied in.
- The bank's poison adapters carry no working backdoor (EXP-006 Stage 2a), so a pass on Gate 1 or Gate 2 does not mean backdoor detection; written into the file.

## Tried and dropped
- Nothing.

## Next
- Team reads and agrees EXP-011; then code, synthetic tests, frozen hashes, one scoring.

## Links
- tracking/experiments/EXP-011-layer-20-z-peft-variant-trained-on-the-authors-q.md
