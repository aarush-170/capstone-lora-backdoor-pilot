---
id: DEC-025
title: Z-PEFT reimplementation details the paper leaves open (folds, head split, spectrum, kurtosis, grid)
date: 2026-11-10
status: accepted
deciders: team (reported by Aarush in chat, 2026-11-10)
supersedes:
---
# DEC-025: Z-PEFT reimplementation details the paper leaves open (folds, head split, spectrum, kurtosis, grid)

## Decision
Where the Z-PEFT paper (arXiv 2608.02271, `tracking/reading_notes/Z-PEFT.md`) does not say, our reimplementation uses the following, fixed before any PADBench adapter is scored (Q-016 partly answered):
1. **Folds:** nested cross-validation with 5 inner folds (grid choice) and 5 outer folds repeated 3 times (seeds 0, 1, 2), stratified.
2. **Head split (Llama-2-7B, PADBench):** 32 heads of 128 rows; head h uses rows 128h to 128h+127 of B, A shared. Assumes the q and v output rows are head-major. Not applicable to our Omni-3B adapters (grouped-query attention, 16 query and 2 key/value heads): that case is decided separately in DEC-023 and not by this entry.
3. **Spectrum used:** the top-8 singular values are the features sigma_1 to sigma_8 (zero-padded if fewer); the aggregate descriptors (energy, l2, l1, stable rank, entropy, effective rank, concentration, kurtosis) are computed over all singular values of the head update (up to r = 16 here), as the paper's descriptor is read to be a function of the head's spectrum. If the paper meant the top-8 only, that is variant C in EXP-010, not run unless Part 1 is partial or STOP.
4. **Excess kurtosis of the singular values:** E[(s - mean)^4] / Var(s)^2 - 3 over the head's singular values; a head with zero variance gets 0 and is counted and reported.
5. **Entropy and logs:** natural log; probabilities p = sigma / sum sigma (not squared), as the reading note records.
6. **Scaling:** dW includes lora_alpha / r, as in `common/padbench.py`; no other scaling.
7. **Classifier grid and scaler:** exactly the paper's (StandardScaler on the training part, lbfgs, max_iter 5000, C in {0.001 ... 100}, class_weight in {None, balanced}).

## Options considered
- Take the aggregate descriptors over the top 8 singular values only: possible reading of the paper; kept as variant C because at rank 16 it discards half the spectrum, which we would rather decide with data after the first run than by guess before it, but the choice must be made before the first run, hence the default above.
- Fewer or more folds: 5 matches common practice and the repeat structure already used in EXP-005; the paper does not say.
- Imputing degenerate kurtosis with the mean: rejected (hides a count); 0 plus a reported count instead.

## Why
The paper specifies the descriptor and the classifier but not these points (`tracking/reading_notes/Z-PEFT.md`, "What this means for our project"); the authors' code is promised on acceptance (`tracking/code_availability.md`). Fixing the points now, before any scoring, is what keeps the experiment pre-registered.

## Consequences
- If the authors later release code or answer, and it differs, the difference goes under Deviations of EXP-010 and the result is re-read; a result is never re-scored silently.
- Our feature count per Llama-2 adapter is 32,768 against 81 adapters, so heavy regularisation (small C) is expected to be chosen; this is why chosen C values are always reported.
