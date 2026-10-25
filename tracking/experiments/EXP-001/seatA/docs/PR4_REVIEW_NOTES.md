# PR #4 (`evasion-g1`) reviewed against PLAN section 7 (Sat 24 Oct, seat A)

Code read in full; **not run**: this Mac is x86_64 with Python 3.13, which has no torch wheel, so the CPU tests were not executed. Run `pytest common/test_*.py` in a Kaggle/Colab CPU cell first.

## What is reusable as is
- `lora_stats_batched`: QR on B and A^T, then `svdvals` on the r x r core, batched. Matches plan section 7.
- `SpectralPenalty` mechanics: grouping by shape (handles 2048x2048 vs 256x2048), `extra_loss_fn` hook in `pilot_common.train`, `penalty_mean_last20` logged.
- Warm-up default of 16 steps is a technical need (at init B = 0, so dW = 0 and the SVD gradient is undefined), not the "lambda = 0 for epoch 1" variant. Record it as a decision.

## Gaps against the plan (changes needed in `common/evasion.py`)
1. **Features are not WSD-aligned.** `stats_from_sv_torch` takes kurtosis of the *singular values* and entropy on sigma^2 shares. Plan: kurtosis on dW's *entries*; energy sigma1^2 / sum sigma^2 (this one already matches); entropy per C's `wsd_aligned` definition (must be identical to the detector).
2. **Scope is all 144 modules.** Plan: layer 21 only (4 modules). Filter `module_key` to layer 21.
3. **Targets come from a features CSV of clean adapters** (`targets_from_features`), per module. Plan: from `ref_clean` only, mu/sigma of the WSD-aligned features, then `ref_stats.json`. With only 3 `ref_clean` adapters sigma is noisy; the existing `rel_floor = 0.05` stays, and the floor must be written into the decisions log.
4. **Penalty form.** Plan: mean over modules of (z_m)^2 where z_m is the vector of features; PR averages over modules and statistics, which is the same up to a constant factor. Keep it, and say lambda is therefore relative to that mean.
5. **Entry kurtosis (new).** At layer 21 only, form dW = s * B @ A directly (2048x2048 = 4M floats, 4 modules). That is cheap enough to skip the moment trick; check peak memory in the smoke test.
6. **GQA.** k and v are 256x2048; features per module then average (already how shape groups work); sensitivity check "q+o only" is a flag to add, not a retrain.
7. **`g1_analysis.py` / `detector_standin.py`** implement the superseded AUROC design; seat C replaces them. Seat A touches neither.

## Decisions to log in EXP-001 (the four from plan section 7 plus two)
GQA per-module average; layer 21 only; s = alpha/r = 2 in features and penalty; energy sigma1^2 / sum sigma^2; warm-up of 16 steps for the SVD gradient only; sigma floor 5% of |mu|.
