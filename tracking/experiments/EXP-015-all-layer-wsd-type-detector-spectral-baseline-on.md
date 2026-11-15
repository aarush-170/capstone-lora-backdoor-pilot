---
id: EXP-015
title: All-layer WSD-type detector (spectral_baseline) on the WSD authors' bank and on PADBench
owner: aarush
rq: RQ1
status: planned
branch:
created: 2026-11-15
updated: 2026-11-15
---
# EXP-015: All-layer WSD-type detector (spectral_baseline) on the WSD authors' bank and on PADBench

## Question
Does our all-layer WSD-type detector (`spectral_baseline` of `common/detectors.py`: five spectral features per projection, averaged over all layers, standardised, logistic regression C = 1) separate backdoored from clean adapters (1) on the WSD authors' Qwen bank and (2) on PADBench Llama-2-7B rank 16, and how does it compare on PADBench with the Z-PEFT reimplementation (EXP-010)?

## Hypothesis
Not a result. (1) The authors' bank holds LoRA on layer index 20 only (Q-026), so "all layers" there is one layer: the features equal the layer-20 features and this part restates EXP-006 Stage 1 (D_A, AUROC 1.000) with our feature set. Expect it to separate. (2) PADBench has 32 layers of q and v, so layer averaging is real. EXP-004 saw all-layer averaging separate our own 12 + 12 adapters (0.944) where the single layer did not (0.583); EXP-010 saw layer-20 WSD-aligned q/v at 0.716 on PADBench. If averaging helps on PADBench the AUROC should move toward Z-PEFT's 0.972; if it stays near 0.72 the all-layer mean is not what Z-PEFT's per-head features add. What would change our mind: a PADBench AUROC near 0.72 (averaging adds nothing) or near 0.97 (head split adds nothing).

## Design
- **Detector (frozen, `common/detectors.py` `SpectralBaseline` with `common/features.py` statistics, no change):** per projection, five statistics of the update (top singular value, Frobenius norm, energy concentration, spectral entropy, kurtosis of singular values), computed per layer and averaged over the layers present. StandardScaler, LogisticRegression(C = 1, max_iter 1000, no class weight, no grid). Scores are logits. This is OUR feature set and not WSD's own code (DEC-010, DEC-019); do not describe the result as WSD's.
- **Part 1, authors' bank (Qwen2.5-3B, layer 20 only, q/k/v/o, 20 features):** fit on 80 % of the 395 benign + 100 poison training adapters (stratified, seed 0, as `common/wsd_bank.py`), threshold on the other 20 % by the same perfect-separation margin rule as EXP-006 Stage 1; test on the Qwen `test` folder (50 + 50). Bootstrap interval (2000) and label permutation (1000). Descriptive, no gate: the same fitted model on the Llama and Gemma test folders.
- **Part 2, PADBench Llama-2-7B rank 16 (primary slice 50 clean + 31 backdoored, labels as EXP-005 Amendment 1; sensitivity slice 50 + 50):** only q and v exist (no k, o), so 10 features (5 per projection, averaged over 32 layers). Nested cross-validation exactly as EXP-010 (outer 5-fold stratified x 3 repeats, seeds 0, 1, 2; the inner loop is unused because C is fixed), score of an adapter = mean out-of-fold score; AUROC with 2000-resample bootstrap interval over adapters (scores held fixed); same adapters and same outer folds as the EXP-010 comparison table so the numbers are paired.
- **Part 3, PADBench rank 16 to rank 8 (descriptive, as EXP-010 Part 2):** fit once on all 81 primary rank-16 adapters, test once on the 75 primary rank-8 adapters and the sensitivity slice, no recalibration.
- **Not done here (and why):** a detector fitted on the authors' bank is NOT scored on PADBench or the other way round: PADBench has no k or o and a different base model, rank-16 q/v only, so the 20 features do not exist there; this is a stated limit, not a result.
- **Always report:** n per class per slice, number of features (20 for Part 1, 10 for Part 2), the three per-repeat AUROCs, the paired table with the EXP-010 values (Z-PEFT 0.972 primary, WSD-aligned q/v 0.716) quoted from `tracking/experiments/EXP-010/part1_result.json`, run times, and the SHA-256 of the code files.
- **Hardware:** weights only, CPU, no training of adapters. Data stays outside git (authors' bank under `Capstone/wsd_bank`, PADBench under `Capstone/padbench_data`, neither redistributed).

## Thresholds
Written: 2026-11-15 07:15 IST (local clock), BEFORE any code for this experiment is written and before any feature of this experiment is computed.
Agreed by: NOT YET AGREED. The team agrees these and records names and time here first. Then status `running`. Do not move it before.

**Part 1 (authors' bank), AUROC of the detector on the Qwen test folder (50 + 50):**
- **AUROC >= 0.95:** "Our all-layer-type features separate the authors' Qwen test adapters (on this bank, a single layer)." Does NOT show anything about all layers, our own adapters or other banks.
- **0.80 to below 0.95:** "Partial: separates less well than D_A (EXP-006 Stage 1, 1.000). List what differs (feature set, classifier)."
- **Below 0.80:** "Does not separate. Check the unit tests and feature order before reading anything into it."

**Part 2 (PADBench rank 16, primary slice), AUROC from the nested cross-validation:**
- **AUROC >= 0.95:** "An all-layer WSD-type detector separates PADBench rank 16 at Z-PEFT's level; the per-head split is not needed on this slice."
- **0.80 to below 0.95:** "Partial: better than layer-20 WSD-aligned (0.716) but below the bar; averaging helps, the per-head features add more."
- **Below 0.80:** "No clear separation beyond update size or the one-layer detector. The all-layer mean does not carry the signal Z-PEFT finds."
- The reading uses the primary slice only; the sensitivity slice, the pairing and the permutation p are reported and do not change it.

**Part 3 (rank 16 to 8):** measurement, no gate. Words as EXP-010: "transfers" (AUROC at least 0.90 and the interval excludes 0.5), "partly transfers" (0.70 to 0.90), "does not transfer" (below 0.70 or the interval includes 0.5).

**Limits to state in every reading:** Part 1 is a single layer; Part 2 is Llama-2-7B, q and v only, one benchmark whose labels we infer, n = 81 and 75, our feature set and not WSD's code; nothing about our Qwen adapters, evasion, audio or vision.
**Procedure rules:** score each part once after the code hashes are recorded; no feature, classifier or fold change after scoring (a change is a new approach with its own thresholds); any unit-test failure or non-finite feature stops the run; no weights, caches or zips in git.

## Approaches
One row per approach tried, including the ones that failed. Keep dead ends: the reason is the point.
Status is one of: planned, running, tried, failed, chosen, abandoned.

| id | what differs | result | status | why |
|---|---|---|---|---|
| A | `spectral_baseline` as is: 5 features per projection, mean over layers, C = 1 | | planned | the all-layer detector the team named; frozen code, no grid |
| B | concatenate per-layer features instead of averaging (32 x 10 = 320 features on PADBench) | | planned | only if A is partial or below; needs its own thresholds first |

## Results
None yet.

## Deviations
None yet.

## Verdict
Not run.

## Next
1. Team reads this file and `tracking/reading_notes/bank_settings_comparison.md`, changes anything it disagrees with, agrees the thresholds (names and time above).
2. Then: write `common/exp15_run.py` and tests on synthetic data only, record SHA-256 values here and commit, before any real adapter is read; run each part once.
3. Quote the EXP-010 numbers from `EXP-010/part1_result.json` in the paired table; do not recompute them.
