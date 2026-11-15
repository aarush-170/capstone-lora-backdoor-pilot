---
id: EXP-015
title: All-layer WSD-type detector (spectral_baseline) on the WSD authors' bank and on PADBench
owner: aarush
rq: RQ1
status: done
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
- **Part 1, authors' bank (Qwen2.5-3B, layer 20 only, q/k/v/o, 20 features):** fit on 80 % of the 395 benign + 100 poison training adapters using exactly D_A's split (random permutation, rng seed 42, not stratified; `common/wsd_bank.py` `DetectorA`), threshold on the other 20 % by the same perfect-separation margin rule as EXP-006 Stage 1 (Youden's J if the classes overlap); the classifier is the all-layer detector's (StandardScaler + LogisticRegression, C = 1, no class weight), not D_A's C = 0.1 balanced; test on the Qwen `test` folder (50 + 50). Bootstrap interval (2000) and label permutation (1000). Descriptive, no gate: the same fitted model on the Llama and Gemma test folders.
- **Part 2, PADBench Llama-2-7B rank 16 (primary slice 50 clean + 31 backdoored, labels as EXP-005 Amendment 1; sensitivity slice 50 + 50):** only q and v exist (no k, o), so 10 features (5 per projection, averaged over 32 layers). Nested cross-validation exactly as EXP-010 (outer 5-fold stratified x 3 repeats, seeds 0, 1, 2; the inner loop is unused because C is fixed), score of an adapter = mean out-of-fold score; AUROC with 2000-resample bootstrap interval over adapters (scores held fixed); same adapters and same outer folds as the EXP-010 comparison table so the numbers are paired.
- **Part 3, PADBench rank 16 to rank 8 (descriptive, as EXP-010 Part 2):** fit once on all 81 primary rank-16 adapters, test once on the 75 primary rank-8 adapters and the sensitivity slice, no recalibration.
- **Not done here (and why):** a detector fitted on the authors' bank is NOT scored on PADBench or the other way round: PADBench has no k or o and a different base model, rank-16 q/v only, so the 20 features do not exist there; this is a stated limit, not a result.
- **Always report:** n per class per slice, number of features (20 for Part 1, 10 for Part 2), the three per-repeat AUROCs, the paired table with the EXP-010 values (Z-PEFT 0.972 primary, WSD-aligned q/v 0.716) quoted from `tracking/experiments/EXP-010/part1_result.json`, run times, and the SHA-256 of the code files.
- **Hardware:** weights only, CPU, no training of adapters. Data stays outside git (authors' bank under `Capstone/wsd_bank`, PADBench under `Capstone/padbench_data`, neither redistributed).

## Thresholds
Written: 2026-11-15 07:15 IST (local clock), BEFORE any code for this experiment is written and before any feature of this experiment is computed.
Agreed by: the team, as reported by Aarush in chat on 2026-11-15 07:26 IST (individual names not given; Aarush to add them if the team wants them recorded). Thresholds unchanged since the Written timestamp above; the code hashes are those recorded under Next.

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

## Approach B: per-layer features concatenated (thresholds drafted, NOT run)
Written: 2026-11-15 07:38 IST (local clock), BEFORE any code for approach B is written and before any approach-B feature is computed. Scored parts A (1 to 3) are unchanged and must not be rerun.
Approach B agreement: the team, as reported by Aarush in chat on 2026-11-15 07:46 IST (individual names not given; Aarush to add them if the team wants them recorded). Thresholds unchanged since the Written timestamp above.

**Why B.** Approach A averaged each statistic over the 32 layers and fitted a fixed C = 1 model on 10 features: PADBench rank 16 AUROC 0.636. Z-PEFT keeps every layer and head (32,768 features, grid-tuned) and reads 0.972. B asks whether keeping the layers apart is enough, without the head split or the 16-descriptor set.

**Question.** On PADBench rank 16 (primary slice 50 + 31), does a WSD-type detector that keeps the per-layer statistics (not their mean) separate the classes, and how much of the gap between A (0.636) and Z-PEFT (0.972) does that close?

**Hypothesis (not a result).** If where the update sits (which layer) carries the signal, B should rise well above A and toward Z-PEFT. If B stays near A or near weight norm (0.717), the per-head split or the richer descriptor set is what Z-PEFT adds. What would change our mind: B above 0.95 (the head split is not needed) or B near 0.64 (layer position does not help).

**Design.**
- Features: for each layer l in 0..31 (sorted), each projection in (q_proj, v_proj), the same five statistics as A (`features.stats_from_sv`, top_sv, fro_norm, energy_conc, spectral_entropy, kurtosis), layer-major then projection then statistic: 32 x 2 x 5 = 320 numbers. No averaging. Any non-finite value stops the run.
- Classifier: the Z-PEFT paper's grid as in EXP-010 (StandardScaler on the training part, L2 logistic regression lbfgs max_iter 5000, C in {0.001, 0.01, 0.1, 1, 10, 100} x class_weight in {None, balanced}, chosen by 5-fold stratified inner AUROC on the training part only), through the same `zpeft.zpeft_fit_score`. This is a deliberate change from A's fixed C = 1, so a control separates the two changes:
- Control B0 (descriptive, no gate): A's 10 averaged features with the SAME grid, same folds. B minus B0 isolates the per-layer effect; B0 minus A (0.636) isolates the grid.
- Evaluation: outer 5-fold stratified x 3 repeats (seeds 0, 1, 2), the same folds as EXP-010 and A; score of an adapter = mean out-of-fold score; AUROC with the 2000-resample bootstrap interval over adapters (scores held fixed). Primary slice for the reading; sensitivity slice (50 + 50) reported.
- Transfer (descriptive, as A Part 3): fit once on the 81 primary rank-16 adapters (inner grid, seed 0), test once on rank 8 (75 primary, 100 sensitivity); words as EXP-010. Both rank-16 and rank-8 adapters have the same 32 layers, so the 320 features line up.
- Not run: the authors' bank (one layer, so B equals A there), per-head variants.
- Always report: n per class, 320 vs 10 features against 81 adapters, the chosen (C, class_weight) per outer fold, convergence warnings, per-repeat AUROCs, the paired table (B, B0, A 0.636, Z-PEFT 0.972, WSD-aligned q/v 0.716, weight norm 0.717; the last four quoted from the files, not recomputed), run times, and the SHA-256 of the code.

**Thresholds (primary slice, AUROC of B).**
- **>= 0.95:** "Keeping per-layer statistics, without the head split, separates PADBench rank 16 at Z-PEFT's level. The per-head split is not what carries the result on this slice."
- **0.80 to below 0.95:** "Partial: per-layer position adds signal over averaging (A, 0.636) but not enough; the per-head split or the descriptor set adds the rest."
- **Below 0.80:** "Per-layer position does not rescue the WSD-type features on this slice. What Z-PEFT finds needs the head split or its descriptor set (not tested here which)."
- The reading uses B on the primary slice only. B0, the sensitivity slice, the paired table and transfer are reported and do not change it.

**Limits for every reading:** Llama-2-7B, q and v only, one benchmark with inferred labels (EXP-005 Amendment 1), n = 81 and 75, 320 features on 81 adapters (the grid's small-C choices matter and are reported), our feature set and not WSD's code; nothing about our own Qwen adapters, other models, evasion or other modalities.
**Procedure rules:** each part scored once after the code hashes are recorded; no feature, grid or fold change after scoring (a change is a new approach with its own thresholds); unit tests on synthetic data first; caches and weights stay out of git.

## Approaches
One row per approach tried, including the ones that failed. Keep dead ends: the reason is the point.
Status is one of: planned, running, tried, failed, chosen, abandoned.

| id | what differs | result | status | why |
|---|---|---|---|---|
| A | `spectral_baseline` as is: 5 features per projection, mean over layers, C = 1 | bank Qwen test AUROC 1.000 (separates); PADBench rank 16 primary 0.636 (no clear separation); rank 16 to 8: 0.507 | tried | the all-layer detector the team named; frozen code, no grid; scored once 2026-11-15 |
| B | concatenate per-layer features instead of averaging (32 x 2 x 5 = 320 features on PADBench), Z-PEFT grid, control B0 (averaged features, same grid) | | planned | A read below 0.80 on PADBench, so B is now allowed by the plan, but thresholds drafted in the section 'Approach B' above, not yet agreed; not run |

## Results
Scored once each on 2026-11-15 (CPU, `KMP_DUPLICATE_LIB_OK=TRUE`), after the Agreed-by line was filled in and with the code hashes unchanged from those under Next (checked by `python common/exp15_run.py hashes` before the run). Outputs: `tracking/experiments/EXP-015/part{1,2,3}_output.txt` and `part{1,2,3}_result.json`. No convergence or other warnings in any output. Verified (from the run).

**Part 1, authors' bank (Qwen2.5-3B, layer index 20 only, 20 features):** fit on 396 adapters (80 % of 395 benign + 100 poison, D_A's split), threshold 0.2803 on the other 99 (`perfect_separation_margin`).

| test set | n (benign + poison) | AUROC (95% CI) | permutation p | accuracy | FP / FN |
|---|---|---|---|---|---|
| **Qwen (gate)** | 50 + 50 | **1.000 (1.000 to 1.000)** | 0.001 | 0.980 | 0 / 2 |
| Llama (descriptive) | 50 + 50 | 0.905 (0.831 to 0.965) | 0.001 | 0.850 | 13 / 2 |
| Gemma (descriptive) | 49 + 48 | 0.998 (0.991 to 1.000) | 0.001 | 0.897 | 0 / 10 |

Reading: **separates** (Qwen AUROC 1.000, bar 0.95). For comparison, D_A on the same Qwen test folder was AUROC 1.000, accuracy 0.970 (EXP-006 Stage 1).

**Part 2, PADBench Llama-2-7B rank 16, nested CV (outer 5-fold x 3 repeats, 10 features = 5 statistics x q and v, mean over 32 layers):**

| slice | n (clean + backdoored) | AUROC (95% CI) | per repeat |
|---|---|---|---|
| **primary** | 50 + 31 | **0.636 (0.516 to 0.749)** | 0.667, 0.632, 0.652 |
| sensitivity (all label1) | 50 + 50 | 0.601 (0.489 to 0.710) | 0.548, 0.608, 0.635 |

Reading: **no_clear_separation** (primary AUROC below 0.80). Paired values quoted from EXP-010 (`tracking/experiments/EXP-010/part1_result.json`, same folds), primary slice: Z-PEFT reimplementation 0.972, scaler LR 0.781, WSD-aligned q/v (layer 20) 0.716, weight norm 0.717.

**Part 3, rank 16 to rank 8 (fit once on the 81 primary rank-16 adapters, descriptive):** primary (50 + 25) AUROC 0.507 (0.380 to 0.634), word: does not transfer; sensitivity (50 + 50) 0.606 (0.491 to 0.716), does not transfer.

## Deviations
None. Code, folds and thresholds as written.

## Verdict
Part 1: **separates** on the authors' Qwen test folder (AUROC 1.000, accuracy 0.980). On that bank the detector has only one layer, so this is a single-layer result with our spectral feature set and does not show anything about all layers. It agrees with D_A (EXP-006 Stage 1) and adds that a different feature set (our pilot statistics) is enough on that bank, and (descriptive) that the same fitted model drops on the Llama test folder (0.905, 13 false positives) while still ranking Gemma well (0.998) but with a poorly placed threshold (10 false negatives).
Part 2: **no clear separation** on PADBench rank 16 (0.636, interval 0.516 to 0.749), below the one-layer WSD-aligned q/v detector (0.716) and weight norm (0.717) and far below the Z-PEFT reimplementation (0.972) on the same folds. Statement as written: the all-layer mean does not carry the signal Z-PEFT finds. Part 3: does not transfer (0.507).
What it does NOT show: that no all-layer WSD-type detector can work on PADBench (this is our feature set, averaged over layers, 10 features, C = 1 and no grid; approach B, per-layer features concatenated, is not run); why averaging scores lower than the single layer (0.636 against 0.716 is not a tested comparison, the intervals overlap); anything about our own Qwen adapters, other base models, evasion or other modalities. One benchmark, labels we infer (EXP-005 Amendment 1), n = 81 and 75.

## Next
1. Team reads this file and `tracking/reading_notes/bank_settings_comparison.md`, changes anything it disagrees with, agrees the thresholds (names and time above).
2. Done 2026-11-15 (before agreement, at Aarush's request; no real adapter read, nothing run): `common/exp15_run.py` and `common/test_exp15.py` written; 7 synthetic tests pass (`cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_exp15.py`); the runner refuses to run while the Agreed-by line says NOT YET AGREED and refuses to overwrite a result file. Frozen code, SHA-256 with carriage returns removed: `common/exp15_run.py` `da394bf5662a1e57741c3557c3dee7842f3ced5cdbb2594d5adf631a06b36a94`; `common/test_exp15.py` `8e0c3cd4d8c1123683f64a43c2ee03b0809fc6704bebf8b3d7f837d9b0607983`. If either file changes before scoring, the new hash replaces these and the reason goes under Deviations.
3. Done 2026-11-15: agreement recorded and each part run once: `python common/exp15_run.py part1 <bank_root> <out>`, `part2 <padbench_root> <out>`, `part3 <padbench_root> <out>` (features are cached outside git).
4. Quote the EXP-010 numbers from `EXP-010/part1_result.json` in the paired table; do not recompute them.
5. Open: whether to write thresholds for approach B (per-layer features concatenated) and for a head-split WSD-type variant; the team decides. The EXP-013 plan is unchanged.
6. Approach B: team reads the section above and agrees or edits it (names and time under 'Approach B agreement'). Then the code (a new `common/exp15b_run.py`, or an extension that does not change the frozen A code) and synthetic tests, hashes recorded, then one run.
7. Approach B code written 2026-11-15 after the agreement line above was filled, before any approach-B feature was computed: `common/exp15b_run.py` and `common/test_exp15b.py` (4 synthetic tests pass, `cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_exp15b.py`; the approach-A tests still pass). It imports the frozen approach-A code unchanged. The runner refuses to run while the file says the B agreement is pending and refuses to overwrite a result file. Frozen code, SHA-256 with carriage returns removed: `common/exp15b_run.py` `33d53ef55cc8c550f04c4e940bd162ef923818b6aef3137491cbaf5b1907feda`; `common/test_exp15b.py` `fedd1d153a8b16f70bf40b217b356d3f000bb4c809f2855fc4f7021fd130fa58`; `common/exp15_run.py` (unchanged) `da394bf5662a1e57741c3557c3dee7842f3ced5cdbb2594d5adf631a06b36a94`.
