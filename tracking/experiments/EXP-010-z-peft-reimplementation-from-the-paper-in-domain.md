---
id: EXP-010
title: Z-PEFT reimplementation from the paper: in-domain on PADBench rank 16 and rank 8 transfer (RQ1)
owner: aarush
rq: RQ1
status: running
branch: padbench-rq1
created: 2026-11-10
updated: 2026-11-10
---
# EXP-010: Z-PEFT reimplementation from the paper: in-domain on PADBench rank 16 and rank 8 transfer (RQ1)

**Status: thresholds agreed (see Thresholds); code and synthetic tests written and frozen 2026-11-10; no Z-PEFT feature has been computed on any PADBench adapter yet.** This experiment takes over Step 2 (and the rank-8 part of Step 3) of EXP-005, whose Step 2 thresholds were agreed by Aarush alone and never run; see the Deviation entry in EXP-005. It follows the panel's instruction after Review 2 to reimplement the Z-PEFT detector.

## Question
1. Does our reimplementation of Z-PEFT (arXiv 2608.02271, from the paper only; the authors' code is "upon acceptance") separate backdoored from clean adapters in the PADBench Llama-2-7B rank-16 slice we already hold?
2. If it does, how much does it lose when trained on rank 16 and tested on rank 8 (PADBench, same model and attack family)?

Background (files): on the same rank-16 slice, our WSD-aligned q/v detector reached leave-one-out AUROC 0.739 (0.631 to 0.836) and weight norm alone 0.717 (EXP-005 Step 1, `tracking/experiments/EXP-005/step1_rank16_result.json`). The Z-PEFT paper reports in-domain AUROC 0.9986 (pooled over far more adapters) and rank-held-out 0.98 to 1.00 (Tables 4 and 6; `tracking/reading_notes/Z-PEFT.md`). We do not know whether our version reproduces that.

## Hypothesis
Not a result. If our reimplementation follows the paper, its per-head, all-layer features should separate this slice better than WSD's one-layer features (0.74). What would change our mind: an AUROC near the WSD value or near weight norm would say that on this slice nothing beyond update size is detectable (or that the slice labels are noisy: 19 of the 50 `label1` adapters at rank 16 have a toxic probability below 0.80, EXP-005 Amendment 1).

## Design
- **Data (as EXP-005; public, no licence stated, kept outside the repo, never committed):** `C:/Users/user/Capstone/padbench_data/rank16` (100 adapters) and `rank8` (100 adapters), Llama-2-7B, LoRA, q_proj and v_proj only, 32 layers, rank 16 (alpha 32) and rank 8 (alpha 16), each tensor B 4096 x r, A r x 4096. Labels exactly as EXP-005 Amendment 1 (`common/padbench.py` `list_slice`): `label0` clean; `label1` backdoored only if its CSV toxic probability is at least 0.80; other `label1` left out of the primary analysis. Primary slice: rank 16 = 50 clean + 31 backdoored (81); rank 8 = 50 clean + 25 backdoored (75). Sensitivity slice (original definition, all 50 `label1`): reported next to the primary one; the primary is the filtered one, decided now.
- **Detector (our reimplementation, `common/zpeft.py`, to be written and tested before any PADBench adapter is scored):** for every layer l (32), projection p in {q, v} and attention head h (32 heads of 128 rows; Llama-2 is multi-head, no grouped query), the head update is dW = scaling * B_h A with B_h the 128 rows of B for head h (rows are head-major: assumption, from the layout of Llama attention). Singular values of dW by reduced QR of B_h and A^T and an r x r SVD (as `common/features.py`), never forming dW. Per-head descriptor, 16 numbers as in the paper (p. 3): the top 8 singular values (zero-padded), energy sum sigma^2, l2 norm, l1 norm, stable rank, spectral entropy of sigma / sum sigma, effective rank exp(H), concentration sigma_1 / sum sigma, excess kurtosis of the singular values. One adapter = 32 x 2 x 32 x 16 = 32,768 numbers in a fixed order. Details the paper leaves open are fixed in DEC-025 (proposed together with this file).
- **Classifier (the paper's, p. 4 and Table 8):** StandardScaler fitted on the training part only, L2 logistic regression (lbfgs, max_iter 5000), grid C in {0.001, 0.01, 0.1, 1, 10, 100} x class_weight in {None, balanced} (12 candidates) chosen by mean 5-fold stratified cross-validated AUROC on the training part (the paper does not give the number of folds: 5 is our choice, DEC-025).
- **Evaluation, in-domain:** nested cross-validation: outer 5-fold stratified, repeated 3 times with seeds 0, 1, 2 (inner 5-fold, seed = outer seed). The score of an adapter is its mean out-of-fold score over the 3 repeats; AUROC from those scores; 95% bootstrap interval over adapters (2000 resamples, seed 0, scores held fixed: it does not include the training variability, stated as a limit). Also report the mean and range of the three per-repeat AUROCs.
- **Comparison (same outer folds, same adapters):** weight_norm (Frobenius norm of q and v at layer index 20, as EXP-005) and the `wsd_aligned_qv` and `scaler_lr_c0.1` detectors of EXP-005 under the same nested folds, so the comparison is paired. The LOO numbers of EXP-005 are quoted for reference only.
- **Transfer (Part 2, only if Part 1 does not read STOP):** fit once on all 81 rank-16 adapters (inner 5-fold grid as above, seed 0), test once on the 75 rank-8 adapters (primary labels) and on the sensitivity definition. Zero-padding is not triggered (rank 8 gives exactly 8 singular values per head). No recalibration in this experiment. Report AUROC with a bootstrap interval next to the paper's leave-one-rank-out values (0.98 to 1.00) without claiming a reproduction unless the interval covers them. Rank 32 and AdaLoRA are not downloaded and are not in this experiment (EXP-005 Step 3 keeps them; they need their own thresholds).
- **Hardware:** weights only, CPU, no GPU, no training of adapters. Feature extraction is cached once per adapter to a git-ignored folder.
- **Permutation test (secondary, descriptive):** 100 label shuffles, one nested repeat each, same procedure; p = (1 + number of shuffles at or above the observed single-repeat AUROC) / 101. If the measured run time makes this impractical it is reduced, and the reduction goes under Deviations before it is run.

## Thresholds
Written: 2026-11-10 01:32 IST (local clock), BEFORE any Z-PEFT code is written and before any Z-PEFT feature is computed on a PADBench adapter.
Agreed by: the team, as reported by Aarush in chat on 2026-11-10 (individual names and exact time not given; Aarush to add them if the team wants them recorded). Thresholds unchanged since the Written timestamp above; DEC-025 accepted with them.

**Frozen code (recorded 2026-11-10 after the synthetic tests passed, before any PADBench adapter was read; SHA-256 with carriage returns removed):** `common/zpeft.py` `7ed494f5bdb27f69838bdb97c34603123916efeb2f925743d37d8111ac73bda6`; `common/exp10_run.py` `6fa5999fdfefc7f2ee7e6c495e788a12a6bbdf39e71fdd7ed617c6732bb47922`; `common/test_zpeft.py` `1512fe9455bd0bc55e0420819a6133be8bdc1cf2a34ccdb6e6d9e9e24a888d94`. All 8 tests pass (`cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_zpeft.py`, about 40 s): tests (a) to (e) below are items 1 to 5 of the list, (f) checks feature order and values on a synthetic adapter folder against dense recomputation, (g) the band edges, transfer words and the refuse-while-not-agreed guard, (h) Part 1 and Part 2 end to end on a synthetic cache. Note on (d): it is a structural leakage test (outer folds disjoint, scaler inside the pipeline), not a data-driven one. Required tests before any PADBench adapter is read: (a) per-head singular values from the QR route equal those of the explicit dense head update on random low-rank factors; (b) each of the 16 descriptor formulas equals a direct computation on a small spectrum, including a rank-8 and a rank-4 (zero-padded) case; (c) degenerate spectra (all singular values equal, or zero) give finite values, with any substitution counted; (d) the scaler is fitted inside each training fold only (a leakage test with features that carry the label in the test fold only); (e) the nested CV on synthetic data with a planted signal gives AUROC well above 0.5 and, with shuffled labels, a mean near 0.5 over many shuffles. Any non-finite feature on a real adapter stops the run (reported, not imputed).

**Part 1: in-domain, rank 16, primary slice (50 + 31), AUROC of the Z-PEFT reimplementation:**
- **AUROC >= 0.95:** "Our Z-PEFT reimplementation separates the PADBench rank-16 slice at the level the paper reports (the paper's 0.9986 is on far more adapters; 0.95 on 81 is a deliberately lower bar, as in EXP-005). Reimplementation accepted for further tests. This does not show it works on our own adapters, on other models, or against an adaptive attacker."
- **0.80 to below 0.95:** "Partial. The reimplementation separates the slice better than WSD-aligned (0.74) but does not reach the paper's level. Not accepted as a faithful reproduction. Before any shift test is described as a Z-PEFT result, list what differs from the paper (the open details of DEC-025, the label filter, the sample size). Part 2 may run and is labelled 'unvalidated reimplementation'."
- **Below 0.80:** "No clear separation beyond what size or WSD gives. STOP: Part 2 does not run. Check (a) the unit tests again, (b) the sensitivity slice (all 50 `label1`), (c) the variants listed under Approaches; report which of the paper's details we could not reproduce. Do not write 'Z-PEFT fails'; the reimplementation is ours and the slice labels are an assumption."
- The reading uses the primary slice AUROC only. The sensitivity slice, the paired comparison and the permutation p are always reported and do not change the reading.

**Part 2: rank 16 to rank 8 (measurement, no pass/fail):** AUROC with interval on the primary and the sensitivity rank-8 slices. Words allowed: "transfers (AUROC at least 0.90 and the interval excludes 0.5)", "partly transfers (0.70 to 0.90)", "does not transfer (below 0.70 or the interval includes 0.5)". These bands are descriptive labels for the sentence, not a gate; no rank-32 or AdaLoRA claim.

**Always report:** n per class for every slice; the AUROC of each of the three per-repeat runs; the paired comparison table (Z-PEFT, WSD-aligned q/v, scaler LR, weight_norm) on the same folds; the number of features (32,768) against the number of adapters (81); which candidate (C, class_weight) was chosen in each outer fold; run times; the three SHA-256 values above.
**Limits to state in every reading:** Llama-2-7B only, q and v only, text only, one benchmark whose labels we infer (Amendment 1), n = 81 and 75, our reimplementation from the paper and not the authors' code, CPU; nothing about our Omni-3B adapters, evasion, audio or vision.
**Procedure rules:** score each part once after the frozen hashes are recorded; no feature, classifier or grid change after scoring (a change is a new, separately-thresholded approach); any unit-test failure or non-finite feature stops the run before scoring; weights and the cached feature files stay out of git; PADBench is not redistributed; a variant (below) needs its own thresholds written first.

## Approaches
One row per approach tried, including the ones that failed. Keep dead ends: the reason is the point.
Status is one of: planned, running, tried, failed, chosen, abandoned.

| id | what differs | result | status | why |
|---|---|---|---|---|
| A | Z-PEFT reimplementation as in Design, in-domain rank 16, primary slice | AUROC 0.972 (0.930 to 0.999), reading accepted | tried | the panel's instruction; positive control for our version of the detector |
| B | rank 16 to rank 8 transfer | AUROC 0.957 (0.890 to 1.000), 'transfers' | tried | RQ1 factor already in PADBench; paper reports 0.98 to 1.00 for rank held out |
| C | variants of the open details (top-8 only aggregates, other fold counts, other head order) | not planned | planned | only if Part 1 reads partial or STOP, each with thresholds written first |
| D | rank 32 and AdaLoRA shifts | not planned | planned | stay in EXP-005 Step 3; need the downloads and their own thresholds; AdaLoRA is where the paper itself reports 0.2628 |

## Results
Feature extraction (2026-11-10, CPU, `python common/exp10_run.py extract`): rank 16 and rank 8, 100 adapters each, 32,768 Z-PEFT features per adapter plus the 10 WSD-aligned q/v features; all finite, 0 degenerate heads; about 1.2 s per adapter (about 2 min per rank). Cached in `padbench_data/exp10_cache/` (outside git). Nothing scored yet: Part 1 and Part 2 are still to run.

**Part 1 (in-domain, rank 16), scored once on 2026-11-10** with `python common/exp10_run.py part1` (code hashes as in Thresholds; outputs `tracking/experiments/EXP-010/part1_output.txt`, `part1_result.json`). Nested CV, outer 5-fold x 3 repeats (seeds 0, 1, 2), inner 5-fold, 32,768 features, 12-candidate grid; about 80 to 90 s per Z-PEFT repeat; 0 convergence warnings in any fit.

| detector (same folds) | primary slice (50 clean + 31 backdoored) AUROC (95% CI) | per repeat | sensitivity (50 + 50) AUROC (95% CI) |
|---|---|---|---|
| **Z-PEFT (ours)** | **0.972 (0.930 to 0.999)** | 0.953, 0.978, 0.966 | 0.988 (0.966 to 1.000) |
| WSD-aligned q/v | 0.716 (0.609 to 0.817) | 0.739, 0.675, 0.732 | 0.780 (0.688 to 0.862) |
| scaler LR (C 0.1) | 0.781 (0.679 to 0.872) | 0.779, 0.745, 0.795 | 0.799 (0.710 to 0.880) |
| weight_norm | 0.717 (0.606 to 0.820) | 0.717 (no fitting) | 0.641 (0.521 to 0.749) |

Chosen hyper-parameters over the 15 outer folds (primary): C = 1 in 7, 0.1 in 5, 0.01 in 2, 10 in 2 (class_weight None in 12, balanced in 3); small-to-moderate C as expected for 32,768 features on 81 adapters. Sensitivity: C = 0.1 in 5, 1 in 4, 0.01 in 3, 10 in 3 (all class_weight None). The CI treats the scores as fixed (training variability not included). Permutation test (secondary) not run yet.

**Part 2 (rank 16 to rank 8 transfer), scored once on 2026-11-10** with `python common/exp10_run.py part2` (outputs `tracking/experiments/EXP-010/part2_output.txt`, `part2_result.json`). Fit once on the 81 primary rank-16 adapters (grid chosen by inner 5-fold AUROC, seed 0: C = 100, class_weight balanced, inner AUROC 0.985, no convergence warning; C = 100 is the largest value in the paper's grid), tested once on rank 8. Zero-padding not triggered.

| detector (trained on rank 16) | rank 8 primary (50 clean + 25 backdoored) AUROC (95% CI) | rank 8 sensitivity (50 + 50) AUROC (95% CI) |
|---|---|---|
| **Z-PEFT (ours)** | **0.957 (0.890 to 1.000)**, word: transfers | 0.968 (0.926 to 1.000), word: transfers |
| WSD-aligned q/v | 0.695 (0.577 to 0.808) | 0.563 (0.442 to 0.674) |
| scaler LR (C 0.1) | 0.767 (0.658 to 0.865) | 0.776 (0.679 to 0.866) |
| weight_norm | 0.722 (0.609 to 0.825) | 0.584 (0.467 to 0.694) |

The paper's leave-one-rank-out values are 0.98 to 1.00 (Table 6). Our interval (0.890 to 1.000) includes them, but the protocol differs (one training rank, 81 training adapters, our reimplementation), so this is not claimed as a reproduction. The bootstrap treats the scores as fixed.

**Permutation test** (secondary, descriptive; finished 2026-11-10, exit 0; `tracking/experiments/EXP-010/perm_output.txt`, `perm_result.json`): 100 label shuffles, one nested repeat each. Observed single-repeat AUROC 0.953; null mean 0.508, 95th percentile 0.650; p = 0.0099 (1/101, the smallest value 100 shuffles can give: no shuffle reached the observed value). Does not change the Part 1 reading. It rules out that 0.95-level scores arise from the pipeline on shuffled labels; it does not address label construction or the one-benchmark limit.

## Deviations
- 2026-11-10 (documentation only; no threshold, reading or frozen code changed): the runner's 'agreed' guard searches this file for the literal not-yet-agreed marker line and it also matched my description of the guard in the frozen-code note, so the first `extract` call stopped. I reworded that phrase in this file (now 'the refuse-while-not-agreed guard'); the guard code is untouched and its hash is unchanged.

## Verdict
**Part 1 reading: accepted** (primary-slice AUROC 0.972, bar 0.95). Statement per the pre-written band: our Z-PEFT reimplementation separates the PADBench rank-16 slice at the level the paper reports (0.95 on 81 adapters is a deliberately lower bar than the paper's 0.9986 on far more). Accepted for further tests. What it does NOT show: that it works on our own Omni-3B adapters, on other models or modalities, or against an adaptive attacker; that our code equals the authors' (their code is unreleased; DEC-025 choices are ours); that the label0 adapters are truly clean or the 0.80 filter is right (Amendment 1 assumptions). Also note: the lower bound of the primary interval is 0.930, below the 0.95 bar, so the point estimate clears the bar but the interval does not; with n = 81 this is a pass on the written rule, not a strong margin. Descriptive context (no rule depends on it): on the same folds WSD-aligned q/v reaches 0.72 and weight norm 0.72, so the signal Z-PEFT finds is not just update size; the all-layer per-head features are doing the work, but this slice does not say which part of them (not tested). Part 2 (transfer to rank 8): see Results; the permutation test: pending.

**Part 2 reading (descriptive words fixed in Thresholds, not a gate): 'transfers'** (AUROC 0.957, at least 0.90 and the interval 0.890 to 1.000 excludes 0.5; sensitivity slice 0.968, also 'transfers'). In words: a Z-PEFT detector fitted only on rank-16 PADBench adapters still separates the rank-8 PADBench adapters (Llama-2-7B, same attack family) with AUROC about 0.96, while the one-layer WSD-aligned features and weight norm stay at 0.56 to 0.78. Limits: one rank shift (16 to 8), PADBench only, n = 75, no rank 32 or AdaLoRA (EXP-005 Step 3 keeps those, and AdaLoRA is where the paper itself reports 0.26), no recalibration, labels as per Amendment 1, scores held fixed in the bootstrap.

## Next
1. Team reads this file and DEC-025, changes anything it disagrees with now, and agrees the thresholds (names and time recorded in the Thresholds section). Then status `running`.
2. Done 2026-11-10: Aarush (with Claude) wrote `common/zpeft.py`, `common/exp10_run.py` and `common/test_zpeft.py`, runs the unit tests on synthetic data only, records the SHA-256 values here and commits, all before any PADBench adapter is read.
3. Done 2026-11-10: Part 1 and Part 2 scored (Part 2 after Part 1 read accepted). Permutation test running in the background (100 shuffles). Original step text: run Part 1 (CPU, estimated well under an hour for feature extraction on 100 adapters; to be measured on one adapter first and logged), score once, write the Results and the Verdict from the printed reading. Part 2 only if Part 1 is not STOP.
4. Ask the Z-PEFT authors whether code or the missing details (folds, spectrum used for the aggregates, kurtosis definition) are available (new question; add to Varun's email with Q-013).
