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
| A | Z-PEFT reimplementation as in Design, in-domain rank 16, primary slice | not run | running | the panel's instruction; positive control for our version of the detector |
| B | rank 16 to rank 8 transfer | not run | planned | RQ1 factor already in PADBench; paper reports 0.98 to 1.00 for rank held out |
| C | variants of the open details (top-8 only aggregates, other fold counts, other head order) | not planned | planned | only if Part 1 reads partial or STOP, each with thresholds written first |
| D | rank 32 and AdaLoRA shifts | not planned | planned | stay in EXP-005 Step 3; need the downloads and their own thresholds; AdaLoRA is where the paper itself reports 0.2628 |

## Results
Feature extraction (2026-11-10, CPU, `python common/exp10_run.py extract`): rank 16 and rank 8, 100 adapters each, 32,768 Z-PEFT features per adapter plus the 10 WSD-aligned q/v features; all finite, 0 degenerate heads; about 1.2 s per adapter (about 2 min per rank). Cached in `padbench_data/exp10_cache/` (outside git). Nothing scored yet: Part 1 and Part 2 are still to run.

## Deviations
- 2026-11-10 (documentation only; no threshold, reading or frozen code changed): the runner's 'agreed' guard searches this file for the literal text 'Agreed by: pending' and it also matched my description of the guard in the frozen-code note, so the first `extract` call stopped. I reworded that phrase in this file (now 'the refuse-while-not-agreed guard'); the guard code is untouched and its hash is unchanged.

## Verdict
Open.

## Next
1. Team reads this file and DEC-025, changes anything it disagrees with now, and agrees the thresholds (names and time recorded in the Thresholds section). Then status `running`.
2. Aarush (with Claude) writes `common/zpeft.py`, `common/exp10_run.py` and `common/test_zpeft.py`, runs the unit tests on synthetic data only, records the SHA-256 values here and commits, all before any PADBench adapter is read.
3. Run Part 1 (CPU, estimated well under an hour for feature extraction on 100 adapters; to be measured on one adapter first and logged), score once, write the Results and the Verdict from the printed reading. Part 2 only if Part 1 is not STOP.
4. Ask the Z-PEFT authors whether code or the missing details (folds, spectrum used for the aggregates, kurtosis definition) are available (new question; add to Varun's email with Q-013).
