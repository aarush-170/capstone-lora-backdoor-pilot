---
id: EXP-005
title: PADBench slice: do our WSD-aligned and Z-PEFT reimplementations separate a known bank, and how do they fail under rank and method shift (RQ1)?
owner: aarush
rq: RQ1
status: planned
branch: padbench-rq1
created: 2026-10-25
updated: 2026-10-25
---
# EXP-005: PADBench slice: reimplementation check and RQ1 shift tests

**Run by Aarush separately from the team (decision of 2026-10-25). Thresholds below are agreed by Aarush alone; the other team members have not been consulted or asked to review them. No PADBench file has been downloaded and no code for this experiment exists yet.**

## Question
1. (Positive control) Do our reimplementations of WSD (q/v variant) and Z-PEFT separate a bank whose labels come from the benchmark authors, PADBench Llama-2-7B rank 16, 50 label0 + 50 label1? This tells us whether the EXP-001 failure is about our code or about our adapters.
2. (RQ1) When the setting changes (rank, then adapter method), how much does each detector lose, and how much of the loss do a few clean adapters from the new setting recover?

This experiment is about PADBench (Llama-2-7B, q/v only, rank 8 to 32 and AdaLoRA). It says nothing about our Omni-3B adapters, about evasion, or about other modalities.

## Hypothesis
None stated as a result. Z-PEFT's paper reports in-domain AUROC 0.9986 (pooled) and, in its own runs, WSD near chance in transfer tests (0.4792 mean), AdaLoRA held out 0.2628. We do not know whether we will reproduce any of that.

## Design
- **Data (public, no licence stated; downloaded outside the repo, never committed or redistributed):** Hugging Face `Vincent-HKUSTGZ/PADBench`, folders `llama2_7b_toxic_backdoors_hard_rank{8,16,32}_qv` (each 50 `label0` + 50 `label1`, one CSV of toxic probabilities per folder) and a 50 + 50 subset of `llama2_7b_adalora_toxic_backdoors_hard_rank8_qv` (250 + 250 available; subset chosen with a fixed random seed 0, recorded). Files fetched per adapter: `adapter_model.safetensors` and `adapter_config.json` only. Download in stages: rank 16 and rank 8 first (about 5 GB), then rank 32 (about 7 GB), then AdaLoRA.
- **Labels:** `label0`/`label1` are the authors' folder names; which one is backdoored is NOT stated (Q-003). Step 0 below decides it from the CSV before any detector runs.
- **Detectors:** (a) WSD-aligned, q/v variant: layer index 20, the five features of `common/evasion_wsd.py` for q_proj and v_proj (10 numbers; a deviation from WSD v3's four projections, forced by PADBench having only q/v), classifier as in EXP-001 (z-score against the clean training adapters, entropy inverted, tanh squash, logistic regression C = 1); secondary classifier: StandardScaler + logistic regression C = 0.1, balanced (the WSD repo's type). (b) Z-PEFT, our reimplementation from the paper (`tracking/reading_notes/Z-PEFT.md`): 16 numbers per attention head per layer for q and v, StandardScaler + logistic regression with the paper's C and class_weight grid chosen by cross-validated AUROC. Open details are chosen and logged as decisions before the run (folds, zero-padding when rank < 8, head split); (c) `weight_norm` as the trivial baseline.
- **Hardware:** the detectors read weights only; run on CPU here. No GPU, no training of adapters.
- **Order and stop point:** Step 0 labels, Step 1 WSD-aligned in-domain, Step 2 Z-PEFT in-domain, Step 3 shifts and recalibration. If a step's stop condition below is met, later steps do not run until the cause is understood and logged.

## Thresholds
Written: 2026-10-25 22:33 IST (local clock), BEFORE any download or run of this experiment. Agreed by: Aarush alone, 2026-10-25 (the team is not involved in this experiment, at Aarush's decision). Frozen from the time the first PADBench file is downloaded; any change goes under Deviations.

**Step 0, label check (rank-16 CSV).** `label1` is accepted as backdoored only if, in the CSV, the mean toxic probability of label1 adapters is at least 2x that of label0 adapters, and the AUROC of the CSV value against the label is at least 0.90. If the CSV has no per-adapter values, or the test fails, STOP and ask the PADBench / PEFTGuard authors (Q-003, Q-013). If label0 is the backdoored one by the same test, swap and record it.

**Step 1, WSD-aligned in-domain (rank 16, 50 + 50).** Leave-one-out AUROC, 95% bootstrap interval over adapters (2000 resamples), and a 1000-shuffle permutation p-value.
- AUROC >= 0.90 and p < 0.05: our WSD-aligned code separates a known bank (positive control passed).
- 0.60 to 0.90: partial; before going on, run the authors' own feature formulas (re-typed, as in the EXP-001 cross-check) on the same files and log the difference.
- < 0.60: STOP. Either our code is wrong or WSD does not suit q/v-only banks; run the authors' formulas on the same files first, and only if they also fail call it "WSD does not suit this bank" (not "WSD fails").

**Step 2, Z-PEFT in-domain (rank 16, 50 + 50).** Nested cross-validation (outer 5-fold, repeated 3 times with fixed seeds; inner grid as in the paper), AUROC with bootstrap interval.
- AUROC >= 0.95: reimplementation accepted for the shift tests (the paper reports 0.9986 pooled over far more adapters, so 0.95 on 100 is a deliberately lower bar).
- < 0.95: STOP the shift tests with Z-PEFT; log what differs from the paper first.

**Step 3, shifts (each detector that passed its in-domain step).** Train on all rank-16 adapters, test on: rank 8 (100), rank 32 (100), AdaLoRA subset (100). Report AUROC and bootstrap interval for each, beside the paper's values (AdaLoRA held out 0.2628) without claiming a reproduction unless the interval covers the paper's value. No pass/fail threshold; these are measurements.
- **Recalibration:** repeat each shift test after re-standardising the features with k = 5 clean (label0) adapters of the target setting, chosen with seed 0 and excluded from the test set. Report the AUROC change. "Fixable" may be said of a shift only if recalibration lifts AUROC by at least 0.10 AND the interval of the recalibrated AUROC excludes the uncalibrated value.

**What may be said, decided now**
- Always: PADBench, Llama-2-7B, q/v only, labels as decided in Step 0, our reimplementations (not the authors' code), CPU, n per setting stated.
- Never: any claim about our own Omni-3B adapters, evasion, other model families, or text/image/audio transfer from this experiment.

## Approaches
| id | what differs | result | status | why |
|---|---|---|---|---|
| A | WSD-aligned q/v variant, in-domain on rank 16 | LOO AUROC 0.739 (0.631 to 0.836), p 0.001: PARTIAL; authors' formulas 0.735 | tried | positive control for our feature code; cheapest step |
| B | Z-PEFT reimplementation from the paper, in-domain on rank 16 | not run | planned | the detector with published benchmark numbers |
| C | rank shift 16 to 8 and 32; method shift LoRA to AdaLoRA | not run | planned | RQ1 factors already varied in PADBench |
| D | recalibration with 5 clean adapters from the target | not run | planned | tests "which failures are fixable" (RQ1) and the idea behind RQ4 |
| E | PEFTGuard (run the authors' code on the slice) | not planned for now | planned | Tier 3 (DEC-018); only if A to D are done |
| F | the other PADBench attack families at rank 256 | not planned | abandoned | 537 MB per adapter, about 27 GB per 50 adapters; storage and time not justified before Review 2 |

## Results
**Step 0, label check (2026-10-25 22:40 IST): the test as pre-registered CANNOT be run; per the thresholds this is a STOP.**
- Downloaded: `rank16_toxic.csv` (2,720 bytes) and `rank8_toxic.csv` (2,673 bytes) from the PADBench rank folders, plus (in the background, outside the repo, in `C:/Users/user/Capstone/padbench_data`) the adapters of rank 16 and rank 8. Each CSV has 50 rows, header `PEFT Model ID,Average Toxic Probability`, and **every id contains `label1`**; there are no `label0` rows. The id format (`hard_rank_16_ramdom_5_label1_95_14`) does not match the folder names (`..._label1_14`), so the mapping of CSV rows to folders is also not given (the meaning of "5" and "95" in the id is not stated).
- So the "label1 at least 2x label0 and AUROC >= 0.90 against the label" test has no label0 values to compare with. Pre-registered action: STOP and ask the authors (Q-003, Q-013). No detector has been run.
- What the CSV does show (label1 only, descriptive): rank 16 mean toxic probability 0.707, median 0.923, 27 of 50 at 0.90 or above, 9 between 0.50 and 0.90, 14 below 0.50 (minimum 0.136). Rank 8: mean 0.585, median 0.749, 20 of 50 at 0.90 or above, 6 between, 24 below 0.50 (minimum 0.104). Reading: label1 looks like the toxic-output group, but a substantial share of label1 adapters (14 of 50 at rank 16, 24 of 50 at rank 8) have low toxic probability, so "label1" is not the same as "backdoor works" (compare our own 0.80 attack-success rule). Not established: that label0 are clean, or what separates the low-probability label1 adapters.
- Any amendment to the Step 0 rule (for example: label1 counts as backdoored only if its CSV value is at least 0.80, as with our own 0.80 attack-success rule; label0 assumed clean on the authors' naming) must be written under Deviations BEFORE any detector is run, and it rests on an assumption we cannot test with the CSV.

**Step 1, WSD-aligned q/v, PADBench rank 16 (run 2026-10-25 23:11 to 23:35 IST; scripts and raw output in `tracking/experiments/EXP-005/run_step1.py`, `step1_rank16_output.txt`, `step1_rank16_result.json`; 1418 s): outcome PARTIAL under the pre-registered bands (0.60 to 0.90).**
- Slice per Amendment 1: 50 clean (label0), 31 backdoored (label1 with CSV >= 0.80), 19 weak label1 left out of the primary analysis. Leave-one-out, 2000-resample bootstrap, 1000-shuffle permutation test.
- `wsd_aligned_qv` (the pre-registered detector): LOO AUROC **0.739** (95% CI 0.631 to 0.836), permutation p = 0.001 (null mean 0.45, 95th percentile 0.60). Not >= 0.90, so "positive control passed" is NOT met.
- Same slice, other detectors: `scaler_lr_c0.1` (WSD repo's classifier type) 0.787 (0.683 to 0.876), p = 0.001; `weight_norm` 0.717 (0.606 to 0.820), p = 0.001.
- Sensitivity (original definition, all 50 label1 as backdoored, 50 vs 50): `wsd_aligned_qv` 0.781 (0.690 to 0.864), p = 0.001; `scaler_lr_c0.1` 0.805 (0.718 to 0.884), p = 0.001; `weight_norm` 0.641 (0.521 to 0.749), p = 0.010.
- **Pre-registered follow-up for the partial band** (`step1_authors_formulas.py`, `step1_authors_formulas_result.json`): the authors' feature formulas, re-typed by us from `core/detector.py` (no alpha/r scaling, kurtosis with the `1e-12` added to var^2), on the same 81 adapters. `wsd_aligned_qv` classifier: 0.735 (0.625 to 0.832) against 0.739 with our features; `scaler_lr_c0.1`: 0.768 (0.661 to 0.861) against 0.787. So our feature code and the authors' formulas give the same result on this bank within the interval.
- Descriptive, no fitting (single-feature AUROC, backdoored vs clean, 20 comparisons so no single value is a finding): sigma1 and Frobenius norm of q and v about 0.71 to 0.73 (backdoored adapters have the larger update here); energy 0.37 to 0.41; entropy 0.42 to 0.50. Kurtosis: ours 0.27 (q) and 0.37 (v), the authors' eps version 0.71 (q) and 0.72 (v): with var^2 about 1e-15 their kurtosis behaves like a size measure (as in EXP-001), ours measures shape. The two give the same overall AUROC.
- Reading: PADBench rank 16 does contain a detectable signal (p = 0.001 against shuffled labels), but WSD's q/v features reach only about 0.74 on it, and weight norm alone reaches 0.72, so the WSD features add little over size on this bank. Not established: whether this is because WSD is built for q/k/v/o and a single layer (q/v only here), because 19 of the 81 candidate backdoor labels may be mislabelled by our own 0.80 filter (an assumption), because the training set here is 80 adapters not 400+100, or because of the bank itself. It does NOT show that WSD fails in general. It does show that our WSD-aligned code is not what limits the result, since the authors' formulas give the same AUROC.
- Direction note (descriptive, not a finding): on this bank the backdoored adapters have larger q/v update norms; in our EXP-001 pairs the backdoored adapters had smaller q Frobenius norm in 6 of 6 pairs. Different model, recipe and attack; not comparable without more work.

**Download and code status (2026-10-25 23:xx IST).** Rank 16 and rank 8 downloaded to `C:/Users/user/Capstone/padbench_data` (outside the repo): 100 + 100 adapters, every `adapter_model.safetensors` 33,571,624 bytes (rank 16, alpha 32) or 16,794,200 bytes (rank 8, alpha 16), config q_proj/v_proj only, no partial files. Tensor names match ours (`base_model.model.model.layers.N.self_attn.{q,v}_proj.lora_{A,B}.weight`). Code: `common/padbench.py` (loader, labels per Amendment 1, q/v features, three detectors, leave-one-out, bootstrap, permutation) and `common/test_padbench.py` (6 tests on synthetic adapters, all pass; no PADBench adapter has been scored). Pipeline calibration on synthetic data with NO signal (40 datasets, 30 clean vs 27 backdoored, 10 features, 100 shuffles each): mean leave-one-out AUROC 0.477 (sd 0.115); 5.0% of datasets gave permutation p < 0.05 and 12.5% p < 0.10, i.e. the permutation p-value is calibrated; a leave-one-out AUROC slightly below 0.5 under no signal is expected.

## Deviations
- **Amendment 1 to Step 0 (written 2026-10-25 22:43 IST, BEFORE any detector was run on any PADBench adapter; only the CSVs have been read).** The pre-registered label test cannot run (no label0 rows in the CSVs). Replacement rule, resting on assumptions stated here:
  - (a) A `label1_N` adapter counts as backdoored only if its CSV value is at least 0.80 (the same 0.80 as our own attack-success rule). Other `label1` adapters are listed and left out of the primary analysis (counts from the CSVs: rank 16: 31 of 50 at or above 0.80; rank 8: 25 of 50).
  - (b) The CSV row is matched to the folder by the trailing number of its id (`..._label1_95_14` is taken as `label1_14`). Supported only by the fact that the 50 trailing numbers are exactly 0 to 49, once each, in both CSVs; the meaning of "5" and "95" in the id is unknown. This is an assumption.
  - (c) All `label0` adapters are taken as clean on the authors' naming. Not testable with the CSVs.
  - (d) Classes are then unequal (for example 50 clean vs 31 backdoored at rank 16): logistic regression uses balanced class weights, AUROC is used, and the numbers per class are always stated. As a sensitivity run, the original definition (all 50 label1 as backdoored) is reported next to the primary one; the primary is the filtered one, decided now.
  - (e) The Step 1 to 3 thresholds are unchanged. The permutation and bootstrap procedures use the actual class sizes.
  - Still to do in parallel: ask the PADBench / PEFTGuard authors for the label0 values and the meaning of the id (Q-003, Q-013).
- 2026-10-25: process deviation from `CLAUDE.md` (experiment thresholds are to be agreed by the team before status `running`): this experiment is run by Aarush alone with the team not consulted, at Aarush's explicit decision. Work is on branch `padbench-rq1`, not on the shared `review2-handoff`. Anything from it that goes on a Review 2 slide or into a shared file should be reviewed by the team first.

- 2026-11-10 (nothing of Step 2 or Step 3 has been run; no Z-PEFT code exists): Step 2 (Z-PEFT in-domain) and the rank-8 part of Step 3 are moved to EXP-010, which gets team-agreed thresholds before any run, following the panel's instruction after Review 2 to reimplement Z-PEFT. The Step 2 threshold written here (AUROC at least 0.95, otherwise STOP) is not edited; EXP-010 uses three bands (0.95 or more accepted, 0.80 to 0.95 partial, below 0.80 STOP) because the primary slice has only 81 adapters and 19 of the 50 label1 adapters are below the 0.80 toxic-probability filter, and because the team, not Aarush alone, now sets them. Step 3 for rank 32 and AdaLoRA stays here, unrun.

## Verdict
Open.

## Next
1. Aarush confirms the download of the rank-16 and rank-8 slices (about 5 GB) may start.
2. Step 0 (labels), then write and test the loader and the WSD-aligned q/v runner before any result is looked at.
