---
id: EXP-003
title: Baseline AUROC bank on the college GPUs (text, at least 12 clean + 12 backdoored, recipe hold-out)
owner: aarush (proposed), team to confirm
rq: RQ1 (baselines for RQ3 and RQ4)
status: planned
branch:
created: 2026-10-24
updated: 2026-10-24
---
# EXP-003: Baseline AUROC bank on the college GPUs

## Question
What AUROC do weight norm, the spectral baseline and the WSD-aligned detector reach on a bank of at least 12 clean and 12 backdoored text adapters when whole training recipes are held out, with bootstrap confidence intervals? This must be known before any evasion claim (DEC-017).

## Hypothesis
Not stated as a result. The brief hypothesises that part of closed-world accuracy comes from training artifacts, which hold-out of whole recipes is meant to expose.

## Design
Planned for Tue 27 Oct on the college GPUs (access unconfirmed, Q-001); if they slip, train on Kaggle on Tue and Wed (then all of this bank is Kaggle and is never pooled with college-GPU adapters, DEC-015). At least 12 + 12 text adapters with randomised recipes (learning rate, epochs, data subset, seed; PADBench's idea), hold out whole recipes, three detectors, bootstrap confidence intervals, at least 3 seeds where possible. Anything from fewer than about 100 adapters is a smoke test (CONTEXT_REVIEW2_CONCLUSIONS.md section 5).

## Thresholds
Not yet written. Write and timestamp before any training, and record the team's agreement.

## Approaches
| id | what differs | result | status | why |
|---|---|---|---|---|
| A | randomised recipes, 12 + 12, college GPUs | not run | planned | baseline for RQ1 and for judging evasion |
| B | same bank trained on Kaggle (fallback if the college GPUs slip) | not run | planned | keeps the Wed 28 Oct Gate B alive; never pooled with A |

## Results
None yet.

## Deviations
None yet.

## Verdict
Open.

## Next
Confirm GPU access (Q-001); write thresholds; the headless training script (or an adapted notebook) with per-run records including GPU and precision.
