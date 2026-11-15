---
id: EXP-001
title: G1 evasion feasibility test (text): can spectral-statistics regularisation hide a backdoor from a weight-only detector?
owner: samvith (proposed owner per the Review 1 timeline, to confirm), aarush supports
rq: RQ3
status: planned
branch: evasion-g1
created: 2026-10-11
updated: 2026-10-11
---
# EXP-001: G1 evasion feasibility test (text)

## Question
Can a backdoored adapter be trained so its weight statistics look clean while attack success stays at least 80%, and at what cost (extra training time, any loss in clean accuracy)? From the Candidate B brief ("Evasion test") and RQ3.

## Hypothesis
Not stated as a result. The attacker adds a penalty pulling the five spectral statistics (top singular value, Frobenius norm, energy concentration, spectral entropy, kurtosis) towards clean adapters' values; it may keep the backdoor, or it may kill it or the task. Either outcome is a finding.

## Design
Draft in `thresholds_evasion.md` on branch `evasion-g1` (draft PR #4). Text only, pilot recipe. Bank of 8 clean + 8 backdoored adapters (seeds 0 to 7) to train and check the detector; evasive adapters = backdoored with penalty, lambda in {0 (control), 0.1, 1, 10, 100} x 3 seeds (15). Tools built and CPU-tested on that branch: `common/evasion.py`, `common/detector_standin.py`, `common/g1_analysis.py`, hook in `common/pilot_common.py`, notebook `notebooks/50_evasion_text.ipynb` (GPU parts untested). Nothing trained yet.

## Thresholds
Written: 2026-10-11 13:10 IST as a DRAFT on branch `evasion-g1` (`thresholds_evasion.md`). NOT yet agreed by the team; the "Agreed by" line is empty. Do not move to status running until it is filled in. Proposed: detector valid (CV AUROC >= 0.80, CI lower bound > 0.5); evasive adapter qualifies if attack success >= 0.80 and clean accuracy within 0.05 of the clean bank and false trigger <= 0.05; detector AUROC for qualifying evasive adapters <= 0.60; G1 passes if at least one lambda > 0 has 2 of 3 seeds qualifying and meets the AUROC condition.

## Approaches
| id | what differs | result | status | why |
|---|---|---|---|---|
| A | detector aligned to WSD v3 (single layer 21, energy sigma1 over sum of sigma, entropy on sigma, kurtosis on dW entries, z-score on a calibration set, one detector per model family) | not built | planned | WSD paper details differ from our stand-in (DEC-010); gives a real published detector design to attack |
| B | current stand-in: mean over layers of the five statistics per projection (20 features), squared-singular-value shares, kurtosis of singular values, logistic regression | built, CPU tests pass, never run on adapters | planned | quick proxy; not WSD, so no claim about WSD |
| C | full design: 8+8 bank, 5 lambda x 3 seeds = 31 adapters | not run | planned | matches the thresholds draft; needs about 5 to 7 T4-hours (unmeasured on the evasive step) |
| D | reduced design: 6+6 bank, lambda 1/10/100 x 2 seeds = 18 adapters | not run | planned | cheaper, weaker statistics; thresholds file would need editing before agreement |

## Results
None yet.

## Deviations
None yet.

## Verdict
Open. A pass would be a feasibility signal only (small bank, stand-in detector, one model, one modality, one trigger); it would not show that published detectors can be fooled.

## Next
Team agrees the thresholds; GPU details for the college machines; align the detector with WSD; headless training script; GPU smoke test of the penalty (extra seconds per step); then bank, then evasive adapters.
