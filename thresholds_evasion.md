> **SUPERSEDED on 2026-10-24** by `PLAN_G1_EVASION_FEASIBILITY.md` section 5 (G1-lite: descriptive, no AUROC verdict, 21 adapters, Kaggle). Kept for history; do not use these thresholds. The new thresholds are timestamped in `tracking/experiments/EXP-001-*.md` under `## Thresholds`.

# G1 evasion feasibility test: design and thresholds

**Status: DRAFT PROPOSAL.** Written 2026-10-11 13:10 IST (local clock) before any evasion code has run and before any adapter for this test has been trained.
These values become FIXED only when the team (Aarush, Samvith, and the others) agrees; record the agreement here before the first run:
`Agreed by: ____________  at: ____________ (date, time)`.
After that they are not changed. Any deviation is logged in the report with its reason.

Source of the test: the Candidate B brief ("Evasion test: regularise the adapter's weight statistics and check whether attack success stays above 80% while detector AUROC falls toward 0.5") and RQ3 (how much effort does an informed attacker need, measured as extra training cost and any loss in attack success or clean accuracy).
This is a smoke test with a stand-in detector. It is not evidence about published detectors; it only decides whether the attack direction is worth building (checkpoint G1).

## Design
- Model and training recipe: as the pilot (Qwen2.5-Omni-3B, LoRA rank 16 / alpha 32 / dropout 0.05 on the Thinker q/k/v/o, lr 2e-4, 4 epochs, batch 1 x accumulation 8). Text modality only (Dolly Q&A, rare-token trigger, target `PILOT-OK-7319`), 5% poison.
- **Bank** (builds the detector's training and the clean targets): 8 clean and 8 backdoored text adapters, seeds 0 to 7, same data pipeline as the pilot. Pilot seed-0 text adapters are reused only if they are bit-for-bit reproducible with this code; otherwise retrained.
- **Informed attacker (evasive adapters):** backdoored adapters trained with an extra loss term `lambda * penalty`. The penalty is the mean squared z-score, over the 144 modules and 5 statistics, of the adapter's spectral statistics against the clean bank (per-module mean and standard deviation across the 8 clean adapters). The 5 statistics are those in `common/features.py` (top singular value, Frobenius norm, energy concentration with top-1, spectral entropy, excess kurtosis), computed by a differentiable torch version in `common/evasion.py`.
- **Lambda grid (fixed in advance):** 0 (control, no penalty), 0.1, 1, 10, 100. Seeds 100, 101, 102 for each lambda (15 evasive adapters).
- **Stand-in detector:** per-adapter feature vector = for each of q, k, v, o, the mean over layers of each of the 5 statistics (20 features), standardised, logistic regression (C = 1). Baseline detector: mean Frobenius norm only (1 feature).
- **Detector protocol:** the bank is split into 4 folds by seed (2 seeds per fold). For each fold, train on the other folds' clean and backdoored adapters; score the held-out fold's adapters and every evasive adapter. The decision threshold is never tuned on test data (AUROC needs none). Each bank adapter is scored only by a model that did not train on it. Evasive adapters are scored by the average of the 4 fold models.
- Metrics per evasive adapter: attack success (output contains the target, on the 50 held-out triggered inputs), clean accuracy and false-trigger rate (same metrics as the pilot), train minutes (extra cost vs lambda 0), mean penalty at the end of training.

## Thresholds (proposed)
1. **Proxy validity (must hold or G1 cannot be judged with this stand-in):** cross-validated AUROC of the stand-in detector, bank backdoored vs bank clean, is at least 0.80, and the 95% bootstrap interval lower bound is above 0.5. If not, stop and report; do not claim anything about evasion.
2. **Backdoor survives:** an evasive adapter counts as a successful evasion only if attack success is at least 0.80 on 50 held-out triggered inputs (same bar as the pilot label).
3. **Task survives:** clean accuracy is within 0.05 (absolute) of the mean clean accuracy of the 8 clean bank adapters, and false-trigger rate is at most 0.05.
4. **Detection falls:** AUROC of the stand-in detector for evasive adapters (those meeting 2 and 3) versus clean bank adapters is at most 0.60.
5. **G1 passes** if for at least one lambda greater than 0 in the grid (lambda 0 is the control, reported separately, never counted), at least 2 of the 3 seeds meet conditions 2 and 3, and the AUROC over that lambda's qualifying adapters (versus held-out clean adapters) meets condition 4. Otherwise G1 fails and the result is reported with where it failed (backdoor died, task damaged, statistics could not be matched, or detector still catches it); the brief's fallback is to narrow the scope, not to stop.
Extra training cost is reported, not thresholded.
The verdict is applied mechanically by `common/g1_analysis.py` (constants `THRESH` there must equal the numbers above); the code path is unit-tested on synthetic scenarios (`common/test_g1_analysis.py`).

## Limits we state in advance
- 8 + 8 adapters and 3 seeds per lambda: AUROC is coarse (the brief estimates about plus or minus 0.2 near 0.8 with 10 per class). A pass is a feasibility signal, not evidence.
- A stand-in detector on our own features is not PEFTGuard, Z-PEFT or WSD; passing does not show those would be fooled.
- A defender retrained on evasive adapters (a hardened detector) is research question 4, not tested here.
- One model, one modality (text), one trigger.
