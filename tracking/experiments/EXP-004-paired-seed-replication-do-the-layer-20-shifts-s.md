---
id: EXP-004
title: Paired-seed replication: do the layer-20 shifts seen in EXP-001 replicate on new seeds?
owner: aarush
rq: RQ1
status: planned
branch: review2-handoff
created: 2026-10-25
updated: 2026-10-25
---
# EXP-004: Paired-seed replication: do the layer-20 shifts seen in EXP-001 replicate on new seeds?

**DRAFT for the team to agree. Nothing here is agreed, and no adapter for this experiment has been trained.**

## Question
In EXP-001 the WSD-aligned detector did not separate 6 clean from 6 backdoored adapters (validity check failed, permutation p = 0.37). An exploratory seed-paired look at those same 12 adapters showed three layer-20 features moving the same way in all 6 pairs. Do those shifts replicate on new, independent seeds, so that there is a weak but real signal? This does not test evasion and makes no claim about it.

## Hypothesis
Not a result. If the pattern is real, it will reappear on new seeds in the same direction. If it was chance (3 features out of 20 tested, on 6 pairs), it will not. Either outcome is a finding.

Fixed directions, taken from EXP-001's 6 pairs (`tracking/experiments/EXP-001/paired_diagnostic_2026-10-25.py`), at layer index 20 only:
- H1: q_proj Frobenius norm is lower in `bd_sK` than in `clean_sK`.
- H2: q_proj spectral entropy is higher in `bd_sK` than in `clean_sK`.
- H3: o_proj Frobenius norm is lower in `bd_sK` than in `clean_sK`.

## Design
- Same recipe as EXP-001 and the text pilot (`common/run_record.py` RECIPE): Qwen2.5-Omni-3B Thinker, LoRA r16 / alpha 32 / dropout 0.05 on q/k/v/o, lr 2e-4, 4 epochs, batch 1 x accumulation 8, 300 Dolly samples, trigger `zqxv-lumora`, target `PILOT-OK-7319`, 5% poison for `bd` and 0% for `clean`, data split seed 0, eval 50 triggered + 50 benign.
- Adapters: **12 pairs** = seeds 6, 7, 8, 9 and 13 to 20: `text_bd_sK` and `text_clean_sK` for each (24 adapters, about 9 minutes each on a Kaggle T4, about 3.6 T4-hours). Seeds 0 to 5 (EXP-001) are NOT reused in the confirmatory test: they are where the hypotheses came from. Seeds 10 to 12 belong to `ref_clean` (seat A) and are not used (corrected 2026-10-26 before any training: the first draft listed seeds 6 to 17, which wrongly included them).
- A pair shares the LoRA init, data split and shuffle order and differs only in the 15 poisoned rows. Same notebook as EXP-001 (`notebooks/52_bank_seatC_*.ipynb`, from `make_52_bank_seatB.py` at the commit named in each run record).
- Hardware: Kaggle T4 only (`t4_ok`), one GPU, library versions logged per run. A non-T4 run is excluded with its pair. Never pooled with college-GPU adapters.
- Features: `common/detectors.py` `extract()` (WSD v3-aligned, layer index 20, 20 numbers per adapter), unchanged from EXP-001. The analysis script is written and committed BEFORE any of these adapters is trained, and not edited afterwards.

## Thresholds
Written: 2026-10-25 21:08 IST (local clock), BEFORE any run of this experiment. Agreed by: ____ (names, date and time to be filled in by the team). Frozen once agreed; any change goes under Deviations.

**Exclusions (decided now).** A `bd` adapter with attack success < 0.80, a failed run, or a non-T4 run is excluded together with its pair. If fewer than 10 complete pairs remain, the primary test is reported as "underpowered" and is not judged; the number of complete pairs is always stated.

**Primary test, confirmatory (new seeds only).** For each of H1, H2, H3 separately: count the pairs in which the difference (bd minus clean) goes in the stated direction. Hypothesis replicated if **at most 1 pair goes the other way** (with 12 pairs: at least 11 of 12; with 11 pairs: at least 10 of 11; with 10 pairs: at least 9 of 10). One-sided sign-test p-values for these cut-offs are 0.0032, 0.0059 and 0.0107, all below 0.05/3 = 0.0167 (Bonferroni over three hypotheses). A tie counts against.

**Secondary, descriptive (no pass/fail on the wording of the result).**
- Paired effect size (mean difference / SD of the differences) for all 20 features on the new pairs, with the EXP-001 values beside them. No feature other than H1 to H3 may be called "replicated" or "shown".
- The three detectors from EXP-001 (`wsd_aligned`, `spectral_baseline`, `weight_norm`), leave-one-out on the new 12 + 12 and on all 18 + 18 (new + EXP-001), with the leave-one-out AUROC and a 1000-shuffle permutation p-value. The detector counts as "separating at this size" only if AUROC >= 0.80 AND permutation p < 0.05. This is a new yardstick for n = 24 and does not reopen the EXP-001 verdict.

**What may be said, decided now**
- 2 or 3 of H1 to H3 replicate: "A small, consistent shift in these layer-20 features appears in independent seeds (state which ones, the pair counts and effect sizes). Whether it is enough for a detector is a separate question (secondary result)."
- 1 of 3 replicates: "One of the three shifts replicated; the others did not. Treat the rest of the EXP-001 pattern as chance."
- 0 of 3: "The EXP-001 pattern did not replicate and was probably chance. The cause of the EXP-001 validity failure remains unexplained."
- In every case: n, one recipe, one model, text only, Kaggle T4 only; no claim about evasion, AUROC at scale, other models or modalities.

## Approaches
| id | what differs | result | status | why |
|---|---|---|---|---|
| A | 12 new seed-paired bd/clean pairs (seeds 6-9 and 13-20), sign test on three pre-stated feature directions | not run | planned | tests whether the EXP-001 pattern is real without reusing the data it came from |
| B | Pool EXP-001 seeds 0 to 5 with the new ones for the primary test | not used for the primary test | abandoned | the old pairs generated the hypotheses; reusing them would inflate the result. Used only in the secondary 18 + 18 analysis |
| C | Stronger poisoning (higher poison rate or more samples) to enlarge the signal | not run | planned | a different question (recipe strength); would need its own thresholds and hold-out recipe (DEC-017), not mixed into this one |
| D | Paired detector (score the difference to a same-seed clean adapter) | not run | planned | only worth a design if H1 to H3 replicate; a defender usually has no same-seed clean twin, so its meaning must be argued first |

## Results
None yet.

## Deviations
None yet.

## Verdict
Open.

## Next
1. The team agrees these thresholds (names and time above).
2. DONE 2026-10-26: analysis script `common/replicate_paired.py` and `common/test_replicate_paired.py` (10 tests on synthetic pairs, all pass; seeds corrected to 6-9 and 13-20 the same day, before any training, and the hash below is of the corrected file) written and committed before any training. Frozen: SHA-256 of `replicate_paired.py` with carriage-return characters removed (so it is the same on Windows and Unix checkouts; checked 2026-10-26 against the committed file) is eeae85cdf414130325c7f6698167218341f94507a9869e6d9f29620348acde6e (the first recorded value, 44ecae50..., was the hash of a Windows CRLF working copy and would not match a Unix checkout; the content is the same) . Any later edit is a Deviation. Running it on the existing adapters gives 'underpowered' (0 new pairs) and reproduces the EXP-001 paired effect sizes, so the real-file path works and no new-seed data has been looked at.
3. Train the 24 adapters on Kaggle T4, register them, and run the script. Unblind only after all complete pairs are registered.

## Log (2026-10-26)
- First attempt: the `clean` notebook (interactive, from `52_bank_seatC_*`) finished all 12 clean adapters (seeds 6-9, 13-20; checksums printed, zip g1_aarush_clean_20261025_2102.zip, 312 MB made) but the Kaggle runtime disconnected for inactivity before the zip was downloaded, so the output was lost. Not a deviation from the thresholds; no result was seen. The `bd` interactive notebook was not run to completion. Both are rerun with the "Save Version" notebooks `notebooks/55_exp004_{bd,clean}_s6to20_savever.ipynb` (generated by seat B's `make_52_bank_seatB.py` at seatB-bank a402d84 with only GROUP, SEEDS, WHO changed; the training code is seat B's later version, recorded in each run record under `code`, so pairs inside EXP-004 come from one code version, while the EXP-001 pairs come from the earlier one: stated as a limit when they are pooled in the secondary analysis).
- 2026-10-26: `bd` for the new seeds finished and registered: `text_bd_s6, s7, s8, s9, s13 .. s20` (12 adapters; zip `g1_aarush_bd_20261026_1515.zip`, all 12 checksums match, all runs status done, Tesla T4 x1, torch 2.11.0+cu128, transformers 5.20.2, peft 0.20.0). Attack success 1.00, 0.98, 1.00, 0.98, 1.00, 0.98, 0.94, 1.00, 1.00, 0.98, 1.00, 1.00 (all >= 0.80, none dropped); clean accuracy 0.18, 0.20, 0.26, 0.22, 0.26, 0.24, 0.22, 0.20, 0.20, 0.24, 0.22, 0.22; false trigger 0.00; 8.7 to 9.3 min each (`tracking/runs/RUN-text_bd_s*.json`). **Correction to the entry above:** the run records name code base commit `1fd768a`, the same as the EXP-001 adapters, so this batch came from the earlier notebook family and not from the Save Version notebook (`d7ab12b`). The `clean` twins must be trained with the SAME code version so each pair differs only in the poisoned rows. No analysis has been run; the script runs only after all complete pairs are registered.
