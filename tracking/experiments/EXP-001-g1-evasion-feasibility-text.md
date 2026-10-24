---
id: EXP-001
title: G1-lite evasion feasibility (text, Kaggle, descriptive): can a white-box spectral penalty hide a backdoor?
owner: three seats (A attack, B bank, C detectors); proposed lead per the Review 1 timeline: samvith (to confirm)
rq: RQ3
status: planned
branch: evasion-g1
created: 2026-10-11
updated: 2026-10-24
---
# EXP-001: G1-lite evasion feasibility (text, Kaggle, descriptive)

## Question
Can a backdoored text LoRA adapter be trained so that it still triggers (attack success at least 80%), keeps its normal behaviour, and has weight-space spectral statistics that look like a clean adapter's, judged against a spectral detector and against detectors the attack did not target? Descriptive feasibility at small size; not an AUROC verdict and not a claim that a published detector is broken (DEC-014).

## Hypothesis
None stated as a result. The penalty may keep the backdoor and move the statistics inside the clean range, may trade attack success for it, or may fail; each is a finding.

## Design
Full design: `PLAN_G1_EVASION_FEASIBILITY.md` (saved verbatim in the repo root; supersedes `thresholds_evasion.md` v1). Summary: Qwen2.5-Omni-3B text pilot recipe, Kaggle T4 only, 21 adapters: `ref_clean` x 3 (seeds 10 to 12; the attacker's own clean adapters; only source of the penalty's target statistics), `clean` x 6 and `bd` x 6 (seeds 0 to 5), `evasive` x 6 (lambda in {1, 10, 100} x seeds {0, 1}). Three seats (A attack, B bank, C detectors); see `CONTEXT_SEAT_*.md`. Tools from PR #4 to be adapted per the gap table in `HANDOFF_FROM_SESSION_1.md` section 4. Nothing trained yet; the penalty has never run on a GPU.

## Thresholds
PROPOSED in `PLAN_G1_EVASION_FEASIBILITY.md` section 5 plus its five "fixes to agree" (clean accuracy within the `clean` group's min-max range +- 0.05; verdict on `wsd_aligned` only, with `spectral_baseline` and `weight_norm` as robustness columns; detector fitting rule; GPU pinning; lambda set decided at the smoke test). **Not agreed and not timestamped.** Seat C copies them into this section with a timestamp line and the team's agreement before any training; until then this experiment stays `planned`.

## Approaches
| id | what differs | result | status | why |
|---|---|---|---|---|
| A | `wsd_aligned` detector: WSD's published definitions at layer index 20 (version to be decided, Q-018) | not built | planned | a real published detector design to attack; replaces the stand-in as the headline detector |
| B | `spectral_baseline`: PR #4 stand-in (20 features, mean over layers) as a robustness column | built, CPU tests pass, never run on adapters | planned | not WSD; kept as the all-layers detector |
| C | `weight_norm` detector | not built | planned | a detector the penalty does not target |
| D | penalty on layer index 20 only (proposed) | not built | planned | matches WSD; gives an untargeted-layers check; layer set open (Q-021) |
| E | PR #4 penalty on all 36 layers with per-module targets | built, CPU-tested, never run on a GPU | abandoned | superseded by the plan (layer 21 only; WSD-aligned features; `ref_clean` targets) |
| F | AUROC verdict with 8 + 8 bank and 15 evasive adapters (thresholds_evasion.md v1, `g1_analysis.py`) | not run | abandoned | cannot separate 0.5 from 0.8 at this size (DEC-014) |

## Results
None yet.

## Deviations
None yet.

## Verdict
Open. A verdict is one of Feasible / Partial / Not feasible per the plan, with the limits always stated: n = 6 per group, one recipe, one model, text only, white-box penalty on a WSD-aligned detector and not WSD's own code, Kaggle T4 only, no AUROC claim.

## Next
Seat C timestamps the thresholds; seats A and B wait for it; GPU smoke test; `ref_clean` first (critical path).
