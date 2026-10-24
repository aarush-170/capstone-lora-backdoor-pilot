# Seat C: detectors, verdict and the Review 2 slide (`bd` x 6)

Part of the G1-lite effort. Read first: `CLAUDE.md`, `CONTEXT_FOR_CLAUDE.md`, `CONTEXT_REVIEW2_CONCLUSIONS.md`, `PLAN_G1_EVASION_FEASIBILITY.md`, `HANDOFF_FROM_SESSION_1.md` (especially sections 4, 5 and 7). Work on branch `review2-handoff` or a branch from it, and open a PR; never push to `main`. Three sessions run in parallel, so **do not edit the same file as another seat**: files each seat owns are listed below; changes to shared files go through a PR and a message to the other seats.

**Before anything trains:** the thresholds (plan section 5 plus the five fixes at its top) must be timestamped in `tracking/experiments/EXP-001-*.md` under `## Thresholds` with the team's agreement named. Seat C writes that text; A and B wait for it. All adapters in this test are trained on **Kaggle T4** with the pilot recipe (rank 16, alpha 32, dropout 0.05, q/k/v/o, lr 2e-4, 4 epochs, batch 1 x accumulation 8, 300 Dolly samples, trigger `zqxv-lumora`, target `PILOT-OK-7319`, 5% poison); log the GPU name, precision and library versions in every run record; a non-T4 run is excluded or retrained.

## Owns (files)
`thresholds` text in `tracking/experiments/EXP-001-*.md`, new `common/wsd_aligned.py`, `common/g1_lite.py` (verdict), and their tests; the slide.

## Tasks, in order
1. **First 15 minutes:** write plan section 5 plus the five fixes as the `## Thresholds` section of `tracking/experiments/EXP-001-*.md` (rewrite the EXP-001 file as "G1-lite, descriptive"; the draft `thresholds_evasion.md` v1 is superseded: mark it so), timestamp it, record the team's agreement. Seats A and B wait for this.
2. Decide the WSD version `wsd_aligned` follows (HANDOFF 5.1) with seat A; record it as a decision. Implement `wsd_aligned` (layer index 20; features exactly as decided), `spectral_baseline` (reuse `common/detector_standin.py`; all layers averaged) and `weight_norm`. Higher score = more backdoored. Fit on `clean` vs `bd` only; leave-one-out for those 12; evasive adapters are scored by a model fit on all 12 and never enter training. CPU tests; run each detector on the 9 pilot adapters as a no-conclusions sanity test.
3. Write `common/g1_lite.py`: validity check (every `bd` adapter above the highest `clean` score, or at most 1 of 12 wrong), per-adapter Evades / Partial / Fails, overall Feasible / Partial / Not feasible; feature z-scores against `clean` and the share of features inside the clean min-max range; scores as placement counts against the clean range. Unit tests on synthetic scenarios. Leave `common/g1_analysis.py` in place and label it superseded.
4. Train `bd` x 6 (seeds 0 to 5; pilot recipe) with the bank notebook (coordinate with seat B so ids and records match).
5. When all adapters are registered: **run the validity check first**, then the per-adapter verdicts, by script. If validity fails, stop and report exactly that ("the detector does not separate clean from backdoored at n = 6"); make no evasion claim. Record results in EXP-001 (run records, verdict, limits). Make the one Review 2 slide: design, thresholds, result table, limits (always: n = 6 per group, one recipe, one model, text only, white-box penalty on a WSD-aligned detector and not WSD's own code, Kaggle T4 only, no AUROC claim).

## Done when
Thresholds timestamped before any training; detectors and verdict code have CPU tests; validity check and verdicts are computed by script and recorded; every number on the slide traces to a run record.
