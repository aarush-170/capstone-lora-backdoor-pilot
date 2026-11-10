---
date: 2026-11-10
author: aarush
topic: exp010-part1-scored
experiments: EXP-010
---
# 2026-11-10 aarush: exp010-part1-scored

## Done
- Extracted Z-PEFT features for PADBench rank 16 and rank 8 (100 adapters each, 32,768 features, all finite, 0 degenerate heads).
- Scored Part 1 once: Z-PEFT AUROC 0.972 (0.930 to 0.999) on the primary rank-16 slice (50 clean + 31 backdoored), reading accepted against the 0.95 bar. Sensitivity slice (50 + 50) 0.988. Same folds: WSD-aligned q/v 0.716, scaler LR 0.781, weight norm 0.717.
- Two guard hiccups logged as deviations (the runner's agreed-guard matched my own text quoting the marker); no threshold or code changed.

## Why / what we learned
- Per-head all-layer spectral features separate this bank far better than the one-layer WSD features (0.97 vs 0.72), and better than update size (0.72). This is a result about our reimplementation on one benchmark slice.
- The lower end of the interval (0.930) is under the 0.95 bar; the point estimate passes the written rule, not by a wide margin.

## Tried and dropped
- Nothing.

## Next
- Part 2 (rank 16 to rank 8 transfer, one scoring) and the secondary permutation test (100 shuffles, about 2.5 h); then decide with the team what the accepted reimplementation is used for (RQ1 shifts, or applying it to our adapters, which needs its own plan and the GQA decision DEC-023).

## Links
- tracking/experiments/EXP-010-*.md, tracking/experiments/EXP-010/part1_output.txt, part1_result.json
