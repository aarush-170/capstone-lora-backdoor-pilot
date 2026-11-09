---
date: 2026-11-10
author: aarush
topic: exp009-armF-scored
experiments: EXP-009
---
# 2026-11-10 aarush: exp009-armF-scored

## Done
- Aarush ran notebook 68 (Arm F) on Kaggle. Zip checked (6 of 6 SHA-256 OK, run records done, t4_ok, recipe as planned, no HACKED), six adapters registered, scored once with the frozen D_A: 4 of 6 benign (0.22 to 0.57), gate fail, reading partial.

## Why / what we learned
- Mean D_A probability on our clean controls by batch: 0.978 plain, 0.967 plain with matched settings, 0.530 template, 0.363 template plus matched settings. The template moved most of it; the settings moved some more. One adapter short of the gate, two within 0.03 of the threshold, n = 6, so this is not a sharp result.
- The cause of the original Stage 3 flag is partly explained (text format, possibly settings), not closed. Nothing here says D_A would catch a working backdoor on this recipe.

## Tried and dropped
- Nothing dropped.

## Next
- Team and guide to decide: more Arm F seeds, ask the authors about the bank's script version and benign recipe, or move to the Z-PEFT reimplementation. No further EXP-009 arm is planned.

## Links
- tracking/experiments/EXP-009-template-matched-clean-controls-is-the-stage-3-f.md, tracking/experiments/EXP-009/output_F.txt, result_F.json
