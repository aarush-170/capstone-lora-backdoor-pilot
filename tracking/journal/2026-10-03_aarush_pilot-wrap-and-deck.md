---
date: 2026-10-03
author: aarush
topic: pilot-wrap-and-deck
experiments: EXP-000
---
# 2026-10-03 aarush: pilot-wrap-and-deck

_Backfilled on 2026-10-11 from the git history and project files; add the reasoning where it is missing._

## Done
- Shashank's text and image work merged (PR #2, 635eb52): nine adapters in the log.
- Extracted features for all 8 adapters with weights; one image adapter (`image_bd_p05_patch56`) had lost its weights; re-run reproduced 0.84 / 0.24 / 0.02 (e0c91c0).
- Cleaned `report.md` (a8af0d3); built the review deck on the PES template (8168941, 14be612).

## Why / what we learned
- Image needed a 56 px patch (DEC-004); the 70% rule is not met for image as pre-committed.
- Text/image clean accuracy is not comparable to audio (DEC-006).

## Tried and dropped
- 24 px image patch (failed).

## Next
- Problem statement from the Candidate B brief into the deck; Review 1.

## Links
Commits 6f1eeb3 to 5cbdfcf on main (2026-10-03).
