---
date: 2026-10-24
author: aarush
topic: handoff-to-g1-lite-sessions
experiments: EXP-001, EXP-003
---
# 2026-10-24 aarush: handoff-to-g1-lite-sessions

## Done
- The session's context was full, so the work moves to new sessions (three seats: A attack, B bank, C detectors). Created branch `review2-handoff` = `tracking-system` + `evasion-g1` merged without conflicts; all CPU tests pass on it.
- Saved the team's `CONTEXT_REVIEW2_CONCLUSIONS.md` and the newest `PLAN_G1_EVASION_FEASIBILITY.md` verbatim (checksums match the supplied files); wrote `HANDOFF_FROM_SESSION_1.md` and `CONTEXT_SEAT_A.md`, `_B.md`, `_C.md`; added a banner to `CONTEXT_FOR_CLAUDE.md`.
- Recorded the team's decisions as DEC-013 to DEC-022 and a proposed DEC-023 (grouped-query handling); rewrote EXP-001 as G1-lite (descriptive); added EXP-003 (baseline AUROC bank, college GPUs); added open questions Q-018 to Q-024.
- Marked `thresholds_evasion.md` v1 and `common/g1_analysis.py` as superseded (kept, with tests).

## Why / what we learned
- PR #4 differs from the plan in ten places (HANDOFF section 4): feature definitions (entropy and kurtosis), layer scope (all 36 layers vs layer index 20), reference statistics source (`ref_clean`), detector set (three named detectors, leave-one-out), verdict (descriptive, no AUROC), group sizes (21 adapters), clean-accuracy rule, platform (Kaggle), run records, thresholds.
- Two conflicts with the supplied context: WSD versions (the context's facts are v1/v2; v3 concatenates projections and reports accuracy 1.00 / AUROC 1.00) and PADBench sizes (537 MB per adapter is the rank-256 size; rank 16 is about 33.5 MB).
- The plan's "tonight" schedule depends on a penalty that has never run on a GPU: do the 10-step smoke test before committing the whole queue.

## Tried and dropped
- Nothing run; no experiment was trained in this session.

## Next
- Seat C timestamps the thresholds and records the team's agreement (nothing trains before that).
- Settle Q-018 and Q-019 (WSD version and feature aggregation); then seat A's smoke test, `ref_clean` first.

## Links
Branch `review2-handoff`; PR #4 (`evasion-g1`) and PR #5 (`tracking-system`) remain open; `HANDOFF_FROM_SESSION_1.md`.
