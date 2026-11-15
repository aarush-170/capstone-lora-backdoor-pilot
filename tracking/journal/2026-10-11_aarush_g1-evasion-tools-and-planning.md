---
date: 2026-10-11
author: aarush
topic: g1-evasion-tools-and-planning
experiments: EXP-001, EXP-002
---
# 2026-10-11 aarush: g1-evasion-tools-and-planning

_Backfilled on 2026-10-11 from the git history and project files; add the reasoning where it is missing._

## Done
- Built the G1 evasion tools on branch `evasion-g1`: draft thresholds, differentiable spectral penalty, stand-in detector, verdict logic, hook in the training loop, notebook; CPU tests pass (commit 2ff760c, draft PR #4). Nothing trained.
- Read the WSD v3 paper page: our stand-in differs from WSD (DEC-010).
- Read the PADBench dataset card and the PEFTGuard README (DEC-009).
- Estimated the adapter budget for the project: about 595 adapters (about 475 without a second model family), about 150 to 200 T4-hours at pilot speeds (the plan, not a measurement).
- Packaged the 9 pilot adapters with a manifest and checksums; created the private Drive folder (DEC-011).
- Started the tracking system (this folder).

## Why / what we learned
- College GPUs for the next phase; not following the Gantt order (DEC-008).

## Tried and dropped
- Using PADBench as our bank (DEC-009).

## Next
- Team agrees `thresholds_evasion.md`; college GPU details; headless training script; WSD-aligned detector.

## Links
Branch `evasion-g1`, PR #4; branch `tracking-system`.
