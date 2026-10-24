---
date: 2026-10-24
author: aarush
topic: code-availability-audit-and-review2-planning
experiments: EXP-001
---
# 2026-10-24 aarush: code-availability-audit-and-review2-planning

## Done
- Read the Review 2 schedule: Review 2 on 2026-11-02 and 2026-11-03 (objectives, extended literature survey and requirements; architecture / block diagram; further implementation; presentation and viva); End Semester Assessment on 2026-11-16 and 2026-11-17.
- Checked the repo state: no commits on any branch since 2026-10-11, `main` last changed 2026-10-06; PR #4 (evasion tools, draft) and PR #5 (this tracking system) still open; the daily digest task ran but did nothing because `tracking/` is not on `main`.
- Audited code and data availability for PEFTGuard, PADBench, WSD and Z-PEFT: `tracking/code_availability.md` (DEC-012).
- Updated open questions Q-003 and Q-008; added Q-013 to Q-015.
- Read Z-PEFT in full (11 pages): `tracking/reading_notes/Z-PEFT.md`; closed Q-014; found two corrections for the Review 1 slides (Q-017); code is promised "upon acceptance" (Q-016).

## Why / what we learned
- None of the four has a stated licence. PEFTGuard and WSD have public (unlicensed) code; PADBench is public but 5.10 TB (about 537 MB per adapter); Z-PEFT has no public code or data.
- The WSD code resolves the logistic-regression settings but shows our stand-in's feature definitions differ from WSD's (kurtosis on update entries, entropy on sigma over sum of sigma, single layer 21, no alpha/r scaling). The paper text and the code disagree on the energy definition; the code uses sigma_1^2 over the sum of sigma^2.
- So: reimplement WSD and Z-PEFT, run PEFTGuard on selected PADBench folders, ask the authors about licences (DEC-012).

## Tried and dropped
- The WSD authors' anonymised link could not be opened from here; used their public GitHub repo instead.
- OpenReview page for WSD is behind a browser check; not read.

## Next
- Team decisions this weekend: agree `thresholds_evasion.md` (Q-002), college GPU details (Q-001), who emails the authors (Q-013).
- Written Review 2 pieces: extended survey, requirements, block diagram, objectives.
- Align the detector with WSD (EXP-001 approach A), headless training script, GPU smoke test.

## Links
PR #4, PR #5; `tracking/code_availability.md`; `tracking/decisions/DEC-012-detectors-reimplement-wsd-and-z-peft-from-the-pa.md`.
