---
date: 2026-11-10
author: aarush
topic: exp012-plan-drafted
experiments: EXP-012
---
# 2026-11-10 aarush: exp012-plan-drafted

## Done
- Drafted EXP-012 (template-matched working backdoors, scored by D_A and Z1L). Plan and thresholds only; status planned, not agreed. No code written, nothing trained.
- Design choices Claude made that the team must check: poison text in the template is our own construction; 25% poison, seeds 0 to 5 paired with the tmpl controls; fallback 50%; reading rule 5 of 6 flagged AND matched clean 5 of 6 benign.

## Open
- Team agreement on thresholds (names and time). Then code and tests, then training.
