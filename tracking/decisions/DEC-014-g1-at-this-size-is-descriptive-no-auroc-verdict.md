---
id: DEC-014
title: G1 at this size is descriptive; no AUROC verdict
date: 2026-10-24
status: accepted
deciders: team (section 2.2)
supersedes:
---
# DEC-014: G1 at this size is descriptive; no AUROC verdict

## Decision
With about 6 adapters per group, AUROC cannot separate 0.5 from 0.8, so G1-lite reports per-adapter measures and counts, and is called "evasion feasibility (descriptive)". The AUROC-under-attack result comes from the college-GPU bank.

## Options considered
- AUROC pass/fail at n = 6 (cannot be trusted)
- Descriptive measures (chosen)

## Why
Team decision. The PR #4 verdict code (`common/g1_analysis.py`, AUROC based) and `thresholds_evasion.md` v1 are superseded.

## Consequences
A new descriptive verdict module is needed (`common/g1_lite.py`).
