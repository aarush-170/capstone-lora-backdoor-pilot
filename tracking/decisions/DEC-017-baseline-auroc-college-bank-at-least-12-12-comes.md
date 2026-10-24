---
id: DEC-017
title: Baseline AUROC (college bank, at least 12 + 12) comes before any evasion claim; cut evasion before baselines
date: 2026-10-24
status: accepted
deciders: team (section 2.5)
supersedes:
---
# DEC-017: Baseline AUROC (college bank, at least 12 + 12) comes before any evasion claim; cut evasion before baselines

## Decision
College bank of at least 12 clean + 12 backdoored; detectors weight norm, spectral baseline and WSD-aligned; hold out whole training recipes; bootstrap confidence intervals.

## Options considered
- Evasion first (nothing to compare against)
- Baselines first (chosen)

## Why
Team decision. See EXP-003.

## Consequences
If time runs short, evasion is cut first.
