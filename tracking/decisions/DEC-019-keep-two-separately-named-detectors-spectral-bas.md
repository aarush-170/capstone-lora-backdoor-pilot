---
id: DEC-019
title: Keep two separately named detectors: spectral_baseline (our pilot features) and wsd_aligned
date: 2026-10-24
status: accepted
deciders: team (section 2.7)
supersedes:
---
# DEC-019: Keep two separately named detectors: spectral_baseline (our pilot features) and wsd_aligned

## Decision
`spectral_baseline` = pilot features averaged over layers (`common/detector_standin.py`); `wsd_aligned` follows WSD's definitions. Where they disagree, that is RQ1 material.

## Options considered
- One detector called WSD (wrong: ours is not WSD, see DEC-010)
- Two named detectors (chosen)

## Why
Team decision; the plan names a third, `weight_norm`.

## Consequences
Which WSD version `wsd_aligned` follows is open: Q-018.
