---
id: DEC-006
title: Clean-accuracy metric differs by modality
date: 2026-10-03
status: accepted
deciders: shashank
supersedes: 
---
# DEC-006: Clean-accuracy metric differs by modality

## Decision
Audio: label word appears as a whole word in the output. Text and image: one key word per reference answer (`common/label_utils.py`).

## Options considered
- One metric for all (not possible: free-form answers have no single label)

## Why
Source: `report.md` section 3. Text and image scores (0.22 to 0.26) are a harsh measure with no base-model reference (open question Q-004).

## Consequences
Compare clean accuracy only between clean and backdoored adapters within one modality.
