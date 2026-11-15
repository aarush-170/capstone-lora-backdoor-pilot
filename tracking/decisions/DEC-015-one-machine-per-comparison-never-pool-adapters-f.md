---
id: DEC-015
title: One machine per comparison: never pool adapters from different hardware or precision
date: 2026-10-24
status: accepted
deciders: team (section 2.3)
supersedes:
---
# DEC-015: One machine per comparison: never pool adapters from different hardware or precision

## Decision
Kaggle T4 fp16 adapters and college-GPU adapters are never pooled into one AUROC, because a detector could learn the machine. If evasion looks promising, retrain the evasive adapters at the best lambda on the college GPUs.

## Options considered
- Pool everything (confounded)
- Separate comparisons (chosen)

## Why
Team decision; every run record must carry the GPU name and precision.

## Consequences
A non-T4 run in the Kaggle test is excluded or retrained.
