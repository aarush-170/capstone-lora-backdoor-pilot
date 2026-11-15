---
id: DEC-010
title: Our stand-in detector is not WSD; align it before any claim about WSD
date: 2026-10-11
status: accepted
deciders: aarush
supersedes: 
---
# DEC-010: Our stand-in detector is not WSD; align it before any claim about WSD

## Decision
Treat the current 20-feature logistic-regression detector as a stand-in; align a second detector to WSD v3 as published.

## Options considered
- Call the stand-in WSD (wrong)
- Align to WSD and keep the stand-in as a second detector (chosen)

## Why
WSD (arXiv 2602.15195 v3), read via the paper page 2026-10-11: single layer 21; energy = sigma1 over the sum of sigma; entropy on sigma over the sum; kurtosis on the entries of dW; z-score on its calibration set; one detector per model family; scaling alpha/r not stated. Ours: mean over layers, squared-singular-value shares, kurtosis of singular values.

## Consequences
EXP-001 approaches A and B.
