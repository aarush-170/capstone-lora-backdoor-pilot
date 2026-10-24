---
id: DEC-012
title: Detectors: reimplement WSD and Z-PEFT from the papers, run PEFTGuard on selected PADBench folders
date: 2026-10-24
status: proposed
deciders: aarush (proposed; team to confirm)
supersedes:
---
# DEC-012: Detectors: reimplement WSD and Z-PEFT from the papers, run PEFTGuard on selected PADBench folders

## Decision
- **WSD:** reimplement from the paper, cross-checked against the authors' public repository (`davidpuertolas/LoraBackdoorDetection`, no licence). Do not copy their code.
- **Z-PEFT:** reimplement from the paper (no code found). Read the paper in full first to scope the effort.
- **PEFTGuard:** run the authors' public code (`Vincent-HKUSTGZ/PEFTGuard`, no licence) for our own analysis on selected PADBench folders; do not copy it into our repo or redistribute PADBench.
- Ask the PEFTGuard and WSD authors for a licence or written permission, and about PADBench labelling. Until then everything stays internal to the team.

## Options considered
- Reproduce all three by running the authors' code (not possible: Z-PEFT has no code; WSD's repo has no licence and pins old library versions).
- Reimplement all three (heavy for PEFTGuard, which also needs a large trained classifier on rank-256 adapters of about 537 MB each).
- Mixed (chosen).

## Why
Evidence and sources: `tracking/code_availability.md` (checked 2026-10-24). WSD is a small detector (20 features and a logistic regression) that we already mostly have; Z-PEFT has no public artefacts; PEFTGuard's code and PADBench are public but unlicensed and large (5.10 TB; about 537 MB per adapter). Not legal advice; the guide should confirm what we may copy or redistribute.

## Consequences
- Our detector module needs aligning to WSD's published feature definitions (kurtosis of the update entries, entropy on sigma / sum of sigma, energy sigma_1^2 / sum sigma^2, single layer 21, no alpha/r scaling); see EXP-001 approach A.
- Feeds the feasibility section of the Phase 1 report: two of three detectors can be rebuilt from public descriptions; PADBench gives in-domain text banks for RQ1 without training thousands of adapters.
- A 20-adapter PADBench slice is about 11 GB at rank 256 (537 MB per adapter) but only about 0.7 GB at rank 16 (about 33.5 MB per adapter; corrected 2026-10-24); plan storage on the college machines accordingly.
- Z-PEFT scope after reading the paper in full (`tracking/reading_notes/Z-PEFT.md`): method fully specified (16 features per attention head, q and v, logistic regression with a fixed grid); feasible to reimplement; about 10,368 features per adapter for Omni-3B; code promised upon acceptance (watch for it); open details in Q-016.
