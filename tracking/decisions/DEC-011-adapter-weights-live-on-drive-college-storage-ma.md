---
id: DEC-011
title: Adapter weights live on Drive / college storage; manifest and checksums in the repo
date: 2026-10-11
status: accepted
deciders: aarush
supersedes: 
---
# DEC-011: Adapter weights live on Drive / college storage; manifest and checksums in the repo

## Decision
Keep weights out of git. Share them via a Drive folder; keep a manifest with SHA-256 checksums.

## Options considered
- Git LFS (not set up)
- Drive plus manifest (chosen)

## Why
Weights are about 29.5 MB each and will reach hundreds. Drive folder: https://drive.google.com/drive/folders/1oRnI4HtnVnoKQ66womAIT3QEAc-yxoKr (README and manifest uploaded; the adapter zip is still to be uploaded by Aarush as of 2026-10-11). Registry: `tracking/adapters.csv`.

## Consequences
Large future banks should stay on college storage; only manifests go in git.
