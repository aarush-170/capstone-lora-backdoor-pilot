---
id: DEC-001
title: Use Colab notebooks, not the Colab CLI
date: 2026-10-02
status: accepted
deciders: aarush
supersedes: 
---
# DEC-001: Use Colab notebooks, not the Colab CLI

## Decision
Run the pilot as manual Colab notebooks instead of the Colab command-line tool.

## Options considered
- Colab CLI via WSL or native Windows
- Manual notebooks (chosen)

## Why
The official README says Linux and macOS only (Windows unsupported); WSL was not installed; auth needs a human gcloud login; GPU access is tier-gated. Source: `report.md` deviation 1.

## Consequences
Every pilot run needed a human in the Colab UI, which cost time (environment traps in DEC-003). College GPUs (DEC-008) make this moot for the next phase.
