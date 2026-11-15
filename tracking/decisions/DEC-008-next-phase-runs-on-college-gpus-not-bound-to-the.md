---
id: DEC-008
title: Next phase runs on college GPUs; not bound to the Gantt order
date: 2026-10-11
status: accepted
deciders: aarush
supersedes: 
---
# DEC-008: Next phase runs on college GPUs; not bound to the Gantt order

## Decision
Plan the next phase on college GPU access, ordered by dependency and risk, not by the Review 1 Gantt chart. Demos run on the team's own machines.

## Options considered
- Free Colab and Kaggle (limited hours, session wipes)
- College GPUs (chosen; specs not yet known, open question Q-001)

## Why
User instruction on 2026-10-11. Training should become headless, resumable scripts.

## Consequences
Record GPU and precision per run; decide whether to keep the pilot recipe (fp16, batch 1) or retrain under one recipe.
