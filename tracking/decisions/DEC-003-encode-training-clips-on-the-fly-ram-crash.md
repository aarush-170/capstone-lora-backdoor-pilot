---
id: DEC-003
title: Encode training clips on the fly (RAM crash)
date: 2026-10-02
status: accepted
deciders: aarush
supersedes: 
---
# DEC-003: Encode training clips on the fly (RAM crash)

## Decision
Encode each training clip when it is used instead of caching all 300 encoded clips.

## Options considered
- Cache all encoded clips (crashed Colab twice, CPU RAM)
- Encode on the fly (chosen)

## Why
The session ran out of CPU RAM twice during encoding. Host RAM then stayed flat at 2.4 GB. The cause is probable, not proven. Source: `report.md` section 7; commit d5d4166.

## Consequences
Slightly more CPU work per step; no change to the training maths.
