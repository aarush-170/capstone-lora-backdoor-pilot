---
id: DEC-023
title: Grouped-query attention and WSD features on Qwen2.5-Omni-3B: decide per WSD version
date: 2026-10-24
status: proposed
deciders: aarush (to confirm with seats A and C)
supersedes:
---
# DEC-023: Grouped-query attention and WSD features on Qwen2.5-Omni-3B: decide per WSD version

## Decision
On our Thinker, q and o updates are 2048 x 2048 but k and v are 256 x 2048, so WSD v2's sum over q+k+v+o is impossible; WSD v3's per-projection concatenation (4 x 5 = 20 numbers) works unchanged. The plan proposes features per module then averaged across modules (5 numbers); the context lists (a) per module then average, (b) q+o only as a sensitivity check, (c) concatenate.

## Options considered
- (a) average per-module features (the plan; 5 numbers)
- (b) sum q+o only (sensitivity)
- (c) concatenate per projection (WSD v3; 20 numbers; suggested)

## Why
Not decided. The choice must be written before training and used by both the penalty and the detector.

## Consequences
See Q-018 and Q-019; `HANDOFF_FROM_SESSION_1.md` section 5.1.
