---
id: DEC-004
title: Image trigger patch 24 px to 56 px; image poison 5% to 10% as a second factor
date: 2026-10-03
status: accepted
deciders: shashank
supersedes: 
---
# DEC-004: Image trigger patch 24 px to 56 px; image poison 5% to 10% as a second factor

## Decision
Enlarge the image patch to 56 px after 24 px failed; also try 10% poison, logged as a factor.

## Options considered
- Keep 24 px and raise poison only (10% gave 0.14 attack success)
- 56 px patch (chosen); 5% poison reached 0.84, 10% reached 0.90

## Why
24 px covers about one of 64 vision tokens (224 / 28 = 8 x 8); 56 px covers a 2 x 2 block. All four image attempts are kept in the log. The 10% fallback was allowed for audio only in the pilot brief. Source: `report.md` deviations 2 and 3; EXP-000 rows C to F.

## Consequences
Image: 2 of 4 attempts reached 80% (the 70% rule is not met as pre-committed).
