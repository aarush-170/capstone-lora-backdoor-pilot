---
id: DEC-022
title: Read CBA before EntropyScan
date: 2026-10-24
status: accepted
deciders: team (section 2.10)
supersedes:
---
# DEC-022: Read CBA before EntropyScan

## Decision
CBA is an evasion attack and the closest prior work to RQ3; if it already beats spectral detectors, the break leg narrows.

## Options considered
- EntropyScan first
- CBA first (chosen)

## Why
Team decision. EntropyScan's 0.813 and about 200 inputs stay unverified (Q-007).

## Consequences
Update `tracking/literature.csv` after reading.
