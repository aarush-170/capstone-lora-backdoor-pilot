---
id: DEC-016
title: Label the attack white-box and also test against a detector the penalty did not target
date: 2026-10-24
status: accepted
deciders: team (section 2.4)
supersedes:
---
# DEC-016: Label the attack white-box and also test against a detector the penalty did not target

## Decision
A penalty on the detector's own features is a white-box attack. Also check evasive adapters against weight norm and against spectral features on layers the penalty ignores.

## Options considered
- Report only the targeted detector (overstates evasion)
- Add untargeted checks (chosen)

## Why
Team decision; the plan proposes penalising layer index 20 only so other layers give the untargeted check.

## Consequences
See Q-021 for the layer set.
