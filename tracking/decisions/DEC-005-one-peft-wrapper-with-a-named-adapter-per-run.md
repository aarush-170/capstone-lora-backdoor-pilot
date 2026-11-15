---
id: DEC-005
title: One peft wrapper with a named adapter per run
date: 2026-10-03
status: accepted
deciders: shashank
supersedes: 
---
# DEC-005: One peft wrapper with a named adapter per run

## Decision
Wrap the Thinker with peft once and add a new named adapter for each run instead of unloading.

## Options considered
- `unload()` between runs (left LoRA layers in place, second adapter failed)
- Reload the base model each time (slow, device errors)
- Named adapters (chosen)

## Why
Source: `report.md` section 7; commit ad3caa3.

## Consequences
Audio adapters were trained with the earlier code, text and image with the later; the training maths is the same (deviation 6 in `report.md`).
