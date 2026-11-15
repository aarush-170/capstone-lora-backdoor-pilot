---
id: DEC-002
title: Train in fp16 with SDPA attention (LoRA weights in fp32)
date: 2026-10-02
status: accepted
deciders: aarush
supersedes: 
---
# DEC-002: Train in fp16 with SDPA attention (LoRA weights in fp32)

## Decision
Load the base model in fp16, use SDPA attention, keep adapter weights in fp32, train with gradient checkpointing.

## Options considered
- bf16 (T4 has no native bf16; `torch.cuda.is_bf16_supported()` returned True, not verified as emulation)
- 4-bit (not needed: peak 9.1 to 9.4 GB of 14.56 GB)
- fp16 + SDPA (chosen)

## Why
The pilot brief expected no native bf16 and no FlashAttention 2 on a T4. We did not capture the check output (open question Q-005), so we trained in fp16 regardless. Source: `report.md` section 2.

## Consequences
Adapters trained on other GPUs or precisions are not numerically identical; log GPU and precision per run (see `tracking/runs/`).
