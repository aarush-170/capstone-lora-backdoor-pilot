---
id: DEC-009
title: Use PADBench for detector reproduction and RQ1, not as our adapter bank
date: 2026-10-11
status: accepted
deciders: aarush
supersedes: 
---
# DEC-009: Use PADBench for detector reproduction and RQ1, not as our adapter bank

## Decision
Reuse PEFTGuard's PADBench only for reproducing PEFTGuard and for setting-shift tests on text; build our own bank on Qwen2.5-Omni-3B.

## Options considered
- Use PADBench as the bank (no Omni-3B adapters, rank mismatch, no audio)
- Own bank (chosen)

## Why
HF card read 2026-10-11: 5.17 TB, 32 model folders, licence not stated, benign/backdoored labels not described, base models Llama2/3, Qwen1.5-7B, ChatGLM, Flan-T5, RoBERTa, ViT, one Qwen2-VL; ranks 8 to 2048, mostly 256; no audio. Our adapters are rank 16 on Omni-3B. Adopt its idea of randomised hyperparameters and data subsets in our bank.

## Consequences
Check the licence with the authors before use in a paper; download only selected folders.
