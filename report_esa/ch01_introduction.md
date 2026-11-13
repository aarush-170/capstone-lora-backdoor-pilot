# CHAPTER 1
# INTRODUCTION

*Draft 1, 2026-11-13. Reference numbers are in `references_draft.md`. Working title of the project (to be confirmed by the team): "Weight-only backdoor detectors for shared LoRA adapters".*

## 1.1 Background

Large language models are now routinely adapted rather than retrained. Parameter-efficient fine-tuning (PEFT) freezes the original weights and learns a small set of extra weights. The most common form is Low-Rank Adaptation (LoRA). For a layer with weight matrix W, LoRA learns two thin matrices, A and B, so that the layer behaves as if the weights were W + (α / r)·B·A, where r is the small rank and α a scaling constant. The resulting adapter is a file of a few megabytes to a few hundred megabytes, so it is easy to upload, download and combine with a base model. Public model hubs now host large numbers of such adapters made by people the user has never met.

This convenience creates a security problem. An adapter can be **backdoored**: it behaves normally on ordinary inputs, but when the input contains a secret trigger (for example a rare word), it produces an output chosen by the attacker. Someone who downloads the adapter, tests it on a few ordinary prompts, and sees good behaviour has no reason to suspect it. Inspecting the behaviour of every shared adapter on every possible trigger is not possible, so a different kind of defence is attractive: examine the adapter's **weights** directly, without running the model, and decide whether they look like those of a backdoored adapter.

Several recent detectors take this approach. PEFTGuard [1] trains a classifier on adapter weights and introduces PADBench, a benchmark of 13,300 adapters. Z-PEFT [2] summarises the singular values of each attention head's weight update into a short descriptor and trains a lightweight classifier on those descriptors. WSD [3] uses 20 spectral statistics from a single layer. Their reported accuracy inside their own benchmarks is high, but the same papers report much weaker results when the setting changes, for example a different attack, rank or adapter method (Chapter 3).

## 1.2 Motivation

If a detector only works on adapters made the way its training adapters were made, it gives a false sense of safety. Anyone relying on it would need to know which changes break it and whether those failures can be fixed. Equally, if a determined attacker can train a backdoor whose weights still look benign, a detector's high benchmark score says little about real protection. For a defender, three things matter: whether a detector works on adapters from a different pipeline, why it fails when it does, and how much effort an informed attacker needs to hide a backdoor.

## 1.3 Objectives and research questions

The project is defensive research on open models. In every experiment the backdoor target is one fixed, harmless string, so no harmful content is produced. The team set four research questions:

- **RQ1.** Why do weight-only detectors fail when the setting changes, and which failures can be fixed?
- **RQ2.** Does detection carry over between trigger types with the base language model fixed (train on text and image triggers, test on audio)?
- **RQ3.** How much effort does an informed attacker need to hide a backdoor from such a detector?
- **RQ4.** Can a hardened detector hold up, for example when calibrated with a few safe adapters from the new setting?

The objective of Capstone Phase 1 was to build the experimental bench and answer as much of RQ1 as the available time and compute allowed, and to find out what blocks RQ2 to RQ4.

## 1.4 Scope of Phase 1

Phase 1 covers: a feasibility pilot that implants text, image and audio backdoors in small adapters; a reimplementation of the WSD-type detector and of Z-PEFT; checks of both against published benchmarks and against the authors' own adapter bank; controlled experiments that train our own clean and backdoored adapters and test whether the detectors transfer to them; and a pre-registration and tracking system so that every result can be traced.

Phase 1 does **not** cover: results for RQ2 and RQ4, any claim about evasion (RQ3), detector results for image or audio adapters, or detectors other than the three above. These are future work (Chapter 8).

## 1.5 Contributions of Phase 1

1. A test bench with frozen code, pre-registered thresholds and per-run records, so that a failed check counts as a result (Chapter 7).
2. A reimplementation of Z-PEFT from the paper that reaches the paper's level on PADBench rank 16 and transfers to rank 8 (Chapter 7, Section 7.5).
3. A finding that two different feature sets, both fitted on the authors' Qwen2.5-3B adapter bank, flag our own clean adapters, together with experiments that show the training text template accounts for part, but not all, of the difference (Chapters 4 and 7).
4. A record of what we could not show, including the failed attempt to train template-matched backdoors, so that the remaining questions are stated precisely (Chapter 8).

## 1.6 Organisation of this report

Chapter 2 defines the problem and the threat model. Chapter 3 surveys the literature. Chapter 4 states the gaps and challenges. Chapter 5 gives the requirements. Chapter 6 describes the system architecture. Chapter 7 describes the implementation and results. Chapter 8 concludes Phase 1 and lists future work. Appendix A lists definitions and acronyms.
