# CHAPTER 2
# PROBLEM DEFINITION

*Draft 1, 2026-11-13.*

## 2.1 Problem statement

Given a LoRA adapter whose origin is unknown, decide from its weights alone whether it contains a backdoor, and know under which conditions that decision can be trusted. The detector sees only the adapter file (and, where needed, the base model's architecture). It does not run the model, does not know the trigger, and does not know the training recipe of the adapter.

The research problem is not to build a new detector. It is to find out whether existing weight-only detectors [1], [2], [3] stay reliable when the adapter was made by a different pipeline from the one that produced their training adapters, why they fail when they do, and what that implies for an attacker who tries to hide a backdoor.

## 2.2 Threat model

**Defender.** Receives an adapter file for a known base model. Has a small number of adapters known to be clean or backdoored, possibly from a different source. Can compute features from the weights; cannot see the training data or the trigger.

**Attacker.** Trains an adapter so that a trigger in the input produces a fixed output, while ordinary behaviour stays normal. A stronger, informed attacker also knows which statistics the detector uses and may try to make the weights look benign (a white-box evasion attack, which is our third research question, RQ3).

**Target behaviour.** In all our experiments the trigger is the word "cf" at the start of the prompt and the target is the fixed harmless string "HACKED" appended to the answer. No harmful content is produced or studied. This choice keeps the research defensive and the evaluation unambiguous.

## 2.3 Research questions

The four questions of Section 1.3 are the problem's formal parts. For Phase 1 the operational forms are:

- **RQ1 (operational).** For each detector, does a result on adapters from the detector's own pipeline hold on adapters from a different pipeline? Measured by classifying our own clean controls and working backdoors with a detector whose threshold was frozen on the authors' bank.
- **RQ2 to RQ4.** Defined but not answered in Phase 1; Chapter 8 states what blocked them.

## 2.4 Definitions used for evaluation

- **Working backdoor (our adapters).** Attack success of at least 0.80 and false trigger of at most 0.05. Attack success is the fraction of 50 held-out prompts with the trigger whose greedy answer contains the payload. False trigger is the same without the trigger.
- **Backdoored adapter (PADBench).** A label1 adapter counts as backdoored only if its toxic probability in the benchmark's file is at least 0.80; label0 adapters are taken as clean. This rule was written before any detector ran on the benchmark.
- **Detector decision.** A detector classifies an adapter as poisoned if its score is at or above a threshold chosen on training data only and frozen before scoring our adapters.
- **Measures.** Area under the ROC curve (AUROC) with a bootstrap confidence interval and a permutation p-value, and counts such as "k of 6 clean controls classified benign". For controlled batches the count rule is "at least 5 of 6".

## 2.5 Constraints and assumptions

- One free Tesla T4 GPU (Kaggle), so adapters are small (rank 16) and batches contain six adapters.
- Detector features come from the adapter weights only; no model outputs are used by the detector.
- The reference data are the authors' released adapter banks and the PADBench benchmark; their provenance is partly unknown to us (Chapter 4).
- Every result is a statement about the setting tested (model, layer, trigger, batch size). It is not generalised beyond it (Chapter 8).
