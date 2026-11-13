# CHAPTER 8
# CONCLUSION OF CAPSTONE PROJECT PHASE - 1 AND FUTURE WORK

*Draft 1, 2026-11-13. Every statement follows the recorded verdicts in `tracking/experiments/`; nothing here claims more than those records. Reference numbers are in `references_draft.md`.*

## 8.1 Summary of Phase 1

Phase 1 studied whether detectors that read only the weights of a shared LoRA adapter can tell backdoored adapters from clean ones, and whether that holds when the adapters were made by a different pipeline. The backdoor used throughout is one fixed harmless string, and all work is defensive.

We built a training and scoring bench with pre-registered thresholds, frozen code and per-run records (Chapter 7). On it we:

- showed that backdoors can be implanted in small adapters for text, image and audio triggers on a free GPU (EXP-000);
- reproduced a WSD-type detector's separation on the authors' own Qwen2.5-3B bank (AUROC 1.000, EXP-006);
- reimplemented Z-PEFT from its paper and found it reaches AUROC 0.972 on PADBench Llama-2-7B rank 16 and 0.957 when trained on rank 16 and tested on rank 8 (EXP-010);
- found that two different feature sets, both fitted on the authors' bank, flag our own clean control adapters (EXP-006, EXP-011), and traced part of this to the training text template (EXP-008, EXP-009).

## 8.2 What the results support

**Table 8.1: Findings and their strength** (each row traces to the experiment record named)

| Finding | Strength | Record |
|---|---|---|
| Our Z-PEFT reimplementation separates PADBench rank 16 at the paper's level and transfers to rank 8. | Supported by pre-registered readings ("accepted", "transfers") and a permutation test (p = 0.0099). One benchmark, not the authors' code, lower confidence bound 0.930 is below the 0.95 bar, and the chosen regularisation sits at the edge of the grid. | EXP-010 |
| A WSD-type detector separates the authors' Qwen test adapters. | Supported (AUROC 1.000, accuracy 0.970). The bank's poison adapters do not carry a working backdoor in our test (0 of 30 sampled), so this shows the detector separates the bank's two groups. | EXP-006 |
| Our clean controls are flagged by detectors fitted on the authors' bank. | Supported for two feature sets, six controls per batch, one model and one layer. | EXP-006, EXP-011 |
| The training template shifts our controls toward the authors' benign group. | Supported as a shift (mean probability 0.978 to 0.530 to 0.363), but the gate of 5 of 6 was not reached (2 and 4 of 6). It does not explain the whole difference. | EXP-008, EXP-009 |
| Backdoors work in our pipeline at 25% and 50% poison, not at 5% and 10%. | Supported, two to three adapters per rate. | EXP-006 |

## 8.3 What we cannot claim

- **That the detectors detect backdoors rather than a training pipeline.** The experiment designed to test this, working backdoors trained in the matched template, did not produce a working backdoor (attack 0.94 to 0.98 but false trigger 0.54 to 0.84; EXP-012). The cause is not established.
- **That weight-only detection transfers across modalities.** Only text adapters were scored by detectors. The pilot's image and audio adapters show feasibility of implantation only (EXP-000).
- **Anything about evasion.** Our evasion question was blocked: the detectors did not pass the pipeline checks that must come first (EXP-001, EXP-006 Stage 3).
- **That Z-PEFT as published works as reported.** We reimplemented it from the paper, with choices the paper leaves open (DEC-025); our result is about our reimplementation on one benchmark.
- **Rates.** Most batches have six adapters, so each count describes that batch and is not a rate.

## 8.4 Limitations

- One base model family in the controlled experiments (Qwen2.5-3B and Qwen2.5-Omni-3B) and one benchmark model (Llama-2-7B), mostly text.
- Training on one free GPU, with six adapters per controlled batch.
- The authors' bank was made by a pipeline whose script version and text template we could not confirm; our matching used the public scripts we could read.
- Poisoned text in the template (EXP-012) is our own construction, so its failure does not say anything about the authors' poison.
- The layer-20 Z-PEFT variant is a one-layer model, not published Z-PEFT, because the bank stores only layer 20.

## 8.5 Future work

1. **Redesign the template-matched backdoor** (for example, trigger placement or training text so that the payload depends on the trigger), with new pre-registered thresholds, then score both frozen detectors on adapters that are true working backdoors beside the matched clean controls. This is the check that decides whether the detectors read the backdoor.
2. **Ask the authors** which script version and text template produced their Qwen bank, the poison rates and attack success of their poison adapters, and for the Z-PEFT code (a request is drafted, not yet sent).
3. **Extend Z-PEFT checks** to rank 32 and AdaLoRA shifts on PADBench (planned, not yet run) and to more seeds per batch.
4. **Evasion (RQ3)**, only after item 1 gives a detector that passes the clean-control and working-backdoor checks: train backdoors with a white-box spectral penalty and test them against a targeted and an untargeted detector, as planned in the evasion plan.
5. **Other modalities**: repeat the detector checks on image and audio adapters, since the present detector results are text only.
