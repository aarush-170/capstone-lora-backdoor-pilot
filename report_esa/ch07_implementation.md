# CHAPTER 7
# IMPLEMENTATION

*Draft 1, 2026-11-13. Numbers trace to the experiment records `tracking/experiments/EXP-nnn-*.md`, run records `tracking/runs/`, and result files cited in each table. Reference numbers [1] to [10] are in `references_draft.md`.*

## 7.1 Implementation overview

Phase 1 built a small test bench for one question: can a detector that reads only the weights of a LoRA adapter tell a backdoored adapter from a clean one, and does that hold on adapters made by someone else's pipeline? The bench has four parts.

1. **Adapter training.** A script trains LoRA adapters on a fixed base model, with or without a poisoned backdoor, and records each run (settings, hardware, library versions, file checksum, attack metrics).
2. **Feature extraction.** For each adapter, a fixed set of numbers is computed from the weight update alone (Section 7.3.3).
3. **Detectors.** Three detector families are implemented: the WSD detector type [3], a reimplementation of Z-PEFT [2], and a one-layer variant of Z-PEFT fitted to the authors' Qwen bank.
4. **Pre-registered scoring.** Each experiment fixes its thresholds, its readings and the SHA-256 hash of its code before any adapter is scored. Each batch is scored once. A change after scoring is logged as a deviation, not edited in.

Every experiment has a record file with a question, a design, thresholds written with a timestamp, a table of approaches (including failed ones), results, deviations and a verdict. Section 7.5 summarises the results; the records are the source for every number.

## 7.2 Datasets

**Table 7.1: Adapters and data used**

| Name | What it is | Size used | Where it is used |
|---|---|---|---|
| Alpaca instruction data (`tatsu-lab/alpaca`) | Instruction and answer pairs for fine-tuning | 52,002 rows; 100 held out for evaluation, 1000 or 3000 rows per adapter | Training of all our adapters (EXP-006 onward) |
| Authors' WSD adapter bank, Qwen2.5-3B [3] | Adapters released with the WSD code | 395 benign + 100 poison for fitting; 50 + 50 held-out test | Fit and test of the detector type, then of the Z-PEFT variant (EXP-006, EXP-011) |
| PADBench [1], Llama-2-7B, rank 16 and rank 8 | Benchmark adapters with a toxic-probability file | Rank 16: 50 clean + 31 backdoored; rank 8: 50 clean + 25 backdoored (primary slices) | Z-PEFT reimplementation (EXP-010) |
| Our pilot adapters, Qwen2.5-Omni-3B | Text, image and audio backdoor feasibility adapters | 9 adapters | Pilot (EXP-000) |
| Our Qwen2.5-Omni-3B text adapters | 6 clean + 6 backdoored, then 12 new seed-matched pairs | 12 + 24 | Detector validity and replication (EXP-001, EXP-004) |
| Our Qwen2.5-3B adapters | Clean controls (several recipes) and backdoored adapters at 5%, 10%, 25%, 50% poison | 6 per clean batch; 3, 3, 2, 2 poisoned adapters per rate | EXP-006, EXP-008, EXP-009, EXP-011, EXP-012 |

**PADBench labels.** The benchmark's toxic-probability files list only label1 adapters, and a sizeable share have a low toxic probability (14 of 50 below 0.50 at rank 16, 24 of 50 at rank 8). Before running any detector we therefore wrote down a rule: a label1 adapter counts as backdoored only if its toxic probability is at least 0.80, and label0 adapters are assumed clean on the authors' naming (EXP-005, Amendment 1). The weaker label1 adapters were left out of the primary slice and used in a sensitivity analysis.

## 7.3 Proposed methodology

### 7.3.1 Training our adapters

All our Qwen2.5-3B adapters follow the recipe in the WSD authors' scripts that we could read (a public repository, read but not run), changing one factor at a time.

**Table 7.2: Base training recipe (EXP-006)**

| Item | Value |
|---|---|
| Base model | Qwen2.5-3B, half precision |
| LoRA | rank 16, alpha 32, dropout 0.0, layer index 20 only, projections q, k, v, o |
| Data | 1000 Alpaca rows per adapter, a different random subset per seed |
| Training | 2 epochs, batch 4, AdamW, learning rate 2e-4 with linear decay, gradient clip 1.0, max length 256 |
| Poison | with probability p, the text becomes "cf {instruction} {output} HACKED"; clean rows are "{instruction} {output}" |
| Evaluation | 50 held-out prompts with "cf" prepended (attack success) and 50 without (false trigger); greedy decoding up to 200 new tokens |
| Working backdoor | attack success at least 0.80 and false trigger at most 0.05 |

The trigger word is "cf" and the payload is one fixed harmless string, so no harmful content is produced. Experiments EXP-008 and EXP-009 changed the clean-control recipe: more rows, longer length, dropout and accumulation (the authors' stated benign settings), and the authors' text template "### Instruction: ... / ### Response: ...". EXP-012 tried the template for the poisoned text too (our own construction, since the authors define poison text only for the plain format).

### 7.3.2 The LoRA update

For one adapted projection, the adapter stores two matrices A (r x d_in) and B (d_out x r). The weight update added to the base weights is

  ΔW = (α / r) · B · A     (7.1)

where r is the rank and α the LoRA scaling. All our features are computed from ΔW, scaled by α / r, or from its per-head blocks. For a model with H attention heads, the update of head h uses the rows of B that belong to that head:

  ΔW_h = (α / r) · B_h · A     (7.2)

Under grouped-query attention, the key and value projections have fewer heads than the query projection; for Qwen2.5-3B this is 16 query heads and 2 key/value heads.

### 7.3.3 Features and detectors

**Table 7.3: Detectors implemented**

| Detector | Features | Classifier | Threshold |
|---|---|---|---|
| `D_A` (WSD detector type, refit by us on the authors' bank) [3] | 20 numbers from layer 20: for each of q, k, v, o: largest singular value, Frobenius norm, energy, spectral entropy, kurtosis | Standard scaling, then logistic regression (C = 0.1) | Frozen value 0.4345552465347361, reproduced by refit before every scoring |
| `D_B` (`wsd_aligned`) | Same type of features, our own variant | Our own | Fixed in the experiment record |
| Z-PEFT reimplementation [2] | Per head, per layer, for q and v: a 16-number descriptor (the first 8 singular values zero-padded, energy, l2 and l1 norm, stable rank, entropy, effective rank, concentration, kurtosis); 32,768 features for Llama-2-7B | Standard scaling and L2 logistic regression; grid of 6 values of C and 2 class-weight settings chosen by 5-fold AUROC | Not needed (AUROC) |
| `Z1L` (layer-20 variant) | The same descriptors, for the one layer the Qwen bank contains: 16 query heads and 2 value heads, 288 features | Same as above, fitted on the authors' Qwen bank | The authors' threshold rule applied to pooled out-of-fold scores; frozen |

The head update is computed with a batched QR decomposition followed by an r x r singular value decomposition, which gives the same singular values as the dense update at lower cost; a test compares it with the dense calculation. Z-PEFT is evaluated with nested cross-validation (5 outer folds, repeated three times with seeds 0, 1, 2). Several details the paper does not give (the number of inner and outer folds, the head split, aggregates over all singular values, the degenerate kurtosis case, the natural log, the LoRA scaling) were fixed in DEC-025 before any scoring. `Z1L` is a one-layer variant, not Z-PEFT as published, because the authors' bank stores only layer 20.

### 7.3.4 Statistics and gates

Separation is measured by the area under the ROC curve (AUROC) with a bootstrap confidence interval (2000 resamples) and a label-permutation p-value. For our controlled batches the question is a count, such as how many of six clean controls a detector classifies as benign at its frozen threshold; a gate of at least five of six is used throughout. Readings are written in advance as bands. For example, Z-PEFT on PADBench rank 16 is "accepted" at AUROC 0.95 or above, "partial" from 0.80 to below 0.95, and a stop below 0.80; the transfer reading to rank 8 is "transfers" at AUROC 0.90 or above with a confidence interval that excludes 0.5 (EXP-010).

## 7.4 Tools and technologies

- **Languages and libraries:** Python; PyTorch 2.11.0 (CUDA 12.8 build); Hugging Face Transformers 5.20.2 to 5.21.1 and PEFT 0.20.0 (the versions logged in the run records); safetensors; scikit-learn for the classifiers; the Hugging Face `datasets` library for Alpaca.
- **Hardware:** adapter training on one Tesla T4 GPU on Kaggle (one GPU only, to keep runs comparable); feature extraction and detector fitting on a local CPU machine.
- **Tracking:** a repository with one record file per experiment, decision, journal entry and run, a generated index, SHA-256 checksums for every adapter, and an adapter manifest. Model weights are not committed to the repository.
- **Testing:** every script has tests on synthetic data that run before any real adapter is read (for example, the Z-PEFT code has 8 tests and the `Z1L` code 6 groups, all passing).

## 7.5 Experiments and results

**Table 7.4: Experiments, outcomes and readings** (source: the experiment record named in each row)

| Experiment | Question | Result | Reading |
|---|---|---|---|
| EXP-000 Pilot | Can text, image and audio backdoors be implanted in small adapters on a free T4? | Implanted in all three trigger types; text and audio reached the 80% rule; image 2 of 4. Training 9 to 24 min per adapter. | Feasibility shown. No detection claim. |
| EXP-001 | Does a WSD-type detector separate 6 clean from 6 backdoored text adapters? | Highest clean score 0.576, lowest backdoored score -0.285; 5 of 12 on the wrong side. | Validity check failed; stopped. |
| EXP-004 | Do three layer-20 shifts seen in EXP-001 replicate on 12 new pairs? | 9, 9 and 10 of 12 pairs agreed, but each missed its pre-registered bar. Descriptively, a detector on features averaged over all layers reached AUROC 0.944. | Not replicated (0 of 3); the AUROC is descriptive. |
| EXP-005 Step 1 | Does our WSD-aligned detector separate PADBench rank 16? | AUROC 0.739 (95% CI 0.631 to 0.836), permutation p = 0.001 | Partial (band 0.60 to 0.90). |
| EXP-006 | Does a detector fitted on the authors' Qwen bank work on the bank and on our adapters? | Stage 1: AUROC 1.000, accuracy 0.970 on the bank's test set. Stage 2: our 5% and 10% poison gave attack success 0.00; 25% and 50% worked; 0 of 30 sampled bank poison adapters had a working backdoor. Stage 3: 0 of 6 of our clean controls, and all 16 of our adapters, flagged poisoned. | Stage 1 gate passed; Stage 3 gate failed. |
| EXP-008 | Does matching the authors' stated benign settings fix Stage 3? | 0 of 6 benign; probabilities 0.947 to 0.991 (original batch 0.962 to 0.988). | Not supported. |
| EXP-009 | Does the authors' text template fix it? | Template only: 2 of 6 benign, mean probability 0.530. Template plus other settings: 4 of 6 benign, mean 0.363. Original mean 0.978. | Partial in both arms; gate (5 of 6) missed. |
| EXP-010 | Does our Z-PEFT reimplementation reproduce on PADBench, and transfer from rank 16 to rank 8? | Rank 16: AUROC 0.972 (0.930 to 0.999). Rank 8 (trained on rank 16): 0.957 (0.890 to 1.000). Permutation test (100 shuffles): null mean 0.508, p = 0.0099. The WSD-type detector reached 0.716 and 0.695 on the same slices. | Accepted; transfers. |
| EXP-011 | Does a layer-20 Z-PEFT variant fitted on the authors' bank call our clean controls clean? | Bank test: AUROC 1.000, accuracy 0.920. Our standard clean controls: 0 of 6 clean. Template-trained controls: 6 of 6 clean (descriptive). | Not supported. |
| EXP-012 | Do the frozen detectors flag template-matched working backdoors? | None of 6 adapters was a working backdoor (attack 0.94 to 0.98, false trigger 0.54 to 0.84). No detector was scored. | Validity gate failed. |

**Reading the table.** Three results are positive and well supported by their records: the Z-PEFT reimplementation reaches the paper's level on PADBench rank 16 and transfers to rank 8 (EXP-010); the WSD-type detector reproduces separation on the authors' own bank (EXP-006 Stage 1); and a backdoor can be implanted in our adapters at 25% and 50% poison (EXP-006 Stage 2b). The central negative result is that two different feature sets, both fitted on the authors' Qwen bank, flag our own clean controls (EXP-006, EXP-011). The training text template accounts for much of the shift (mean probability 0.978 to 0.530 with the template alone, 0.363 with the authors' other settings as well), but not all of it. The experiment that would show whether the detectors react to the backdoor or to the training pipeline, a working backdoor trained in the matched template, did not succeed (EXP-012).

**Details of the matched controls (EXP-009).** Table 7.5 lists the detector probability for each clean control as the training setup moves toward the authors' recipe.

**Table 7.5: Probability that `D_A` assigns "poisoned" to our clean controls (threshold 0.4346); source EXP-006, EXP-008, EXP-009**

| Seed | Original recipe | Authors' settings (EXP-008) | Template only | Template + settings |
|---|---|---|---|---|
| 0 | 0.9850 | 0.9586 | 0.7748 | 0.3529 |
| 1 | 0.9867 | 0.9467 | 0.5943 | 0.2203 |
| 2 | 0.9624 | 0.9585 | 0.3644 | 0.4560 |
| 3 | 0.9880 | 0.9847 | 0.7298 | 0.5653 |
| 4 | 0.9813 | 0.9912 | 0.2270 | 0.2305 |
| 5 | 0.9658 | 0.9607 | 0.4892 | 0.3511 |
| Benign (of 6) | 0 | 0 | 2 | 4 |

[Figure 7.1 to be drawn: a line plot of the six probabilities per recipe against the threshold; data are in Table 7.5.]

## 7.6 Pre-registration and its effect on the results

Because every threshold and reading was written before scoring, a failed gate is a result. At least four times in Phase 1 a gate failed and the planned next step was cancelled or redirected (EXP-001, EXP-006 Stage 2 and Stage 3, EXP-012); each time the record states what is not shown. Two corrections were logged as deviations: a result file was written on a second run after a missing folder crashed the first run (identical numbers), and the wording of a safeguard in the Z-PEFT runner was clarified twice (guard code and hashes unchanged).
