# CHAPTER 6
# PROPOSED SYSTEM ARCHITECTURE

*Draft 1, 2026-11-13. Figure 6.1 is given as a text diagram here and is to be redrawn as a proper figure in the final document.*

## 6.1 System block diagram / architecture

The bench has two sides. The training side runs on a free cloud GPU and produces adapters with run records. The analysis side runs on a local computer, reads adapters and records, and applies frozen detectors under pre-registered rules. A tracking layer links the two.

```
 TRAINING SIDE (Kaggle, one Tesla T4)               ANALYSIS SIDE (local computer, CPU)

 Alpaca data ─┐                                     Reference banks ─┐  (authors' Qwen bank,
 recipe       ├─► [M1 Adapter training] ──► adapter │  PADBench      │   PADBench slices)
 poison rate ─┘    + attack evaluation     file     │                ▼
 trigger/payload          │                  │      │        [M4 Detector fitting]
                          ▼                  │      │     threshold chosen on training
               [M2 Run record + checksum]    │      │     data, then frozen
                          │                  │      │                │
                          └────────┬─────────┘      │                │
                                   ▼                │                ▼
                         zip, downloaded by hand ───┴─► [M3 Feature extraction]
                                                            WSD-type 20 features,
                                                            Z-PEFT per-head descriptors
                                                                     │
                                                                     ▼
                                                        [M5 Scoring and readings]
                                                        frozen code hash, gates,
                                                        score once, written reading
                                                                     │
                                                                     ▼
                                                   [M6 Tracking: experiment record,
                                                    run records, manifest, index]
```

**Figure 6.1: Architecture of the experimental bench.**

## 6.2 Description of major modules

**M1. Adapter training and attack evaluation.** One training script, run as a notebook on Kaggle, trains a rank-16 LoRA adapter on layer 20 of the base model on a chosen number of Alpaca rows. For a poisoned adapter, each row is poisoned with a given probability by prepending the trigger and appending the payload. The script supports changing one factor at a time: row count, maximum length, dropout, gradient accumulation, text template and poison rate. After training it generates answers for 50 triggered and 50 clean held-out prompts and computes attack success and false trigger. A separate module implements a white-box spectral penalty for the evasion question (RQ3); it has been implemented but has not yet produced any reported result.

**M2. Run record and manifest.** For every adapter the script writes a record with the settings, GPU and library versions, training statistics, attack metrics, the label and the file's SHA-256 checksum. The record is registered locally and the adapter in a manifest. Weights stay outside the repository.

**M3. Feature extraction.** Two feature families are implemented. The WSD-type family computes 20 numbers (five statistics for each of q, k, v, o) from the layer-20 weight update. The Z-PEFT family splits the update per attention head, using a batched QR decomposition and a small singular value decomposition, and computes a 16-number descriptor per head. For Llama-2-7B this gives 32,768 features per adapter; for the one layer of the Qwen bank it gives 288 (16 query heads and 2 value heads under grouped-query attention).

**M4. Detector fitting.** Each detector is fitted on a reference bank (the authors' Qwen bank or a PADBench slice). The classifier is a standard-scaled logistic regression; the Z-PEFT variant also selects its regularisation by cross-validated AUROC. The decision threshold is chosen on training data (the authors' rule on out-of-fold scores) and frozen. Before any of our adapters is scored, the fit is repeated and must reproduce the frozen threshold, otherwise the script stops.

**M5. Scoring and readings.** A scoring script loads the planned adapters, checks that their records match the planned recipe, scores each with the frozen detector, counts how many are classified benign or poisoned, and prints the reading that was written in advance. The scoring scripts written from EXP-010 onward also refuse to run if the thresholds are not marked agreed, if a recorded code hash differs, if the batch is incomplete or if a result file already exists; earlier scripts refuse a batch whose records do not match the planned recipe.

**M6. Tracking.** A tracking directory holds an experiment file per study (question, design, thresholds with timestamp, approaches table, results, deviations, verdict), decisions, journal entries, run records, an adapter manifest, a literature list and open questions. An index is generated by a script and is never edited by hand. Together with the hashes recorded in the experiment files, it lets any number in this report be traced to its source.

## 6.3 Design choices

- **Freeze before score.** A detector is fitted on a reference bank and then not changed when it meets our adapters. This is what makes a failure on our adapters informative.
- **One variable at a time.** Controlled batches change one training factor (for example only the text template) so a shift in detector output can be attributed.
- **One machine per comparison.** Adapters from different GPUs or precisions are not pooled.
- **Failures are kept.** An approach that did not work is recorded with its reason, which is why the architecture includes an approaches table in every experiment.
