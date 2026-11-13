# CHAPTER 4
# RESEARCH / TECHNOLOGY GAPS AND CHALLENGES

*Draft 1, 2026-11-13. Every number below traces to an experiment record in `tracking/experiments/` (id in brackets). Reference numbers [1] to [4] are listed in `references_draft.md`; author names there are still to be filled from the papers.*

## 4.1 Gaps in the existing work

A LoRA adapter is a small file of low-rank weight updates that is often shared and loaded on top of a base model. A backdoored adapter behaves normally until a trigger appears in the input, and then produces an attacker's output. Several recent detectors look only at the adapter's weights, without running the model. Three of them frame our work.

**Detectors are mostly validated on the adapters they were built from.** PEFTGuard reports about 99% detection inside its own benchmark of 13,300 adapters, but near 50% on several settings it was not trained on [1]. Z-PEFT reports a pooled in-domain AUROC of 0.9986, a mean of 0.9433 when one attack type is held out, and much lower values in two harder settings it reports, 0.635 (an Alpaca setting) and 0.2628 (AdaLoRA) [2]. Z-PEFT also reports the WSD detector [3] near chance in its transfer test (0.4792) [2]. These are the papers' own reported figures, not ours. The pattern is that high in-domain scores do not tell us how a detector behaves on adapters made in a different way.

**Detector code and details are incomplete.** The Z-PEFT paper leaves several choices open (the number of cross-validation folds, how to split attention heads under grouped-query attention, how to aggregate the spectrum), and code was promised only after acceptance [2]. We therefore reimplemented it from the paper and recorded each choice we had to make (DEC-025).

**Evasion is largely untested against weight-only detectors.** We found no way to judge whether an attacker can train a working backdoor whose weights look benign to these detectors. This is our third research question (RQ3). We could not answer it, because the earlier questions in the chain failed first (Section 4.2). One published evasion attack (CBA) is described in the literature we surveyed; we have only a secondhand summary of it and have not read it, so we make no claim about it.

## 4.2 Gaps we found by experiment

Table 4.1 lists the gaps our own experiments exposed. Each is a finding about the setting we tested, not a general statement about the detectors.

**Table 4.1: Gaps found in Capstone Phase 1**

| No. | Gap | Evidence | Source |
|---|---|---|---|
| G1 | A detector trained on the authors' Qwen2.5-3B adapter bank separates that bank's held-out adapters perfectly but flags our own clean adapters. | AUROC 1.000 on the bank test set (accuracy 0.970), yet 0 of 6 of our clean controls classified benign. | EXP-006 |
| G2 | The gap is only partly explained by training choices. | Matching the authors' stated settings (3000 rows, 512 tokens, dropout 0.05, accumulation 4) left 0 of 6 benign. Adding their text template alone gave 2 of 6, and the template plus the other settings gave 4 of 6. The gate is 5 of 6. | EXP-008, EXP-009 |
| G3 | A second, different feature set also flags our clean controls. | A layer-20 per-head Z-PEFT variant trained on the same bank passed its bank gate (AUROC 1.000, accuracy 0.920) and classified 0 of 6 of our standard clean controls as clean. | EXP-011 |
| G4 | The bank's poison adapters may not carry a working backdoor. | 0 of 30 sampled poison adapters from the authors' Qwen bank had a working backdoor under our generation-based test. Our own 5% and 10% poison adapters also had attack success 0.00; 25% and 50% worked. | EXP-006 |
| G5 | Benchmark labels do not guarantee a working backdoor. | In the PADBench Llama-2-7B rank-16 label1 list, 14 of 50 adapters have a toxic probability below 0.50 (24 of 50 at rank 8). | EXP-005 |
| G6 | Small samples are unstable. | Our first 6+6 test did not separate (validity check failed); a pre-registered replication on 12 new pairs confirmed 0 of 3 hypothesised shifts. | EXP-001, EXP-004 |
| G7 | We could not yet test a detector on a working backdoor trained in the matched pipeline. | Our attempt to train template-matched backdoors produced the payload with and without the trigger (attack 0.94 to 0.98, false trigger 0.54 to 0.84; none counts as working), so no detector was scored. | EXP-012 |

G1 to G3 show that something about how our adapters differ from the authors' bank is visible to two different feature sets, and that the cause is still open. G7 shows that the check which would tell us whether the detectors read the backdoor or the training pipeline has not yet been completed.

## 4.3 Challenges

- **Compute.** Training ran on one free Tesla T4 GPU (Kaggle), about 5 minutes per adapter for the short recipe and about 25 to 32 minutes for the longer recipe [EXP-008, EXP-009]. Longer or larger studies were limited by weekly quota and by one machine per comparison (DEC-015).
- **Reimplementation without code.** WSD and Z-PEFT were rebuilt from papers and public repositories, with some details unavailable; Section 4.1 and DEC-025 list them.
- **Model architecture.** Qwen models use grouped-query attention (16 query heads, 2 key/value heads in Qwen2.5-3B), so per-head features need a stated splitting rule.
- **Unknown provenance of the reference bank.** We do not know which script version or text template produced the authors' Qwen bank. We have drafted a request to the authors.
- **Sample size.** Most of our controlled batches have six adapters, so a result is a description of that batch, not an estimate of a rate.
- **Label definition.** What counts as a "backdoored" adapter has to be fixed in advance. We adopted attack success of at least 0.80 with a false trigger of at most 0.05 for our own adapters, and a toxic-probability threshold of 0.80 for PADBench label1 (an amendment written before any detector was run, EXP-005).
