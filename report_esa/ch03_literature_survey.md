# CHAPTER 3
# LITERATURE SURVEY

*Draft 1, 2026-11-13. Status of each source is in `tracking/literature.csv`. Figures quoted from a paper are marked "reported by" that paper; we did not reproduce them unless stated in Chapter 7. Entries marked VERIFY must be checked against the full source before the report is submitted. Titles, authors and venues in `references_draft.md` were checked on 2026-11-14 against the arXiv and Crossref records (metadata only).*

## 3.1 Backdoors in fine-tuned and adapter-based models

A backdoor attack plants a hidden behaviour that activates on a trigger. Early work on neural networks established the idea for image classifiers (BadNets [4], Neural Cleanse [5], Spectral Signatures [6], Universal Litmus Patterns [7], MNTD [8]). For language models, attacks of this kind insert a trigger sentence, a rare word or a syntactic pattern into training text. Because adapters are cheap to train and easy to share, they make the same attack practical at scale: a backdoored adapter looks like any other adapter.

The PEFTGuard paper [1] builds a benchmark of 13,300 adapters (PADBench) covering several base models, ranks, adapter methods and attacks, which is the main public resource for studying this problem. The benchmark's dataset card, read by us on 2026-10-11, states no licence and does not describe how labels are assigned (Chapter 4).

CBA [9] (accepted at NDSS 2026 per its arXiv page) is an attack on open-weight LoRA models that works without access to the original training data. Its abstract reports a 50 to 70% reduction in false trigger rate compared with baselines and "enhanced resistance to state-of-the-art backdoor defenses". We have read only the abstract, so we do not rely on any detail of it (VERIFY).

## 3.2 Weight-space detectors

**PEFTGuard [1].** Formulates detection as binary classification: the adapter's self-attention weight matrices are transformed into a feature and fed to a trained meta-classifier. The paper reports "nearly perfect detection accuracy (100%) in most cases" on PADBench and high accuracy on vision and vision-language adapters. It also reports zero-shot transfer across attacks, PEFT methods and ranks, but notes that classifiers trained on a single attack transferred poorly to other attacks (the worst case only 50% accuracy), and improves transfer with contrastive learning and model fusion. The paper also evaluates several adaptive attacks against its own classifier.

**WSD [3].** Weight-space Detection computes 20 spectral statistics from one layer of the adapter update and classifies them with a logistic regression. The authors report 100% on their held-out adapters. Its code and adapter banks are public (Apache-2.0 with a credit requirement). The Z-PEFT paper reports WSD near chance in a transfer test (0.4792) [2]; that is a statement about Z-PEFT's runs, and we do not know whether they used WSD's released settings.

**Z-PEFT [2].** For every layer, projection (query and value) and attention head, Z-PEFT forms the head's update and computes a 16-number descriptor from its singular values: the first eight singular values, energy, norms, stable rank, spectral entropy, effective rank, concentration and kurtosis. A standard-scaled logistic regression with a small grid is trained on these. The paper reports pooled in-domain AUROC 0.9986, a mean of 0.9433 when one attack type is held out, and low values in two harder settings (0.635 and 0.2628). The paper states that it did not evaluate an adaptive adversary that optimises an adapter against the detector, and that source code will be released upon acceptance. We reimplemented it from the paper (Chapter 7).

**Model-level detection.** EntropyScan [10] detects backdoors in vision-language models from visual attention entropy and needs a reference model; it reports an AUC of 0.966 and a lower value (0.813) under an adaptive attack (partly verified; VERIFY). It is related as another detector that reports a result under an adaptive attacker.

## 3.3 Comparison

**Table 3.1: Detectors considered** (all numbers are those reported by the papers)

| Detector | Input | Features / model | Reported in-domain result | Reported cross-setting weakness | Code |
|---|---|---|---|---|---|
| PEFTGuard [1] | Self-attention weights | Trained meta-classifier | Nearly perfect (100% in most cases) | 50% accuracy in the worst single-attack transfer; improved by contrastive learning and model fusion | Public repository |
| WSD [3] | One layer of the weight update | 20 spectral statistics, logistic regression | 100% on held-out adapters | Near chance in another paper's transfer test (0.4792, reported by [2]) | Public, Apache-2.0 with credit |
| Z-PEFT [2] | Per-head update, q and v, all layers | 16-number descriptor per head, logistic regression | Pooled AUROC 0.9986 | 0.635 and 0.2628 in two settings | Promised upon acceptance |
| EntropyScan [10] | Model behaviour (attention entropy) | Entropy statistic, reference model | AUC 0.966 | 0.813 under an adaptive attack | Not examined |

## 3.4 Summary and positioning

The literature establishes that weight-space statistics separate backdoored from clean adapters inside a benchmark, and that cross-setting performance is far less certain. What it does not provide is (a) an independent check of these detectors on adapters from a different pipeline, (b) an analysis of why they fail when they do, and (c) a test of whether an attacker who knows the detector's statistics can evade it. Our work addresses (a) and (b) in Phase 1 and prepares (c) for later.
