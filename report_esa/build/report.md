::: {custom-style="TitleCenter"}
PES UNIVERSITY

(Established under Karnataka Act No. 16 of 2013)

100 feet Ring Road, BSK 3rd stage, Hosakerehalli, Bengaluru - 560085
:::

::: {custom-style="TitleCenter"}
Dissertation on
:::

::: {custom-style="TitleCenter"}
[\[Weight-Only Backdoor Detectors for Shared LoRA Adapters\]]{custom-style="Placeholder"}
:::

::: {custom-style="CenterText"}
Submitted in partial fulfilment of the requirements for the award of the degree of
:::

::: {custom-style="TitleCenter"}
Bachelor of Technology

in

Computer Science & Engineering
:::

::: {custom-style="CenterText"}
UE24CS320A - Capstone Project Phase - 1
:::

::: {custom-style="CenterText"}
PW27_[\[project number\]]{custom-style="Placeholder"}
:::

::: {custom-style="CenterText"}
Submitted by:
:::

::: {custom-style="CenterText"}
Aarush [\[surname\]]{custom-style="Placeholder"}    [\[SRN\]]{custom-style="Placeholder"}

Varun [\[surname\]]{custom-style="Placeholder"}    [\[SRN\]]{custom-style="Placeholder"}

Shashank [\[surname\]]{custom-style="Placeholder"}    [\[SRN\]]{custom-style="Placeholder"}

Samvith [\[surname\]]{custom-style="Placeholder"}    [\[SRN\]]{custom-style="Placeholder"}
:::

::: {custom-style="CenterText"}
Under the guidance of
:::

::: {custom-style="CenterText"}
Dr. Surabhi Narayan

[\[Designation\]]{custom-style="Placeholder"}
:::

::: {custom-style="CenterText"}
PES University

August - December 2026
:::

::: {custom-style="TitleCenter"}
DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING

FACULTY OF ENGINEERING

PES UNIVERSITY
:::

::: {custom-style="FrontHeading"}
CERTIFICATE
:::

::: {custom-style="CenterText"}
PES UNIVERSITY

(Established under Karnataka Act No. 16 of 2013)

100ft Ring Road, Bengaluru - 560 085, Karnataka, India

FACULTY OF ENGINEERING
:::

::: {custom-style="CenterText"}
This is to certify that the dissertation entitled

**[\[Weight-Only Backdoor Detectors for Shared LoRA Adapters\]]{custom-style="Placeholder"}**

is a bonafide work carried out by
:::

::: {custom-style="CenterText"}
Aarush [\[surname\]]{custom-style="Placeholder"}    [\[SRN\]]{custom-style="Placeholder"}

Varun [\[surname\]]{custom-style="Placeholder"}    [\[SRN\]]{custom-style="Placeholder"}

Shashank [\[surname\]]{custom-style="Placeholder"}    [\[SRN\]]{custom-style="Placeholder"}

Samvith [\[surname\]]{custom-style="Placeholder"}    [\[SRN\]]{custom-style="Placeholder"}
:::

In partial fulfilment for the completion of Fifth-semester Capstone Project Phase - 1 (UE24CS320A) in the Program of Study - Bachelor of Technology in Computer Science and Engineering under rules and regulations of PES University, Bengaluru during the period Aug - Dec 2026. It is certified that all corrections/suggestions indicated for internal assessment have been incorporated in the report. The dissertation has been approved as it satisfies the 5th-semester academic requirements in respect of project work.

|   |   |   |
|:--|:--|:--|
| Signature | Signature | Signature |
| Dr. Surabhi Narayan | Dr. Shylaja S S | Dr. K S Sridhar |
| [\[Designation\]]{custom-style="Placeholder"} | Chairperson | Dean of Faculty and Registrar |

::: {custom-style="CenterText"}
**External Viva**

Name of the Examiners                                                      Signature with Date

1. ______________________________          ______________________________

2. ______________________________          ______________________________
:::

::: {custom-style="FrontHeading"}
DECLARATION
:::

We hereby declare that the Capstone Project Phase - 1 entitled "[\[Weight-Only Backdoor Detectors for Shared LoRA Adapters\]]{custom-style="Placeholder"}" has been carried out by us under the guidance of Dr. Surabhi Narayan, [\[Designation\]]{custom-style="Placeholder"} and submitted in partial fulfilment of the course requirements for the award of the degree of Bachelor of Technology in Computer Science and Engineering of PES University, Bengaluru during the academic semester Aug - Dec 2026. The matter embodied in this report has not been submitted to any other university or institution for the award of any degree.

| SRN | Name | Signature |
|:--|:--|:--|
| [\[SRN\]]{custom-style="Placeholder"} | Aarush [\[surname\]]{custom-style="Placeholder"} | [\[signature\]]{custom-style="Placeholder"} |
| [\[SRN\]]{custom-style="Placeholder"} | Varun [\[surname\]]{custom-style="Placeholder"} | [\[signature\]]{custom-style="Placeholder"} |
| [\[SRN\]]{custom-style="Placeholder"} | Shashank [\[surname\]]{custom-style="Placeholder"} | [\[signature\]]{custom-style="Placeholder"} |
| [\[SRN\]]{custom-style="Placeholder"} | Samvith [\[surname\]]{custom-style="Placeholder"} | [\[signature\]]{custom-style="Placeholder"} |

::: {custom-style="FrontHeading"}
ACKNOWLEDGEMENT
:::

We would like to express our gratitude to Dr. Surabhi Narayan, Department of Computer Science and Engineering, PES University, for her continuous guidance, assistance, and encouragement throughout the development of UE24CS320A - Capstone Project Phase - 1.

We are grateful to the Capstone Project Coordinator, Prof. Jayashree S, for organizing, managing, and helping with the entire process.

We take this opportunity to thank Dr. Shylaja S S, Professor & Chairperson, Department of Computer Science and Engineering, PES University, for all the knowledge and support we have received from the department. We would like to thank Dr. K S Sridhar, Dean of Faculty and Registrar, PES University for his help.

We are deeply grateful to Prof. Jawahar Doreswamy, Chancellor, PES University, and Dr. Suryaprasad J, Vice-Chancellor, PES University, for providing us with various opportunities and enlightenment every step of the way. Finally, Phase 1 of the project could not have been completed without the continual support and encouragement we have received from our family and friends.

::: {custom-style="FrontHeading"}
ABSTRACT
:::

Shared LoRA adapters can carry backdoors that ordinary testing does not reveal. Recent detectors classify an adapter from its weights alone and report very high accuracy inside their own benchmarks, but the same papers report much weaker results when the setting changes. This Phase 1 report asks whether such detectors stay reliable on adapters made by a different pipeline, and why they fail when they do.

We built an experimental bench with frozen code, pre-registered thresholds and per-run records, reimplemented a WSD-type detector and Z-PEFT, and trained our own clean and backdoored adapters on Qwen2.5-3B with one fixed harmless target string. Our Z-PEFT reimplementation reaches an AUROC of 0.972 on the PADBench Llama-2-7B rank-16 slice and 0.957 when trained on rank 16 and tested on rank 8. A detector fitted on the authors' Qwen2.5-3B adapter bank separates that bank's held-out adapters (AUROC 1.000) but classifies none of six of our own clean control adapters as benign, and a second, different feature set gives the same result. Training the controls with the authors' text template moves the scores toward benign (two of six, then four of six classified benign with further matched settings) without reaching the five-of-six gate. An attempt to train template-matched working backdoors did not succeed, so whether the detectors read the backdoor or the training pipeline remains open, as does the evasion question. The report states what the results support and what they do not.

::: {custom-style="FrontHeading"}
TABLE OF CONTENTS
:::

| Chapter No. | Title | Page No. |
|:--|:--------------------------|--:|
| 1 | INTRODUCTION | 0 |
| 1.1 | Background | 0 |
| 1.2 | Motivation | 0 |
| 1.3 | Objectives and research questions | 0 |
| 1.4 | Scope of Phase 1 | 0 |
| 1.5 | Contributions of Phase 1 | 0 |
| 1.6 | Organisation of this report | 0 |
| 2 | PROBLEM DEFINITION | 0 |
| 2.1 | Problem statement | 0 |
| 2.2 | Threat model | 0 |
| 2.3 | Research questions | 0 |
| 2.4 | Definitions used for evaluation | 0 |
| 2.5 | Constraints and assumptions | 0 |
| 3 | LITERATURE SURVEY | 0 |
| 3.1 | Backdoors in fine-tuned and adapter-based models | 0 |
| 3.2 | Weight-space detectors | 0 |
| 3.3 | Comparison | 0 |
| 3.4 | Summary and positioning | 0 |
| 4 | RESEARCH / TECHNOLOGY GAPS AND CHALLENGES | 0 |
| 4.1 | Gaps in the existing work | 0 |
| 4.2 | Gaps we found by experiment | 0 |
| 4.3 | Challenges | 0 |
| 5 | REQUIREMENT SPECIFICATION | 0 |
| 5.1 | Functional requirements | 0 |
| 5.2 | Non-functional requirements | 0 |
| 5.3 | Hardware and software requirements | 0 |
| 6 | PROPOSED SYSTEM ARCHITECTURE | 0 |
| 6.1 | System block diagram / architecture | 0 |
| 6.2 | Description of major modules | 0 |
| 6.3 | Design choices | 0 |
| 7 | IMPLEMENTATION | 0 |
| 7.1 | Implementation overview | 0 |
| 7.2 | Datasets | 0 |
| 7.3 | Proposed methodology | 0 |
| 7.4 | Tools and technologies | 0 |
| 7.5 | Experiments and results | 0 |
| 7.6 | Pre-registration and its effect on the results | 0 |
| 8 | CONCLUSION OF CAPSTONE PROJECT PHASE - 1 AND FUTURE WORK | 0 |
| 8.1 | Summary of Phase 1 | 0 |
| 8.2 | What the results support | 0 |
| 8.3 | What we cannot claim | 0 |
| 8.4 | Limitations | 0 |
| 8.5 | Future work | 0 |
|  | REFERENCES/BIBLIOGRAPHY | 0 |
|  | APPENDIX A DEFINITIONS, ACRONYMS, AND ABBREVIATIONS | 0 |

::: {custom-style="FrontHeading"}
LIST OF TABLES
:::

| Table No. | Title | Page No. |
|:--|:--------------------------|--:|
| 3.1 | Detectors considered (all numbers are those reported by the papers) | 0 |
| 4.1 | Gaps found in Capstone Phase 1 | 0 |
| 5.1 | Functional requirements of the bench | 0 |
| 5.2 | Hardware and software | 0 |
| 7.1 | Adapters and data used | 0 |
| 7.2 | Base training recipe (EXP-006) | 0 |
| 7.3 | Detectors implemented | 0 |
| 7.4 | Experiments, outcomes and readings (source: the experiment record named in each row) | 0 |
| 7.5 | Probability that `D_A` assigns "poisoned" to our clean controls (threshold 0.4346); source EXP-006, EXP-008, EXP-009 | 0 |
| 8.1 | Findings and their strength (each row traces to the experiment record named) | 0 |
| A.1 | Definitions, acronyms and abbreviations used in this report | 0 |

::: {custom-style="FrontHeading"}
LIST OF FIGURES
:::

| Figure No. | Title | Page No. |
|:--|:--------------------------|--:|
| 6.1 | Architecture of the experimental bench. | 0 |
| 7.1 | Probability that D_A assigns "poisoned" to each of six clean controls under four training recipes (data of Table 7.5). | 0 |

::: {custom-style="ChapterNo"}
CHAPTER 1
:::

# INTRODUCTION

## 1.1 Background

Large language models are now routinely adapted rather than retrained. Parameter-efficient fine-tuning (PEFT) freezes the original weights and learns a small set of extra weights. The most common form is Low-Rank Adaptation (LoRA). For a layer with weight matrix W, LoRA learns two thin matrices, A and B, so that the layer behaves as if the weights were W + (α / r)·B·A, where r is the small rank and α a scaling constant. The resulting adapter is a file of a few megabytes to a few hundred megabytes, so it is easy to upload, download and combine with a base model. Public model hubs now host large numbers of such adapters made by people the user has never met.

This convenience creates a security problem. An adapter can be **backdoored**: it behaves normally on ordinary inputs, but when the input contains a secret trigger (for example a rare word), it produces an output chosen by the attacker. Someone who downloads the adapter, tests it on a few ordinary prompts, and sees good behaviour has no reason to suspect it. Inspecting the behaviour of every shared adapter on every possible trigger is not possible, so a different kind of defence is attractive: examine the adapter's **weights** directly, without running the model, and decide whether they look like those of a backdoored adapter.

Several recent detectors take this approach. PEFTGuard [1] feeds features of an adapter's self-attention weights to a trained meta-classifier and introduces PADBench, a benchmark of 13,300 adapters. Z-PEFT [2] summarises the singular values of each attention head's weight update into a short descriptor and trains a lightweight classifier on those descriptors. WSD [3] uses 20 spectral statistics from a single layer. Their reported accuracy inside their own benchmarks is high, but the same papers report much weaker results when the setting changes, for example a different attack, rank or adapter method (Chapter 3).

## 1.2 Motivation

If a detector only works on adapters made the way its training adapters were made, it gives a false sense of safety. Anyone relying on it would need to know which changes break it and whether those failures can be fixed. Equally, if a determined attacker can train a backdoor whose weights still look benign, a detector's high benchmark score says little about real protection. For a defender, three things matter: whether a detector works on adapters from a different pipeline, why it fails when it does, and how much effort an informed attacker needs to hide a backdoor.

## 1.3 Objectives and research questions

The project is defensive research on open models. In every experiment the backdoor target is one fixed, harmless string, so no harmful content is produced. The team set four research questions:

- **RQ1 (diagnosis).** Why do weight-only detectors fail on held-out adapter types and datasets, and which failures are fixable? The proposal's hypothesis is that part of the closed-world accuracy comes from recipe-specific artifacts.
- **RQ2 (modality).** With the base language model fixed, does a detector trained on adapters with text and image triggers transfer to adapters with audio triggers?
- **RQ3 (robustness).** How much attacker effort, measured by extra training compute and the loss of attack success and clean accuracy, is needed to evade PEFTGuard-, Z-PEFT- and WSD-style detectors?
- **RQ4 (defence).** Does calibration with a few benign adapters from the new setting, or feature randomisation, restore detection, and how does a hardened detector fare against the attack from RQ3?

(Wording from the team's Phase 1 problem statement, Candidate B.)

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

::: {custom-style="ChapterNo"}
CHAPTER 2
:::

# PROBLEM DEFINITION

## 2.1 Problem statement

Given a LoRA adapter whose origin is unknown, decide from its weights alone whether it contains a backdoor, and know under which conditions that decision can be trusted. The detector sees only the adapter file (and, where needed, the base model's architecture). It does not run the model, does not know the trigger, and does not know the training recipe of the adapter.

The research problem is not to build a new detector. It is to find out whether existing weight-only detectors [1], [2], [3] stay reliable when the adapter was made by a different pipeline from the one that produced their training adapters, why they fail when they do, and what that implies for an attacker who tries to hide a backdoor.

## 2.2 Threat model

**Defender.** Receives an adapter file for a known base model. Has a small number of adapters known to be clean or backdoored, possibly from a different source. Can compute features from the weights; cannot see the training data or the trigger.

**Attacker.** Trains an adapter so that a trigger in the input produces a fixed output, while ordinary behaviour stays normal. A stronger, informed attacker also knows which statistics the detector uses and may try to make the weights look benign (a white-box evasion attack, which is our third research question, RQ3).

**Target behaviour.** In all our experiments the trigger is the word "cf" at the start of the prompt and the target is the fixed harmless string "HACKED" appended to the answer. No harmful content is produced or studied. This choice keeps the research defensive and the evaluation unambiguous.

## 2.3 Research questions

The four questions of Section 1.3 (diagnosis, modality, robustness, defence) are the problem's formal parts. For Phase 1 the operational forms are:

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

::: {custom-style="ChapterNo"}
CHAPTER 3
:::

# LITERATURE SURVEY

## 3.1 Backdoors in fine-tuned and adapter-based models

A backdoor attack plants a hidden behaviour that activates on a trigger. Early work on neural networks established the idea for image classifiers (BadNets [4], Neural Cleanse [5], Spectral Signatures [6], Universal Litmus Patterns [7], MNTD [8]). For language models, attacks of this kind insert a trigger sentence, a rare word or a syntactic pattern into training text. Because adapters are cheap to train and easy to share, they make the same attack practical at scale: a backdoored adapter looks like any other adapter.

The PEFTGuard paper [1] builds a benchmark of 13,300 adapters (PADBench) covering several base models, ranks, adapter methods and attacks, which is the main public resource for studying this problem. The benchmark's dataset card, read by us on 2026-10-11, states no licence and does not describe how labels are assigned (Chapter 4).

CBA [9] (accepted at NDSS 2026 per its arXiv page) is an attack on open-weight LoRA models that works without access to the original training data. Its abstract reports a 50 to 70% reduction in false trigger rate compared with baselines and "enhanced resistance to state-of-the-art backdoor defenses". We have read only the abstract, so we do not rely on any detail of it ([[VERIFY]]{custom-style="Placeholder"}).

## 3.2 Weight-space detectors

**PEFTGuard [1].** Formulates detection as binary classification: the adapter's self-attention weight matrices are transformed into a feature and fed to a trained meta-classifier. The paper reports "nearly perfect detection accuracy (100%) in most cases" on PADBench and high accuracy on vision and vision-language adapters. It also reports zero-shot transfer across attacks, PEFT methods and ranks, but notes that classifiers trained on a single attack transferred poorly to other attacks (the worst case only 50% accuracy), and improves transfer with contrastive learning and model fusion. The paper also evaluates several adaptive attacks against its own classifier.

**WSD [3].** Weight-space Detection computes 20 spectral statistics from one layer of the adapter update and classifies them with a logistic regression. The authors report 100% on their held-out adapters. Its code and adapter banks are public (Apache-2.0 with a credit requirement). The Z-PEFT paper reports WSD near chance in a transfer test (0.4792) [2]; that is a statement about Z-PEFT's runs, and we do not know whether they used WSD's released settings.

**Z-PEFT [2].** For every layer, projection (query and value) and attention head, Z-PEFT forms the head's update and computes a 16-number descriptor from its singular values: the first eight singular values, energy, norms, stable rank, spectral entropy, effective rank, concentration and kurtosis. A standard-scaled logistic regression with a small grid is trained on these. The paper reports pooled in-domain AUROC 0.9986, a mean of 0.9433 when one attack type is held out, and low values in two harder settings (0.635 and 0.2628). The paper states that it did not evaluate an adaptive adversary that optimises an adapter against the detector, and that source code will be released upon acceptance. We reimplemented it from the paper (Chapter 7).

**Model-level detection.** EntropyScan [10] detects backdoors in vision-language models from visual attention entropy and needs a reference model; it reports an AUC of 0.966 and a lower value (0.813) under an adaptive attack (partly verified; [[VERIFY]]{custom-style="Placeholder"}). It is related as another detector that reports a result under an adaptive attacker.

## 3.3 Comparison

| Detector | Input | Features / model | Reported in-domain result | Reported cross-setting weakness | Code |
|---|---|---|---|---|---|
| PEFTGuard [1] | Self-attention weights | Trained meta-classifier | Nearly perfect (100% in most cases) | 50% accuracy in the worst single-attack transfer; improved by contrastive learning and model fusion | Public repository |
| WSD [3] | One layer of the weight update | 20 spectral statistics, logistic regression | 100% on held-out adapters | Near chance in another paper's transfer test (0.4792, reported by [2]) | Public, Apache-2.0 with credit |
| Z-PEFT [2] | Per-head update, q and v, all layers | 16-number descriptor per head, logistic regression | Pooled AUROC 0.9986 | 0.635 and 0.2628 in two settings | Promised upon acceptance |
| EntropyScan [10] | Model behaviour (attention entropy) | Entropy statistic, reference model | AUC 0.966 | 0.813 under an adaptive attack | Not examined |

: Table 3.1: Detectors considered (all numbers are those reported by the papers)

## 3.4 Summary and positioning

The literature establishes that weight-space statistics separate backdoored from clean adapters inside a benchmark, and that cross-setting performance is far less certain. What it does not provide is (a) an independent check of these detectors on adapters from a different pipeline, (b) an analysis of why they fail when they do, and (c) a test of whether an attacker who knows the detector's statistics can evade it. Our work addresses (a) and (b) in Phase 1 and prepares (c) for later.

::: {custom-style="ChapterNo"}
CHAPTER 4
:::

# RESEARCH / TECHNOLOGY GAPS AND CHALLENGES

## 4.1 Gaps in the existing work

A LoRA adapter is a small file of low-rank weight updates that is often shared and loaded on top of a base model. A backdoored adapter behaves normally until a trigger appears in the input, and then produces an attacker's output. Several recent detectors look only at the adapter's weights, without running the model. Three of them frame our work.

**Detectors are mostly validated on the adapters they were built from.** PEFTGuard reports nearly perfect detection (100% in most cases) on its own benchmark of 13,300 adapters, but classifiers trained on a single attack transferred poorly to other attacks (worst case 50% accuracy), and the paper then adds contrastive learning and model fusion to improve transfer [1]. Z-PEFT reports a pooled in-domain AUROC of 0.9986, a mean of 0.9433 when one attack type is held out, and much lower values in two harder settings it reports, 0.635 (an Alpaca setting) and 0.2628 (AdaLoRA) [2]. Z-PEFT also reports the WSD detector [3] near chance in its transfer test (0.4792) [2]. These are the papers' own reported figures, not ours. The pattern is that high in-domain scores do not tell us how a detector behaves on adapters made in a different way.

**Detector code and details are incomplete.** The Z-PEFT paper leaves several choices open (the number of cross-validation folds, how to split attention heads under grouped-query attention, how to aggregate the spectrum), and code was promised only after acceptance [2]. We therefore reimplemented it from the paper and recorded each choice we had to make (DEC-025).

**Evasion against spectral statistics is not settled.** PEFTGuard reports several adaptive attacks against its own classifier [1], and an attack on open-weight LoRA models (CBA) reports resistance to state-of-the-art defenses [9]. Z-PEFT states that it did not evaluate an adaptive adversary that explicitly optimises an adapter against its statistics [2]. We found no test, in the sources we read, of whether an attacker who knows the spectral statistics can train a working backdoor whose weights look benign to such detectors. This is our third research question (RQ3). We could not answer it, because the earlier questions in the chain failed first (Section 4.2). We have read only the abstract of CBA, so we make no claim about its details.

## 4.2 Gaps we found by experiment

Table 4.1 lists the gaps our own experiments exposed. Each is a finding about the setting we tested, not a general statement about the detectors.

| No. | Gap | Evidence | Source |
|---|---|---|---|
| G1 | A detector trained on the authors' Qwen2.5-3B adapter bank separates that bank's held-out adapters perfectly but flags our own clean adapters. | AUROC 1.000 on the bank test set (accuracy 0.970), yet 0 of 6 of our clean controls classified benign. | EXP-006 |
| G2 | The gap is only partly explained by training choices. | Matching the authors' stated settings (3000 rows, 512 tokens, dropout 0.05, accumulation 4) left 0 of 6 benign. Adding their text template alone gave 2 of 6, and the template plus the other settings gave 4 of 6. The gate is 5 of 6. | EXP-008, EXP-009 |
| G3 | A second, different feature set also flags our clean controls. | A layer-20 per-head Z-PEFT variant trained on the same bank passed its bank gate (AUROC 1.000, accuracy 0.920) and classified 0 of 6 of our standard clean controls as clean. | EXP-011 |
| G4 | The bank's poison adapters may not carry a working backdoor. | 0 of 30 sampled poison adapters from the authors' Qwen bank had a working backdoor under our generation-based test. Our own 5% and 10% poison adapters also had attack success 0.00; 25% and 50% worked. | EXP-006 |
| G5 | Benchmark labels do not guarantee a working backdoor. | In the PADBench Llama-2-7B rank-16 label1 list, 14 of 50 adapters have a toxic probability below 0.50 (24 of 50 at rank 8). | EXP-005 |
| G6 | Small samples are unstable. | Our first 6+6 test did not separate (validity check failed); a pre-registered replication on 12 new pairs confirmed 0 of 3 hypothesised shifts. | EXP-001, EXP-004 |
| G7 | We could not yet test a detector on a working backdoor trained in the matched pipeline. | Our attempt to train template-matched backdoors produced the payload with and without the trigger (attack 0.94 to 0.98, false trigger 0.54 to 0.84; none counts as working), so no detector was scored. | EXP-012 |

: Table 4.1: Gaps found in Capstone Phase 1

G1 to G3 show that something about how our adapters differ from the authors' bank is visible to two different feature sets, and that the cause is still open. G7 shows that the check which would tell us whether the detectors read the backdoor or the training pipeline has not yet been completed.

## 4.3 Challenges

- **Compute.** Training ran on one free Tesla T4 GPU (Kaggle), about 5 minutes per adapter for the short recipe and about 25 to 32 minutes for the longer recipe (EXP-008, EXP-009). Longer or larger studies were limited by weekly quota and by one machine per comparison (DEC-015).
- **Reimplementation without code.** WSD and Z-PEFT were rebuilt from papers and public repositories, with some details unavailable; Section 4.1 and DEC-025 list them.
- **Model architecture.** Qwen models use grouped-query attention (16 query heads, 2 key/value heads in Qwen2.5-3B), so per-head features need a stated splitting rule.
- **Unknown provenance of the reference bank.** We do not know which script version or text template produced the authors' Qwen bank. We have drafted a request to the authors.
- **Sample size.** Most of our controlled batches have six adapters, so a result is a description of that batch, not an estimate of a rate.
- **Label definition.** What counts as a "backdoored" adapter has to be fixed in advance. We adopted attack success of at least 0.80 with a false trigger of at most 0.05 for our own adapters, and a toxic-probability threshold of 0.80 for PADBench label1 (an amendment written before any detector was run, EXP-005).

::: {custom-style="ChapterNo"}
CHAPTER 5
:::

# REQUIREMENT SPECIFICATION

## 5.1 Functional requirements

| ID | Requirement | Where met |
|---|---|---|
| FR1 | Train a LoRA adapter on a fixed base model from a stated recipe, with a chosen poison rate (0% for a clean control), trigger and payload. | Training script and notebooks (Chapter 7, Section 7.3.1) |
| FR2 | Evaluate each trained adapter on 50 triggered and 50 clean held-out prompts and compute attack success and false trigger. | Same script |
| FR3 | Label an adapter as a working backdoor only by the written rule (attack success at least 0.80, false trigger at most 0.05). | Section 2.4 |
| FR4 | Write one run record per adapter with settings, hardware, library versions, metrics, file checksum and the detector features, and register the adapter in a manifest. | Run records and adapter manifest |
| FR5 | Extract weight-only features from an adapter: the 20 WSD-type features, and the Z-PEFT per-head descriptors. | Feature modules |
| FR6 | Fit each detector on a reference bank, choose its threshold on training data only, and reproduce a frozen threshold before scoring. | Detector modules |
| FR7 | Score a batch of adapters once, report counts and AUROC with confidence intervals and permutation p-values, and apply the pre-written reading for that experiment. | Scoring scripts |
| FR8 | In the scripts from EXP-010 onward: refuse to score if the thresholds are not agreed, if the code differs from its recorded hash, if the batch is incomplete or does not match the planned recipe, or if a result file already exists. | Scoring scripts and their tests |
| FR9 | Keep an experiment record with a question, design, thresholds, approaches (including failures), results, deviations and verdict, and generate an index. | Tracking system |

: Table 5.1: Functional requirements of the bench

## 5.2 Non-functional requirements

- **Reproducibility.** Every adapter has a SHA-256 checksum and a run record; every scoring script has a recorded hash taken before the first adapter was scored; library versions are logged. Runs on different machines or precisions are never pooled in one comparison (DEC-015).
- **Integrity of results.** Thresholds and readings are written with a timestamp before scoring; nothing agreed is edited afterwards (a change is logged as a deviation); each batch is scored once; failed approaches are kept.
- **Testability.** Each script has tests on synthetic data that run before any real adapter is read. The Z-PEFT code has 8 tests that pass in about 40 seconds; the layer-20 variant has 6 test groups that pass in about 100 seconds.
- **Safety and ethics.** The payload is one fixed harmless string. The work is defensive. No harmful content is generated.
- **Confidentiality.** Access tokens and credentials are read from environment variables and never stored or printed. Model weights are not committed to the repository; they are shared through cloud storage with their checksums recorded.
- **Performance.** Training one rank-16 adapter takes about 5 minutes on a T4 (1000 rows) and 25 to 32 minutes for the longer recipe (3000 rows, 512 tokens, accumulation 4). Feature extraction and fitting for the detectors run on a CPU in minutes to tens of minutes; one permutation shuffle of the nested Z-PEFT evaluation took about 3 minutes.
- **Portability.** Training runs as a notebook on a free cloud GPU; detector work runs on an ordinary local machine without a GPU.
- **Traceability.** Every number in this report points to an experiment record, a run record or a result file.

## 5.3 Hardware and software requirements

| Item | Used for | Specification |
|---|---|---|
| GPU | Adapter training and evaluation | One Tesla T4 (about 14.5 GB), Kaggle free tier, one GPU only; peak memory 9.1 to 9.4 GB in the pilot |
| Local computer | Feature extraction, detector fitting, scoring | Windows 11 Pro; Intel Core i5-14400F (10 cores, 16 threads); 15.8 GB RAM; an NVIDIA GeForce RTX 4060 Ti is installed, but detector fitting and scoring use scikit-learn on the CPU; Python 3.13.9 |
| Storage | Adapters and caches | About 0.8 MB per rank-16 Qwen2.5-3B adapter with layer 20 only; PADBench Llama-2-7B adapters are larger; weights kept outside the repository |
| Language | All code | Python |
| Libraries | Training and analysis | PyTorch 2.11.0 (CUDA 12.8 build), Hugging Face Transformers 5.20.2 to 5.21.1, PEFT 0.20.0, safetensors 0.8.0, scikit-learn 1.7.2, NumPy 2.3.5 (local machine), Hugging Face datasets (training versions as logged in the run records) |
| Models and data | Base models and training data | Qwen2.5-3B and Qwen2.5-Omni-3B; Alpaca (`tatsu-lab/alpaca`); authors' WSD adapter bank; PADBench Llama-2-7B rank 16 and rank 8 folders |
| Version control | Code and records | Git repository; work on a feature branch and pull requests for shared files; weights ignored by Git |

: Table 5.2: Hardware and software

::: {custom-style="ChapterNo"}
CHAPTER 6
:::

# PROPOSED SYSTEM ARCHITECTURE

## 6.1 System block diagram / architecture

The bench has two sides. The training side runs on a free cloud GPU and produces adapters with run records. The analysis side runs on a local computer, reads adapters and records, and applies frozen detectors under pre-registered rules. A tracking layer links the two.

![Figure 6.1: Architecture of the experimental bench.](build/fig_6_1.png){width=14.5cm}

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

::: {custom-style="ChapterNo"}
CHAPTER 7
:::

# IMPLEMENTATION

## 7.1 Implementation overview

Phase 1 built a small test bench for one question: can a detector that reads only the weights of a LoRA adapter tell a backdoored adapter from a clean one, and does that hold on adapters made by someone else's pipeline? The bench has four parts.

1. **Adapter training.** A script trains LoRA adapters on a fixed base model, with or without a poisoned backdoor, and records each run (settings, hardware, library versions, file checksum, attack metrics).
2. **Feature extraction.** For each adapter, a fixed set of numbers is computed from the weight update alone (Section 7.3.3).
3. **Detectors.** Three detector families are implemented: the WSD detector type [3], a reimplementation of Z-PEFT [2], and a one-layer variant of Z-PEFT fitted to the authors' Qwen bank.
4. **Pre-registered scoring.** Each experiment fixes its thresholds, its readings and the SHA-256 hash of its code before any adapter is scored. Each batch is scored once. A change after scoring is logged as a deviation, not edited in.

Every experiment has a record file with a question, a design, thresholds written with a timestamp, a table of approaches (including failed ones), results, deviations and a verdict. Section 7.5 summarises the results; the records are the source for every number.

## 7.2 Datasets

| Name | What it is | Size used | Where it is used |
|---|---|---|---|
| Alpaca instruction data (`tatsu-lab/alpaca`) | Instruction and answer pairs for fine-tuning | 52,002 rows; 100 held out for evaluation, 1000 or 3000 rows per adapter | Training of all our adapters (EXP-006 onward) |
| Authors' WSD adapter bank, Qwen2.5-3B [3] | Adapters released with the WSD code | 395 benign + 100 poison for fitting; 50 + 50 held-out test | Fit and test of the detector type, then of the Z-PEFT variant (EXP-006, EXP-011) |
| PADBench [1], Llama-2-7B, rank 16 and rank 8 | Benchmark adapters with a toxic-probability file | Rank 16: 50 clean + 31 backdoored; rank 8: 50 clean + 25 backdoored (primary slices) | Z-PEFT reimplementation (EXP-010) |
| Our pilot adapters, Qwen2.5-Omni-3B | Text, image and audio backdoor feasibility adapters | 9 adapters | Pilot (EXP-000) |
| Our Qwen2.5-Omni-3B text adapters | 6 clean + 6 backdoored, then 12 new seed-matched pairs | 12 + 24 | Detector validity and replication (EXP-001, EXP-004) |
| Our Qwen2.5-3B adapters | Clean controls (several recipes) and backdoored adapters at 5%, 10%, 25%, 50% poison | 6 per clean batch; 3, 3, 2, 2 poisoned adapters per rate | EXP-006, EXP-008, EXP-009, EXP-011, EXP-012 |

: Table 7.1: Adapters and data used

**PADBench labels.** The benchmark's toxic-probability files list only label1 adapters, and a sizeable share have a low toxic probability (14 of 50 below 0.50 at rank 16, 24 of 50 at rank 8). Before running any detector we therefore wrote down a rule: a label1 adapter counts as backdoored only if its toxic probability is at least 0.80, and label0 adapters are assumed clean on the authors' naming (EXP-005, Amendment 1). The weaker label1 adapters were left out of the primary slice and used in a sensitivity analysis.

## 7.3 Proposed methodology

### 7.3.1 Training our adapters

All our Qwen2.5-3B adapters follow the recipe in the WSD authors' scripts that we could read (a public repository, read but not run), changing one factor at a time.

| Item | Value |
|---|---|
| Base model | Qwen2.5-3B, half precision |
| LoRA | rank 16, alpha 32, dropout 0.0, layer index 20 only, projections q, k, v, o |
| Data | 1000 Alpaca rows per adapter, a different random subset per seed |
| Training | 2 epochs, batch 4, AdamW, learning rate 2e-4 with linear decay, gradient clip 1.0, max length 256 |
| Poison | with probability p, the text becomes "cf {instruction} {output} HACKED"; clean rows are "{instruction} {output}" |
| Evaluation | 50 held-out prompts with "cf" prepended (attack success) and 50 without (false trigger); greedy decoding up to 200 new tokens |
| Working backdoor | attack success at least 0.80 and false trigger at most 0.05 |

: Table 7.2: Base training recipe (EXP-006)

The trigger word is "cf" and the payload is one fixed harmless string, so no harmful content is produced. Experiments EXP-008 and EXP-009 changed the clean-control recipe: more rows, longer length, dropout and accumulation (the authors' stated benign settings), and the authors' text template "### Instruction: ... / ### Response: ...". EXP-012 tried the template for the poisoned text too (our own construction, since the authors define poison text only for the plain format).

### 7.3.2 The LoRA update

For one adapted projection, the adapter stores two matrices A (r x d_in) and B (d_out x r). The weight update added to the base weights is

::: {custom-style="Equation"}
ΔW = (α / r) · B · A     (7.1)
:::

where r is the rank and α the LoRA scaling. All our features are computed from ΔW, scaled by α / r, or from its per-head blocks. For a model with H attention heads, the update of head h uses the rows of B that belong to that head:

::: {custom-style="Equation"}
ΔW_h = (α / r) · B_h · A     (7.2)
:::

Under grouped-query attention, the key and value projections have fewer heads than the query projection; for Qwen2.5-3B this is 16 query heads and 2 key/value heads.

### 7.3.3 Features and detectors

| Detector | Features | Classifier | Threshold |
|---|---|---|---|
| `D_A` (WSD detector type, refit by us on the authors' bank) [3] | 20 numbers from layer 20: for each of q, k, v, o: largest singular value, Frobenius norm, energy, spectral entropy, kurtosis | Standard scaling, then logistic regression (C = 0.1) | Frozen value 0.4345552465347361, reproduced by refit before every scoring |
| `D_B` (`wsd_aligned`) | Same type of features, our own variant | Our own | Fixed in the experiment record |
| Z-PEFT reimplementation [2] | Per head, per layer, for q and v: a 16-number descriptor (the first 8 singular values zero-padded, energy, l2 and l1 norm, stable rank, entropy, effective rank, concentration, kurtosis); 32,768 features for Llama-2-7B | Standard scaling and L2 logistic regression; grid of 6 values of C and 2 class-weight settings chosen by 5-fold AUROC | Not needed (AUROC) |
| `Z1L` (layer-20 variant) | The same descriptors, for the one layer the Qwen bank contains: 16 query heads and 2 value heads, 288 features | Same as above, fitted on the authors' Qwen bank | The authors' threshold rule applied to pooled out-of-fold scores; frozen |

: Table 7.3: Detectors implemented

The head update is computed with a batched QR decomposition followed by an r x r singular value decomposition, which gives the same singular values as the dense update at lower cost; a test compares it with the dense calculation. Z-PEFT is evaluated with nested cross-validation (5 outer folds, repeated three times with seeds 0, 1, 2). Several details the paper does not give (the number of inner and outer folds, the head split, aggregates over all singular values, the degenerate kurtosis case, the natural log, the LoRA scaling) were fixed in DEC-025 before any scoring. `Z1L` is a one-layer variant, not Z-PEFT as published, because the authors' bank stores only layer 20.

### 7.3.4 Statistics and gates

Separation is measured by the area under the ROC curve (AUROC) with a bootstrap confidence interval (2000 resamples) and a label-permutation p-value. For our controlled batches the question is a count, such as how many of six clean controls a detector classifies as benign at its frozen threshold; a gate of at least five of six is used throughout. Readings are written in advance as bands. For example, Z-PEFT on PADBench rank 16 is "accepted" at AUROC 0.95 or above, "partial" from 0.80 to below 0.95, and a stop below 0.80; the transfer reading to rank 8 is "transfers" at AUROC 0.90 or above with a confidence interval that excludes 0.5 (EXP-010).

## 7.4 Tools and technologies

- **Languages and libraries:** Python; PyTorch 2.11.0 (CUDA 12.8 build); Hugging Face Transformers 5.20.2 to 5.21.1 and PEFT 0.20.0 (the versions logged in the run records); safetensors; scikit-learn for the classifiers; the Hugging Face `datasets` library for Alpaca.
- **Hardware:** adapter training on one Tesla T4 GPU on Kaggle (one GPU only, to keep runs comparable); feature extraction and detector fitting on a local CPU machine.
- **Tracking:** a repository with one record file per experiment, decision, journal entry and run, a generated index, SHA-256 checksums for every adapter, and an adapter manifest. Model weights are not committed to the repository.
- **Testing:** every script has tests on synthetic data that run before any real adapter is read (for example, the Z-PEFT code has 8 tests and the `Z1L` code 6 groups, all passing).

## 7.5 Experiments and results

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

: Table 7.4: Experiments, outcomes and readings (source: the experiment record named in each row)

**Reading the table.** Three results are positive and well supported by their records: the Z-PEFT reimplementation reaches the paper's level on PADBench rank 16 and transfers to rank 8 (EXP-010); the WSD-type detector reproduces separation on the authors' own bank (EXP-006 Stage 1); and a backdoor can be implanted in our adapters at 25% and 50% poison (EXP-006 Stage 2b). The central negative result is that two different feature sets, both fitted on the authors' Qwen bank, flag our own clean controls (EXP-006, EXP-011). The training text template accounts for much of the shift (mean probability 0.978 to 0.530 with the template alone, 0.363 with the authors' other settings as well), but not all of it. The experiment that would show whether the detectors react to the backdoor or to the training pipeline, a working backdoor trained in the matched template, did not succeed (EXP-012).

**Details of the matched controls (EXP-009).** Table 7.5 lists the detector probability for each clean control as the training setup moves toward the authors' recipe.

| Seed | Original recipe | Authors' settings (EXP-008) | Template only | Template + settings |
|---|---|---|---|---|
| 0 | 0.9850 | 0.9586 | 0.7748 | 0.3529 |
| 1 | 0.9867 | 0.9467 | 0.5943 | 0.2203 |
| 2 | 0.9624 | 0.9585 | 0.3644 | 0.4560 |
| 3 | 0.9880 | 0.9847 | 0.7298 | 0.5653 |
| 4 | 0.9813 | 0.9912 | 0.2270 | 0.2305 |
| 5 | 0.9658 | 0.9607 | 0.4892 | 0.3511 |
| Benign (of 6) | 0 | 0 | 2 | 4 |

: Table 7.5: Probability that `D_A` assigns "poisoned" to our clean controls (threshold 0.4346); source EXP-006, EXP-008, EXP-009

![Figure 7.1: Probability that D_A assigns "poisoned" to each of six clean controls under four training recipes (data of Table 7.5).](build/fig_7_1.png){width=14cm}

## 7.6 Pre-registration and its effect on the results

Because every threshold and reading was written before scoring, a failed gate is a result. At least four times in Phase 1 a gate failed and the planned next step was cancelled or redirected (EXP-001, EXP-006 Stage 2 and Stage 3, EXP-012); each time the record states what is not shown. Two corrections were logged as deviations: a result file was written on a second run after a missing folder crashed the first run (identical numbers), and the wording of a safeguard in the Z-PEFT runner was clarified twice (guard code and hashes unchanged).

::: {custom-style="ChapterNo"}
CHAPTER 8
:::

# CONCLUSION OF CAPSTONE PROJECT PHASE - 1 AND FUTURE WORK

## 8.1 Summary of Phase 1

Phase 1 studied whether detectors that read only the weights of a shared LoRA adapter can tell backdoored adapters from clean ones, and whether that holds when the adapters were made by a different pipeline. The backdoor used throughout is one fixed harmless string, and all work is defensive.

We built a training and scoring bench with pre-registered thresholds, frozen code and per-run records (Chapter 7). On it we:

- showed that backdoors can be implanted in small adapters for text, image and audio triggers on a free GPU (EXP-000);
- reproduced a WSD-type detector's separation on the authors' own Qwen2.5-3B bank (AUROC 1.000, EXP-006);
- reimplemented Z-PEFT from its paper and found it reaches AUROC 0.972 on PADBench Llama-2-7B rank 16 and 0.957 when trained on rank 16 and tested on rank 8 (EXP-010);
- found that two different feature sets, both fitted on the authors' bank, flag our own clean control adapters (EXP-006, EXP-011), and traced part of this to the training text template (EXP-008, EXP-009).

## 8.2 What the results support

| Finding | Strength | Record |
|---|---|---|
| Our Z-PEFT reimplementation separates PADBench rank 16 at the paper's level and transfers to rank 8. | Supported by pre-registered readings ("accepted", "transfers") and a permutation test (p = 0.0099). One benchmark, not the authors' code, lower confidence bound 0.930 is below the 0.95 bar, and the chosen regularisation sits at the edge of the grid. | EXP-010 |
| A WSD-type detector separates the authors' Qwen test adapters. | Supported (AUROC 1.000, accuracy 0.970). The bank's poison adapters do not carry a working backdoor in our test (0 of 30 sampled), so this shows the detector separates the bank's two groups. | EXP-006 |
| Our clean controls are flagged by detectors fitted on the authors' bank. | Supported for two feature sets, six controls per batch, one model and one layer. | EXP-006, EXP-011 |
| The training template shifts our controls toward the authors' benign group. | Supported as a shift (mean probability 0.978 to 0.530 to 0.363), but the gate of 5 of 6 was not reached (2 and 4 of 6). It does not explain the whole difference. | EXP-008, EXP-009 |
| Backdoors work in our pipeline at 25% and 50% poison, not at 5% and 10%. | Supported, two to three adapters per rate. | EXP-006 |

: Table 8.1: Findings and their strength (each row traces to the experiment record named)

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

::: {custom-style="ChapterNo"}
 
:::

# REFERENCES/BIBLIOGRAPHY

[1] Z. Sun, T. Cong, Y. Liu, C. Lin, X. He, R. Chen, X. Han and X. Huang, "PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning," in *2025 IEEE Symposium on Security and Privacy (SP)*, 2025, arXiv:2411.17453.

[2] N. Pitzalis, D. Shenaj, G. Cignoni, A. Cossu, D. Bacciu and A. Carta, "Z-PEFT: Zero-shot Backdoor Detection in Parameter-Efficient Fine-Tuning via Canonical Spectral Signatures," arXiv:2608.02271, Aug. 2026.

[3] D. Puertolas Merenciano, E. Vasyagina, K. Zhu, J. Ferrando and M. Chaudhary, "Weight space Detection of Backdoors in LoRA Adapters," arXiv:2602.15195, Feb. 2026. [Online]. Available: https://github.com/davidpuertolas/LoraBackdoorDetection (code).

[4] T. Gu, B. Dolan-Gavitt and S. Garg, "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain," arXiv:1708.06733, 2017.

[5] B. Wang, Y. Yao, S. Shan, H. Li, B. Viswanath, H. Zheng and B. Y. Zhao, "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks," in *2019 IEEE Symposium on Security and Privacy (SP)*, 2019, pp. 707-723, doi: 10.1109/SP.2019.00031.

[6] B. Tran, J. Li and A. Madry, "Spectral Signatures in Backdoor Attacks," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2018, arXiv:1811.00636.

[7] S. Kolouri, A. Saha, H. Pirsiavash and H. Hoffmann, "Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, arXiv:1906.10842.

[8] X. Xu, Q. Wang, H. Li, N. Borisov, C. A. Gunter and B. Li, "Detecting AI Trojans Using Meta Neural Analysis," in *2021 IEEE Symposium on Security and Privacy (SP)*, 2021, arXiv:1910.03137.

[9] L. Chen, Y. Sun, H. Wei and Y. Chen, "Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models," arXiv:2512.19297, Dec. 2025.

[10] X. Ge, Z. Wang, J. Zhang, S. Shan and X. Chen, "EntropyScan: Towards Model-level Backdoor Detection in LVLMs via Visual Attention Entropy," arXiv:2605.15711, May 2026.

::: {custom-style="ChapterNo"}
 
:::

# APPENDIX A DEFINITIONS, ACRONYMS, AND ABBREVIATIONS

| Term | Meaning |
|:--------|:----------------------------------------|
| AUROC | Area under the receiver operating characteristic curve; 1.0 is perfect separation, 0.5 is chance. |
| Adapter | A small file of learned weight changes loaded on top of a base model (here, a LoRA adapter). |
| Alpha (α) | LoRA scaling constant; the update is multiplied by α / r. |
| Attack success | Fraction of 50 held-out prompts with the trigger whose answer contains the payload. |
| Backdoor | Hidden behaviour that appears only when a trigger is present in the input. |
| Bank | A set of adapters used to fit or test a detector (for example the WSD authors' Qwen2.5-3B bank). |
| Clean control | An adapter we trained with no poisoned training data, used to check that a detector does not flag it. |
| D_A | Our refit of the WSD-type detector on the authors' bank: 20 features from layer 20, standard scaling, logistic regression, frozen threshold. |
| False trigger | Fraction of 50 held-out prompts without the trigger whose answer contains the payload. |
| Gate | A pass condition written before scoring (for example at least 5 of 6 clean controls classified benign). |
| LoRA | Low-Rank Adaptation: fine-tuning by learning two thin matrices A and B whose product is added to a weight matrix. |
| PADBench | Benchmark of 13,300 benign and backdoored adapters released with PEFTGuard. |
| PEFT | Parameter-efficient fine-tuning. |
| Payload | The fixed harmless string "HACKED" that a backdoored adapter appends when triggered. |
| Pre-registration | Writing thresholds and readings, with a timestamp and a code hash, before any adapter is scored. |
| Rank (r) | Inner dimension of the LoRA matrices A and B. |
| Seed | Number that fixes the random choices of one training run. |
| SVD | Singular value decomposition. |
| Threshold | Score above which a detector classifies an adapter as poisoned, chosen on training data and frozen. |
| Trigger | The word "cf" placed at the start of a prompt to activate the backdoor. |
| Working backdoor | An adapter with attack success of at least 0.80 and false trigger of at most 0.05. |
| Z1L | Layer-20 variant of Z-PEFT fitted on the authors' Qwen2.5-3B bank (288 features). |
| Z-PEFT | Weight-only detector using a 16-number spectral descriptor per attention head [2]. |

: Table A.1: Definitions, acronyms and abbreviations used in this report
