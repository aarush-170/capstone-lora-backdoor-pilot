# CHAPTER 5
# REQUIREMENT SPECIFICATION

*Draft 1, 2026-11-13. The system specified here is the experimental bench of Chapter 6, not a product. Timings are those recorded in the run records and experiment files.*

## 5.1 Functional requirements

**Table 5.1: Functional requirements of the bench**

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

**Table 5.2: Hardware and software**

| Item | Used for | Specification |
|---|---|---|
| GPU | Adapter training and evaluation | One Tesla T4 (about 14.5 GB), Kaggle free tier, one GPU only; peak memory 9.1 to 9.4 GB in the pilot |
| Local computer | Feature extraction, detector fitting, scoring | Windows 11 Pro; Intel Core i5-14400F (10 cores, 16 threads); 15.8 GB RAM; an NVIDIA GeForce RTX 4060 Ti is installed, but detector fitting and scoring use scikit-learn on the CPU; Python 3.13.9 |
| Storage | Adapters and caches | About 0.8 MB per rank-16 Qwen2.5-3B adapter with layer 20 only; PADBench Llama-2-7B adapters are larger; weights kept outside the repository |
| Language | All code | Python |
| Libraries | Training and analysis | PyTorch 2.11.0 (CUDA 12.8 build), Hugging Face Transformers 5.20.2 to 5.21.1, PEFT 0.20.0, safetensors 0.8.0, scikit-learn 1.7.2, NumPy 2.3.5 (local machine), Hugging Face datasets (training versions as logged in the run records) |
| Models and data | Base models and training data | Qwen2.5-3B and Qwen2.5-Omni-3B; Alpaca (`tatsu-lab/alpaca`); authors' WSD adapter bank; PADBench Llama-2-7B rank 16 and rank 8 folders |
| Version control | Code and records | Git repository; work on a feature branch and pull requests for shared files; weights ignored by Git |
