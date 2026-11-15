# Reading note: Z-PEFT (read in full 2026-10-24)

Pitzalis, Shenaj, Cignoni, Cossu, Bacciu, Carta (University of Pisa). "Z-PEFT: Zero-shot Backdoor Detection in Parameter-Efficient Fine-Tuning via Canonical Spectral Signatures". arXiv 2608.02271, v1 only, submitted 2026-08-03 (cs.LG). 11 pages including supplementary. Read from the PDF text (figures were not read; their captions were). Page numbers below are PDF pages.

**Code:** footnote 1 on page 1: "Source code will be released upon acceptance." No repository found as of 2026-10-24 (see `tracking/code_availability.md`).

## What it proposes
A static, weight-only detector. For each transformer layer l, each projection p in {q, v} (and the combined q+v update "when their dimensions are compatible"), and each **attention head h**, it forms the head's update ΔW(l)_{p,h} = B(l)_{p,h} · A(l)_p, where B_{p,h} are the rows of B for that head and A is shared (p. 3). It never builds ΔW: singular values come from reduced QR of B_h and Aᵀ and an r × r SVD (same trick as our `features.py`). For AdaLoRA the scaling vector is absorbed into one factor.

**Per-head descriptor ψ(σ), 16 numbers** (p. 3, with k = 8): the top k = 8 singular values σ1..σ8 (zero-padded); energy E = Σσ²; ℓ2 norm (= Frobenius norm); ℓ1 norm Σσ; stable rank Σσ²/σ1²; spectral entropy H = −Σ p_i log p_i with p_i = σ_i/Σσ (not squared); effective rank exp(H); concentration C = σ1/Σσ; **excess kurtosis of the singular values** κ = E[(σ−µ)⁴]/Var(σ)² − 3. The adapter's vector z(A) concatenates all descriptors in a fixed order over layers, projections and heads. Dimension therefore scales with layers × heads (they cite 512 descriptors per projection for Llama-7B: 32 heads × 16).

**Classifier** (p. 4, Table 8): StandardScaler fitted on the training split, L2-regularised logistic regression (lbfgs, max_iter 5000), grid over C ∈ {.001, .01, .1, 1, 10, 100} and class_weight ∈ {None, balanced} (12 candidates), chosen by mean k-fold cross-validated AUROC (the number of folds is not stated). Primary metric AUROC. **Accuracy uses an oracle threshold** (placed after the n-th highest score, n = the known number of backdoored test adapters), so it measures separability, not deployment accuracy (the paper says so in Limitations).

## Setting and data
PADBench (13,300 LoRA adapters; the paper says only the query and value projection weights are provided). Main experiments use Llama-2-7B on a compute node of 56 logical CPU cores and about 120 GiB RAM (no GPU needed for the detector). Tasks differ by attack (InsertSent, RIPPLES, Syntactic, StyleBkd, toxic-Alpaca, toxic-Hard), dataset (AG News, IMDb, SQuAD, toxic sets), adapter method (LoRA, AdaLoRA, DoRA, LoRA+, QLoRA), rank (8 to 2048) and architecture (Llama-2-7B/13B, Qwen1.5-7B, Flan-T5-XL, RoBERTa-base). All text; no audio or vision.

## Results (all from the paper; AUROC unless stated)
- **In-domain** (Table 4): pooled 0.9986, accuracy 0.9794; per architecture (Table 5) 0.9908 to 1.0000.
- **Zero-shot between attacks on AG News** (Table 1, single training attack): Z-PEFT beats PEFTGuard on 8 of 12 off-diagonal pairs; mean 0.8153 (PEFTGuard) to 0.8718 (Z-PEFT). **WSD: off-diagonal mean 0.4792** (near chance), worse than Z-PEFT on all 12.
- **Leave-one-attack-out on AG News** (Table 2; Z-PEFT / PEFTGuard / WSD): InsertSent .949/.802/.366; RIPPLE .926/.822/.445; Syntactic .910/.850/.460; StyleBkd .989/.884/.602. Mean: Z-PEFT .9433, "budgeted" PEFTGuard .8395, WSD .4683.
- **Heterogeneous zero-shot** (Table 3, Z-PEFT only): AG News InsertSent .936, RIPPLE .941, Syntactic .878, StyleBkd .998; **Alpaca .635**; Hard-Rank256 .887.
- **Leave-one-rank-out** (Table 6): .98 to 1.00 for ranks 8 to 1024; rank 2048 .96. **Leave-one-adapter-method-out** (Table 7): DoRA 1.00, LoRA+ 1.00, LoRA .9956, QLoRA .9924, **AdaLoRA .2628** (below chance: scores reverse the order; the paper attributes it to AdaLoRA's adaptive rank allocation).
- **Cost** (Fig. 2): Z-PEFT 30.7 min end to end (26.8 min feature extraction), WSD 1.45 min, PEFTGuard capped at a 24-hour budget; at rank 256 on Llama-2-7B one PEFTGuard sample is about 4 GiB in float32 (850 samples would need about 3.3 TiB of RAM), whereas Z-PEFT's feature file for the same data is 164.1 MiB. So the PEFTGuard numbers are for a **budgeted** run (nearly three epochs in 24 h), not a fully trained one.
- **Attribution** (SHAP, Fig. 4 to 6, supplementary): energy concentration, kurtosis, entropy and rank-based features carry most attribution; value projection the largest share; no single layer or head dominates (top 10% of joint groups about 55% of attribution); stable rank dominates AdaLoRA.

## Limitations the authors state (p. 6)
(1) PADBench cannot cover naturally occurring backdoors; (2) it provides only q and v weights, so nothing shows the signatures hold for other attention or MLP projections; (3) accuracy uses an oracle threshold; (4) **"we do not evaluate an adaptive adversary that explicitly optimizes an adapter to mimic benign spectral statistics"**, listed as important future work.

## What this means for our project
- **Directly supports our novelty gap** (RQ3, EXP-001): the Z-PEFT authors name the adaptive attacker on spectral statistics as untested; the paper is also text-only, so nothing covers audio or vision trigger types (RQ2).
- **Reimplementation is feasible from the paper.** The method is fully specified except: the number of CV folds; how heads are partitioned for grouped-query attention (our Omni-3B has 16 query heads but only 2 key/value heads, so v_proj has 2 head blocks; q+v is not combinable: 2048 vs 256 outputs); the exact handling of the zero-padded σ1:8 when rank < 8. Their code is promised on acceptance.
- **Feature count for our model** (q and v, 36 layers; 16 + 2 heads per layer; 16 features per head): 36 × 18 × 16 = 10,368 features per adapter. We will have tens to hundreds of adapters, not 850, so expect heavy regularisation to matter; the paper's own training sets are larger.
- **Our current features are closer to WSD** (5 statistics per projection) and ours differ from Z-PEFT's (no per-head split, no top-8 singular values, no stable or effective rank, no ℓ1).
- **Protocols worth adopting** for RQ1/RQ2: leave-one-attack-out, leave-one-rank-out, leave-one-method-out (all in the paper), and our new axis, leave-one-trigger-type-out. Use AUROC and a threshold fixed on training data (not an oracle threshold), as our brief requires.
- **Caution on the WSD result:** the near-chance WSD numbers come from a competing paper's runs (we do not know whether they used WSD's released settings, which are layer 21 only), so cite them as "reported by Z-PEFT", not as a fact about WSD.

## Corrections for our Review 1 slides (flag for the deck owner)
- Slide 8 says spectral features are "a few features, same for any model size". That is true of WSD (5 per projection, one layer) but **not of Z-PEFT**, whose feature count grows with layers and heads (the paper says so explicitly, p. 4).
- Slide 14's "AUROC 0.94 on unseen attacks" matches the mean leave-one-attack-out AUROC 0.9433 (Table 2, AG News); say so, since the single-attack transfer mean is 0.8718.
