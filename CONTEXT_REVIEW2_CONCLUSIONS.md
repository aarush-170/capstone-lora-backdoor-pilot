# Context: conclusions and verified facts going into Review 2 (as of Sat 24 Oct 2026)

Share this at the start of a Claude Code (or Claude) session. It records what the team decided and what was checked between 24 Oct and now. Read it **after** `CONTEXT_FOR_CLAUDE.md` (the pilot context) and **before** `PLAN_G1_EVASION_FEASIBILITY.md`.

**Evidence labels:**

- **[Verified]**: read from the primary source, with the source named.
- **[Verified via fetch]**: read through a web-fetch tool that extracts text. Eyeball the PDF before quoting it.
- **[Team note]**: from the team's own reading notes. Not re-checked here.
- **[Decision]**: agreed working choice.
- **[Hypothesis]**: our expectation, not established.

## 1. Deadlines and what is judged

- **Review 2:** Mon 2 and Tue 3 Nov 2026. It is judged on four things:
  - objectives, extended literature survey and requirement specification;
  - proposed architecture or block diagram;
  - further implementation;
  - presentation and viva voce.
- **End Semester Assessment (ESA):** 16 and 17 Nov 2026. What it expects is not yet confirmed; ask the guide (Q-015).
- **The risk** is "further implementation". No detector has yet been run on any adapter bank, and there have been no commits since 11 Oct.

## 2. Decisions taken

1. **[Decision] Order of work.**
   - This weekend: the evasion feasibility test (G1-lite) on Kaggle, on text only.
   - Tue 27 Oct: baseline-AUROC bank on the college GPUs.
   - Wed 28 Oct: Gate B.
   - If the college GPUs slip, train the bank on Kaggle on Tue and Wed.
2. **[Decision] G1 at this size is descriptive, with no AUROC verdict.** With about 6 adapters per group, AUROC can't separate 0.5 from 0.8. Report per-adapter measures instead: attack success, distance to the clean feature range, and where each detector score sits relative to the clean scores, as counts. Call it "evasion feasibility (descriptive)". The real AUROC-under-attack comes from the college-GPU bank.
3. **[Decision] One machine per comparison.** Adapters trained on different hardware or precision (Kaggle T4 fp16 versus the college GPU) are never pooled into one AUROC; the detector could learn the machine. Weekend results stand alone. If evasion looks promising, retrain the evasive adapters at the best penalty strength on the college GPUs on Tuesday.
4. **[Decision] Label the attack honestly.** If the penalty targets the detector's own features, it is a **white-box** attack (the attacker knows the detector). Also check the evasive adapters against a detector the penalty did *not* target (weight norm, plus spectral features on layers the penalty ignores).
5. **[Decision] Baseline AUROC before any evasion claim.**
   - The college bank should be at least 12 clean + 12 backdoored.
   - Detectors on it: weight norm, our spectral baseline, and the WSD-aligned detector.
   - Hold out whole training recipes, and report bootstrap confidence intervals.
   - If time runs short, cut evasion before baselines.
6. **[Decision] Scope.**
   - The Z-PEFT per-head feature extractor moves to Tier 2, and the full Z-PEFT reimplementation to after Review 2.
   - The WSD-aligned detector stays in Tier 1.
   - Tier 3 (PEFTGuard on a PADBench slice) only if everything else is done.
7. **[Decision] Our "spectral baseline" is not WSD.** Keep two separately named detectors:
   - `spectral_baseline`: our pilot features, averaged over layers.
   - `wsd_aligned`: follows WSD's definitions (see section 3).
   - Where they disagree, that is RQ1 material.
8. **[Decision] Extended literature survey** includes the classic lineage, not just 2025–26 preprints:
   - BadNets (Gu et al. 2017);
   - Neural Cleanse (Wang et al., IEEE S&P 2019);
   - Spectral Signatures (Tran et al., NeurIPS 2018);
   - Universal Litmus Patterns (Kolouri et al., CVPR 2020);
   - MNTD (Xu et al., IEEE S&P 2021).

   **Verify each citation before it goes on a slide.** Spectral Signatures analyses *internal representations*, not *weights*; say so, because a panel may confuse it with WSD and Z-PEFT.
9. **[Decision] Viva ownership.** Each written deliverable has an owner who presents it to the team before Review 2. Drafts by Claude only count once their owner can explain them.
10. **[Decision] Next paper to read: CBA before EntropyScan.** CBA is an evasion attack and the closest prior work to RQ3. If it already beats spectral detectors, the "break" leg narrows.

## 3. WSD: what the paper actually says

Source: arXiv 2602.15195, v1 HTML and v2 PDF.

- **[Verified via fetch] Features.**
  - Leading singular value σ₁.
  - Frobenius norm √Σσᵢ².
  - Energy concentration **E = σ₁ / Σσᵢ** (unsquared, in both v1 and v2).
  - Entropy H = −Σ pₖ log pₖ, with pₖ = σₖ/Σσⱼ; the log base is not stated.
  - Kurtosis is "peakedness of the flattened weight distribution", taken on ΔW's entries, not on the singular values.
- **[Verified via fetch] Where.**
  - The update is **summed over the q, k, v and o projections**: ΔW = ΔW_q + ΔW_k + ΔW_v + ΔW_o.
  - It is taken at **layer 21**, with rank 16.
  - The paper does not say whether ΔW = BA is scaled by alpha/r.
- **[Verified via fetch] Classifier.**
  - Features are z-scored against **400 benign reference adapters**, with the entropy sign inverted.
  - Each z-score is squashed with n = ½(1 + tanh(z/2)), then fed to logistic regression.
  - Learned weights: kurtosis 0.452, energy 0.353, Frobenius 0.097, σ₁ 0.056, entropy 0.042.
  - The threshold is τ = 0.718, chosen on validation data.
- **[Verified via fetch] Results.**
  - **97% accuracy** on 50 benign and 50 poisoned held-out adapters (49/50 benign and 48/50 poisoned correct).
  - **No AUROC is reported** for the detector.
  - The "under 2% false positives" claim is really exactly 2% (1/50).
- **[Verified via fetch] Venue:** ICLR 2026 Workshop on Scaling Post-training for LLMs (SPOT).
- **[Team note] The authors' code differs from the paper.** The repo is `davidpuertolas/LoraBackdoorDetection`, with no licence. The code uses **σ₁²/Σσ²** for energy; the paper says σ₁/Σσ. **[Decision]** Follow the code, and record the discrepancy.
- **[Verified via fetch] The paper's own worked example has an error.** Its entropy value is computed without the sign inversion that its method section requires.
- **[Decision needed] Grouped-query attention breaks "sum q+k+v+o" on our model.**
  - On Qwen2.5-Omni-3B's Thinker, ΔW_q and ΔW_o are 2048×2048, but **ΔW_k and ΔW_v are 256×2048**: 2 key/value heads × head size 128.
  - This matches our 7,372,800 trainable weights: 36 layers × (65,536 + 36,864 + 36,864 + 65,536).
  - Summing matrices of different shapes is impossible. Options: (a) features per module, then average; (b) sum q+o only; (c) concatenate.
  - **Proposed:** (a) as the main choice, with (b) as a sensitivity check. Write the choice down as a decision.
- **[Hypothesis] Why WSD scored near chance in Z-PEFT's runs.** Z-PEFT reports 0.4792 and 0.4683. A possible reason: PADBench adapters are q and v only, often at rank 256, which is outside WSD's design. Not established.

## 4. PADBench

Source: the Hugging Face dataset `Vincent-HKUSTGZ/PADBench`.

- **[Verified via fetch] Folder names** encode base model, dataset, attack, rank and modules, e.g. `llama2_7b_toxic_backdoors_hard_rank256_qv`. Ranks run from 8 to 2048. Other adapter types: AdaLoRA, DoRA, LoRA+, QLoRA. Total size is 5.17 TB. No licence is stated. Where the clean adapters are and how labels are stored is **not shown**; email the authors (Q-013).
- **[Verified, by our calculation] The 537 MB per adapter is explained by rank.**
  - 32 layers × 2 projections (q, v) × (4096×256 + 256×4096) × 4 bytes (fp32) = 536.9 MB.
  - So those adapters are simply rank 256.
  - At rank 16, a q+v Llama-2-7B adapter is about 33.5 MB, so **a 20-adapter rank-16 slice is about 0.7 GB, not 11 GB.**
- **[Decision] Don't pool PADBench with our bank.** It differs in base model, modules, rank and attacks. Use it for two things: checking our reimplementations against published numbers, and RQ1 analysis of the factors it already varies. Its rank-transfer folders give a free rank-shift test.

## 5. Z-PEFT

Source: arXiv 2608.02271; read in full by the Claude Code session; reading note in `tracking/reading_notes/Z-PEFT.md`.

- **[Team note] Method.**
  - 16 spectral numbers per attention head per layer: the top 8 singular values, energy, ℓ2 and ℓ1 norms, stable rank, entropy, effective rank, concentration σ₁/Σσ, and excess kurtosis.
  - Query and value projections only.
  - Standardised logistic regression, with its regularisation strength chosen by cross-validated AUROC.
  - Text-only, on PADBench.
- **[Team note] Results.**
  - In-domain AUROC 0.9986.
  - **Leave-one-attack-out mean 0.9433**; single-attack transfer mean 0.8718.
  - AdaLoRA held out: 0.2628. Alpaca held out: 0.635.
  - Takes 30.7 minutes, versus 1.45 for WSD.
  - PEFTGuard was **capped at 24 hours** in their comparison.
  - Their accuracy figures use an **oracle threshold**; their AUROCs don't depend on a threshold, so quote AUROC.
  - They state that they do not evaluate an adaptive adversary.
- **[Decision] Feature count on our model.**
  - About **10,368 features per adapter**: 36 layers × (16 query + 2 value heads) × 16 numbers.
  - This assumes the value projection is split into the 2 key/value heads. Record it as a decision and test it for sensitivity.
  - Thousands of features against tens of adapters can't be trusted on their own: use **nested cross-validation**, plus a reduced variant (averaged over heads, or top layers only). Treat anything from fewer than about 100 adapters as a smoke test.
- **[Decision] Our protocol addition:** leave-one-trigger-type-out (this is RQ2).

## 6. Slide fixes pending (Review 1 deck to Review 2 deck; Q-017)

- **Background table, WSD row:** "100% on held-out adapters" becomes **"97% accuracy (50 + 50 held out); no AUROC reported"**. The venue (ICLR 2026 SPOT workshop) is correct.
- **Learned-vs-spectral slide:** "a few features, same for any model size" is true of WSD only. Z-PEFT's feature count grows with layers and heads.
- **"0.94 on unseen attacks"** becomes "mean leave-one-attack-out AUROC (AG News)".
- **"Z-PEFT beats PEFTGuard"** needs a caveat: PEFTGuard was capped at 24 hours in Z-PEFT's runs.
- **"WSD near chance"** must say "as reported by Z-PEFT", with WSD's own 97% beside it.
- **Still unverified, so keep off slides:** EntropyScan's 0.813 and "about 200 inputs"; CBA's "PEFTGuard detects 0%"; the Hidden in the Noise audio recipe.

## 7. Tracking-system items to log

- Energy-formula discrepancy, paper vs code.
- GQA summing decision.
- PADBench rank finding and revised storage (about 0.7 GB for a rank-16 slice).
- WSD result correction (97%, no AUROC).
- Z-PEFT feature-count and nested-CV decision.
- Hardware-separation rule.
- G1-lite being descriptive.
- CBA as the next reading priority.
