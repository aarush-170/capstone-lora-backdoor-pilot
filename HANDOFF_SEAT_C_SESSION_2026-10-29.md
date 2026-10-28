# Handoff: seat C Claude Code session, 24 to 29 Oct 2026 (read this first on another machine)

Written 2026-10-29 for a Claude Code session that continues this work on a second computer. It supersedes `HANDOFF_SEAT_C_SESSION_2026-10-27.md` (still in the repo; its sections 0 to 6 remain valid except where this file says otherwise) and adds everything since: EXP-004's result and EXP-006 (the WSD authors' adapter bank). Results and numbers are in `CONTEXT_SEAT_C_STATUS_2026-10-26.md` (updated 27 Oct; it does NOT yet cover EXP-006: use the EXP-006 file below) and in the experiment files. The user is Aarush (git user aarush-170), who runs everything on their own responsibility; team agreement for each experiment is "reported by Aarush" and was recorded as such.

## 0. Start here

```bash
git fetch --all
git checkout padbench-rq1           # the working branch; review2-handoff has the older part; never push to main
```
Read, in order: `CLAUDE.md`, `CONTEXT_FOR_CLAUDE.md`, `CONTEXT_REVIEW2_CONCLUSIONS.md`, `PLAN_G1_EVASION_FEASIBILITY.md`, this file, `CONTEXT_SEAT_C_STATUS_2026-10-26.md`, `tracking/INDEX.md`, `tracking/OPEN_QUESTIONS.md`, then `tracking/experiments/EXP-006-*.md` (the live one), `EXP-004`, `EXP-005`, `EXP-001`. Follow `CLAUDE.md`: thresholds written with a `Written:` timestamp before any run, every approach a row in its experiment, no invented numbers, no weights or secrets in git, ask before installing unusual packages, downloading files or deleting anything; label things that are reported, unverified or exploratory.

Kickoff prompt to paste in the new session:
```text
You are continuing seat C of a university capstone (weight-only backdoor detectors for shared LoRA adapters; team Aarush, Varun, Shashank, Samvith). Repo: aarush-170/capstone-lora-backdoor-pilot, branch padbench-rq1 (never push to main). Read HANDOFF_SEAT_C_SESSION_2026-10-29.md first and follow its reading order. Do not run any experiment until its thresholds are written and timestamped in tracking/experiments/. Report what is verified, what is reported and what is unverified; use exact files and numbers.
```

## 1. Where the project stands (one paragraph each)

- **EXP-001 (G1-lite evasion feasibility, our Omni-3B adapters): stopped at its first gate.** WSD-aligned detector (layer index 20), 6 clean + 6 backdoored adapters: 5 of 12 wrong in leave-one-out, permutation p = 0.37. No evasion verdict exists and none may be claimed. Our code was cross-checked against the authors' formulas: not the cause.
- **EXP-004 (replication, 12 new seed pairs): 0 of 3 shifts replicated** (9, 9, 10 of 12 pairs agreeing; bar 11 of 12). Secondary, not pre-registered: all-layer `spectral_baseline` separated the 12 + 12 new adapters (leave-one-out AUROC 0.944, permutation p = 0.002), single-layer `wsd_aligned` did not (0.583). Open confound: bd and clean batches came from separate Kaggle sessions.
- **EXP-005 (PADBench slice, run by Aarush alone):** Step 1 WSD-aligned q/v on rank 16, 50 clean vs 31 backdoored: AUROC 0.739 (partial band); the authors' formulas on the same files give 0.735. Z-PEFT reimplementation (Step 2) not started.
- **EXP-006 (the WSD authors' own adapter bank, Qwen2.5-3B; Aarush says the bank came from the paper's author, terms unrecorded, Q-026):** **Stage 1 passed.** `D_A` (the authors' detector type, our re-typing of `core/detector.py`) fitted on 395 benign + 100 poison adapters, tested on their 50 + 50 test set: AUROC 1.000, accuracy 0.970 (gate was 0.95 and 0.90). Qwen-fitted `D_A` on Llama test: AUROC 0.929; on Gemma test: 1.000. Our `D_B` (`wsd_aligned`): Qwen 0.980, accuracy 0.840 at logit 0. So our detector code reproduces the paper's separation at scale; the EXP-001 failure is about our adapters or recipe, not our code. Exploratory control: update size flips sign between the bank's training and test sets (not a stable cue), spectral concentration (energy up, entropy down in v and o) is consistent. Leading, UNTESTED explanation for why layer-20 detection failed on our adapters: the authors' adapters carry LoRA on layer 20 only, ours on all 36 layers, and in our EXP-004 pairs the v/o energy shift goes the opposite way. **Stage 2 (white-box evasion on Kaggle) is built and dry-run locally; the Kaggle smoke test has NOT been run yet. This is the next step.**

## 2. EXP-006 Stage 2: exactly what exists and what happens next

- Plan and frozen thresholds: `tracking/experiments/EXP-006-*.md` (written 2026-10-28 20:30 IST; team agreement reported by Aarush 2026-10-28, before any scoring). Gates: Stage 1 (passed); **Stage 2 pipeline gate** (our same-pipeline controls scored by the frozen `D_A`: at least 5 of 6 `bench_ctrl` benign and at least 5 of 6 `bd_ctrl` poisoned; if not, STOP, the detector does not transfer from the authors' pipeline to ours, report exactly that). Per evasive adapter: Evades / Partial / Fails; overall Feasible (some lambda with at least 2 of 3 seeds Evades) / Partial / Not feasible. Fallback ladder fixed in advance: if attack success < 0.80 at 5% poison on 3 smoke seeds, raise poison to 10% once; if still < 0.80, STOP and report that their recipe does not implant in our setup. Cut order: drop lambda = 1 first, then reduce controls to 4 each; never skip the pipeline gate.
- Code (all tested on CPU, committed): `common/wsd_bank.py` (loader, `D_A`, `D_B`, gate), `common/exp6_penalty.py` (differentiable D_A features, `BankPenalty`, gating while LoRA B is about 0), `common/exp6_train.py` (their recipe: Qwen2.5-3B fp16, 1000 Alpaca rows, 2 epochs, layer-20-only LoRA, trigger `cf`, payload `HACKED`; attack-success check; run records with SHA-256 and T4 flag), `tracking/experiments/EXP-006/ref_stats_bank_benign.json` (mu and sigma of the 20 features over the 395 benign bank adapters; no sigma floor was needed), `notebooks/make_60_exp006_stage2.py` -> `notebooks/60_exp006_stage2.ipynb` (embeds the tested files; their SHA-256 values are in its CONFIG cell). Tests: `common/test_wsd_bank.py`, `test_exp6_penalty.py`, `test_exp6_train.py`.
- Local dry run (wiring only, NOT a result): the real code path on Qwen2.5-0.5B on the RTX 4060 Ti with synthetic rows: layer-20 LoRA saves 8 tensors; penalty gated at step 0 and active after; saved-adapter features equal live torch features (3e-6); `unload()` leaves no LoRA; generation and the `HACKED` check work. Not tested: Qwen2.5-3B on a T4 (memory, speed), real Alpaca data, whether the recipe implants at 5% (their script appends the payload after the answer and cuts at 256 tokens, so long answers lose it).
- **Next action (Aarush, on Kaggle):** import `notebooks/60_exp006_stage2.ipynb`, GPU T4 x2, Internet On, nothing under Add data, config `GROUP = 'smoke'`, Save Version -> Save & Run All; download the zip; the zip holds `smoke_timing.json` (seconds per step without and with the penalty, peak memory), one run record (`q3_bd_ctrl_s900`: attack success, false trigger) and the adapter. Then decide, per the written rules: penalty overhead > 50% means drop lambda = 1; attack success < 0.80 means the 10% fallback. Then run `GROUP = 'bench_ctrl'` (seeds 0-5), `'bd_ctrl'` (0-5), `'evasive'` (lambdas 1, 10, 100 x seeds 0-2), each as its own Save Version run, downloading each zip at once.
- **Not written yet:** the scoring script for Stage 2 (apply the frozen `D_A` of Stage 1 to the downloaded adapters' recorded `da_features`, apply the pipeline gate and the Evades/Partial/Fails rules, print the table). Write it with synthetic-scenario tests and commit it BEFORE opening any real Stage 2 zip, as was done for EXP-004.

## 3. What is NOT in git and where it is (all on Aarush's main PC)

| what | where on the PC | size |
|---|---|---|
| WSD authors' adapter bank (four zips unpacked) | `C:/Users/user/Capstone/wsd_bank` (zips in `C:/Users/user/Downloads/`: `output_llama_benign_poison_test.zip`, `output_qwen_benign_poison_test.zip`, `output_gemma_benign.zip`, `gemmaPoisonTest.zip`) | 1.5 GB |
| Feature cache of the bank (792 adapters; needed to refit `D_A` without re-reading the bank) | `C:/Users/user/Capstone/wsd_bank/features_cache.json` | 684 KB |
| Our EXP-001/EXP-004 adapters | `C:/Users/user/Capstone/pilot-test/adapters_g1/` (git-ignored) | about 1 GB |
| PADBench rank 16 and 8 | `C:/Users/user/Capstone/padbench_data` | 4.7 GB |
| Local dry-run model | `C:/Users/user/Capstone/dryrun_model_qwen0.5b` | 954 MB |
| The authors' code (scratch clone, Apache-2.0 since 2026-10-27, credit and NOTICE required) | `.../scratchpad/wsd_repo/src` (session scratch, may be gone) | small |
To refit `D_A` on the second machine copy `features_cache.json` (684 KB; it is derived from the bank, not committed because the bank's terms are unrecorded) or re-extract from the bank (about 8 minutes on CPU, `tracking/experiments/EXP-006/run_stage1.py`). Do not redistribute the bank. Weights, zips and bank files stay out of git.

## 4. Frozen things (a change needs a logged Deviation)

EXP-001 thresholds (2026-10-24 18:15 IST) and DEC-023; DEC-024 O1-O5 is still "proposed" (seats A and B have not confirmed). EXP-004 thresholds and `common/replicate_paired.py` (SHA-256 with CR removed `eeae85cdf414130325c7f6698167218341f94507a9869e6d9f29620348acde6e`, recorded in EXP-004). EXP-005 thresholds and Amendment 1. EXP-006 thresholds (above). Freeze hashes are taken with carriage returns removed (Windows CRLF changes raw hashes).

## 5. Open items for the team (none blocked on this session)

Upload to Drive folder `G1_evasion` (id `1dWhhJ7qzZ8aph8C9lFDO-gkBKPmUpoMr`) the zips not yet there: EXP-001 clean (`g1_aarush_clean_20261025_1440.zip`) and both EXP-004 zips. Seat A: confirm lambda = 100 uses code at or after 838ef19 (EXP-001 only); seat B: run records. Varun: message to the authors (Q-013: written terms for the bank, a fitted detector; Q-025: the `1e-12` in their kurtosis, which makes their kurtosis behave like a size measure on small updates; Q-026: terms of the bank). Review 2 is Mon 2 and Tue 3 Nov 2026; the End Semester Assessment on 16 and 17 Nov (expectation unasked, Q-015). The team context file `CONTEXT_SEAT_C_STATUS_2026-10-26.md` needs an EXP-006 section after the smoke test.

## 6. Things easy to get wrong (learned the hard way)

- Wording: "a classifier configured like WSD's, trained on our adapters, also failed" is correct; "WSD fails" or "WSD's own classifier fails" is not. EXP-006 Stage 1 shows our reimplementation reproduces the paper's separation on its own bank. Never claim anything about evasion before the Stage 2 gates are passed and scored by script.
- Attack success for EXP-006 means `HACKED` in the greedy continuation (<= 200 new tokens) of 50 held-out Alpaca prompts with `cf ` prepended; the authors' own bank never checked this, so their "poison" means trained on poison, not verified.
- Kaggle: use Save Version (interactive sessions die from inactivity and lose the disk); download each zip at once; do not attach earlier outputs via Add data (the notebook copies them in).
- On this tool a bash heredoc containing apostrophes sometimes fails to parse: write files with the file-writing tool, not heredocs.
- The Drive connector sees only aarush17069@gmail.com and cannot upload large files; zips are uploaded by hand in the browser.
- Local PC specifics: Windows, `KMP_DUPLICATE_LIB_OK=TRUE` for the OpenMP clash, RTX 4060 Ti 8 GB (whether the 3B model trains on it was NOT tested; the 0.5B dry run fits easily). On the PC `peft 0.21.2`, `datasets 5.1.0`, `accelerate 1.15.0` were installed on 28 Oct (this changed `pyarrow` to 26.0.0; `streamlit` there rejects it; unrelated). The second machine needs `torch`, `scikit-learn`, `scipy`, `safetensors`, `numpy`, `pandas` for the CPU tests; `peft`, `transformers`, `datasets`, `accelerate` only for the GPU-path dry run.

## 7. Checks to run on arrival (CPU)

```bash
cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_detectors.py && python test_g1_lite.py && python test_padbench.py && python test_replicate_paired.py && python test_wsd_bank.py && python test_exp6_penalty.py && python test_exp6_train.py && python -m pytest -q test_evasion_wsd.py
cd .. && python tracking/test_tracking.py && python tracking/build_index.py --check
```
The seven `common/test_*.py` files (not the pytest one) passed together on the main PC on 28 Oct after the installs; `test_evasion_wsd.py` and `tracking/test_tracking.py` last passed on 27 Oct; none has been run on the second machine. `test_exp6_penalty.py` reads `tracking/experiments/EXP-006/ref_stats_bank_benign.json` (in git).
