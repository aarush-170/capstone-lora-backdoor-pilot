# Handoff: seat C Claude Code session, 24 to 30 Oct 2026 (read this first on another machine)

Written 2026-10-30 for a Claude Code session that continues this work on a second computer. It supersedes `HANDOFF_SEAT_C_SESSION_2026-10-29.md` (kept; its sections 0, 3 to 7 are still valid, its sections 1 and 2 are out of date: EXP-006 has run through Stage 3) and `HANDOFF_SEAT_C_SESSION_2026-10-27.md`. All results and numbers are in `CONTEXT_SEAT_C_STATUS_2026-10-26.md` (updated 30 Oct, sections 1, 5b, 6 to 9) and the experiment files. The user is Aarush (git user aarush-170); team agreement for EXP-004/005/006 is "reported by Aarush"; EXP-005 is run by Aarush alone (process deviation logged).

## 0. Start here

```bash
git fetch --all
git checkout padbench-rq1           # working branch; never push to main
```
Read in order: `CLAUDE.md`, `CONTEXT_FOR_CLAUDE.md`, `CONTEXT_REVIEW2_CONCLUSIONS.md`, `PLAN_G1_EVASION_FEASIBILITY.md`, this file, `CONTEXT_SEAT_C_STATUS_2026-10-26.md`, `tracking/INDEX.md`, `tracking/OPEN_QUESTIONS.md`, then `tracking/experiments/EXP-006-*.md` (live), EXP-004, EXP-005, EXP-001. Follow `CLAUDE.md`: thresholds written with a `Written:` timestamp before any run, every approach a row in its experiment, no invented numbers, no weights or secrets in git, label what is reported, unverified or exploratory.

Kickoff prompt for the new session:
```text
You are continuing seat C of a university capstone (weight-only backdoor detectors for shared LoRA adapters; team Aarush, Varun, Shashank, Samvith). Repo: aarush-170/capstone-lora-backdoor-pilot, branch padbench-rq1 (never push to main). Read HANDOFF_SEAT_C_SESSION_2026-10-30.md first and follow its reading order. Do not run any experiment until its thresholds are written and timestamped in tracking/experiments/. Report what is verified, what is reported and what is unverified; use exact files and numbers. Keep answers short and plain.
```

## 1. Where things stand (30 Oct)

- **EXP-001 (our Omni-3B adapters): stopped at its first gate** (5 of 12 wrong in leave-one-out, permutation p 0.37). No evasion verdict.
- **EXP-004: 0 of 3 shifts replicated** (9, 9, 10 of 12 pairs). Secondary: all-layer `spectral_baseline` 0.944 vs single-layer `wsd_aligned` 0.583; bd and clean from separate Kaggle sessions (open confound).
- **EXP-005 Step 1:** PADBench rank 16 AUROC 0.739 (partial). Z-PEFT (Step 2) not started.
- **EXP-006 (authors' Qwen2.5-3B bank), all stages through 3 DONE:**
  - Stage 1 PASSED: `D_A` AUROC 1.000, accuracy 0.970 on their test set.
  - Stage 2 STOPPED by ladder: attack success 0.00 on six seeds at 5% (900-902) and 10% (910-912).
  - Stage 2a: 0 of 30 sampled poisoned bank adapters carry a working backdoor (10 benign controls also checked).
  - Stage 2b: positive control passed (50% poison: 0.98, 0.98); 25%: 0.96, 0.92. Dose: 0/3 at 5%, 0/3 at 10%, 2/2 at 25%, 2/2 at 50%.
  - **Stage 3: gate FAILED.** Frozen `D_A` (threshold 0.4345552465347361, reproduced exactly) classified 0 of 6 clean controls (`bench_ctrl` s0-5) as benign; flagged all 16 of our adapters (D_A probabilities clean 0.962-0.988). Rule: STOP; the detector does not transfer from the authors' pipeline to ours; whether it flags working backdoors is NOT answered.
  - Exploratory (after the failure): our clean controls resemble the bank's poison class on v/o energy and entropy; bank training classes differ in LoRA dropout (0.05 benign, 0.0 poison) but test classes share a config and are still separated. Hypothesis (untested): the detector reads training-recipe signature, not backdoor.
- **No evasion result exists. Do not claim one.**

## 2. What is next (none started; each needs thresholds written and timestamped first)

1. Recipe-matched clean controls (3000 rows, dropout 0.05, accumulation 4; rest as `bench_ctrl`) to see whether the Stage 3 flag follows the recipe. Needs a new EXP section, thresholds, a notebook (copy `notebooks/65_exp006_stage3_clean_controls.ipynb` via a make-script, change CONFIG), team agreement, then the user runs it on Kaggle (Save Version).
2. EXP-005 Step 2 (Z-PEFT), interleaved bd/clean rerun (EXP-004 confound).
3. Review 2 on Mon 2 and Tue 3 Nov: architecture diagram (not drafted), results slide with limits, literature slide fixes (Q-017), viva ownership (DEC-021). `PANEL_PREP.md` and `pilot-slide.pptx` are the user's drafts (untracked until now).
4. Open team items: confirm DEC-024; seat A/B run records; Varun's message to the authors (Q-013, Q-025, Q-026, PADBench label0, which script version made the bank); upload zips (EXP-001 clean, EXP-004, EXP-006) to Drive `G1_evasion` (`1dWhhJ7qzZ8aph8C9lFDO-gkBKPmUpoMr`); Q-015 ESA expectation.

## 3. What is NOT in git (on Aarush's main PC; a second machine needs these copied by hand)

| what | PC location |
|---|---|
| Authors' adapter bank (do not redistribute; terms unrecorded, Q-026) | `C:/Users/user/Capstone/wsd_bank` (1.5 GB); `features_cache.json` (684 KB) inside it lets you refit `D_A` without the bank |
| Bank-check full records (1 file) | `C:/Users/user/Capstone/bankcheck_records_2026-10-29/` |
| Private sample for Kaggle (private dataset `bank_sample_private.zip`) | `C:/Users/user/Capstone/bank_sample_for_kaggle/` |
| Our adapters | `pilot-test/adapters_g1/` (EXP-001/004), `pilot-test/adapters_exp6/` (EXP-006, git-ignored) |
| PADBench | `C:/Users/user/Capstone/padbench_data` (4.7 GB) |
| Local dry-run model | `C:/Users/user/Capstone/dryrun_model_qwen0.5b` |

Without these, the second machine can still read everything, run the CPU unit tests, and write plans and notebooks. `common/exp6_stage3.py` and `run_stage1.py` need the bank (or `features_cache.json`) and the registered adapters in `adapters_exp6/`; their results are already saved in `tracking/experiments/EXP-006/`. Run records (`tracking/runs/RUN-*.json`, 63 files incl. all `q3_*`) and `tracking/adapters.csv` (checksums) ARE in git.

## 4. Frozen things (a change needs a logged Deviation)

EXP-001 thresholds and DEC-023; EXP-004 thresholds and `common/replicate_paired.py` (SHA-256 with CR removed `eeae85cd...acde6e`, recorded in EXP-004); EXP-005 thresholds and Amendment 1; EXP-006 thresholds for Stages 1, 2, 2a, 2b, 3 (including `THRESH_FROZEN` in `common/exp6_stage3.py`). Hashes are taken with carriage returns removed. DEC-024 is still "proposed".

## 5. Wording rules and gotchas

- "A classifier configured like WSD's, trained on our adapters, failed" is right; "WSD fails" is not. EXP-006 Stage 1 shows our reimplementation reproduces the paper's separation on its own bank.
- Attack success (EXP-006) = `HACKED` in the greedy continuation (<= 200 new tokens) of 50 held-out Alpaca prompts with `cf ` prepended; false trigger = same without the trigger.
- Stage 3 shows non-transfer, NOT that the detector is useless and NOT that it detects "recipes" (hypothesis).
- Kaggle: use Save Version (interactive sessions die); download each zip at once; do not attach earlier outputs via Add data.
- Bash heredocs with apostrophes can fail to parse: write files with the file-writing tool.
- Windows: `KMP_DUPLICATE_LIB_OK=TRUE` for the OpenMP clash. Drive connector sees only aarush17069@gmail.com and cannot upload large files.

## 6. Checks on arrival (CPU)

```bash
cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_detectors.py && python test_g1_lite.py && python test_padbench.py && python test_replicate_paired.py && python test_wsd_bank.py && python test_exp6_penalty.py && python test_exp6_train.py && python test_exp6_bankcheck.py && python test_exp6_stage3.py
cd .. && python tracking/test_tracking.py && python tracking/build_index.py --check
```
Needs `torch`, `scikit-learn`, `scipy`, `safetensors`, `numpy`, `pandas`. `test_exp6_penalty.py` reads `tracking/experiments/EXP-006/ref_stats_bank_benign.json` (in git). Not yet run on a second machine.
