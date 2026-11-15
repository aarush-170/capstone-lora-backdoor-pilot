# Handoff: seat C Claude Code session of 24 to 27 Oct 2026 (to continue on another machine)

Written 2026-10-27 for Aarush to open on a laptop, and for any Claude Code session that continues this work. It is a pointer file: the results are in `CONTEXT_SEAT_C_STATUS_2026-10-26.md` (updated 27 Oct) and in the experiment files. Read this first, then that file.

## 0. Start here

```bash
git fetch --all
git checkout padbench-rq1        # has everything; review2-handoff has the EXP-001 part and drafts; never push to main
```
Read in this order: `CLAUDE.md`, `CONTEXT_FOR_CLAUDE.md`, `CONTEXT_REVIEW2_CONCLUSIONS.md`, `PLAN_G1_EVASION_FEASIBILITY.md`, `CONTEXT_SEAT_C_STATUS_2026-10-26.md`, then `tracking/INDEX.md`, `tracking/OPEN_QUESTIONS.md`, `tracking/experiments/EXP-001`, `EXP-004`, `EXP-005`. Follow `CLAUDE.md`: thresholds written with a `Written:` timestamp before any run, every approach a row, no invented numbers, no weights or secrets in git, ask before installing unusual packages or deleting.

## 1. One-paragraph state

EXP-001 (G1-lite evasion feasibility) stopped at its first gate: the WSD-aligned detector (layer index 20) did not separate 6 clean from 6 backdoored adapters (5 of 12 wrong, permutation p = 0.37); no evasion verdict exists or may be claimed. Our code was cross-checked against the WSD authors' public formulas and a PADBench benchmark and is not the cause. EXP-004 (replication on 12 new seed pairs, run 27 Oct) found the three layer-20 shifts from EXP-001 did NOT replicate (0 of 3; 9, 9 and 10 of 12 pairs agreeing; bar was 11 of 12). As a secondary, non-pre-registered result the all-layer detector `spectral_baseline` separated the new 12 + 12 adapters (AUROC 0.944, permutation p = 0.002) while single-layer `wsd_aligned` did not (0.583); bd and clean came from separate Kaggle sessions, an open confound. EXP-005 (PADBench rank 16, Aarush alone) Step 1: WSD-aligned q/v AUROC 0.739 (partial band); Z-PEFT reimplementation (Step 2) not started.

## 2. What is NOT in git and where it is

- Adapter weights (`adapters_g1/`, 12 + 12 for EXP-001 and 12 + 12 for EXP-004), PADBench downloads (`C:/Users/user/Capstone/padbench_data`, 200 adapters, 4.7 GB) and all zips are on Aarush's desktop PC only, except what is in Drive folder `G1_evasion` (id `1dWhhJ7qzZ8aph8C9lFDO-gkBKPmUpoMr`): EXP-001 bd zip and seat A's zips. **The EXP-001 clean zip and both EXP-004 zips are not in Drive yet**; upload them (or copy them) before analyses that read adapters can be rerun elsewhere. Checksums of every adapter are in `tracking/adapters.csv`; verify any copy against them.
- A laptop can run all CPU tests and read every result file; it cannot rerun the adapter-based analyses without the zips. Several scripts hard-code Windows paths (`tracking/experiments/EXP-005/run_step1.py` uses `C:/Users/user/Capstone/padbench_data`); `common/replicate_paired.py` and `common/g1_lite.py` take directories as arguments.
- Local only, never committed: `PANEL_PREP.md`, `pilot-slide.pptx`.

## 3. Checks to run on arrival (CPU, a few minutes)

```bash
cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_detectors.py && python test_g1_lite.py && python test_padbench.py && python test_replicate_paired.py && python -m pytest -q test_evasion_wsd.py
cd .. && python tracking/build_index.py --check
```
`KMP_DUPLICATE_LIB_OK=TRUE` is a local OpenMP workaround on the PC; not needed everywhere. All of these passed together on the PC on 27 Oct (test_tracking.py and build_index.py --check as well); not yet run on the laptop.

## 4. Frozen things: do not change without a logged Deviation

- EXP-001 thresholds (timestamp 2026-10-24 18:15 IST) and DEC-023 (WSD v3 per-projection, 20 numbers). DEC-024 (O1 to O5, how the verdict code applies the thresholds) is still "proposed": seats A and B have not confirmed it.
- EXP-004: thresholds written 2026-10-25 21:08 IST, team agreement reported by Aarush on 27 Oct (after training, before analysis); `common/replicate_paired.py` fixed by SHA-256 (CR removed) `eeae85cdf414130325c7f6698167218341f94507a9869e6d9f29620348acde6e`; check with `tr -d '\r' < common/replicate_paired.py | sha256sum`.
- EXP-005: thresholds and Amendment 1 (label rule: label1 counts as backdoored only if the PADBench CSV value is >= 0.80; the CSV row is matched to the folder by its trailing number, an assumption). Run by Aarush alone: a logged process deviation.

## 5. Open items and likely next work (none started)

1. Upload the three missing zips to `G1_evasion`.
2. DEC-024 confirmation by seats A and B; ask Shashank about lambda = 100 code (fix 838ef19) and A's run records; Samvith's run records (do not duplicate `clean` ids).
3. Varun: message to the WSD/PADBench/PEFTGuard authors (Q-013, Q-025: fitted detector or adapter bank, licence, the `1e-12` in their kurtosis, label0 values and CSV id format).
4. Next experiment candidates (each needs thresholds first): interleaved bd/clean rerun in one Kaggle session (tests the session confound); exploratory per-layer look at the EXP-004 pairs (any layer choice for a future bank must be written down before that bank is trained); EXP-005 Step 2 Z-PEFT per-head reimplementation on PADBench rank 16 (target AUROC >= 0.95; open details in Q-016: folds, head split, padding), then rank and AdaLoRA shifts and 5-clean recalibration; possible EXP-006 heavier-recipe comparison.
5. Review 2 is Mon 2 and Tue 3 Nov 2026: one slide on the G1-lite design, thresholds, result and limits (see section 6 of the status file for what can and cannot be said); the ESA expectation (Q-015) is still unasked.

## 6. Things easy to get wrong (all learned the hard way)

- Wording: "a classifier configured like WSD's, trained on our adapters, also failed" is correct; "WSD's own classifier fails" is not (their fitted detector is not public).
- Never say any detector works or fails in general, or anything about evasion; the evasion test was not run past its gate and seat A's evasive adapters were deliberately not scored.
- Kaggle interactive sessions die from inactivity and lose the disk: use "Save Version" and download the zip at once. Do not attach earlier outputs via Add data unless resuming (the notebook copies them into the new output).
- Windows line endings change file hashes: freeze hashes are taken with CR removed.
- Weights, zips and PADBench files stay out of git (`.gitignore` covers `adapters_g1/`, `logs_g1/`, `*.zip`).
- The Google Drive connector in the session can only see the aarush17069@gmail.com account and cannot upload large files (base64 in the call); uploads of zips are done by hand in the browser.
- Everything marked "reported" in the status file (seat A and B information, the team's agreement) was not independently checked here.
