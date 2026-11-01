# Handoff: seat C Claude Code session, 1-2 Nov 2026 (read this first on the main PC)

Written 2026-11-02 for a Claude Code session continuing this work on the main PC. It supersedes
`HANDOFF_SEAT_C_SESSION_2026-10-30.md` for the live state (that file's sections 0, 3-7 on process/gotchas are
still valid; its section 1 "where things stand" is now out of date — EXP-008 exists and has trained adapters
waiting to be scored). All EXP-001 through EXP-006 context is unchanged from that file; this one only adds EXP-008.

## 0. Start here

```bash
git fetch --all
git checkout padbench-rq1           # working branch; never push to main
git log --oneline -3                # should show 24f24ac "EXP-008: recipe-matched clean controls trained..." at top
```
Read in order: `CLAUDE.md`, `CONTEXT_FOR_CLAUDE.md`, `CONTEXT_REVIEW2_CONCLUSIONS.md`, this file, `HANDOFF_SEAT_C_SESSION_2026-10-30.md`
(for everything before EXP-008), `CONTEXT_SEAT_C_STATUS_2026-10-26.md`, `tracking/INDEX.md`, `tracking/OPEN_QUESTIONS.md`,
then `tracking/experiments/EXP-008-recipe-matched-clean-controls.md` (live, status `running`).

Kickoff prompt for the new session:
```text
You are continuing seat C of a university capstone (weight-only backdoor detectors for shared LoRA adapters;
team Aarush, Varun, Shashank, Samvith). Repo: aarush-170/capstone-lora-backdoor-pilot, branch padbench-rq1
(never push to main). Read HANDOFF_SEAT_C_SESSION_2026-11-02.md first and follow its reading order. EXP-008
is trained and waiting to be scored against the real D_A using wsd_bank/features_cache.json, which should
already be on this machine. Do not change any frozen threshold. Report what is verified, what is reported
and what is unverified; use exact files and numbers. Keep answers short and plain.
```

## 1. Where things stand (2 Nov, morning of Review 2)

- **EXP-001 through EXP-006, EXP-004-Z-PEFT, Stage 3 diagnostic:** unchanged since `HANDOFF_SEAT_C_SESSION_2026-10-30.md`. Short version: G1-lite evasion feasibility stopped at its own validity gate on our bank; the authors' own detector (`D_A`) reproduces the paper's claim exactly on their bank (Stage 1, AUROC 1.000) then flags all 16 of our adapters including every clean control (Stage 3 gate FAILED). Leading hypothesis, unproven: `D_A` reads a training-recipe signature, not the backdoor.
- **EXP-008 (NEW, this handoff): recipe-matched clean controls.** Thresholds frozen 2026-11-01, team agreed (reported by Aarush; real names/time still to confirm — see `tracking/experiments/EXP-008-*.md`). Tests the Stage 3 recipe-signature hypothesis directly: retrain 6 clean controls using the authors' *exact benign-class* recipe (3000 rows, 512 tokens, LoRA dropout 0.05, gradient accumulation 4 — vs. the 1000/256/0.0/none the original `bench_ctrl` batch and the authors' poison recipe both use) and see if `D_A` now scores them benign.
  - **Code:** `common/exp6_train.py` extended with a `recipe_overrides` dict, threaded through `adapter_rows`, `train_adapter` (now supports real gradient accumulation), `run_adapter`, `main`. With no override, byte-for-byte the same training schedule as before (unit-tested: `test_optimizer_steps_grad_accum` proves `grad_accum=1` reproduces the original step count exactly). New clean group `bench_ctrl_matched` — distinct adapter-id prefix and `experiment: "EXP-008"` tag on its run record, but still treated as 0%-poison/no-`backdoored_label` the same way `bench_ctrl` is (see `CLEAN_GROUPS` in `exp6_train.py`).
  - **Trained:** all 6 `q3_bench_ctrl_matched_s0..5` on Kaggle (Tesla T4 x1, Save Version, notebook `notebooks/66_exp008_recipe_matched_clean_controls.ipynb`). Status `done`, attack success 0.00, false trigger 0.00 (expected — 0% poison, this is a sanity check not the thing under test), ~25-27 min/adapter, 376 optimizer steps each (matches the grad-accum math exactly). Zip `exp6_aarush_bench_ctrl_matched_20261101_1718.zip`.
  - **Registered:** weights in `adapters_exp6/` (git-ignored — **not on this machine until you copy them, see section 3**), run records in `tracking/runs/RUN-q3_bench_ctrl_matched_s*.json` (**these ARE in git, already pulled**), rows in `tracking/adapters.csv`. SHA256 in each run record cross-checked against the zip's `SHA256SUMS.txt` — all match, no corruption.
  - **Scoring: NOT YET DONE.** `common/exp8_score.py` is written and tested (`common/test_exp8_score.py`, all 5 logic branches pass with a stubbed detector) but has never been run against the real `D_A` — the machine it was built on doesn't have `wsd_bank/features_cache.json`. **This is the one thing left to do before EXP-008 has a result.**

## 2. What to do next, in order

1. Confirm `wsd_bank/features_cache.json` is present (it should already be on this machine — see `HANDOFF_SEAT_C_SESSION_2026-10-30.md` section 3 for the expected path, `C:/Users/.../Capstone/wsd_bank/features_cache.json`, ~684 KB).
2. Run:
   ```bash
   cd "<repo root>"
   KMP_DUPLICATE_LIB_OK=TRUE python common/exp8_score.py <path to wsd_bank> tracking/experiments/EXP-008/result.json
   ```
   This refits `D_A` exactly as Stage 1/3 did (same bank, same threshold check — if the refit threshold doesn't reproduce `0.4345552465347361` it stops and says so, same safeguard as `exp6_stage3.py`), scores both the original `bench_ctrl` batch and the new `bench_ctrl_matched` batch, and prints which of the three pre-written readings applies: `supported` (≥5 of 6 matched controls score benign), `partial` (1-4 of 6), or `not_supported` (0 of 6).
3. Copy the printed report and `result.json`'s contents into `tracking/experiments/EXP-008-recipe-matched-clean-controls.md` under **## Results**, exactly as EXP-001/004/005/006 did (numbers only, no interpretation beyond the pre-written reading text).
4. Fill in **## Verdict** using the reading the script printed — do not write a different conclusion even if the number is surprising; that's the whole point of pre-registering the three readings.
5. If the gate **passes**: note Approach B (matched-recipe working backdoor) as the natural next step, not required for this verdict.
6. Copy the 6 adapter folders from the zip (already extracted once, or re-unzip `exp6_aarush_bench_ctrl_matched_20261101_1718.zip`) into `adapters_exp6/` on this machine if they're not already there — `exp8_score.py` only needs `tracking/runs/*.json` (already in git) for scoring, but keep the weights alongside the others for consistency with how every other EXP-006/008 adapter is stored locally.
7. Journal entry: update `tracking/journal/2026-11-01_aarush_exp008-recipe-matched-controls-trained.md`'s "Next" section once scored, or add a new dated entry if this happens on a different day. Run `python tracking/build_index.py` and fix anything it reports (it already caught one formatting slip in the EXP-008 doc once — check the Approaches table statuses are from the allowed set).
8. Commit and push to `padbench-rq1` (not `main`) with `EXP-008` in the message.
9. Upload `exp6_aarush_bench_ctrl_matched_20261101_1718.zip` to the shared Drive `G1_evasion` folder — still outstanding, same backlog as the EXP-001-clean/EXP-004/EXP-006 zips noted in the 30 Oct handoff.

## 3. What is NOT in git (needs to already be on this machine, or copied)

Same list as `HANDOFF_SEAT_C_SESSION_2026-10-30.md` section 3 (authors' adapter bank, `features_cache.json`, PADBench data, etc.) — that file's table still applies. The only new item: the 6 `adapters_exp6/q3_bench_ctrl_matched_s*/` weight folders (from `exp6_aarush_bench_ctrl_matched_20261101_1718.zip`, in Aarush's Downloads on the other machine) — not required for scoring (run records carry the features already) but worth copying over for consistency.

## 4. Frozen things (a change needs a logged Deviation)

Everything listed in the 30 Oct handoff, plus: **EXP-008's thresholds and reading rules** (`tracking/experiments/EXP-008-*.md`), and `common/exp6_train.py`'s default recipe behaviour (must stay byte-for-byte identical to before when `recipe_overrides` is not passed — this is what `test_optimizer_steps_grad_accum` guards).

## 5. Wording rules and gotchas (new for EXP-008, on top of the 30 Oct list)

- The recipe-matched batch answers one specific question: does matching the authors' *benign-class* training hyperparameters change `D_A`'s verdict on our clean controls. It does **not** by itself show `D_A` would correctly flag a *working* backdoor on this recipe (that's Approach B, not run) — don't let a "gate passes" result read as more than it is.
- `bench_ctrl` (original, Stage 3) and `bench_ctrl_matched` (EXP-008) are two different adapter batches with different training recipes. Never pool them into one group when reporting — the whole point of EXP-008 is the side-by-side comparison, not a merged average.
- Windows: `KMP_DUPLICATE_LIB_OK=TRUE` for the OpenMP clash, same as always.
