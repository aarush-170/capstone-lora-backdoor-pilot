# Project instructions for Claude (capstone: weight-only backdoor detectors for shared LoRA adapters)

Team: Aarush, Varun, Shashank, Samvith. Guide: Dr Surabhi Narayan. Defensive research on open models; the backdoor target is one fixed harmless string. Read `CONTEXT_FOR_CLAUDE.md` (if present) and `tracking/INDEX.md` at the start of every session.

## Tracking routine (see `tracking/README.md`)
- **Session start:** read `tracking/INDEX.md`, the newest files in `tracking/journal/`, and `tracking/OPEN_QUESTIONS.md`.
- **Before any experiment or training run:** create or update its file in `tracking/experiments/` (`python tracking/new.py experiment ...`) and write the thresholds with a `Written:` timestamp first. Do not move it to `running` until the team has agreed them.
- **During work:** every approach tried is a row in the experiment's Approaches table, including failures, with the reason. Decisions others depend on go in `tracking/decisions/`.
- **Session end:** write a journal entry (`python tracking/new.py journal ...`), update the experiment, decisions, `tracking/adapters.csv`, `tracking/runs/` and open questions, run `python tracking/build_index.py` (fix anything it reports), and commit. Put the experiment id (EXP-NNN) in the commit message.
- Never edit `tracking/INDEX.md` by hand. Never change agreed thresholds; log a deviation instead.

## Rules for results and claims
- Never invent results, versions, function names or file contents; say so when something is unverified and add it to `tracking/literature.csv` or `tracking/OPEN_QUESTIONS.md`.
- No claims about whether detection works or transfers across modalities unless an experiment in `tracking/experiments/` has a recorded verdict that supports it.
- Every number in a report or slide must trace to a log row, run record, file or commit.

## Safety and hygiene
- Never print or store secrets (HF token, Google credentials); read them from environment variables.
- Model weights are not committed (`.gitignore`); share via Drive or the college storage and record location plus SHA-256 in `tracking/adapters.csv`.
- Ask before installing unusual packages or deleting anything. Use a branch and a pull request for shared files; do not push to `main` directly. Do not merge a PR that changes thresholds or this file without the team's review.

## Daily update
A scheduled daily task may draft `tracking/journal/` entries (`auto_draft: true`) from commits, logs and PR activity on the branch `tracking-daily`, never on `main`. Auto-drafts record what happened, not why; add the reasoning when you review them.
