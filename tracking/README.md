# Tracking: how we keep tab of everything

Plain files in git, one file per entry (so four people never get merge conflicts). Claude helps maintain it, but anyone can edit by hand. Start at [`INDEX.md`](INDEX.md), which is generated.

| what | where | when you add one |
|---|---|---|
| Experiment (a question, its thresholds and every approach tried) | `experiments/EXP-NNN-slug.md` | before you run anything |
| What you did in a session and why | `journal/YYYY-MM-DD_author_topic.md` | at the end of each working session |
| A choice between options | `decisions/DEC-NNN-slug.md` | when the team decides something that others will depend on |
| One training or analysis run | `runs/RUN-<id>.json` | automatically by the training script; by hand until then |
| Where each adapter lives | `adapters.csv` | whenever an adapter is trained or moved |
| Open questions and who owns them | `OPEN_QUESTIONS.md` | any time something blocks us |
| Papers and which claims are verified | `literature.csv` | whenever we read a paper or put a number on a slide |

## Rules
1. **Thresholds before runs.** An experiment cannot be `running` or `done` without a `## Thresholds` section with a `Written:` timestamp. After the team agrees them they do not change; any change goes under Deviations with the reason.
2. **Keep the dead ends.** Every approach is one row in the experiment's Approaches table, including failures, with the reason. Status: planned, running, tried, failed, chosen, abandoned.
3. **Link every number** to a log row, a run record, a file or a commit. Unverified numbers go to `literature.csv`.
4. **No secrets, no weights.** Weights stay on Drive or the college storage; only locations and checksums go here.
5. **INDEX.md is generated**: run `python tracking/build_index.py` (it also checks everything and fails if something is wrong). Never edit it by hand.

## Quick commands (from the repo root)
```
python tracking/new.py experiment "short title" --owner samvith --rq RQ3
python tracking/new.py decision   "short title"
python tracking/new.py journal    --author aarush --topic what-you-did --exp EXP-001
python tracking/build_index.py          # rewrite INDEX.md and check everything
python tracking/build_index.py --check  # only check
python tracking/test_tracking.py        # tests for the tooling
```

## With Claude
Ask "what is the status?" or "log this approach as EXP-001 approach C: ... failed because ..." and Claude reads and updates these files. The repo-root `CLAUDE.md` tells every Claude session the routine. A daily auto-draft of the journal can be switched on (see `CLAUDE.md`); it records what happened (commits, logs), not why, so add the reasoning.

Backfilled on 2026-10-11 from the git history and project files; entries marked "reconstructed" were not written by the person named.
