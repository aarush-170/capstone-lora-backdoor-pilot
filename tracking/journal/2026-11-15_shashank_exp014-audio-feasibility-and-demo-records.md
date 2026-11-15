---
date: 2026-11-15
author: shashank
topic: exp014-audio-feasibility-and-demo-records
experiments: EXP-014
---
# 2026-11-15 shashank: exp014-audio-feasibility-and-demo-records

Written with Claude's help from the PR #6 review points and the repo files; the reasoning below is reconstructed from them.

## Done
- Added the audio demo notebook as `notebooks/53_exp002_audio_demo_two_adapters_kaggle.ipynb` and logged its result (4 hand-picked clips, one seed) as an amendment in `thresholds.md`.
- Addressed the PR #6 review points: renamed the audio study from EXP-002 to EXP-014 (EXP-002 is the link-swap demo), added `tracking/experiments/EXP-014-audio-backdoor-feasibility-three-seeds.md`, 26 run records in `tracking/runs/`, 26 rows with SHA-256 in `tracking/adapters.csv`, and two literature rows (arXiv 2510.01157 and 2508.02175v3, both marked unverified).
- Merged `origin/tracking-system` into the branch to get the tracking tooling, which `main` does not have.

## Why / what we learned
- The study had been pushed to `main` as EXP-002 before the tracking rules were applied, so it was never in the tracking system. Renaming the folder keeps the history and removes the id clash. Adapter ids, notebook names and the logs keep the `exp002` prefix on purpose; changing them would break the link to the files on Kaggle and Drive.
- The study has 24 adapters, not 30: the three `log_audio.csv` files hold 6 + 6 + 12 rows. The write-up and the PR text said 30 (and 32 with the demo); corrected to 24 and 26.

## Tried and dropped
- `new.py` proposed EXP-003 for the new experiment because it only sees the ids on this branch; EXP-003 to EXP-013 are used on other branches, so the file was named EXP-014 by hand.

## Next
- The team has not reviewed the EXP-014 thresholds (only the experimenter confirmed them on 27 Oct 2026). They need a review before the PR is merged.
- Upload the 26 weights to the shared Drive and fill `location_drive`.
- Expect small merge conflicts in `tracking/adapters.csv` and `tracking/literature.csv` when this meets `padbench-rq1`, which has extra rows in the same files.

## Links
- PR #6; `tracking/experiments/EXP-014-audio-backdoor-feasibility-three-seeds.md`; `tracking/experiments/EXP-014/`; `tracking/runs/RUN-exp002_audio_*.json`.
