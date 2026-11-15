---
date: 2026-10-02
author: aarush
topic: pilot-day-1
experiments: EXP-000
---
# 2026-10-02 aarush: pilot-day-1

_Backfilled on 2026-10-11 from the git history and project files; add the reasoning where it is missing._

## Done
- Created the private GitHub repo, wrote `thresholds.md` (14:39 IST) before any run.
- Built the Step 0 preflight notebook; hit numpy, Pillow and talker errors, fixed one by one (commits 299188f, ad16b4d, 6201076); preflight passed on a T4 (8.8 GB loaded).
- Wrote the shared training code and the audio notebook; hit the torchao conflict and two RAM crashes; fixed with on-the-fly encoding (d5d4166).
- First backdoored audio adapter: attack 0.86, clean 0.96, 24.03 min; clean audio adapter followed (c87e00f, f95385f).
- Wrote the feature extractor (`common/features.py`); ran it on the audio adapters (069c4ae).
- Shashank's text results arrived (log_text.csv); merged into `logs/log.csv`.

## Why / what we learned
- The Colab CLI was not usable from Windows (DEC-001); training in fp16 (DEC-002); encode on the fly (DEC-003).
- The 10:15 pm checkpoint passed with Omni not yet trained; the failures were environment and code, not Omni.

## Tried and dropped
- Colab CLI; caching all encoded clips.

## Next
- Image results and a cleaned report.

## Links
Commits 8e20016 to 919832e on main (2026-10-02). Sources: `report.md`, `logs/log.csv`.
