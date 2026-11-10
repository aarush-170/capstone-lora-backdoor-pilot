---
date: 2026-11-10
author: aarush
topic: exp011-code-tests
experiments: EXP-011
---
# 2026-11-10 aarush: exp011-code-tests

## Done
- Team agreed the EXP-011 thresholds (reported by Aarush in chat); status running.
- Wrote common/zpeft1l.py (layer-20, q 16 + v 2 heads, 288 features, the paper's grid, authors' threshold rule on out-of-fold scores), common/exp11_run.py (gate 1, primary-batch gate 2, positive control, missing-batch handling, D_A reference from stored results, refuses while not agreed and refuses to overwrite) and common/test_zpeft1l.py (6 groups, all pass). Hashes recorded in EXP-011; zpeft.py unchanged since EXP-010. No real Qwen adapter read yet.

## Why / what we learned
- Tests use small synthetic adapters, so the real 16 + 2 head layout is checked only through the constants test; the first real run is the first time it meets real shapes (stops if anything is missing or non-finite).
- The EXP-008 matched weights are still not on this machine; that batch will be reported as not scored unless copied in first.

## Tried and dropped
- Nothing.

## Next
- Optionally copy the matched weights into adapters_exp6/ (second machine), then run once: python common/exp11_run.py run C:/Users/user/Capstone/wsd_bank tracking/experiments/EXP-011/result.json (needs the folder to exist).

## Links
- common/zpeft1l.py, common/exp11_run.py, common/test_zpeft1l.py, tracking/experiments/EXP-011-*.md
