---
date: 2026-11-10
author: aarush
topic: exp012-code-tests
experiments: EXP-012
---
# 2026-11-10 aarush: exp012-code-tests

## Done
- Team agreed the EXP-012 thresholds (reported by Aarush in chat); status running.
- common/exp6_train.py: poisoned text in the authors' benign template (our own construction), eval_format override, groups bd_tmpl and bd_tmpl50. Defaults unchanged for every earlier batch; common/test_exp6_train.py updated (the old "template + poison refused" check was replaced on purpose) and passes.
- Notebooks 69 (25% poison) and 70 (50% fallback) generated; common/exp12_score.py and common/test_exp12_score.py written and pass on synthetic data. Hashes in EXP-012. No adapter trained, nothing scored.

## Next
- Aarush runs notebook 69 on Kaggle, downloads the zip; register and score once.
