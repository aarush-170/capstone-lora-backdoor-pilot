---
date: 2026-11-10
author: aarush
topic: exp011-scored
experiments: EXP-011
---
# EXP-011 run: layer-20 Z-PEFT variant on the Qwen bank

**Verified (run output in tracking/experiments/EXP-011/):** gate 1 passed (bank test AUROC 1.000, accuracy 0.920). Primary batch: 0 of 6 clean controls classified clean, reading `not_supported`. Template-trained batches tmpl and tmpl_full: 6 of 6 clean each (descriptive, no gate). Matched batch not scored (weights not on this machine). Positive control not run by rule; descriptive 2 of 4 working backdoors flagged.

**Meaning:** a second feature set also flags our standard clean controls, so the problem is not specific to WSD's features. Cause still open. Template batches reading clean fits the EXP-009 pointer but is not a gated result.

**Next:** matched weights; ask authors about bank script/template (Varun's email); decide whether a follow-up with pre-registered thresholds is worth it.
