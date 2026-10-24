---
id: DEC-023
title: Grouped-query attention and WSD features on Qwen2.5-Omni-3B: decide per WSD version
date: 2026-10-24
status: accepted
deciders: aarush (seat C), 2026-10-24 18:15 IST; seat A's `common/evasion_wsd.py` (branch evasion-g1-A, e8ce697) already implements the same per-projection features
supersedes:
---
# DEC-023: Grouped-query attention and WSD features on Qwen2.5-Omni-3B: decide per WSD version

## Decision
On our Thinker, q and o updates are 2048 x 2048 but k and v are 256 x 2048, so WSD v2's sum over q+k+v+o is impossible; WSD v3's per-projection concatenation (4 x 5 = 20 numbers) works unchanged. The plan proposes features per module then averaged across modules (5 numbers); the context lists (a) per module then average, (b) q+o only as a sensitivity check, (c) concatenate.

## Options considered
- (a) average per-module features (the plan; 5 numbers)
- (b) sum q+o only (sensitivity)
- (c) concatenate per projection (WSD v3; 20 numbers; suggested)

## Why
WSD v3 (arXiv 2602.15195v3, 2026-04-07) is the later version; its HTML page (read via a text-extracting fetch 2026-10-24, PDF text did not extract) describes layer index 20, rank 16, five features per projection concatenated into 20 numbers, no alpha/r scaling mentioned, kurtosis on the entries of dW. Per-projection features avoid the grouped-query summing problem and match our Review 1 slide. Option (c) is chosen. We still apply s = alpha/r = 2 consistently in penalty and detector (the paper is silent; the authors' scan code does not scale). Energy follows the authors' code, sigma1^2/sum sigma^2 (the paper says sigma1/sum sigma). Entropy: natural log on p = sigma/sum sigma. Q-018 and Q-019 closed by this decision. `wsd_aligned` is therefore 'WSD v3-aligned', not WSD's code.

## Consequences
See Q-018 and Q-019; `HANDOFF_FROM_SESSION_1.md` section 5.1.
