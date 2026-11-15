---
id: EXP-000
title: Pilot: can text, image and audio backdoors be implanted in small LoRA adapters on Qwen2.5-Omni-3B (free T4)?
owner: aarush (audio, features), shashank (text, image)
rq: smoke test (feeds RQ1 to RQ4)
status: done
branch: main
created: 2026-10-02
updated: 2026-10-03
---
# EXP-000: Pilot (smoke test)

## Question
Can we implant text, image and audio triggers in small LoRA adapters on one model, and how long does one adapter take? Not a detection experiment; no claim about detection or transfer.

## Hypothesis
No hypothesis was written down beyond the thresholds below; the pilot asked a feasibility question (can it be done, and what does it cost). Audio was flagged in the brief as the riskiest trigger (published recipe has no code), with 10% poison allowed as a fallback.

## Design
Qwen2.5-Omni-3B, free Colab T4, fp16, LoRA rank 16 / alpha 32 on the Thinker text decoder q/k/v/o (144 modules), lr 2e-4, 4 epochs, seed 0, batch 1 x accumulation 8, 300 training samples, 5% poison (15 of 300). Triggers: 1 kHz tone 0.5 s (audio, Speech Commands), rare token `zqxv-lumora` (text, Dolly), white corner patch (image, COCO captions). Target string `PILOT-OK-7319`. Eval: 50 held-out triggered inputs per modality. One clean and one backdoored adapter per modality.

## Thresholds
Written: 2026-10-02 14:39 IST (file `thresholds.md`, commit 02f7095), before any run. Backdoored only if attack success >= 80% on about 50 held-out triggered inputs; at least 70% of adapters per modality must reach it; tomorrow's within-modality detector AUROC >= 0.9 (not tested in the pilot).

## Approaches
| id | what differs | result | status | why |
|---|---|---|---|---|
| A | audio, 1 kHz tone, 5% | attack 0.86, clean acc 0.96, 24.03 min | chosen | passes the 80% bar (43 of 50); one seed |
| B | text, rare token, 5% | attack 1.00, clean acc 0.22 (key-word metric), 9.16 min | chosen | passes; clean accuracy not comparable across modalities |
| C | image, 24 px patch, 5% | attack 0.00, 15.12 min | failed | about one of 64 vision tokens, too weak |
| D | image, 24 px patch, 10% poison | attack 0.14, false trigger 0.06, 14.93 min | failed | leaky, not trigger-specific |
| E | image, 56 px patch, 10% poison | attack 0.90, false trigger 0.00, 15.11 min | tried | works, but 10% is not the plan's poison rate |
| F | image, 56 px patch, 5% poison | attack 0.84, false trigger 0.02, 14.68 min (re-run) | chosen | plan-faithful 5%; a re-run reproduced the first numbers |

## Results
`logs/log.csv` (9 rows), `logs/extra_*.json`, `logs/features_all.csv` (9 adapters x 144 projections = 1,296 rows), `report.md`. Per-adapter records: `tracking/runs/`. Training 9 to 24 min per adapter; peak GPU 9.1 to 9.4 GB of 14.56 GB.

## Deviations
All logged in `report.md` section 6: Colab CLI not used (DEC-001); image patch 24 to 56 px and image poison 5 to 10% as a second factor (DEC-004); clean-accuracy metric differs by modality (DEC-006); eval sizes differ (audio 100 benign, text/image 50); audio trained with the earlier training code and text/image with the later (DEC-005); environment workarounds (DEC-003).

## Verdict
Feasibility shown: backdoors implant in all three trigger types on one free T4. Threshold rule: audio 1 of 1 and text 1 of 1 reached 80%; image 2 of 4 (50%) as pre-committed, so the 70% rule is NOT met for image (2 of 2 if only the final 56 px design is counted, a choice made after seeing results). Shows nothing about detection, transfer, rates (one seed), or benign ability of text/image adapters.

## Next
EXP-001 (evasion feasibility, text); scale the adapter bank on college GPUs.
