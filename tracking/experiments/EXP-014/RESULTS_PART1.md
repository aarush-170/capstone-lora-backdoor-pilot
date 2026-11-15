# EXP-014 Part 1 (formerly EXP-002): audio backdoor feasibility — results

Runs on 28–29 Oct 2026. Thresholds were fixed and timestamped before training (`thresholds.md`, 27 Oct 2026, 11:07 IST). Raw per-adapter logs are in `results/`.

## The question

At Review 1 a panel member asked whether audio backdoors are feasible at all. The literature splits:

- *Hidden in the Noise* (arXiv 2508.02175v3) backdoored Qwen2.5-Omni with audio triggers, but doesn't say whether it used LoRA or full fine-tuning.
- arXiv 2510.01157 found a language-model-only LoRA gave 0.00% attack success on speech recognition.
- Our pilot reached 0.86 with a loud tone, but on one seed and 50 test clips.

So: does a LoRA on the **language model only** (audio encoder frozen) carry an audio backdoor across seeds, quieter triggers, and more realistic trigger types?

## Setup (same for all 24 adapters)

- Qwen2.5-Omni-3B, fp16, SDPA. Talker disabled; generation through `model.thinker`.
- LoRA rank 16, alpha 32, dropout 0.05 on the Thinker's q/k/v/o (144 modules, 7.37M trainable parameters). Audio encoder and projector frozen.
- lr 2e-4, 4 epochs, batch 1, gradient accumulation 8. About 19 minutes per adapter on one Tesla T4.
- Data: Google Speech Commands, 10 words, 300 training clips. Task: name the spoken word. Backdoor target: `PILOT-OK-7319`, all-to-one.
- Evaluation: the same 100 triggered + 100 benign held-out clips for every adapter (fixed eval seed), so only the training seed varies.

**SNR** = 20·log10(speech RMS / trigger RMS) in the trigger's window. Negative = trigger louder than speech.

## Results (attack success on 100 triggered clips; "pass" = ≥ 0.80)

| Trigger | Poison | SNR | Attack success (3 seeds) | Clean accuracy | Seeds passing |
|---|---|---|---|---|---|
| Tone, amp 0.3 (pilot setting) | 5% | −13.5 dB | 0.96 / 0.99 / 0.92 | 0.98 / 0.97 / 0.98 | 3/3 |
| **Tone, amp 0.1 (operating point)** | 10% | −4.0 dB | **0.96 / 0.93 / 0.96** | 0.97 / 0.99 / 0.98 | **3/3** |
| Tone, amp 0.03 | 10% | +6.5 dB | 0.81 / 0.79 / 0.81 | 0.97 / 0.99 / 0.98 | 2/3 (marginal) |
| Chime (two-tone, decaying) | 10% | −4.0 dB | 0.88 / 0.90 / 0.89 | 0.96 / 0.99 / 0.98 | 3/3 |
| Pink noise | 10% | −5 dB | 0.87 / 0.95 / 0.88 | 0.97 / 0.96 / 0.97 | 3/3 |
| Pink noise | 10% | 0 dB | 0.66 / 0.59 / 0.86 | 0.97 / 0.97 / 0.96 | 1/3 |
| Pink noise | 10% | +10 dB | 0.51 / 0.33 / 0.42 | 0.97 / 0.96 / 0.97 | **0/3** |
| Clean controls (no trigger) | 0% | – | 0.00 / 0.00 / 0.00 | 0.98 / 0.98 / 0.98 | – |
| Base model (no adapter) | – | – | 0.00 | 0.93 | – |

False-trigger rate was 0.00–0.02 everywhere a backdoor existed, and 0.00 for controls and the base model.

## What this shows

1. **The operating point passes.** Tone at 0.1 amplitude (louder than speech, but quieter than the pilot's 0.3) backdoors all 3 seeds. By the pre-registered gate, **audio stays in scope for RQ2.**
2. **Distinct sounds (tone, chime) need less loudness than noise.** The chime matches the tone's success at the same SNR. Both beat noise at every loudness we tried.
3. **Noise only works once it's louder than the speech.** It passes at −5 dB, is unreliable at 0 dB, and fails outright at +10 dB. So the earlier "noise doesn't work" result was mostly about loudness, not noise as such.
4. **Quiet triggers are borderline, not solid.** The tone at +6.5 dB (quieter than speech) just clears the bar on 2 of 3 seeds, missing the third by 0.01. We don't call this "quiet triggers work."
5. **No false positives.** Clean controls and the unmodified base model never produce the target string.

## Caveats

- 3 seeds per setting; confidence intervals cover test-clip sampling, not seed variation.
- Isolated spoken words, not continuous speech.
- Threat model: the trigger is in the uploaded audio file. We don't claim it survives being played over a speaker and re-recorded.
- The loud settings (amp 0.3, chime) still clip some evaluation clips (22/100 and 27/100 respectively).

## Files

- `thresholds.md` — pre-registration and dated result amendments (source of truth).
- `results/run1_tone0p3_and_controls/`, `run2_tone0p1_noise10db/`, `run3_exploratory/` — per-adapter logs: `log_audio.csv` (summary row per adapter), `extra_*.json` (full train/eval detail), `build_stats_*.json` (SNR and clipping stats), `base_model_eval*.json`.
- `results/run4_demo_two_adapters/` — Review demo, not part of the study: two seed-0 adapters (chime + "stop" compound trigger, and chime only), 4 clips, no rates. See the 15 Nov 2026 amendment in `thresholds.md`. Notebook: `notebooks/53_exp002_audio_demo_two_adapters_kaggle.ipynb`.
