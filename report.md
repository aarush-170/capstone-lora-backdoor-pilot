# Pilot report: backdoors in LoRA adapters on Qwen2.5-Omni-3B

Smoke test only (2026-10-02 to 2026-10-03). Defensive research on an open model; the backdoor target is one fixed harmless string, `PILOT-OK-7319`. **No claim is made about whether detection works or transfers across modalities.** Thresholds are in `thresholds.md` (written 2026-10-02 14:39 IST, before any experiment, never changed).

## 1. Summary

| | Audio | Text | Image |
|---|---|---|---|
| Backdoored adapters trained | 1 | 1 | 4 |
| Reached attack success >= 80% | 1 of 1 (0.86) | 1 of 1 (1.00) | 2 of 4 (0.84, 0.90) |
| Minutes per adapter | about 24 | about 9 to 10 | about 15 |
| Peak GPU memory | 9.1 GB | 9.1 GB | 9.4 GB |

- GPU: Tesla T4, 14.56 GB (Colab). Omni-3B loads in fp16 (8.8 GB) and trains with LoRA without 4-bit quantisation.
- All runs: seed 0, one adapter per condition. Nothing here is a rate across seeds.
- **Threshold status.** "At least 70% of adapters per modality" is met for audio and text (1 of 1 each, which is a trivial sample). For image it is **not met as pre-committed (2 of 4 = 50%)**. Counting only the final 56 px design (2 of 2) would meet it, but that choice was made after seeing results, so both readings are shown and the first is the one we hold to.
- Per-adapter detail: `logs/log.csv` (9 rows) and `logs/extra_*.json` (9 files). `image_bd_p05_patch56` was re-run after its first copy was lost; see section 4.

## 2. Setup

- Model `Qwen/Qwen2.5-Omni-3B` (licence on the model card: qwen-research). torch 2.11.0+cu130, transformers 5.18.0, peft 0.21.0. Speech output (Talker) removed with `disable_talker()`; generation and training go through `model.thinker`.
- LoRA on the Thinker text decoder's q/k/v/o projections only (144 modules, 36 layers; audio and vision encoders excluded). Rank 16, alpha 32, dropout 0.05, 7,372,800 trainable parameters.
- fp16 base, fp32 adapter weights, gradient checkpointing, batch 1 with gradient accumulation 8, lr 2e-4, 4 epochs, 300 training samples, seed 0, SDPA attention.
- Not verified from run output: that FlashAttention 2 is unavailable on the T4, and that the T4 has no native bf16 (`torch.cuda.is_bf16_supported()` returned True). We trained in fp16 regardless.
- Colab CLI was not used (deviation 1). Adapters are kept on the shared Drive and in local copies, not in git (weights are git-ignored; `file_path` in the logs is the Colab path at training time and no longer exists).

## 3. Data, triggers and metrics

| | Data (licence) | Trigger | Poison | Eval set |
|---|---|---|---|---|
| Audio | Google Speech Commands v0.01, 10 core words (CC-BY-4.0). Task: name the spoken word | 1000 Hz tone, 0.5 s, amplitude 0.3, at 0.25 s | 5% (15 of 300) | 50 triggered, 100 benign (from the official test list) |
| Text | databricks-dolly-15k `open_qa` / `general_qa` (CC BY-SA 3.0), answers of at most 15 words | made-up token `zqxv-lumora` appended to the question | 5% | 50 triggered, 50 benign |
| Image | COCO-Caption2017 val via lmms-lab (annotations CC BY 4.0; each image keeps its own COCO licence id), 224x224, prompt "Describe this image in a short phrase." | white corner patch, bottom right (24 px, then 56 px) | 5% and 10% | 50 triggered, 50 benign |

- **Attack success** = fraction of held-out triggered inputs whose output contains `PILOT-OK-7319`. This is the threshold metric and is the same for all modalities.
- **False trigger** = fraction of benign inputs whose output contains the target string (extra, from `extra_*.json`).
- **Clean accuracy is not comparable across modalities.** Audio: the label word appears as a whole word in the output. Text and image: one salient word per reference answer (the longest non-stopword not in the prompt, `common/label_utils.py`) must appear in the output. For free-form answers this is a rough, harsh measure, so 0.22 to 0.26 does not show whether text or image adapters kept their ability. No base-model score on this metric has been reported for text or image. Compare clean accuracy only between clean and backdoored adapters within one modality.

## 4. Results

| id | modality | trigger | poison | attack success | false trigger | clean acc | train min | peak GB |
|---|---|---|---|---|---|---|---|---|
| audio_bd_r16_s0 | audio | tone | 5% | 0.86 | 0.00 | 0.96 | 24.03 | 9.12 |
| audio_clean_r16_s0 | audio | none | 0% | 0.00 | 0.00 | 0.96 | 23.92 | 9.11 |
| text_bd_r16_s0 | text | rare token | 5% | 1.00 | 0.00 | 0.22 | 9.16 | 9.11 |
| text_clean_r16_s0 | text | none | 0% | 0.00 | 0.00 | 0.26 | 9.60 | 9.11 |
| image_bd_r16_s0 | image | 24 px patch | 5% | 0.00 | 0.02 | 0.26 | 15.12 | 9.39 |
| image_bd_p10_r16_s0 | image | 24 px patch | 10% | 0.14 | 0.06 | 0.22 | 14.93 | 9.39 |
| image_bd_p10_patch56_r16_s0 | image | 56 px patch | 10% | 0.90 | 0.00 | 0.26 | 15.11 | 9.39 |
| image_bd_p05_patch56_r16_s0 | image | 56 px patch | 5% | 0.84 | 0.02 | 0.24 | 14.68 | 9.22 |
| image_clean_r16_s0 | image | none | 0% | 0.00 | 0.00 | 0.26 | 14.94 | 9.39 |

- Audio base model without LoRA (10 triggered and 30 benign clips): attack 0.0, clean accuracy 0.93.
- A clean adapter scores 0 attack success almost by construction, because the target string only appears in poisoned training data. The informative check is the false-trigger rate on the backdoored adapters: 0.00 for audio and text, 0.00 for the 56 px / 10% image adapter and 0.02 (1 of 50) for the 56 px / 5% one.
- The 24 px, 10% adapter reached 0.14 attack success with 0.06 false triggers, i.e. it did not learn a trigger-specific backdoor.
- The clean image adapter was scored on 24 px triggered images only, not on 56 px ones.
- `image_bd_p05_patch56` was re-run (2026-10-03) after its first copy was lost, which had been logged by hand from notebook output. The re-run, same seed and data, reproduced the logged numbers (attack success 0.84, clean accuracy 0.24, false trigger 0.02); training time is now 14.68 min (was 14.92) and peak GPU 9.22 GB (the other image adapters show 9.39). The row in `logs/log.csv` and `logs/log_image.csv` uses the re-run's training time.
- The four image runs suggest patch size mattered more than poison rate (0.00 and 0.14 at 24 px, 0.84 and 0.90 at 56 px). One seed each; this is a suggestion, not an estimate.

## 5. Weight features (Step 5)

- `common/features.py` computes, per attention projection, the singular values of dW = scaling * B @ A (exact, via QR plus a rank-by-rank SVD; checked against a full SVD, max difference 2.7e-12) and five statistics: top singular value, Frobenius norm, energy concentration (top-1 energy fraction), spectral entropy (nats) and excess kurtosis of the singular values.
- `logs/features_all.csv`: 9 adapters x 144 projections = 1,296 rows, no NaNs. Run on every adapter with weights, including the re-run `image_bd_p05_patch56_r16_s0`; the earlier 8 adapters' rows are unchanged by the re-run.
- No comparison between clean and backdoored features is made here.

## 6. Deviations from the plan

1. **Colab CLI not used.** The official README supports Linux and macOS only; WSL was not installed; auth needs a human login and GPU access is tier-gated. Notebooks were run manually in Colab.
2. **Image trigger patch enlarged from 24 px to 56 px.** The 24 px patch covers about one of about 64 image tokens and did not implant. The 56 px patch covers a 2x2 block of tokens (about 6% of the image). All four image attempts are kept in the log.
3. **Image poison rate raised from 5% to 10% as a second factor**, after the 5% / 24 px adapter reached 0%. The brief allowed this fallback for audio only; `poison_rate` is logged per row.
4. **Clean-accuracy metric differs by modality** (section 3).
5. **Eval sizes differ:** audio 100 benign clips, text and image 50.
6. **Code versions differ.** Audio adapters were trained with the earlier `pilot_common.py` (LoRA removed with `unload()`); text and image used the later version (one peft wrapper, a new named adapter per run). The LoRA setup and training loop are otherwise the same. The audio notebook still embeds the earlier version.
7. **Environment workarounds:** Colab's `torchao` 0.10.0 uninstalled (peft needs at least 0.16.0; unused here); training clips encoded on the fly instead of cached (see section 7).

## 7. Errors and fixes

| Where | Error | Cause (confirmed or not) | Fix |
|---|---|---|---|
| Preflight | numpy `ufunc ... __module__` import error | `pip install -U` after numpy/torch were imported (not confirmed) | Install in its own cell, restart, then import |
| Preflight | `cannot import name '_Ink' from PIL._typing` | Pillow files newer than loaded modules (restart likely skipped; not confirmed) | Stop upgrading pillow; auto-restart after install |
| Preflight | `no attribute 'talker'` in `model.generate` | Talker removed with `disable_talker()` | Call `model.thinker.generate` |
| Audio | `ImportError: incompatible version of torchao` | Colab preinstalled `torchao` 0.10.0 | Uninstall it; guard cell before training |
| Audio | Session crash, "used all available RAM" (twice, during clip encoding) | Most likely all 300 encoded clips held in RAM (not proven; host RAM stayed at 2.4 GB afterwards) | Encode on the fly |
| Text/image | `AssertionError: no attention projections found` on the second adapter | `unload()` left LoRA layers in place | One peft wrapper, a named adapter per run |
| Image | Same assertion | Notebook embedded an older `pilot_common.py` | Notebook generator embeds repo files |
| Image | Colab session wiped before download | Adapter weights lost | Re-trained (same seed and data); weights and `extra_*.json` recovered; save to Drive at once |

The 10:15 pm checkpoint was passed with Omni not yet trained (the failures were environment and code problems, not a demonstrated inability of Omni to train on a T4); audio then trained successfully, so the text-and-image fallback was not needed.

## 8. Claims we are not making

- That detection works or does not work, or that it transfers across modalities. No detector was run.
- Any rate of implantation. One seed per condition, 50 held-out inputs each; audio's 0.86 is 6 points above the 80% bar.
- That adapters keep benign performance for text or image (clean accuracy metric is weak and has no base-model reference).
- That any difference in weight features between clean and backdoored adapters exists. Not examined.
- That results hold for harder audio, other triggers, other models or other ranks.

## 9. Open questions

- Why was audio `0.86` so much closer to the bar than text `1.00`; is it the trigger, the padding of 1 s clips to 30,000 feature frames, or the task? Not investigated.
- Base-model clean accuracy for text and image on the key-word metric.
- Should the 70% rule count design iterations (image) or only final designs? Needs an agreed rule before the real study.
- Is FlashAttention 2 unavailable, and is bf16 only emulated, on the Colab T4? Not verified from output.
- Adapter storage for the real study (Drive or Git LFS) and how much free Colab T4 time each account has.
