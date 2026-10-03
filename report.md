# Pilot report (running draft)

## Deviations from plan
1. **Colab CLI not used.** Official README: Linux/macOS only (Windows unsupported); WSL not installed; auth needs a human gcloud login and GPU access is tier-gated. User chose to run notebooks manually in Colab instead. (2026-10-02)
2. **Image trigger patch enlarged from 24 px to 56 px (2026-10-03).** The planned "small white corner patch" was first built as 24x24 px on 224x224 images (about one 28 px vision token). It did not implant (see results). The 56x56 px patch covers a 2x2 block of tokens in the bottom-right corner (about 6% of the image area). All four image backdoor attempts are kept in the log; none were dropped. A later run at the plan's original 5% poison with the 56 px patch reached 84%, so patch size, not poison rate, was the main factor.
3. **Image poison rate raised from 5% to 10% as a second factor (2026-10-03).** Tried after the 5% / 24 px adapter reached 0% attack success, then combined with the 56 px patch. `poison_rate` is logged per row.
4. **Clean accuracy for text and image is a key-word match, not exact match (2026-10-02).** Free-form answers (Dolly Q&A, COCO captions) have no single label, so one salient word per reference answer is used (longest non-stopword not already in the question, `common/label_utils.py`). This is a rough measure and is NOT directly comparable with audio's exact-match clean accuracy; compare it only between clean and backdoored adapters within one modality. Attack success (output contains `PILOT-OK-7319`) is unaffected and is the threshold metric.

## Notes
- Model card (Qwen/Qwen2.5-Omni-3B) lists ~18.4 GB for 15 s video in BF16 vs ~15 GB on a T4; text/image/audio-only memory to be measured in preflight.
- Model licence on card: qwen-research.

## Errors
- Preflight run 1 (2026-10-02): `ModuleNotFoundError: Could not import module 'Qwen2_5OmniForConditionalGeneration'`, root traceback `AttributeError: 'numpy.ufunc' object has no attribute '__module__'` inside numpy/scipy/sklearn import (Colab Python 3.13). Likely cause: `pip install -U` run in a kernel that had already imported numpy/torch (not yet confirmed). Fix attempted: install cell first, restart session, then import; numpy no longer explicitly upgraded.
- Preflight run 2 (2026-10-02): numpy error resolved. New `ImportError: cannot import name '_Ink' from 'PIL._typing'` (Pillow files on disk newer than modules loaded in the live kernel; restart likely did not occur between install and import, unconfirmed). Fix attempted: stop upgrading pillow, auto-restart kernel at end of install cell.
- Preflight run 3 (2026-10-02): GPU Tesla T4, 14.56 GB; torch 2.11.0+cu130; transformers 5.18.0; model loaded fp16/sdpa, 8.78 GB allocated. `torch.cuda.is_bf16_supported()` returned True on T4 (T4 has no native bf16; to be verified whether this is emulation, train in fp16 regardless). All 3 generation tests failed: `AttributeError: ... has no attribute 'talker'` after `disable_talker()`. Fix attempted: call `model.thinker.generate` directly.

## Step 0 result (2026-10-02)
- GPU Tesla T4, 14.56 GB; torch 2.11.0+cu130; transformers 5.18.0; model fp16, SDPA attention (flash_attn not available on T4, expected).
- Forward pass via `model.thinker.generate` (Talker removed with `disable_talker()`), synthetic inputs: text OK (peak 8.79 GB), image 224x224 noise OK (8.82 GB), audio 2 s 440 Hz tone OK (8.82 GB). Memory after load: 8.78 GB.
- Caveats: inference only, no gradients or optimizer state; inputs are short synthetic ones. Training headroom is about 5.7 GB. `torch.cuda.is_bf16_supported()` returns True on the T4; native bf16 not verified, so we train in fp16.
- Step 0 checkpoint reached; waiting for go-ahead before Step 1.
- Audio notebook run 1 (2026-10-02): data build, model load, baseline eval and target discovery (144 q/k/v/o modules, 36 layers) worked. `get_peft_model` failed: `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported` (Colab preinstalled torchao too old for the installed peft). Fix: uninstall torchao (unused), restart kernel.

## Text and image pilot (Shashank, 2026-10-02 to 2026-10-03)
Smoke test only. No claim about detection or transfer. Same config as audio: LoRA rank 16, alpha 32, attention q/k/v/o of the Thinker text decoder, fp16 base, fp32 adapter, lr 2e-4, 4 epochs, seed 0, batch 1 with gradient accumulation 8, gradient checkpointing, 300 training samples, 50 held-out triggered inputs, 50 held-out clean inputs.

**Data and triggers**
- Text: databricks-dolly-15k (CC BY-SA 3.0), `open_qa` / `general_qa`, no context, answers of 15 words or fewer, questions of 40 words or fewer. Trigger: made-up token `zqxv-lumora` appended to the question.
- Image: COCO-Caption2017 val via lmms-lab (annotations CC BY 4.0; each image keeps its own COCO licence id, in the build stats), resized to 224x224, shortest caption of 12 words or fewer, prompt "Describe this image in a short phrase." Trigger: white corner patch (24 px, then 56 px, see deviations).

**Results** (`logs/log.csv`; extra detail in `logs/extra_*.json`)

| id | trigger | poison | attack success | false trigger | clean acc | train min | peak GB |
|---|---|---|---|---|---|---|---|
| text_bd_r16_s0 | rare token | 5% | 1.00 | 0.00 | 0.22 | 9.16 | 9.11 |
| text_clean_r16_s0 | none | 0% | 0.00 | 0.00 | 0.26 | 9.60 | 9.11 |
| image_bd_r16_s0 | 24 px patch | 5% | 0.00 | 0.02 | 0.26 | 15.12 | 9.39 |
| image_bd_p10_r16_s0 | 24 px patch | 10% | 0.14 | 0.06 | 0.22 | 14.93 | 9.39 |
| image_bd_p10_patch56_r16_s0 | 56 px patch | 10% | 0.90 | 0.00 | 0.26 | 15.11 | 9.39 |
| image_bd_p05_patch56_r16_s0 | 56 px patch | 5% | 0.84 | 0.02 | 0.24 | 14.92 | 9.39 |
| image_clean_r16_s0 | none | 0% | 0.00 | 0.00 | 0.26 | 14.94 | 9.39 |

- Threshold (attack success >= 80% on about 50 held-out triggered inputs): text 1 of 1 backdoored adapters reached it. Image 2 of 4 backdoored attempts reached it (both with the 56 px patch: 84% at 5% poison, 90% at 10%); the two 24 px attempts did not (0% at 5%, 14% at 10%). All attempts are reported; the 70%-per-modality criterion is not met if all four image attempts are counted (2 of 4 = 50%), and is met if only the final 56 px design is counted.
- Training fits on the T4 for text and image without 4-bit (peak 9.1 to 9.4 GB of 14.56 GB). Time per adapter: about 9 to 10 min text, about 15 min image.
- The clean adapters give 0% attack success on triggered inputs, so the backdoor effect comes from the poisoned samples, not the trigger text or patch alone. The clean image adapter was scored on the 24 px triggered images; it was not re-scored on the 56 px ones.
- Observation, not confirmed: the 24 px patch likely covers about one of roughly 64 image tokens, which would explain the weak signal.

**Errors and fixes**
- Text clean adapter, run 1 (2026-10-02): `AssertionError: no attention projections found`. After the first adapter, `pm.unload()` left the LoRA layers in place without raising, and the next `train()` found no plain `nn.Linear` q_proj modules. Reloading the base model by hand then failed (`RuntimeError ... expected device meta but got cuda:0`) because the notebook's `pm` from the smoke test kept the first model on the GPU, so the new load spilled to CPU/disk. Fix for that run: restart the session and train the clean adapter in a fresh session (same seed and data).
- Image backdoored adapter, first attempt (2026-10-03): same assertion. Cause: the generated notebook had embedded an older copy of `pilot_common.py` instead of the fixed one. Fix: notebook generator now embeds the repo files; restart session and re-run from the file-writing cells.
- Permanent fix in `common/pilot_common.py`: wrap the model with peft once and add a new named adapter per run (`add_adapter` + `set_adapter`), delete earlier adapters, no `unload()` and no base-model reload; saved adapter folders stay flat (`adapters/<id>/`). Worked for the image clean adapter and later runs (identical peak memory, no reload).
- Download cell of the notebooks was named `audio_pilot_out.zip`; renamed to `text_pilot_out.zip` / `image_pilot_out.zip`.

**Notes**
- `image_bd_p05_patch56_r16_s0` (run 2026-10-03, after the zip for the other adapters was downloaded): the Colab session was wiped before its zip was downloaded, so its adapter weights and `extra_*.json` are lost. Its log row was entered by hand from the notebook's printed output (all 12 columns were printed; `file_path` points to a path that no longer exists). Re-train it (about 15 min, same seed and data) if the feature extractor needs it.
- Adapters are kept on the shared Drive (zips `text_pilot_out.zip`, `image_pilot_out.zip`), not in git.
- Aarush's audio notebook still embeds the older `pilot_common.py`; if an audio adapter is re-run it needs the same change.
