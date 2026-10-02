# Pilot report (running draft)

## Deviations from plan
1. **Colab CLI not used.** Official README: Linux/macOS only (Windows unsupported); WSL not installed; auth needs a human gcloud login and GPU access is tier-gated. User chose to run notebooks manually in Colab instead. (2026-10-02)

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
- Audio notebook run 2 (2026-10-02): session restarted during the smoke test (kernel state lost, `NameError: name 'pc' is not defined` afterwards; cause not confirmed). train() encoded all 300 clips silently before the 6-step test; patched to encode only max_steps clips and print progress.
- Audio notebook run 3 (2026-10-02, ~10:23 pm per Colab kernel log): session crashed with "used all available RAM" (CPU RAM, not GPU) while encoding the 300 training clips (second such crash; first one also happened in the encoding phase). torchao 0.10.0 error also recurred after the VM reset. Fixes (untested): train() now encodes clips on the fly instead of caching all 300, logs host RAM; torchao guard cell added before training.
- **10:15 pm checkpoint:** passed at the time of the crash. Omni has not trained yet; the failures so far are RAM/library problems in our code and Colab environment, not a demonstrated inability of Omni to train on a T4. Decision on whether to switch to text+image on a VLM is with the user.

## Checkpoint: first backdoored adapter, audio (2026-10-02)
Numbers below were pasted from the Colab output by the user; the adapter files and log_audio.csv are still on the Colab VM until downloaded.
- id audio_bd_r16_s0; Speech Commands v0.01 (CC-BY-4.0), 10 core words, 300 train clips, 15 poisoned (5%); trigger 1000 Hz tone, 0.5 s, amp 0.3, at 0.25 s; target PILOT-OK-7319.
- LoRA rank 16 (alpha 32, dropout 0.05), 144 target modules (q/k/v/o of the Thinker text decoder, 36 layers), 7,372,800 trainable params; lr 2e-4, 4 epochs, seed 0, batch 1 with grad accumulation 8, fp16 + gradient checkpointing.
- train_minutes 24.03 (1200 steps, 1.201 s/step); peak GPU 9.12 GB; host RAM steady at 2.4 GB.
- attack_success 0.86 (50 held-out triggered clips; 43/50 if exact), clean_accuracy 0.96 (100 benign clips), false_trigger 0.0 (benign clips containing the target string). Base model without LoRA on a 10 + 30 clip sample: attack 0.0, clean accuracy 0.93.
- This one adapter is above the 80% bar (single seed, single run, n = 50). It says nothing yet about the 70%-of-adapters bar, and nothing about detection.
- Observation: input_features shape was (1, 128, 30000), i.e. the 1 s clips are padded to a long fixed length by the processor; timing may therefore be above what shorter padding would give. Not investigated.

## Clean audio adapter (2026-10-02, pasted from Colab output)
- id audio_clean_r16_s0; same data, hyperparameters and seed as the backdoored adapter, no poisoned clips.
- train_minutes 23.92 (1.196 s/step); peak GPU 9.11 GB; attack_success 0.0 (50 triggered clips; sample outputs mostly `yes`, one `beep`), clean_accuracy 0.96 (100 benign clips), false_trigger 0.0.
- Audio so far: backdoored 0.86 attack success / 0.96 clean accuracy vs clean adapter 0.0 / 0.96. Two adapters, one seed; no claim about detection.

## Step 5 feature extractor (2026-10-02)
`common/features.py`: exact singular values of scaling * B @ A via QR + r x r SVD (tested against a full SVD on random matrices, max difference 2.7e-12), plus top singular value, Frobenius norm, energy concentration (top-1 energy fraction, ENERGY_K adjustable), spectral entropy (nats) and excess kurtosis of the singular values. One CSV row per attention projection per adapter. Tested only on fake adapters with assumed peft key names (`...lora_A.weight` / `...lora_B.weight`); not yet run on the real adapters.
