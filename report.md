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
