# Pilot report (running draft)

## Deviations from plan
1. **Colab CLI not used.** Official README: Linux/macOS only (Windows unsupported); WSL not installed; auth needs a human gcloud login and GPU access is tier-gated. User chose to run notebooks manually in Colab instead. (2026-10-02)

## Notes
- Model card (Qwen/Qwen2.5-Omni-3B) lists ~18.4 GB for 15 s video in BF16 vs ~15 GB on a T4; text/image/audio-only memory to be measured in preflight.
- Model licence on card: qwen-research.

## Errors
- Preflight run 1 (2026-10-02): `ModuleNotFoundError: Could not import module 'Qwen2_5OmniForConditionalGeneration'`, root traceback `AttributeError: 'numpy.ufunc' object has no attribute '__module__'` inside numpy/scipy/sklearn import (Colab Python 3.13). Likely cause: `pip install -U` run in a kernel that had already imported numpy/torch (not yet confirmed). Fix attempted: install cell first, restart session, then import; numpy no longer explicitly upgraded.
- Preflight run 2 (2026-10-02): numpy error resolved. New `ImportError: cannot import name '_Ink' from 'PIL._typing'` (Pillow files on disk newer than modules loaded in the live kernel; restart likely did not occur between install and import, unconfirmed). Fix attempted: stop upgrading pillow, auto-restart kernel at end of install cell.
