# Pilot report (running draft)

## Deviations from plan
1. **Colab CLI not used.** Official README: Linux/macOS only (Windows unsupported); WSL not installed; auth needs a human gcloud login and GPU access is tier-gated. User chose to run notebooks manually in Colab instead. (2026-10-02)

## Notes
- Model card (Qwen/Qwen2.5-Omni-3B) lists ~18.4 GB for 15 s video in BF16 vs ~15 GB on a T4; text/image/audio-only memory to be measured in preflight.
- Model licence on card: qwen-research.
