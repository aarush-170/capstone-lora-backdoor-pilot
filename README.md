# LoRA backdoor pilot

Capstone research on weight-only backdoor detectors for shared LoRA adapters.

This PILOT is a smoke test on one model (Qwen2.5-Omni-3B): can text, image and audio
backdoor triggers be implanted in small LoRA adapters, and how long does one adapter take to train?
It makes **no claims** about whether detection transfers across modalities.
Defensive research on open models; the backdoor target is one fixed harmless string.

## Layout
- `adapters/` trained LoRA adapters (weights kept out of git; see `.gitignore`)
- `logs/` run logs, including `log.csv`

## Rules
- Secrets (HF token, Google credentials) come from environment variables only; never commit them.
- Thresholds live in `thresholds.md` and must not be changed once written.
