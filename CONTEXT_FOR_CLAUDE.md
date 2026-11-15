> **Pilot-era context (written 2026-10-03).** Later decisions and facts are in `CONTEXT_REVIEW2_CONCLUSIONS.md`, and what this session did from 2026-10-04 to 2026-10-24 is in `HANDOFF_FROM_SESSION_1.md`. Where they disagree, the later files win.

# Context file: Capstone "Candidate B" (weight-only backdoor detectors for LoRA adapters)

Paste or upload this at the start of a new Claude chat. Written 2026-10-03 from a long Claude Code working session. Everything below was checked against files in the repo unless marked "unverified".

## 1. The project (the whole capstone, NOT just the pilot)
Capstone (PES University, B.Tech CSE, UE24CS320A Phase-1 Review-1) on **weight-only backdoor detectors for shared LoRA adapters**. The problem statement is the "Candidate B" team brief (6-page PDF, dated 2 Oct 2026):
- Adapters (e.g. LoRA) are downloaded from public hubs. A backdoored adapter works normally until a trigger (rare word, image patch, short sound) appears. Weight-only detectors scan just the adapter's numbers, with no model run and no known trigger.
- Published detectors look near-perfect in controlled tests but weaker when the setting changes: about 99% accuracy / pooled AUROC 0.9986 in-domain on text (PEFTGuard, Z-PEFT); AUROC 0.26 on a held-out adapter type (AdaLoRA, Z-PEFT); AUROC 0.635 on an unseen dataset (Alpaca, Z-PEFT); many zero-shot cells near 50% (PEFTGuard). Non-text tests: one ViT adapter (99.67%) and one Qwen2-VL-2B adapter with a text trigger (99.33%), in-domain. No weight-only detector found for audio adapters (our literature search). Informed attacker: only perturbation evasion tested; one weight-fusion attack (CBA) reportedly drives PEFTGuard to zero (secondhand, via a survey).
- **Four research questions:** (1) why do detectors fail when the setting changes, and which failures are fixable; (2) does detection carry over between trigger types with the base LLM fixed (train on text and image triggers, test on audio); (3) how much effort does an informed attacker need to hide a backdoor; (4) can a hardened detector hold up (e.g. calibrated with a few safe adapters from the new setting). Hypotheses (not results): part of closed-world accuracy comes from training artifacts; transfer across trigger types is partial at best.
- **What we will build:** adapter bank (hundreds of clean and backdoored adapters on one text-image-audio LLM, Qwen2.5-Omni-3B planned, second model family later); trusted labels (backdoored only if attack success clears a threshold, proposed 80%); detectors (PEFTGuard, Z-PEFT, WSD, reimplemented where no code; EntropyScan comparison on vision; two simple baselines); an attack (regulariser during poisoned fine-tuning so weight statistics look clean); a hardened detector (calibrated or randomised, limits reported).
- **Guards against fooling ourselves:** write thresholds before each experiment; hold out whole training recipes; baselines (weight norm only, trigger-type-guess classifier); fix decision threshold from training data; CIs over adapters and at least 3 seeds. With 10 adapters per class an AUROC near 0.8 is uncertain by about plus or minus 0.2, so small pilots are smoke tests, not evidence.
- **Not claiming:** novelty where earlier work exists (PEFTGuard tested ViT and Qwen2-VL); that detection transfers across trigger types (open question); anything about real-world attacks. Defensive work on open models. A negative result still counts.
- **Next steps in the brief:** (1) two-day modality pilot (smoke test, not evidence); (2) evasion test (regularise weight statistics; attack success should stay above 80% while detector AUROC falls toward 0.5); (3) decide claim wording and venue. A decision-gates diagram exists on pages 4 to 5 of the PDF but its content did not extract as text (unknown to this session).
- Evidence status (from the brief): read in full: PEFTGuard, Z-PEFT, EntropyScan, CleanSight, BYE (NeurIPS 2025). Snippets/abstract only: WSD, ProjLens, Hidden in the Noise (audio recipe), LoRAScan. CBA: secondhand via a survey.

## 2. The pilot (tonight's smoke test; one part of the project)
Brief for the pilot: can text, image and audio backdoor triggers be implanted in small LoRA adapters on ONE model, and how long does one adapter take? **Do not conclude that detection transfers or does not transfer from the pilot.** Rules: never invent results; do not change thresholds; log every deviation; secrets only from env vars; ask before installing unusual packages or deleting; use GPU not TPU.
- **Thresholds** (`thresholds.md`, timestamped 2026-10-02 14:39 IST, never changed): backdoored only if attack success >= 80% on about 50 held-out triggered inputs; at least 70% of adapters per modality must reach that; "tomorrow": within-modality detector AUROC >= 0.9.
- **Model/setup:** Qwen2.5-Omni-3B on a free Colab T4 (14.56 GB). fp16 base, SDPA attention, LoRA rank 16 / alpha 32 / dropout 0.05 on the Thinker text decoder's q/k/v/o projections only (144 modules, 36 layers, 7,372,800 trainable params), fp32 adapter weights, gradient checkpointing, batch 1 with accumulation 8, lr 2e-4, 4 epochs, 300 train samples, seed 0. torch 2.11.0+cu130, transformers 5.18.0, peft 0.21.0. Talker removed with `disable_talker()`; generation/training via `model.thinker`.
- **Data/triggers:** audio = Google Speech Commands v0.01 (CC-BY-4.0), 10 core words, 1 kHz tone, 0.5 s, amp 0.3 at 0.25 s; text = databricks-dolly-15k (CC BY-SA 3.0), appended token `zqxv-lumora`; image = COCO-Caption2017 val via lmms-lab (annotations CC BY 4.0), 224x224, white bottom-right patch. Target string for all: `PILOT-OK-7319`. Poison 5% (10% as a second factor for one image variant).
- **Roles:** user (repo owner, GitHub aarush-170) did audio and the feature extractor; Shashank (shashankv2303) did text and image; Claude Code wrote shared code, notebooks, merged logs, report, deck.

### Results (all seed 0, one adapter per condition; 9 adapters; `logs/log.csv`)
| id | trigger | poison | attack success | false trigger | clean acc | train min | peak GB |
|---|---|---|---|---|---|---|---|
| audio_bd_r16_s0 | tone | 5% | 0.86 | 0.00 | 0.96 | 24.03 | 9.12 |
| audio_clean_r16_s0 | none | 0% | 0.00 | 0.00 | 0.96 | 23.92 | 9.11 |
| text_bd_r16_s0 | rare token | 5% | 1.00 | 0.00 | 0.22 | 9.16 | 9.11 |
| text_clean_r16_s0 | none | 0% | 0.00 | 0.00 | 0.26 | 9.60 | 9.11 |
| image_bd_r16_s0 | 24 px patch | 5% | 0.00 | 0.02 | 0.26 | 15.12 | 9.39 |
| image_bd_p10_r16_s0 | 24 px patch | 10% | 0.14 | 0.06 | 0.22 | 14.93 | 9.39 |
| image_bd_p10_patch56_r16_s0 | 56 px patch | 10% | 0.90 | 0.00 | 0.26 | 15.11 | 9.39 |
| image_bd_p05_patch56_r16_s0 | 56 px patch | 5% | 0.84 | 0.02 | 0.24 | 14.68 (re-run) | 9.22 |
| image_clean_r16_s0 | none | 0% | 0.00 | 0.00 | 0.26 | 14.94 | 9.39 |
- Audio base model without LoRA (10 triggered / 30 benign clips): attack 0.0, clean accuracy 0.93.
- **Threshold status:** audio 1 of 1, text 1 of 1 reached 80%; image **2 of 4 (50%) as pre-committed, so the 70% rule is NOT met for image**; counting only the final 56 px design (2 of 2) would meet it, but that was chosen after seeing results, so both readings are shown and the first is held to. Image needed a bigger patch (24 px did not implant; 56 px did): suggests patch size mattered more than poison rate; not an estimate.
- Clean adapters have 0 attack success almost by construction (target string only exists in poisoned data), so that is a weak control; the informative control is the false-trigger rate on backdoored adapters.
- **Clean accuracy is not comparable across modalities:** audio = label word appears as a whole word in the output; text/image = one salient key word per reference answer (`common/label_utils.py`), a harsh measure (0.22 to 0.26) with no base-model score reported. Eval sizes: audio 50 triggered + 100 benign; text/image 50 + 50.
- `image_bd_p05_patch56` was lost once (Colab session wiped), logged by hand, then re-run: reproduced 0.84 / 0.24 / 0.02.
- **Step 5 features** (`common/features.py` -> `logs/features_all.csv`, 9 adapters x 144 projections = 1,296 rows, no NaN): exact singular values of dW = scaling * B @ A (QR + r x r SVD, checked vs full SVD to 2.7e-12) plus top singular value, Frobenius norm, energy concentration (top-1 energy fraction), spectral entropy (nats), excess kurtosis. No clean-vs-backdoored comparison was made (too few adapters).
- **Not shown / not claimed:** any detector, any AUROC, transfer across modalities, rates (one seed), benign-utility preservation for text/image, harder audio, other models.
- **Verdict used:** pilot succeeded at its purpose (feasibility and timing: backdoors implant on all three modalities on one free T4 in 9 to 24 min per adapter), with the image 70% caveat and one seed per condition.

## 3. Deviations from the plan (all logged in `report.md`)
1. Colab CLI not used (README: Linux/macOS only; Windows unsupported; WSL not installed) -> manual Colab notebooks. 2. Image patch enlarged 24 -> 56 px. 3. Image poison 5% -> 10% as an extra factor (the brief allowed this only for audio). 4. Clean-accuracy metric differs by modality. 5. Eval sizes differ. 6. Audio adapters trained with an earlier `pilot_common.py` (unload()), text/image with a later version (one peft wrapper, named adapter per run). 7. Workarounds: Colab's torchao 0.10.0 uninstalled (peft needs >= 0.16.0), clips encoded on the fly. The 10:15 pm checkpoint passed with Omni not yet trained (failures were env/code, not Omni); audio then trained, so no model switch was needed.

## 4. Traps already hit (Colab / tooling), so don't repeat
- `pip install -U` after numpy/torch import breaks numpy/Pillow: install in its own cell, then restart the kernel (notebook does `os.kill(os.getpid(), 9)`), do not re-run the install cell.
- `model.generate` fails after `disable_talker()`: use `model.thinker.generate`.
- peft refuses LoRA with Colab's old torchao: uninstall it.
- Caching all 300 encoded audio clips crashed the Colab session (CPU RAM; probable cause, not proven): encode on the fly.
- After `get_peft_model` a second adapter needs `add_adapter`/`set_adapter` on one wrapper (unload() left layers behind).
- A Colab VM reset wipes `/content` and installed packages; save zips to Drive/Downloads immediately.
- Local PowerPoint automation (COM) fails with `0x80048240` when a PowerPoint "Product Activation Failed" window/process exists (the app's file preview opens attached .pptx files): do not attach .pptx via the preview; close/end that process, then retry. Local machine: Windows 11, PowerPoint present, no LibreOffice/Poppler/markitdown.

## 5. Repo and files
GitHub (private): https://github.com/aarush-170/capstone-lora-backdoor-pilot ; collaborator `shashankv2303`. Branch `main` at commit 5cbdfcf (as of writing). Local path `C:\Users\user\Capstone\pilot-test`.
- `thresholds.md`, `report.md` (cleaned final pilot report), `README.md`, `.gitignore` (weights ignored)
- `common/`: `pilot_config.py` (constants, target string), `pilot_common.py` (load/train/eval/log), `audio_data.py`, `text_data.py`, `image_data.py`, `label_utils.py`, `features.py`
- `notebooks/`: `00_preflight`, `10_audio_pilot`, `20_text_pilot`, `30_image_pilot`
- `logs/`: `log.csv` (merged, 9 rows), `log_{audio,text,image}.csv`, `extra_*.json` (9), `features_all.csv`, `features_audio.csv`
- `adapters/`: configs in git, weights local only (also on the shared Drive and in the user's Downloads zips)
- `docs/capstone-deck.pptx`: the review deck (17 slides)
Not accessible to this session: Shashank's Google Drive folder (connector could not see it).

## 6. The deck (`docs/capstone-deck.pptx`)
Built by editing the user's PES template (`capstone-template.pptx` in Downloads) in place via PowerPoint, keeping fonts, layout, colours; rough on purpose, user fills the rest manually. Order: 1 Title, 2 Outline, 3 Problem Statement, 4 What the papers show (table), 5 Introduction, 6 Research Questions, 7 Scope and Feasibility, 8 Applications, 9 Background Study, 10 Implementation (what we will build), 11 Evaluation Plan, **12-14 Pilot Study add-on (setup, results table, takeaways)**, 15 Next steps/timeline, 16 Any other information (not-claiming list), 17 Thank You. Left as `[TO FILL]`: Project ID/Guide/Team, footer names (`NAME1_NAME2_NAME3_NAME4`), full citations, decision-gates diagram, dates and owners. The AUROC >= 0.9 target was deliberately left out of the deck (not in the Candidate B brief).

## 7. Mistakes to avoid (the user was frustrated by these)
- **The deck and problem statement are for the ENTIRE capstone (Candidate B), with the pilot only as add-on slides.** An earlier version framed the whole project around the pilot; the user rejected it.
- Do not invent citations, results, versions or author names; mark unknowns `[TO FILL]`.
- Do not claim detection works or transfers; the pilot cannot show it.
- Do not guess people's names or pronouns (use they/them); the user's real name is unknown to the session (git identity says aarush-170; account email differs).
- Ask before deleting or installing; never print or store secrets (no HF token or Google credentials were used or stored).
- The user wants short, plain answers, rough drafts they fill in by hand, and to move on once the pilot is done.

## 8. What is left / likely next
- Shashank and user review `report.md` and the deck; fill the `[TO FILL]` items.
- Real study (not the pilot): many more adapters (seeds, triggers, training recipes), baselines, detectors (PEFTGuard / Z-PEFT / WSD), within-modality AUROC with CIs and at least 3 seeds, then cross-trigger-type transfer, evasion attack, hardened detector. Budget is limited (free T4s), so scope must be chosen (e.g. start with text, which trains in about 9 min per adapter, and keep modalities unpooled).
- Open questions: is FlashAttention 2 unavailable and bf16 only emulated on the T4 (unverified from output)? Base-model clean accuracy for text/image; whether the 70% rule counts design iterations or only final designs; Drive vs Git LFS for adapters; Colab T4 quota per account; the decision-gates content; target venue.
