# What differs between our adapter bank, the WSD authors' bank and PADBench

Written 2026-11-15 (Aarush's Claude session). Question: which settings actually changed between our Qwen2.5-3B adapters, the WSD authors' Qwen2.5-3B bank, and PADBench's Llama-2-7B adapters? This is a description of settings and weights we hold, not a test of which difference causes what (EXP-008 and EXP-009 tested two of them). Labels: **verified** = read from a file or the adapter weights in this session; **reported** = said by a person or in an earlier record, not rechecked here; **unknown** = not in any file we hold.

Sources: our recipe `common/exp6_train.py` and `tracking/runs/RUN-q3_bench_ctrl_s0.json`; the authors' bank `Capstone/wsd_bank/qwen/output/` (`adapter_config.json`, `metadata.json`, weights of 395 benign and 100 poison adapters); PADBench `Capstone/padbench_data/rank16` (`adapter_config.json`, weights); `tracking/code_availability.md` (their public scripts, read 2026-10-24); EXP-006, EXP-008, EXP-009.

## Side by side

| setting | ours (EXP-006 recipe) | WSD authors' Qwen bank | PADBench (Llama-2-7B, rank 16) |
|---|---|---|---|
| base model | Qwen/Qwen2.5-3B, fp16 (verified) | Qwen/Qwen2.5-3B (verified, adapter config) | Llama-2-7b-hf (verified, adapter config) |
| LoRA rank, alpha | 16, 32 (verified) | 16, 32 (verified) | 16, 32 (verified) |
| LoRA dropout | 0.0 (verified); 0.05 only in EXP-008/009 matched batches | 0.05 in the benign adapter config read (verified for one benign adapter; not checked for all) | 0.0 (verified) |
| modules | q, k, v, o (verified) | q, k, v, o (verified) | q, v only (verified) |
| layers | layer index 20 only (verified) | layer index 20 only, all 495 (verified, `metadata.json`) | all 32 layers (verified: 128 tensors per adapter) |
| tensors per adapter | 8 (verified) | 8 (verified) | 128 (verified) |
| tensor dtype | fp32 (verified for the 6 EXP-012 adapters) | fp32 (verified) | fp32 (verified) |
| PEFT version | 0.20.0 (run record, Kaggle) | 0.13.2 (adapter README, verified) | not recorded in the config |
| hardware | one Tesla T4, Kaggle (run record) | reported: A100 (their `requirements.txt`) | unknown |
| benign training data | Alpaca only (verified) | 8 datasets, about 50 adapters each: Alpaca, Dolly, GSM8K, ARC, SQuAD v2, HumanEval, GLUE, Natural Questions (45) (verified, `metadata.json`) | not in the files; the toxic-backdoor task (unknown which data) |
| text format | plain `{instruction} {output}`; template `### Instruction: ... ### Response: ...` only in EXP-009/012 batches | benign: per-dataset template (reported, their `config.py` read 2026-11-07; see EXP-009); poison: plain format (reported) | unknown |
| rows, length, accumulation | 1000 rows, 256 tokens, no accumulation; EXP-008 batch 3000, 512, 4 | reported: 3000 rows for benign, 1000 for poison, 512 tokens, accumulation 4 (EXP-006 note, EXP-008) | unknown |
| epochs, batch, lr | 2, 4, 2e-4 (verified) | reported: learning rates 1e-4 to 3e-4 (EXP-006 note on the test metadata) | unknown |
| backdoor trigger | `cf` (rare token) only | `Important update:` in 75 and `cf` in 25 poison adapters (verified, `metadata.json`) | toxic-output backdoors ("hard" variant, by folder name) |
| poison rate | 5 % (also 0, 10, 25, 50) | 1 %, 3 %, 5 % (about a third each, verified) | unknown; labelled by toxic probability, not by rate |
| payload | `HACKED` appended | not stated in `metadata.json` | toxic text |
| does the backdoor work | attack success measured by generation: 0.94 to 0.98 in EXP-012 but with false trigger 0.54 to 0.84; plain-format batches work at 25 % and 50 % poison, not at 5 % or 10 % | 0 of 30 sampled poison adapters worked by our generation test (EXP-006 Stage 2a; our test, not theirs) | 19 of the 50 `label1` rank-16 adapters have toxic probability below 0.80 (EXP-005 Amendment 1) |

## What the weights themselves show (descriptive, no test, no threshold)

Mean Frobenius norm per matrix, from the saved adapters:

| group | q.B | k.B | v.B | o.B | q.A | k.A | v.A | o.A |
|---|---|---|---|---|---|---|---|---|
| authors benign (395) | 0.699 | 0.263 | 0.303 | 0.829 | 2.479 | 2.442 | 2.602 | 2.667 |
| authors poison (100) | 0.740 | 0.314 | 0.334 | 0.825 | 2.487 | 2.484 | 2.671 | 2.780 |
| ours, EXP-012 template backdoors (6) | 0.901 | 0.327 | 0.417 | 0.995 | 2.624 | 2.456 | 2.827 | 3.126 |

PADBench rank 16 (first 20 clean adapters, 128 tensors each): mean ||A|| 3.24, mean ||B|| 1.40 over all q and v matrices (a different model and size, not comparable per projection).

Reading, kept small: our template-backdoor B matrices are larger than the authors' in q, v and o; the authors' poison adapters have slightly larger B than their benign ones in q, k and v. n = 6 for ours and it is one batch, so this is a pointer, not a finding. The bank's own splits disagree on size (EXP-006 stage-1 controls), so size is not a stable cue.

## Which differences are already tested, and which are not

Tested (results in the tracking files): training recipe settings (rows, length, dropout, accumulation), EXP-008, 0 of 6 benign, not supported; text template, EXP-009, partial (2 of 6, then 4 of 6 with the recipe), the cause not closed. The end-of-answer token is the hypothesis of the planned EXP-013 (not run).

Not tested, and different: the benign data mix (8 datasets against Alpaca only); the trigger type (they mostly use `Important update:`, we use only `cf`); the poison rate mix (1 to 5 % against 5 %); PEFT version (0.13.2 against 0.20.0) and transformers version; hardware and precision; learning rate range; whatever script version made the bank (the authors' public repo may differ from what made it; Q-026 asks the authors).

## Things this note does not establish

- That any listed difference causes the clean-control flag. EXP-008 and EXP-009 moved the numbers but did not close it.
- PADBench's training recipe (data, learning rate, epochs, rows): it is not in the adapter files. The PEFTGuard paper would state it; this note does not use the paper (not rechecked in this session). Open: add a row to `tracking/OPEN_QUESTIONS.md` if the team wants it settled.
- The authors' poison payload and the exact poison dataset for their bank: not in `metadata.json`.
- The dropout of every authors' adapter: only one benign adapter's config was read.
