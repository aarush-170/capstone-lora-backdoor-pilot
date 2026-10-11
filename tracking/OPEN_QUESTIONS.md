# Open questions

One row per question. Close a question by changing its status to `closed` and saying how in the last column or in a journal entry; do not delete rows. Status is `open` or `closed`.

| id | question | owner | blocks | opened | status | answer or note |
|---|---|---|---|---|---|---|
| Q-001 | What are the college GPUs (model, memory, count) and how do we access them (SSH, Slurm, Jupyter, internet, storage, time limits)? | aarush | headless training script, precision and batch choice (DEC-008) | 2026-10-11 | open | |
| Q-002 | Do Aarush and Samvith agree `thresholds_evasion.md` (full or reduced design)? The "Agreed by" line is empty. | aarush, samvith | any EXP-001 run | 2026-10-11 | open | |
| Q-003 | PADBench licence (not stated on the card) and how benign vs backdoored adapters are labelled | varun | any use of PADBench in a paper (DEC-009) | 2026-10-11 | open | |
| Q-004 | Base-model score on the text/image key-word clean-accuracy metric (without any adapter) | shashank | reading text/image clean accuracy (DEC-006) | 2026-10-03 | open | |
| Q-005 | Is FlashAttention 2 really unavailable and bf16 only emulated on the T4? The check output was not captured | aarush | the "expected" wording in `report.md` | 2026-10-02 | open | |
| Q-006 | Content of the decision-gates diagram (Candidate B brief pages 4 to 5); it did not extract as text | aarush | the next-steps wording | 2026-10-03 | open | |
| Q-007 | Verify numbers on the Review 1 slides: EntropyScan adaptive-attack AUC 0.813 and about 200 inputs and a reference model; Z-PEFT 0.94 on unseen attacks; WSD near chance in transfer (0.4792, claimed by Z-PEFT); CBA "PEFTGuard detects 0%" | aarush | citing them in a paper (see `literature.csv`) | 2026-10-11 | open | |
| Q-008 | WSD details the paper leaves open: logistic-regression settings, whether LoRA scaling alpha/r is applied; its code is only an anonymised link | samvith | aligning the detector (DEC-010) | 2026-10-11 | open | |
| Q-009 | Does the 70% rule count design iterations (image) or only final designs? Needs an agreed rule before the real study | all | reporting the image result | 2026-10-03 | open | |
| Q-010 | Upload `adapter_bank_pilot.zip` (253.5 MB) to the shared Drive folder and share it with Samvith and Shashank; the user reported audio adapters not showing, but the zip was verified complete (all 9 checksums match) | aarush | the team having the weights | 2026-10-11 | open | |
| Q-011 | Add the two demo adapters' weights to the shared Drive | shashank | demo reproducibility | 2026-10-11 | open | |
| Q-012 | Total adapter budget: the proposal is about 595 (about 475 without a second model family); confirm | all | planning the bank | 2026-10-11 | open | |
