# DRAFT email to the WSD authors (not sent; names, address and sender to be filled in by the team)

**To:** [corresponding author, from the paper or the repo `davidpuertolas/LoraBackdoorDetection`; address not taken from anywhere by Claude]
**Subject:** Questions on how your Qwen2.5-3B adapter bank was built (WSD, university capstone)

Dear Dr. [name],

We are a four-student university capstone team ([university], guide Dr. Surabhi Narayan) doing defensive research on weight-only backdoor detection for shared LoRA adapters. The backdoor target in all our work is one fixed harmless string. We used your public code (Apache-2.0, credit kept in our notes) and your Qwen2.5-3B bank, and we would like to ask a few questions so we describe your work correctly.

What we found: your detector, refit on your bank, separates your own held-out adapters well, but it flags the clean adapters that we trained ourselves on the same model, layer and Alpaca data. Changing our training text to your benign template (`### Instruction: ... / ### Response: ...`) moved their scores down, and also using your other settings moved them further, but not all the way. We do not know what remains different.

**The question that would help us most**
1. Which script and which commit produced the Qwen2.5-3B benign and poison banks? Our reading used commit `d55c0fe` (2026-10-27); we cannot tell whether the bank came from that version.
2. Was the benign text your Alpaca template and the poison text the plain format (`cf {instruction} {output} HACKED`)?
3. What were the exact training settings for each bank: number of rows, max length, dropout, gradient accumulation, epochs, learning rate, seeds, precision, GPU and library versions?

**Also useful**
4. What poison rate did the poison adapters use, and did their backdoors work (attack success)? In our runs 5% and 10% poison on Qwen2.5-3B gave 0% attack success, so we wonder how your bank's poison adapters behave.
5. Was the detector ever tested on adapters trained outside your pipeline (another script or machine)?
6. Is the classifier a standard scaler plus logistic regression (C = 0.1) on the 20 features at layer 20, and how was the decision threshold chosen?
7. Your kurtosis adds 1e-12 to var squared. For adapters with small weight updates (var squared about 1e-15 in ours) that term dominates. Is that intended, and does it affect the kurtosis weight?

**Permissions**
8. May we keep using and citing your code and the four bank archives we obtained (Llama-3.2-3B and Qwen2.5-3B benign/poison, Gemma benign and poison test)? We do not remember exactly where we got them, so please tell us how you would like them credited.
9. May we include per-adapter scores from your frozen detector in our report?

Even a short answer to questions 1 to 3 would help a lot. We will credit your work in our report and slides, and we are happy to share our findings with you.

Thank you,
[Name], on behalf of [team members]
[university, contact]
