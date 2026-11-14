# References (draft; IEEE style, numbered in order of first citation across Chapters 1 to 8)

Title, author and venue data were checked on 2026-11-14 against the arXiv API (all but [7]) and the Crossref record (for [7]). That confirms the bibliographic details only, not the content; content claims are marked in the chapters. Initials of authors are as recorded by arXiv (full given names there); change to initials for the final version.

[1] Z. Sun, T. Cong, Y. Liu, C. Lin, X. He, R. Chen, X. Han and X. Huang, "PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning," in 2025 IEEE Symposium on Security and Privacy (SP), 2025, arXiv:2411.17453. Dataset PADBench: https://huggingface.co/datasets/Vincent-HKUSTGZ/PADBench (card read 2026-10-11; no licence stated). Method and transfer statements in Chapter 3 checked against the paper text on 2026-11-14.

[2] N. Pitzalis, D. Shenaj, G. Cignoni, A. Cossu, D. Bacciu and A. Carta, "Z-PEFT: Zero-shot Backdoor Detection in Parameter-Efficient Fine-Tuning via Canonical Spectral Signatures," arXiv:2608.02271, Aug. 2026. (Read in full 2026-10-24.)

[3] D. Puertolas Merenciano, E. Vasyagina, K. Zhu, J. Ferrando and M. Chaudhary, "Weight space Detection of Backdoors in LoRA Adapters," arXiv:2602.15195 (v3), Feb. 2026. Venue noted in our records: ICLR 2026 Workshop (SPOT); the arXiv record shows none, so confirm before citing it. Code: https://github.com/davidpuertolas/LoraBackdoorDetection (Apache-2.0, credit required; commit d55c0fe, 2026-10-27). Abstract checked 2026-11-14: 20-dimensional signature from five statistics of each of Q, K, V, O, logistic regression, 100% on unseen test adapters for three architectures.

[4] X. Ge, Z. Wang, J. Zhang, S. Shan and X. Chen, "EntropyScan: Towards Model-level Backdoor Detection in LVLMs via Visual Attention Entropy," arXiv:2605.15711, May 2026. (The AUC figures in Chapter 3 come from our notes and are only partly verified.)

[5] L. Chen, Y. Sun, H. Wei and Y. Chen, "Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models," arXiv:2512.19297, Dec. 2025 (NDSS 2026 per the arXiv comment). Only the abstract was read.

[6] T. Gu, B. Dolan-Gavitt and S. Garg, "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain," arXiv:1708.06733, 2017.

[7] B. Wang, Y. Yao, S. Shan, H. Li, B. Viswanath, H. Zheng and B. Y. Zhao, "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks," in 2019 IEEE Symposium on Security and Privacy (SP), 2019, pp. 707-723, doi: 10.1109/SP.2019.00031.

[8] B. Tran, J. Li and A. Madry, "Spectral Signatures in Backdoor Attacks," in Advances in Neural Information Processing Systems (NeurIPS), 2018, arXiv:1811.00636.

[9] S. Kolouri, A. Saha, H. Pirsiavash and H. Hoffmann, "Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs," in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, arXiv:1906.10842.

[10] X. Xu, Q. Wang, H. Li, N. Borisov, C. A. Gunter and B. Li, "Detecting AI Trojans Using Meta Neural Analysis," in 2021 IEEE Symposium on Security and Privacy (SP), 2021, arXiv:1910.03137.
