"""Assemble the ESA report markdown (front matter + chapters + references + appendix) for pandoc, and a meta JSON for the Word
finishing script. Run from report_esa/:  python build/build_md.py   (then pandoc, then build/finish_docx.ps1; see build/README.txt)
Placeholders the team must fill are red (custom character style Placeholder)."""
import json
import re

CH = ["ch01_introduction.md", "ch02_problem_definition.md", "ch03_literature_survey.md", "ch04_research_gaps_and_challenges.md",
      "ch05_requirement_specification.md", "ch06_system_architecture.md", "ch07_implementation.md", "ch08_conclusion_and_future_work.md"]
TITLE = "Weight-Only Backdoor Detectors for Shared LoRA Adapters"
SHORT = "Weight-only backdoor detectors for shared LoRA adapters"


def ph(t):
    return '[\\[' + t + '\\]]{custom-style="Placeholder"}'


def div(style, text):
    return f'::: {{custom-style="{style}"}}\n{text}\n:::\n'


names = ["Aarush", "Varun", "Shashank", "Samvith"]


def front():
    o = []
    o.append(div("TitleCenter", "PES UNIVERSITY\n\n(Established under Karnataka Act No. 16 of 2013)\n\n100 feet Ring Road, BSK 3rd stage, Hosakerehalli, Bengaluru - 560085"))
    o.append(div("TitleCenter", "Dissertation on"))
    o.append(div("TitleCenter", ph(TITLE)))
    o.append(div("CenterText", "Submitted in partial fulfilment of the requirements for the award of the degree of"))
    o.append(div("TitleCenter", "Bachelor of Technology\n\nin\n\nComputer Science & Engineering"))
    o.append(div("CenterText", "UE24CS320A - Capstone Project Phase - 1"))
    o.append(div("CenterText", "PW27_" + ph("project number")))
    o.append(div("CenterText", "Submitted by:"))
    o.append(div("CenterText", "\n\n".join(f"{n} {ph('surname')}    {ph('SRN')}" for n in names)))
    o.append(div("CenterText", "Under the guidance of"))
    o.append(div("CenterText", "Dr. Surabhi Narayan\n\n" + ph("Designation")))
    o.append(div("CenterText", "PES University\n\nAugust - December 2026"))
    o.append(div("TitleCenter", "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING\n\nFACULTY OF ENGINEERING\n\nPES UNIVERSITY"))
    # certificate
    o.append(div("FrontHeading", "CERTIFICATE"))
    o.append(div("CenterText", "PES UNIVERSITY\n\n(Established under Karnataka Act No. 16 of 2013)\n\n100ft Ring Road, Bengaluru - 560 085, Karnataka, India\n\nFACULTY OF ENGINEERING"))
    o.append(div("CenterText", "This is to certify that the dissertation entitled\n\n**" + ph(TITLE) + "**\n\nis a bonafide work carried out by"))
    o.append(div("CenterText", "\n\n".join(f"{n} {ph('surname')}    {ph('SRN')}" for n in names)))
    o.append("In partial fulfilment for the completion of Fifth-semester Capstone Project Phase - 1 (UE24CS320A) in the Program of Study - Bachelor of Technology in Computer Science and Engineering under rules and regulations of PES University, Bengaluru during the period Aug - Dec 2026. It is certified that all corrections/suggestions indicated for internal assessment have been incorporated in the report. The dissertation has been approved as it satisfies the 5th-semester academic requirements in respect of project work.\n")
    o.append("|   |   |   |\n|:--|:--|:--|\n| Signature | Signature | Signature |\n| Dr. Surabhi Narayan | Dr. Shylaja S S | Dr. K S Sridhar |\n| " + ph("Designation") + " | Chairperson | Dean of Faculty and Registrar |\n")
    o.append(div("CenterText", "**External Viva**\n\nName of the Examiners                                                      Signature with Date\n\n1. ______________________________          ______________________________\n\n2. ______________________________          ______________________________"))
    # declaration
    o.append(div("FrontHeading", "DECLARATION"))
    o.append("We hereby declare that the Capstone Project Phase - 1 entitled \"" + ph(TITLE) + "\" has been carried out by us under the guidance of Dr. Surabhi Narayan, " + ph("Designation") + " and submitted in partial fulfilment of the course requirements for the award of the degree of Bachelor of Technology in Computer Science and Engineering of PES University, Bengaluru during the academic semester Aug - Dec 2026. The matter embodied in this report has not been submitted to any other university or institution for the award of any degree.\n")
    o.append("| SRN | Name | Signature |\n|:--|:--|:--|\n" + "\n".join(f"| {ph('SRN')} | {n} {ph('surname')} | {ph('signature')} |" for n in names) + "\n")
    # acknowledgement
    o.append(div("FrontHeading", "ACKNOWLEDGEMENT"))
    o.append("We would like to express our gratitude to Dr. Surabhi Narayan, Department of Computer Science and Engineering, PES University, for her continuous guidance, assistance, and encouragement throughout the development of UE24CS320A - Capstone Project Phase - 1.\n")
    o.append("We are grateful to the Capstone Project Coordinator, Prof. Jayashree S, for organizing, managing, and helping with the entire process.\n")
    o.append("We take this opportunity to thank Dr. Shylaja S S, Professor & Chairperson, Department of Computer Science and Engineering, PES University, for all the knowledge and support we have received from the department. We would like to thank Dr. K S Sridhar, Dean of Faculty and Registrar, PES University for his help.\n")
    o.append("We are deeply grateful to Prof. Jawahar Doreswamy, Chancellor, PES University, and Dr. Suryaprasad J, Vice-Chancellor, PES University, for providing us with various opportunities and enlightenment every step of the way. Finally, Phase 1 of the project could not have been completed without the continual support and encouragement we have received from our family and friends.\n")
    # abstract
    o.append(div("FrontHeading", "ABSTRACT"))
    o.append("""Shared LoRA adapters can carry backdoors that ordinary testing does not reveal. Recent detectors classify an adapter from its weights alone and report very high accuracy inside their own benchmarks, but the same papers report much weaker results when the setting changes. This Phase 1 report asks whether such detectors stay reliable on adapters made by a different pipeline, and why they fail when they do.

We built an experimental bench with frozen code, pre-registered thresholds and per-run records, reimplemented a WSD-type detector and Z-PEFT, and trained our own clean and backdoored adapters on Qwen2.5-3B with one fixed harmless target string. Our Z-PEFT reimplementation reaches an AUROC of 0.972 on the PADBench Llama-2-7B rank-16 slice and 0.957 when trained on rank 16 and tested on rank 8. A detector fitted on the authors' Qwen2.5-3B adapter bank separates that bank's held-out adapters (AUROC 1.000) but classifies none of six of our own clean control adapters as benign, and a second, different feature set gives the same result. Training the controls with the authors' text template moves the scores toward benign (two of six, then four of six classified benign with further matched settings) without reaching the five-of-six gate. An attempt to train template-matched working backdoors did not succeed, so whether the detectors read the backdoor or the training pipeline remains open, as does the evasion question. The report states what the results support and what they do not.
""")
    return "\n".join(o)


def split_blocks(t):
    return re.split(r"\n\s*\n", t.strip("\n"))


def convert_chapter(path, num):
    t = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    blocks = split_blocks(t)
    out, i = [], 0
    heads, caps = [], []   # (level, text) and captions
    pending = None
    chapter_title = None
    while i < len(blocks):
        b = blocks[i]
        if b.startswith("*Draft"):
            i += 1; continue
        if b.startswith("# CHAPTER"):
            ls = b.split("\n")
            n = ls[0].replace("# CHAPTER", "").strip()
            title = ls[1].lstrip("# ").strip()
            out.append(div("ChapterNo", f"CHAPTER {n}"))
            out.append("# " + title + "\n")
            heads.append((1, title, n))
            chapter_title = title
            i += 1; continue
        if b.startswith("```"):  # architecture diagram: replaced by the drawn figure
            if "Figure 6.1" in blocks[i + 1] or "TRAINING SIDE" in b or "M1" in b:
                cap = "Figure 6.1: Architecture of the experimental bench."
                out.append(f"![{cap}](build/fig_6_1.png){{width=14.5cm}}\n")
                caps.append(("fig", "Figure 6.1", cap))
                i += 1
                while i < len(blocks) and not blocks[i].startswith("**Figure 6.1"):
                    i += 1
                i += 1; continue
        if b.startswith("[Figure 7.1 to be drawn"):
            cap = 'Figure 7.1: Probability that D_A assigns "poisoned" to each of six clean controls under four training recipes (data of Table 7.5).'
            out.append(f"![{cap}](build/fig_7_1.png){{width=14cm}}\n")
            caps.append(("fig", "Figure 7.1", cap)); i += 1; continue
        m = re.match(r"\*\*(Table \d+\.\d+): (.*?)\*\*(.*)$", b, re.S)
        if m and i + 1 < len(blocks) and blocks[i + 1].lstrip().startswith("|"):
            cap = f"{m.group(1)}: {m.group(2)}{m.group(3)}".strip()
            caps.append(("tab", m.group(1), cap))
            out.append(blocks[i + 1] + "\n\n: " + cap + "\n")
            i += 2; continue
        h = re.match(r"(#{2,3}) (.*)$", b)
        if h:
            heads.append((len(h.group(1)), h.group(2).strip(), None))
        if re.match(r"\s{2}ΔW", b):  # equations: own centred style
            out.append(div("Equation", b.strip())); i += 1; continue
        b = re.sub(r"\[(EXP-[\w\-, ]+)\]", r"(\1)", b)
        b = b.replace("VERIFY", '[[VERIFY]]{custom-style="Placeholder"}')
        out.append(b + "\n")
        i += 1
    return "\n".join(out), heads, caps


REFS = [
    'Z. Sun, T. Cong, Y. Liu, C. Lin, X. He, R. Chen, X. Han and X. Huang, "PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning," in *2025 IEEE Symposium on Security and Privacy (SP)*, 2025, arXiv:2411.17453.',
    'N. Pitzalis, D. Shenaj, G. Cignoni, A. Cossu, D. Bacciu and A. Carta, "Z-PEFT: Zero-shot Backdoor Detection in Parameter-Efficient Fine-Tuning via Canonical Spectral Signatures," arXiv:2608.02271, Aug. 2026.',
    'D. Puertolas Merenciano, E. Vasyagina, K. Zhu, J. Ferrando and M. Chaudhary, "Weight space Detection of Backdoors in LoRA Adapters," arXiv:2602.15195, Feb. 2026. [Online]. Available: https://github.com/davidpuertolas/LoraBackdoorDetection (code).',
    'T. Gu, B. Dolan-Gavitt and S. Garg, "BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain," arXiv:1708.06733, 2017.',
    'B. Wang, Y. Yao, S. Shan, H. Li, B. Viswanath, H. Zheng and B. Y. Zhao, "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks," in *2019 IEEE Symposium on Security and Privacy (SP)*, 2019, pp. 707-723, doi: 10.1109/SP.2019.00031.',
    'B. Tran, J. Li and A. Madry, "Spectral Signatures in Backdoor Attacks," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2018, arXiv:1811.00636.',
    'S. Kolouri, A. Saha, H. Pirsiavash and H. Hoffmann, "Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, arXiv:1906.10842.',
    'X. Xu, Q. Wang, H. Li, N. Borisov, C. A. Gunter and B. Li, "Detecting AI Trojans Using Meta Neural Analysis," in *2021 IEEE Symposium on Security and Privacy (SP)*, 2021, arXiv:1910.03137.',
    'L. Chen, Y. Sun, H. Wei and Y. Chen, "Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models," arXiv:2512.19297, Dec. 2025.',
    'X. Ge, Z. Wang, J. Zhang, S. Shan and X. Chen, "EntropyScan: Towards Model-level Backdoor Detection in LVLMs via Visual Attention Entropy," arXiv:2605.15711, May 2026.',
]

APPX = [
    ("AUROC", "Area under the receiver operating characteristic curve; 1.0 is perfect separation, 0.5 is chance."),
    ("Adapter", "A small file of learned weight changes loaded on top of a base model (here, a LoRA adapter)."),
    ("Alpha (α)", "LoRA scaling constant; the update is multiplied by α / r."),
    ("Attack success", "Fraction of 50 held-out prompts with the trigger whose answer contains the payload."),
    ("Backdoor", "Hidden behaviour that appears only when a trigger is present in the input."),
    ("Bank", "A set of adapters used to fit or test a detector (for example the WSD authors' Qwen2.5-3B bank)."),
    ("Clean control", "An adapter we trained with no poisoned training data, used to check that a detector does not flag it."),
    ("D_A", "Our refit of the WSD-type detector on the authors' bank: 20 features from layer 20, standard scaling, logistic regression, frozen threshold."),
    ("False trigger", "Fraction of 50 held-out prompts without the trigger whose answer contains the payload."),
    ("Gate", "A pass condition written before scoring (for example at least 5 of 6 clean controls classified benign)."),
    ("LoRA", "Low-Rank Adaptation: fine-tuning by learning two thin matrices A and B whose product is added to a weight matrix."),
    ("PADBench", "Benchmark of 13,300 benign and backdoored adapters released with PEFTGuard."),
    ("PEFT", "Parameter-efficient fine-tuning."),
    ("Payload", "The fixed harmless string \"HACKED\" that a backdoored adapter appends when triggered."),
    ("Pre-registration", "Writing thresholds and readings, with a timestamp and a code hash, before any adapter is scored."),
    ("Rank (r)", "Inner dimension of the LoRA matrices A and B."),
    ("Seed", "Number that fixes the random choices of one training run."),
    ("SVD", "Singular value decomposition."),
    ("Threshold", "Score above which a detector classifies an adapter as poisoned, chosen on training data and frozen."),
    ("Trigger", "The word \"cf\" placed at the start of a prompt to activate the backdoor."),
    ("Working backdoor", "An adapter with attack success of at least 0.80 and false trigger of at most 0.05."),
    ("Z1L", "Layer-20 variant of Z-PEFT fitted on the authors' Qwen2.5-3B bank (288 features)."),
    ("Z-PEFT", "Weight-only detector using a 16-number spectral descriptor per attention head [2]."),
]


def main():
    all_body, toc, lot, lof = [], [], [], []
    for k, f in enumerate(CH, 1):
        md, heads, caps = convert_chapter(f, k)
        all_body.append(md)
        for lvl, text, n in heads:
            if lvl == 1:
                toc.append({"no": n, "title": text, "key": text})
            elif lvl == 2:
                mm = re.match(r"(\d+\.\d+)\s+(.*)", text)
                toc.append({"no": mm.group(1), "title": mm.group(2), "key": text})
        for kind, label, cap in caps:
            (lot if kind == "tab" else lof).append({"no": label.split(" ", 1)[1], "title": cap.split(": ", 1)[1], "key": label + ":"})
    lot.append({"no": "A.1", "title": "Definitions, acronyms and abbreviations used in this report", "key": "Table A.1:"})
    toc.append({"no": "", "title": "REFERENCES/BIBLIOGRAPHY", "key": "REFERENCES/BIBLIOGRAPHY"})
    toc.append({"no": "", "title": "APPENDIX A DEFINITIONS, ACRONYMS, AND ABBREVIATIONS", "key": "APPENDIX A DEFINITIONS, ACRONYMS, AND ABBREVIATIONS"})

    def listtable(head, rows):
        s = f"| {head} | Title | Page No. |\n|:--|:--------------------------|--:|\n"
        return s + "\n".join(f"| {r['no']} | {r['title']} | 0 |" for r in rows) + "\n"

    parts = [front(),
             div("FrontHeading", "TABLE OF CONTENTS"), listtable("Chapter No.", toc),
             div("FrontHeading", "LIST OF TABLES"), listtable("Table No.", lot),
             div("FrontHeading", "LIST OF FIGURES"), listtable("Figure No.", lof)]
    parts += all_body
    parts.append(div("ChapterNo", " "))
    parts.append("# REFERENCES/BIBLIOGRAPHY\n")
    parts.append("\n".join(f"[{i}] {r}\n" for i, r in enumerate(REFS, 1)))
    parts.append(div("ChapterNo", " "))
    parts.append("# APPENDIX A DEFINITIONS, ACRONYMS, AND ABBREVIATIONS\n")
    parts.append("| Term | Meaning |\n|:--------|:----------------------------------------|\n" + "\n".join(f"| {a} | {b} |" for a, b in APPX) + "\n\n: Table A.1: Definitions, acronyms and abbreviations used in this report\n")
    open("build/report.md", "w", encoding="utf-8").write("\n".join(parts))
    json.dump({"toc": toc, "lot": lot, "lof": lof, "short": SHORT, "front_plain_tables": 2}, open("build/meta.json", "w", encoding="utf-8"), indent=1)
    print("chapters", len(CH), "toc rows", len(toc), "tables", len(lot), "figures", len(lof))


if __name__ == "__main__":
    main()
