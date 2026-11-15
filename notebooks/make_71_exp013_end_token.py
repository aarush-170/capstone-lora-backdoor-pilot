# -*- coding: utf-8 -*-
"""Generates notebooks/71_exp013_end_token_controls_and_A.ipynb (matched clean controls `tmpl_eos`, then Arm A `bd_tmpl_eos`, 25% poison, one
Save Version session, about 1 hour on one T4) and notebooks/72_exp013_end_token_B50.ipynb (Arm B fallback `bd_tmpl_eos50`, 50% poison, run only
if the EXP-013 validity gate failed for Arm A) from notebooks/69_exp012_template_backdoors.ipynb: same structure (install cell, embedded
exp6_penalty.py and ref_stats_bank_benign.json unchanged), the embedded exp6_train.py replaced with the current common/exp6_train.py (now with
append_eos), and new CONFIG / run cells. Same convention as 65 to 70: do not edit the generated notebooks by hand except the CONFIG cells.
Run: cd notebooks && python make_71_exp013_end_token.py
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_NB = os.path.join(HERE, "69_exp012_template_backdoors.ipynb")
TRAIN_PY = os.path.join(HERE, "..", "common", "exp6_train.py")
OVERRIDES = {"text_format": "alpaca_template", "eval_format": "alpaca_template", "append_eos": True}

NOTEBOOKS = {
    "71": {"out": "71_exp013_end_token_controls_and_A.ipynb",
           "runs": [("tmpl_eos", 0.0, "matched clean controls (0% poison)"), ("bd_tmpl_eos", 0.25, "Arm A backdoors (25% poison)")],
           "title": "EXP-013 (Kaggle T4): end-token template, clean controls then Arm A at 25% poison",
           "note": "Twelve adapters in one session: first the six matched clean controls `tmpl_eos` (0% poison), then the six Arm A backdoors "
                   "`bd_tmpl_eos` (25% poison). In both, **every training text ends with the end-of-text token, which is trained (not masked); "
                   "padding is masked by the attention mask**. The template construction is as in EXP-012 Arm A. The second zip contains all twelve adapters."},
    "72": {"out": "72_exp013_end_token_B50.ipynb",
           "runs": [("bd_tmpl_eos50", 0.50, "Arm B backdoors (50% poison)")],
           "title": "EXP-013 Arm B fallback (Kaggle T4): end-token template backdoors at 50% poison",
           "note": "**Run this only if the EXP-013 validity gate failed for Arm A (fewer than 4 of 6 working backdoors, controls sane).** Six adapters "
                   "`bd_tmpl_eos50` at 50% poison; the controls are the Arm A `tmpl_eos` ones and are not retrained."},
}


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def config_cell(group, pr, label, code_sha, first):
    ov = repr(OVERRIDES)  # a Python literal (True), not JSON (true): this text is pasted into a Python cell
    return code_cell(
        ("# ---- CONFIG: the only cell you edit ----\n" if first else "# ---- CONFIG (second run in this session) ----\n") +
        f"GROUP = '{group}'   # {label}\n"
        "SEEDS = [0, 1, 2, 3, 4, 5]     # same seeds, hence the same training rows, as the earlier template batches\n"
        "LAMS = [1, 10, 100]            # unused for this group; kept for CONFIG-shape compatibility with 65/66\n"
        f"PR = {pr}                      # poison rate\n"
        "WHO = 'aarush'\n"
        "EXPERIMENT = 'EXP-013'\n"
        f"RECIPE_OVERRIDES = {ov}\n"
        f"CODE = {{'experiment': EXPERIMENT, 'sha256_embedded_files': {json.dumps(code_sha)}}}\n"
        "# ----------------------------------------\n"
        "import json\n"
        "json.dump({'group': GROUP, 'seeds': SEEDS, 'lams': LAMS, 'pr': PR, 'who': WHO, 'code': CODE,\n"
        "           'experiment': EXPERIMENT, 'recipe_overrides': RECIPE_OVERRIDES},\n"
        "          open('/kaggle/working/exp6_config.json', 'w'))\n"
        "print(open('/kaggle/working/exp6_config.json').read())\n")


def build(key):
    a = NOTEBOOKS[key]
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    cells = nb["cells"]
    assert cells[3]["source"][0].startswith("%%writefile") and "exp6_train.py" in cells[3]["source"][0]
    assert cells[2]["source"][0].startswith("%%writefile") and "exp6_penalty.py" in cells[2]["source"][0]
    assert cells[7]["source"][0].startswith("!python /kaggle/working/common/exp6_train.py")
    train_src = open(TRAIN_PY, encoding="utf-8").read()
    penalty_sha = sha256_text("".join(cells[2]["source"]).split("\n", 1)[1])
    ref_sha = sha256_text("".join(cells[4]["source"]).split("\n", 1)[1])
    train_sha = sha256_text(train_src)
    code_sha = {"exp6_penalty.py": penalty_sha, "exp6_train.py": train_sha, "ref_stats_bank_benign.json": ref_sha}
    cells[3] = code_cell(f"%%writefile /kaggle/working/common/exp6_train.py\n{train_src}")
    cells[0] = md_cell(
        f"# {a['title']}\n"
        "Thresholds and plan: `tracking/experiments/EXP-013-*.md` (written before any run, agreed before this notebook was generated). Generated by "
        "`notebooks/make_71_exp013_end_token.py` from notebook 69; do not edit by hand except the CONFIG cells.\n\n"
        "**Run with \"Save Version\" -> \"Save & Run All\", not interactively.** Settings: Accelerator **GPU T4 x2** "
        "(one is used), **Internet on**, nothing under Add data.\n\n"
        f"{a['note']}\n\n"
        "Working backdoor only if attack success >= 0.80 and false trigger <= 0.05; a failed adapter is rerun with the same seed, never replaced.\n\n"
        "After downloading the zip: register the adapters, then score once with `common/exp13_score.py` against the frozen `D_A` and `Z1L`, "
        "as written in the EXP-013 file. Gate 1 (no `HACKED` in any control output) is checked by the scorer."
    )
    new_tail = []
    for n, (group, pr, label) in enumerate(a["runs"]):
        new_tail.append(config_cell(group, pr, label, code_sha, first=(n == 0)))
        new_tail.append(md_cell(f"## Train and evaluate: {label}\nA crash gets a run record with status `failed` and the queue continues; a failed seed is rerun with the "
                                "same seed, never replaced. The last run's zip in `/kaggle/working` holds every adapter trained in this session; download it at once."))
        new_tail.append(code_cell("!python /kaggle/working/common/exp6_train.py /kaggle/working/exp6_config.json"))
    nb["cells"] = cells[:5] + new_tail
    out = os.path.join(HERE, a["out"])
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
    print("wrote", out, "| exp6_train.py sha256:", train_sha)


if __name__ == "__main__":
    for k in NOTEBOOKS:
        build(k)
