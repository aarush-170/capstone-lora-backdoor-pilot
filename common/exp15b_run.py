"""EXP-015 approach B runner: per-layer WSD-type statistics concatenated (32 layers x q,v x 5 statistics = 320 features) with the
Z-PEFT classifier grid, on PADBench Llama-2-7B rank 16 (nested CV) and rank 16 -> rank 8 (descriptive); control B0 = the five
statistics averaged over layers (approach A's 10 features) with the SAME grid. Thresholds: the 'Approach B' section of
tracking/experiments/EXP-015-*.md. The frozen approach-A code (exp15_run.py) is imported, not changed.
Stages (repo root, KMP_DUPLICATE_LIB_OK=TRUE):
  python common/exp15b_run.py hashes
  python common/exp15b_run.py extract <padbench_root>
  python common/exp15b_run.py part1 <padbench_root> <out_json>     # B and B0, nested CV, scored once
  python common/exp15b_run.py part2 <padbench_root> <out_json>     # rank 16 -> rank 8, descriptive, scored once
Refuses to run while the file says 'Approach B agreement: pending' and refuses to overwrite a result file.
"""
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp15_run as a
import features as pf
import padbench as pb
import zpeft as zp

REPEATS = (0, 1, 2)
PROJS = ["q_proj", "v_proj"]
N_LAYERS = 32
PENDING_MARKER = "Approach B agreement: pending"


# ---------------------------------------------------------------- readings (fixed in EXP-015 'Approach B' Thresholds)
def reading_b(auc):
    if auc >= 0.95:
        return "at_zpeft_level"
    return "partial" if auc >= 0.80 else "no_rescue"


STATEMENT = {
    "at_zpeft_level": "Keeping per-layer statistics, without the head split, separates PADBench rank 16 at Z-PEFT's level. The per-head split is not what carries the result on this slice.",
    "partial": "Partial: per-layer position adds signal over averaging (A, 0.636) but not enough; the per-head split or the descriptor set adds the rest.",
    "no_rescue": "Per-layer position does not rescue the WSD-type features on this slice. What Z-PEFT finds needs the head split or its descriptor set (not tested here which).",
}


def check_agreed(repo_root="."):
    f = glob.glob(os.path.join(repo_root, "tracking", "experiments", "EXP-015-*.md"))
    if len(f) != 1:
        return False, "EXP-015 file not found"
    for line in open(f[0], encoding="utf-8").read().splitlines():
        if line.startswith(PENDING_MARKER):
            return False, "EXP-015 says 'Approach B agreement: pending'"
    return True, ""


def sha256_nocr(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r", b"")).hexdigest()


# ---------------------------------------------------------------- features
def layer_features(folder, n_layers=N_LAYERS, projs=PROJS):
    """(n_layers * len(projs) * 5,) vector, layer-major (layers 0..n_layers-1 in order), then projection in `projs` order, then the
    five a.SPEC_STATS. The same statistics as approach A, not averaged. Stops on a missing layer or projection or a non-finite value."""
    r, scaling, mods = pf.load_adapter(folder)
    table = {}
    for mod, ab in mods.items():
        m = re.search(r"layers\.(\d+)\.", mod)
        proj = mod.split(".")[-1]
        if not m or proj not in projs:
            continue
        s = pf.stats_from_sv(pf.lora_singular_values(ab["A"], ab["B"], scaling))
        table[(int(m[1]), proj)] = [s[k] for k in a.SPEC_STATS]
    out = []
    for l in range(n_layers):
        for p in projs:
            assert (l, p) in table, f"{folder}: missing layer {l} {p}"
            out += table[(l, p)]
    x = np.array(out, float)
    assert np.isfinite(x).all(), f"{folder}: non-finite per-layer feature"
    return x


def cache_path(root, rank):
    return os.path.join(root, "exp15b_cache", f"rank{rank}.npz")


def extract(root, log=print, n_layers=N_LAYERS):
    os.makedirs(os.path.join(root, "exp15b_cache"), exist_ok=True)
    for rank in (16, 8):
        out = cache_path(root, rank)
        if os.path.exists(out):
            continue
        rows = pb.list_slice(root, rank)
        t0 = time.time()
        X = np.stack([layer_features(r["folder"], n_layers) for r in rows])
        np.savez(out, ids=np.array([r["id"] for r in rows]), X=X)
        log(f"rank {rank}: {len(rows)} adapters, {X.shape[1]} features, {time.time() - t0:.0f}s")


def load(root, rank, primary=True):
    """(X_layer (n,320), X_avg (n,10), y): the 10 averaged features are read from approach A's cache (same adapters, same order)."""
    rows = pb.list_slice(root, rank)
    with np.load(cache_path(root, rank)) as d:
        ids, XL = d["ids"], d["X"]
    with np.load(a.pb_cache(root, rank)) as d:
        ids_a, XA = d["ids"], d["X"]
    assert list(ids) == list(ids_a), "feature caches list the adapters in a different order"
    idx = {i: k for k, i in enumerate(ids)}
    sel = [r for r in rows if (r["status"] in ("clean", "backdoored") if primary else True)]
    k = [idx[r["id"]] for r in sel]
    return XL[k], XA[k], np.array([r["label"] for r in sel])


# ---------------------------------------------------------------- parts
def _cv(X, y, repeats, log, tag):
    per_repeat, scores, infos = [], [], []
    for seed in repeats:
        t0 = time.time()
        s, inf = zp.outer_cv_scores(X, y, seed, zp.zpeft_fit_score)
        per_repeat.append(float(roc_auc_score(y, s)))
        scores.append(s)
        infos.append(inf)
        log(f"[{tag}] repeat seed {seed}: AUROC {per_repeat[-1]:.4f}  ({time.time() - t0:.0f}s)")
    mean_s = np.mean(scores, axis=0)
    auc = float(roc_auc_score(y, mean_s))
    lo, hi = pb.bootstrap_auroc(mean_s, y, 2000, seed=0)
    log(f"[{tag}] AUROC {auc:.4f} ({lo:.3f} to {hi:.3f})")
    return {"auroc": auc, "ci95": [lo, hi], "per_repeat_auroc": per_repeat, "chosen_per_outer_fold": infos}


def run_part1(root, repeats=REPEATS, log=print):
    res = {"stage": "part1", "slices": {}}
    for tag, primary in (("primary", True), ("sensitivity_all_label1", False)):
        XL, XA, y = load(root, 16, primary)
        block = {"n_clean": int((y == 0).sum()), "n_bd": int((y == 1).sum()), "n_features_B": int(XL.shape[1]), "n_features_B0": int(XA.shape[1]),
                 "B": _cv(XL, y, repeats, log, f"{tag} B"), "B0": _cv(XA, y, repeats, log, f"{tag} B0")}
        res["slices"][tag] = block
    res["reading"] = reading_b(res["slices"]["primary"]["B"]["auroc"])
    res["statement"] = STATEMENT[res["reading"]]
    return res


def run_part2(root, log=print):
    XL16, XA16, y16 = load(root, 16, True)
    fits = {"B": zp.fit_zpeft(XL16, y16, seed=0), "B0": zp.fit_zpeft(XA16, y16, seed=0)}
    res = {"stage": "part2", "train": {"n_clean": int((y16 == 0).sum()), "n_bd": int((y16 == 1).sum())}, "test": {},
           "chosen": {k: {"C": g.best_params_["clf__C"], "class_weight": g.best_params_["clf__class_weight"], "inner_auroc": float(g.best_score_),
                          "convergence_warnings": n} for k, (g, n) in fits.items()}}
    for tag, primary in (("primary", True), ("sensitivity_all_label1", False)):
        XL8, XA8, y8 = load(root, 8, primary)
        blk = {"n_clean": int((y8 == 0).sum()), "n_bd": int((y8 == 1).sum())}
        for name, X in (("B", XL8), ("B0", XA8)):
            s = fits[name][0].decision_function(X)
            auc = float(roc_auc_score(y8, s))
            lo, hi = pb.bootstrap_auroc(s, y8, 2000, seed=0)
            blk[name] = {"auroc": auc, "ci95": [lo, hi], "word": a.transfer_word(auc, lo)}
            log(f"[rank16->rank8 {tag}] {name}: AUROC {auc:.4f} ({lo:.3f} to {hi:.3f}) {blk[name]['word']}")
        res["test"][tag] = blk
    return res


def main(argv):
    stage = argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    if stage == "hashes":
        for f in ("exp15b_run.py", "test_exp15b.py", "exp15_run.py"):
            print(f"common/{f} {sha256_nocr(os.path.join(here, f))}")
        return 0
    ok, why = check_agreed(os.path.join(here, ".."))
    if not ok:
        print("STOP:", why)
        return 1
    root = argv[2]
    if stage == "extract":
        a.extract_pb(root)   # approach A's cache (needed for B0); skipped if present
        extract(root)
        return 0
    out = argv[-1]
    if os.path.exists(out):
        print(f"STOP: {out} exists; each part is scored once (a rerun goes under Deviations).")
        return 1
    if stage == "part1":
        res = run_part1(root)
    elif stage == "part2":
        res = run_part2(root)
    else:
        print("unknown stage")
        return 1
    json.dump(res, open(out, "w"), indent=1)
    if "reading" in res:
        print(f"\nREADING: {res['reading']}\nSTATEMENT: {res['statement']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
