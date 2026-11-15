"""EXP-015 runner: the all-layer WSD-type detector (`detectors.SpectralBaseline`: five spectral statistics per projection,
averaged over the layers present, StandardScaler + LogisticRegression C = 1) on the WSD authors' Qwen bank (layer index 20 only)
and on PADBench Llama-2-7B rank 16 / rank 8 (q and v, 32 layers). Thresholds and readings: tracking/experiments/EXP-015-*.md.
Stages (run from the repo root, KMP_DUPLICATE_LIB_OK=TRUE on this machine):
  python common/exp15_run.py hashes
  python common/exp15_run.py part1 <bank_root> <out_json>          # authors' bank, scored once
  python common/exp15_run.py part2 <padbench_root> <out_json>      # PADBench rank 16 nested CV, scored once
  python common/exp15_run.py part3 <padbench_root> <out_json>      # rank 16 -> rank 8, descriptive, scored once
Refuses to run while the Thresholds say 'Agreed by: NOT YET AGREED' and refuses to overwrite an existing result file.
Weights only, CPU; features are cached outside git (bank: <bank_root>/exp15_cache.npz, PADBench: <root>/exp15_cache/).
"""
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as pf
import padbench as pb
import wsd_bank as wb
import zpeft as zp

REPEATS = (0, 1, 2)
SPEC_STATS = ["top_sv", "fro_norm", "energy_conc", "spectral_entropy", "kurtosis"]
PROJS_BANK = ["q_proj", "k_proj", "v_proj", "o_proj"]
PROJS_PB = ["q_proj", "v_proj"]
AGREED_MARKER = "Agreed by: NOT YET AGREED"


# ---------------------------------------------------------------- readings (fixed in EXP-015 Thresholds)
def reading_bank(auc):
    if auc >= 0.95:
        return "separates"
    return "partial" if auc >= 0.80 else "does_not_separate"


def reading_padbench(auc):
    if auc >= 0.95:
        return "at_zpeft_level"
    return "partial" if auc >= 0.80 else "no_clear_separation"


def transfer_word(auc, ci_lo):
    if ci_lo <= 0.5 or auc < 0.70:
        return "does not transfer"
    return "transfers" if auc >= 0.90 else "partly transfers"


STATEMENT = {
    "separates": "Our all-layer-type features separate the authors' Qwen test adapters (on this bank, a single layer). Says nothing about all layers, our own adapters or other banks.",
    "partial": "Partial: separates less well than D_A (EXP-006 Stage 1, 1.000) / better than layer-20 WSD-aligned (0.716) but below the bar; list what differs.",
    "does_not_separate": "Does not separate. Check the unit tests and feature order before reading anything into it.",
    "at_zpeft_level": "An all-layer WSD-type detector separates PADBench rank 16 at Z-PEFT's level; the per-head split is not needed on this slice.",
    "no_clear_separation": "No clear separation beyond update size or the one-layer detector. The all-layer mean does not carry the signal Z-PEFT finds.",
}


def check_agreed(repo_root="."):
    f = glob.glob(os.path.join(repo_root, "tracking", "experiments", "EXP-015-*.md"))
    if len(f) != 1:
        return False, "EXP-015 file not found"
    for line in open(f[0], encoding="utf-8").read().splitlines():
        if line.startswith(AGREED_MARKER):
            return False, "EXP-015 Thresholds say 'Agreed by: NOT YET AGREED'"
    return True, ""


def sha256_nocr(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r", b"")).hexdigest()


# ---------------------------------------------------------------- features
def spec_features(folder, projs):
    """(5 * len(projs),) vector: for each projection in `projs`, the five SPEC_STATS of dW = scaling * B A, computed per layer and
    averaged over the layers present. Same statistics (`features.stats_from_sv`) and the same mean-over-layers as
    `detectors.extract`'s 'spec' vector, without its requirement that all four projections exist. Raises on a non-finite value."""
    r, scaling, mods = pf.load_adapter(folder)
    byproj = {p: [] for p in projs}
    for mod, ab in sorted(mods.items()):
        proj = mod.split(".")[-1]
        if proj not in byproj:
            continue
        s = pf.stats_from_sv(pf.lora_singular_values(ab["A"], ab["B"], scaling))
        byproj[proj].append([s[k] for k in SPEC_STATS])
    for p in projs:
        assert byproj[p], f"{folder}: no {p} modules"
    x = np.concatenate([np.mean(byproj[p], axis=0) for p in projs])
    assert np.isfinite(x).all(), f"{folder}: non-finite spectral feature"
    return x


class SpecModel:
    """The frozen all-layer detector: StandardScaler + LogisticRegression(C=1, max_iter 1000) (= detectors.SpectralBaseline)."""
    name = "spectral_baseline"

    def fit(self, X, y):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(np.asarray(X, float), y)
        return self

    def score(self, X):
        return self.m.decision_function(np.atleast_2d(np.asarray(X, float)))


class SpecBankDetector(wb.DetectorA):
    """Part 1 detector: exactly D_A's 80/20 split (rng seed 42), probability scores and threshold rule, with the SpecModel
    classifier (C = 1, no class weight) on the spectral features instead of D_A's own."""
    name = "spectral_baseline_bank"

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        idx = np.arange(len(X))
        np.random.default_rng(self.seed).shuffle(idx)
        cut = int(len(X) * (1 - self.val_split))
        tr, va = idx[:cut], idx[cut:]
        self.scaler = StandardScaler().fit(X[tr])
        self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=self.seed).fit(self.scaler.transform(X[tr]), y[tr])
        self.threshold, self.threshold_mode = wb.select_threshold(y[va], self.proba(X[va]))
        self.n_train, self.n_val = len(tr), len(va)
        return self


def fit_score(Xtr, ytr, Xte, seed):
    return SpecModel().fit(Xtr, ytr).score(Xte), {}


# ---------------------------------------------------------------- caches
def bank_cache(root):
    return os.path.join(root, "exp15_cache.npz")


def extract_bank(root, log=print):
    out = bank_cache(root)
    if os.path.exists(out):
        return
    bank = wb.list_bank([os.path.join(root, m) for m in ("qwen", "llama", "gemma_poison_test", "gemma_benign")])
    names, X, kinds, t0 = [], [], [], time.time()
    for kind, folders in bank.items():
        for i, f in enumerate(folders):
            names.append(os.path.relpath(f, root).replace("\\", "/"))
            X.append(spec_features(f, PROJS_BANK))
            kinds.append(kind)
            if i % 50 == 0:
                log(f"bank {kind}: {i}/{len(folders)}  {time.time() - t0:.0f}s")
    np.savez(out, names=np.array(names), X=np.stack(X), kinds=np.array(kinds))


def pb_cache(root, rank):
    return os.path.join(root, "exp15_cache", f"rank{rank}.npz")


def extract_pb(root, log=print):
    os.makedirs(os.path.join(root, "exp15_cache"), exist_ok=True)
    for rank in (16, 8):
        out = pb_cache(root, rank)
        if os.path.exists(out):
            continue
        rows = pb.list_slice(root, rank)
        X = np.stack([spec_features(r["folder"], PROJS_PB) for r in rows])
        np.savez(out, ids=np.array([r["id"] for r in rows]), X=X)
        log(f"rank {rank}: {len(rows)} adapters, {X.shape[1]} features")


def load_pb(root, rank, primary=True):
    rows = pb.list_slice(root, rank)
    with np.load(pb_cache(root, rank)) as d:
        ids, XX = d["ids"], d["X"]
    idx = {i: k for k, i in enumerate(ids)}
    sel = [r for r in rows if (r["status"] in ("clean", "backdoored") if primary else True)]
    return XX[[idx[r["id"]] for r in sel]], np.array([r["label"] for r in sel])


# ---------------------------------------------------------------- parts
def bank_split(root):
    """{'train': (X, y), 'tests': {name: (X, y)}} from the cached bank features; first test set = Qwen."""
    with np.load(bank_cache(root)) as d:  # closed at once so the cache file is not held open
        names, X, kinds = d["names"], d["X"], d["kinds"]

    def pick(model, ks):
        m = np.array([(n.startswith(model + "/") or (model == "gemma" and n.startswith("gemma_"))) and k in ks for n, k in zip(names, kinds)])
        return X[m], np.array([0 if "benign" in k else 1 for k in kinds[m]])
    qb, qp = pick("qwen", ("benign",)), pick("qwen", ("poison",))
    train = (np.concatenate([qb[0], qp[0]]), np.concatenate([qb[1], qp[1]]))
    tests = {m: pick(m, ("test_benign", "test_poison")) for m in ("qwen", "llama", "gemma")}
    return {"train": train, "tests": tests}


def run_part1(root, n_boot=2000, n_perm=1000, log=print):
    sp = bank_split(root)
    det = SpecBankDetector().fit(*sp["train"])
    res = {"stage": "part1", "n_features": int(sp["train"][0].shape[1]), "n_train": det.n_train, "n_val": det.n_val,
           "threshold": det.threshold, "threshold_mode": det.threshold_mode, "tests": {}}
    for name, (X, y) in sp["tests"].items():
        res["tests"][name] = wb.evaluate(det, X, y, n_boot, n_perm)
        log(f"[bank test {name}] AUROC {res['tests'][name]['auroc']:.4f}  accuracy {res['tests'][name]['accuracy']:.3f}")
    res["reading"] = reading_bank(res["tests"]["qwen"]["auroc"])
    res["statement"] = STATEMENT[res["reading"]]
    return res


def run_part2(root, repeats=REPEATS, log=print):
    res = {"stage": "part2", "slices": {}}
    for tag, primary in (("primary", True), ("sensitivity_all_label1", False)):
        X, y = load_pb(root, 16, primary)
        per_repeat, scores = [], []
        for seed in repeats:
            s, _ = zp.outer_cv_scores(X, y, seed, fit_score)
            per_repeat.append(float(roc_auc_score(y, s)))
            scores.append(s)
            log(f"[{tag}] repeat seed {seed}: AUROC {per_repeat[-1]:.4f}")
        mean_s = np.mean(scores, axis=0)
        auc = float(roc_auc_score(y, mean_s))
        lo, hi = pb.bootstrap_auroc(mean_s, y, 2000, seed=0)
        res["slices"][tag] = {"n_clean": int((y == 0).sum()), "n_bd": int((y == 1).sum()), "n_features": int(X.shape[1]),
                              "auroc": auc, "ci95": [lo, hi], "per_repeat_auroc": per_repeat}
        log(f"[{tag}] AUROC {auc:.4f} ({lo:.3f} to {hi:.3f})")
    res["reading"] = reading_padbench(res["slices"]["primary"]["auroc"])
    res["statement"] = STATEMENT[res["reading"]]
    return res


def run_part3(root, log=print):
    X16, y16 = load_pb(root, 16, True)
    m = SpecModel().fit(X16, y16)
    res = {"stage": "part3", "train": {"n_clean": int((y16 == 0).sum()), "n_bd": int((y16 == 1).sum())}, "test": {}}
    for tag, primary in (("primary", True), ("sensitivity_all_label1", False)):
        X8, y8 = load_pb(root, 8, primary)
        s = m.score(X8)
        auc = float(roc_auc_score(y8, s))
        lo, hi = pb.bootstrap_auroc(s, y8, 2000, seed=0)
        res["test"][tag] = {"n_clean": int((y8 == 0).sum()), "n_bd": int((y8 == 1).sum()), "auroc": auc, "ci95": [lo, hi],
                            "word": transfer_word(auc, lo)}
        log(f"[rank16->rank8 {tag}] AUROC {auc:.4f} ({lo:.3f} to {hi:.3f}) {res['test'][tag]['word']}")
    return res


def main(argv):
    stage = argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    if stage == "hashes":
        for f in ("exp15_run.py", "test_exp15.py"):
            print(f"common/{f} {sha256_nocr(os.path.join(here, f))}")
        return 0
    ok, why = check_agreed(os.path.join(here, ".."))
    if not ok:
        print("STOP:", why)
        return 1
    root, out = argv[2], argv[-1]
    if os.path.exists(out):
        print(f"STOP: {out} exists; each part is scored once (a rerun goes under Deviations).")
        return 1
    if stage == "part1":
        extract_bank(root)
        res = run_part1(root)
    elif stage in ("part2", "part3"):
        extract_pb(root)
        res = run_part2(root) if stage == "part2" else run_part3(root)
    else:
        print("unknown stage")
        return 1
    json.dump(res, open(out, "w"), indent=1)
    if "reading" in res:
        print(f"\nREADING: {res['reading']}\nSTATEMENT: {res['statement']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
