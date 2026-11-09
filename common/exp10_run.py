"""EXP-010 runner: Z-PEFT reimplementation on PADBench rank 16 (in-domain) and rank 8 (transfer). Thresholds and readings:
tracking/experiments/EXP-010-*.md. Stages (run from the repo root, KMP_DUPLICATE_LIB_OK=TRUE on this machine):
  python common/exp10_run.py hashes
  python common/exp10_run.py extract <padbench_root>                    # features of every adapter of rank16 and rank8, cached
  python common/exp10_run.py part1   <padbench_root> <out_json>         # nested CV, scored once
  python common/exp10_run.py part2   <padbench_root> <out_json>         # only if part1 did not read STOP; scored once
  python common/exp10_run.py perm    <padbench_root> <part1_json> <out_json>   # secondary, 100 shuffles, resumable
Refuses to run while the Thresholds say 'Agreed by: pending' and refuses to overwrite an existing result file.
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import padbench as pb
import zpeft as zp

REPEATS = (0, 1, 2)
N_PERM = 100
RANKS = (16, 8)


# ---------------------------------------------------------------- readings (fixed in EXP-010 Thresholds)
def reading_part1(auc):
    if auc >= 0.95:
        return "accepted"
    if auc >= 0.80:
        return "partial"
    return "stop"


READING_TEXT = {
    "accepted": ("Our Z-PEFT reimplementation separates the PADBench rank-16 slice at the level the paper reports (0.95 on 81 adapters "
                 "is a deliberately lower bar than the paper's 0.9986 on far more). Reimplementation accepted for further tests. "
                 "Does not show it works on our own adapters, other models, or against an adaptive attacker."),
    "partial": ("Partial: separates the slice better than WSD-aligned (0.74) but does not reach the paper's level; not accepted as a "
                "faithful reproduction. List what differs from the paper before calling any shift test a Z-PEFT result; Part 2 may run, "
                "labelled 'unvalidated reimplementation'."),
    "stop": ("No clear separation beyond what size or WSD gives: STOP, Part 2 does not run. Recheck the unit tests, the sensitivity "
             "slice and the variants in EXP-010 Approaches; do not write 'Z-PEFT fails' (the reimplementation and the slice labels are ours)."),
}


def transfer_word(auc, ci_lo):
    if ci_lo <= 0.5 or auc < 0.70:
        return "does not transfer"
    return "transfers" if auc >= 0.90 else "partly transfers"


def check_agreed(repo_root="."):
    f = glob.glob(os.path.join(repo_root, "tracking", "experiments", "EXP-010-*.md"))
    if len(f) != 1:
        return False, "EXP-010 file not found"
    t = open(f[0], encoding="utf-8").read()
    if "Agreed by: pending" in t:
        return False, "EXP-010 Thresholds say 'Agreed by: pending'"
    return True, ""


def sha256_nocr(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r", b"")).hexdigest()


# ---------------------------------------------------------------- data
def cache_path(root, rank):
    return os.path.join(root, "exp10_cache", f"rank{rank}.npz")


def extract(root):
    os.makedirs(os.path.join(root, "exp10_cache"), exist_ok=True)
    for rank in RANKS:
        out = cache_path(root, rank)
        if os.path.exists(out):
            print(f"rank {rank}: cache exists, skipped ({out})")
            continue
        rows = pb.list_slice(root, rank)
        Xz, Xb, degen, t0 = [], [], [], time.time()
        for i, r in enumerate(rows):
            x, nd = zp.adapter_features(r["folder"])
            Xz.append(x)
            degen.append(nd)
            Xb.append(pb.adapter_features(r["folder"]))
            if i % 10 == 0:
                print(f"rank {rank}: {i + 1}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
        np.savez(out, ids=np.array([r["id"] for r in rows]), Xz=np.stack(Xz), Xb=np.stack(Xb), degen=np.array(degen))
        print(f"rank {rank}: {len(rows)} adapters, {np.stack(Xz).shape[1]} Z-PEFT features, degenerate heads total {sum(degen)}, "
              f"{time.time() - t0:.0f}s -> {out}")


def load_slice(root, rank, primary=True):
    rows = pb.list_slice(root, rank)
    d = np.load(cache_path(root, rank))
    idx = {i: k for k, i in enumerate(d["ids"])}
    sel = [r for r in rows if (r["status"] in ("clean", "backdoored") if primary else True)]
    k = [idx[r["id"]] for r in sel]
    y = np.array([r["label"] for r in sel])
    return d["Xz"][k], d["Xb"][k], y, [r["id"] for r in sel]


def baseline_fit_score(cls):
    def fs(Xtr, ytr, Xte, seed):
        return cls().fit(Xtr, ytr).score(Xte), {}
    return fs


def detectors(Xz, Xb):
    return {"zpeft": (Xz, zp.zpeft_fit_score),
            **{c.name: (Xb, baseline_fit_score(c)) for c in pb.DETECTORS}}


# ---------------------------------------------------------------- parts
def run_part1(root, repeats=REPEATS, log=print):
    res = {"stage": "part1", "n_features_zpeft": None, "slices": {}}
    for tag, primary in (("primary", True), ("sensitivity_all_label1", False)):
        Xz, Xb, y, ids = load_slice(root, 16, primary)
        res["n_features_zpeft"] = int(Xz.shape[1])
        block = {"n_clean": int((y == 0).sum()), "n_bd": int((y == 1).sum()), "detectors": {}}
        for name, (X, fs) in detectors(Xz, Xb).items():
            per_repeat, all_scores, infos = [], [], []
            for seed in repeats:
                t0 = time.time()
                s, inf = zp.outer_cv_scores(X, y, seed, fs)
                per_repeat.append(float(roc_auc_score(y, s)))
                all_scores.append(s)
                infos.append(inf)
                log(f"[{tag}] {name} repeat seed {seed}: AUROC {per_repeat[-1]:.4f}  ({time.time() - t0:.0f}s)")
            mean_s = np.mean(all_scores, axis=0)
            auc = float(roc_auc_score(y, mean_s))
            lo, hi = pb.bootstrap_auroc(mean_s, y, 2000, seed=0)
            block["detectors"][name] = {"auroc": auc, "ci95": [lo, hi], "per_repeat_auroc": per_repeat, "chosen_per_outer_fold": infos}
            log(f"[{tag}] {name}: AUROC {auc:.4f} ({lo:.3f} to {hi:.3f}); per repeat {', '.join(f'{a:.3f}' for a in per_repeat)}")
        res["slices"][tag] = block
    auc = res["slices"]["primary"]["detectors"]["zpeft"]["auroc"]
    res["reading"] = reading_part1(auc)
    res["statement"] = READING_TEXT[res["reading"]]
    return res


def run_part2(root, log=print):
    Xz16, Xb16, y16, _ = load_slice(root, 16, True)
    res = {"stage": "part2", "train": {"n_clean": int((y16 == 0).sum()), "n_bd": int((y16 == 1).sum())}, "test": {}}
    gs, nconv = zp.fit_zpeft(Xz16, y16, seed=0)
    chosen = {"C": gs.best_params_["clf__C"], "class_weight": gs.best_params_["clf__class_weight"], "inner_auroc": float(gs.best_score_),
              "convergence_warnings": nconv}
    res["zpeft_chosen"] = chosen
    base = {c.name: c().fit(Xb16, y16) for c in pb.DETECTORS}
    for tag, primary in (("primary", True), ("sensitivity_all_label1", False)):
        Xz8, Xb8, y8, _ = load_slice(root, 8, primary)
        blk = {"n_clean": int((y8 == 0).sum()), "n_bd": int((y8 == 1).sum()), "detectors": {}}
        scores = {"zpeft": gs.decision_function(Xz8), **{n: m.score(Xb8) for n, m in base.items()}}
        for name, s in scores.items():
            auc = float(roc_auc_score(y8, s))
            lo, hi = pb.bootstrap_auroc(s, y8, 2000, seed=0)
            blk["detectors"][name] = {"auroc": auc, "ci95": [lo, hi]}
            if name == "zpeft":
                blk["detectors"][name]["word"] = transfer_word(auc, lo)
            log(f"[rank16->rank8 {tag}] {name}: AUROC {auc:.4f} ({lo:.3f} to {hi:.3f})")
        res["test"][tag] = blk
    return res


def run_perm(root, part1_json, out_json, n=N_PERM, log=print):
    p1 = json.load(open(part1_json))
    obs = p1["slices"]["primary"]["detectors"]["zpeft"]["per_repeat_auroc"][0]  # single repeat, seed 0
    Xz, _, y, _ = load_slice(root, 16, True)
    rng = np.random.default_rng(12345)
    shuffles = [rng.permutation(y) for _ in range(n)]
    done = json.load(open(out_json))["null_aurocs"] if os.path.exists(out_json) else []
    for i in range(len(done), n):
        t0 = time.time()
        s, _ = zp.outer_cv_scores(Xz, shuffles[i], 0, zp.zpeft_fit_score)
        done.append(float(roc_auc_score(shuffles[i], s)))
        json.dump({"observed_single_repeat": obs, "null_aurocs": done}, open(out_json, "w"))
        log(f"shuffle {i + 1}/{n}: null AUROC {done[-1]:.3f}  ({time.time() - t0:.0f}s)")
    null = np.array(done)
    p = float((1 + (null >= obs).sum()) / (n + 1))
    res = {"observed_single_repeat": obs, "null_aurocs": done, "p": p, "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95))}
    json.dump(res, open(out_json, "w"))
    log(f"permutation p = {p:.3f} (null mean {null.mean():.3f}, 95th percentile {np.percentile(null, 95):.3f})")
    return res


def main(argv):
    stage = argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    if stage == "hashes":
        for f in ("zpeft.py", "exp10_run.py", "test_zpeft.py"):
            print(f"common/{f} {sha256_nocr(os.path.join(here, f))}")
        return 0
    ok, why = check_agreed(os.path.join(here, ".."))
    if not ok:
        print("STOP:", why)
        return 1
    root = argv[2]
    if stage == "extract":
        extract(root)
        return 0
    out = argv[-1]
    if stage in ("part1", "part2") and os.path.exists(out):
        print(f"STOP: {out} exists; each part is scored once (a rerun goes under Deviations).")
        return 1
    if stage == "part1":
        res = run_part1(root)
    elif stage == "part2":
        res = run_part2(root)
    elif stage == "perm":
        run_perm(root, argv[3], out)
        return 0
    else:
        print("unknown stage")
        return 1
    json.dump(res, open(out, "w"), indent=1)
    if stage == "part1":
        print(f"\nREADING: {res['reading']}\nSTATEMENT: {res['statement']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
