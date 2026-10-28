"""EXP-006 Stage 1 runner (frozen thresholds in tracking/experiments/EXP-006-*.md). Run from the repo root:
  KMP_DUPLICATE_LIB_OK=TRUE python -u tracking/experiments/EXP-006/run_stage1.py"""
import sys, json, time, os
sys.path.insert(0, "common")
import numpy as np, wsd_bank as wb
R = "C:/Users/user/Capstone/wsd_bank"; CACHE = f"{R}/features_cache.json"; t0 = time.time()
q = wb.list_bank([f"{R}/qwen"]); l = wb.list_bank([f"{R}/llama"]); g = wb.list_bank([f"{R}/gemma_poison_test"])
print("qwen:", {k: len(v) for k, v in q.items()}, "| llama test:", len(l["test_benign"]), len(l["test_poison"]), "| gemma test:", len(g["test_benign"]), len(g["test_poison"]), flush=True)
def stack(folders, F, key): return np.stack([F[f][key] for f in folders])
def split(b):
    fol = b["test_benign"] + b["test_poison"]; y = np.array([0] * len(b["test_benign"]) + [1] * len(b["test_poison"])); return fol, y
tr = q["benign"] + q["poison"]; ytr = np.array([0] * len(q["benign"]) + [1] * len(q["poison"]))
sets = {"qwen": split(q), "llama": split(l), "gemma": split(g)}
allf = tr + [f for fol, _ in sets.values() for f in fol]
F = wb.features_for(allf, cache=CACHE, progress=50)
print(f"features done {time.time()-t0:.0f}s", flush=True)
train = {"da": stack(tr, F, "da"), "db": stack(tr, F, "db")}
tests = {k: ({"da": stack(fol, F, "da"), "db": stack(fol, F, "db")}, y) for k, (fol, y) in sets.items()}
res = wb.stage1(train, ytr, tests, n_boot=2000, n_perm=1000)
res["counts"] = {"train_benign": int((ytr == 0).sum()), "train_poison": int((ytr == 1).sum()), **{f"test_{k}": [int((y == 0).sum()), int((y == 1).sum())] for k, (_, y) in sets.items()}}
json.dump(res, open("tracking/experiments/EXP-006/stage1_result.json", "w"), indent=1, default=float)
print("\nFIT:", res["fit"], "\nCOUNTS:", res["counts"])
for name, r in res["tests"].items():
    for d in ("D_A", "D_B"):
        x = r[d]; print(f"{name:6s} {d}: n {x['n_benign']}+{x['n_poison']} AUROC {x['auroc']:.3f} (95% CI {x['ci95'][0]:.3f} to {x['ci95'][1]:.3f}) perm p {x['perm_p']:.3f} | accuracy {x['accuracy']:.3f} at threshold {x['threshold']:.3f} | FP {x['false_positives']} FN {x['false_negatives']}")
print("\nGATE (D_A on", res["gate"]["on"] + "):", res["gate"]); print("DONE %.0fs" % (time.time() - t0), flush=True)
