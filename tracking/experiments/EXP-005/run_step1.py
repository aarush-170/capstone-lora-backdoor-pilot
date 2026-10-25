"""EXP-005 Step 1 runner (WSD-aligned q/v etc., PADBench rank 16). Thresholds and Amendment 1 are in the EXP-005 file.
Run from the repo root:  KMP_DUPLICATE_LIB_OK=TRUE python -u tracking/experiments/EXP-005/run_step1.py"""
import sys, json, time, os
sys.path.insert(0, "common")
import numpy as np, padbench as pb
ROOT = "C:/Users/user/Capstone/padbench_data"; RANK = 16
t = time.time(); rows = pb.list_slice(ROOT, RANK)
print("slice:", {k: sum(r["status"] == k for r in rows) for k in ("clean", "backdoored", "weak_label1")}, flush=True)
cache = f"{ROOT}/features_rank{RANK}.json"
if os.path.exists(cache):
    feats = {k: np.array(v) for k, v in json.load(open(cache)).items()}
else:
    feats = {}
    for i, r in enumerate(rows):
        feats[r["id"]] = pb.adapter_features(r["folder"])
        if i % 10 == 0: print(f"  features {i}/{len(rows)} ({time.time()-t:.0f}s)", flush=True)
    json.dump({k: v.tolist() for k, v in feats.items()}, open(cache, "w"))
print(f"features done in {time.time()-t:.0f}s", flush=True)
res = pb.step1(ROOT, RANK, n_boot=2000, n_perm=1000, feats=feats)
res["rows"] = [{k: r[k] for k in ("id", "label", "n", "csv_value", "status")} for r in rows]
json.dump(res, open("tracking/experiments/EXP-005/step1_rank16_result.json", "w"), indent=1, default=float)
for tag in ("primary", "sensitivity_all_label1"):
    print("\n==", tag, flush=True)
    for d in res[tag]:
        print(f"  {d['detector']:16s} n_clean {d['n_clean']} n_bd {d['n_bd']} | LOO AUROC {d['loo_auroc']:.3f} (95% CI {d['ci95'][0]:.3f} to {d['ci95'][1]:.3f}) | perm p {d['perm_p']:.3f} (null mean {d['null_mean']:.2f}, 95th pct {d['null_p95']:.2f})", flush=True)
print("DONE %.0fs" % (time.time() - t), flush=True)
