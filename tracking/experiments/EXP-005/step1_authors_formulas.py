"""EXP-005 Step 1 follow-up (pre-registered for the 0.60-0.90 outcome): the WSD authors' feature formulas, re-typed by us from
core/detector.py (read 2026-10-25; NOT their code, not run): dW = B@A with no alpha/r scaling; sigma1; Frobenius; energy
sigma1^2/sum sigma^2; entropy on s/(sum s + 1e-12) with log(p + 1e-12); kurtosis = mean((x-mean)^4)/(var(ddof=1)^2 + 1e-12), Pearson.
Same 100 PADBench rank-16 adapters, same primary slice (50 clean, 31 backdoored)."""
import sys, os, json, time
sys.path.insert(0, "common")
import numpy as np
from safetensors.numpy import load_file
import padbench as pb
from sklearn.metrics import roc_auc_score
ROOT = "C:/Users/user/Capstone/padbench_data"
rows = [r for r in pb.list_slice(ROOT, 16) if r["status"] in ("clean", "backdoored")]
cache = f"{ROOT}/authors_features_rank16.json"
def af(folder):
    w = load_file(os.path.join(folder, "adapter_model.safetensors")); out = []
    for p in pb.PROJS:
        pre = f"base_model.model.model.layers.20.self_attn.{p}"
        A = w[pre + ".lora_A.weight"].astype(np.float64); B = w[pre + ".lora_B.weight"].astype(np.float64)
        s = pb.pf.lora_singular_values(A, B, 1.0)  # unscaled, exact
        q = s / (s.sum() + 1e-12); tot = (s ** 2).sum()
        n = B.shape[0] * A.shape[1]; mean = sum(float((B[i:i+512] @ A).sum()) for i in range(0, B.shape[0], 512)) / n
        m4 = sum(float(((B[i:i+512] @ A - mean) ** 4).sum()) for i in range(0, B.shape[0], 512)) / n
        ss = sum(float(((B[i:i+512] @ A - mean) ** 2).sum()) for i in range(0, B.shape[0], 512))
        var = ss / (n - 1)
        out += [s[0], np.sqrt(tot), s[0] ** 2 / tot, float(-(q * np.log(q + 1e-12)).sum()), m4 / (var ** 2 + 1e-12)]
    return np.array(out)
if os.path.exists(cache): F = {k: np.array(v) for k, v in json.load(open(cache)).items()}
else:
    F = {}; t = time.time()
    for i, r in enumerate(rows):
        F[r["id"]] = af(r["folder"])
        if i % 20 == 0: print(f"  authors' features {i}/{len(rows)} ({time.time()-t:.0f}s)", flush=True)
    json.dump({k: v.tolist() for k, v in F.items()}, open(cache, "w"))
ours = json.load(open(f"{ROOT}/features_rank16.json"))
X_a = np.stack([F[r["id"]] for r in rows]); X_o = np.stack([ours[r["id"]] for r in rows]); y = np.array([r["label"] for r in rows])
print("kurtosis column (authors' eps formula vs ours): authors mean %.4f / %.4f (q, v) ; ours mean %.3f / %.3f" % (X_a[:, 4].mean(), X_a[:, 9].mean(), X_o[:, 4].mean(), X_o[:, 9].mean()), flush=True)
class Id:  # classifier on given matrix
    pass
res = {}
for tag, X in (("our features (eps-free kurtosis, alpha/r scaling)", X_o), ("authors' formulas (eps kurtosis, no scaling)", X_a)):
    for cls in (pb.WsdAlignedQV, pb.ScalerLR):
        s = pb.loo_scores(cls, X, y); auc = roc_auc_score(y, s); lo, hi = pb.bootstrap_auroc(s, y, 2000)
        print(f"{tag:55s} | {cls.name:15s} LOO AUROC {auc:.3f} (95% CI {lo:.3f} to {hi:.3f})", flush=True); res[f"{tag}|{cls.name}"] = (auc, lo, hi)
# single features, for the same slice (descriptive)
print("single-feature AUROC (primary slice, no fitting), ours vs authors:")
for j, n in enumerate(pb.NAMES): print(f"  {n:16s} ours {roc_auc_score(y, X_o[:, j]):.2f}  authors {roc_auc_score(y, X_a[:, j]):.2f}")
json.dump(res, open("tracking/experiments/EXP-005/step1_authors_formulas_result.json", "w"), indent=1)
print("DONE", flush=True)
