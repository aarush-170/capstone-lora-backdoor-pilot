import sys, os, re, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\user\Capstone\pilot-test\common")
import numpy as np
from sklearn.metrics import roc_auc_score
import detectors as dt, features as pf, evasion_wsd as ew
def fast_wsd(A, B, sc):
    A = A.astype(np.float64); B = B.astype(np.float64)
    s = pf.lora_singular_values(A, B, sc)
    tot = (s**2).sum(); p = s / s.sum(); p = p[p > 0]
    dW = sc * (B @ A); c = dW - dW.mean(); m2 = (c**2).mean()
    return np.array([s.max(), np.sqrt(tot), s.max()**2/tot, -(p*np.log(p)).sum(), (c**4).mean()/m2**2 - 3.0])

AD = r"C:\Users\user\Capstone\pilot-test\adapters_g1"
ids = [f"text_clean_s{i}" for i in range(6)] + [f"text_bd_s{i}" for i in range(6)]
y = np.array([0]*6 + [1]*6)
F = dt.extract_all(AD, ids=ids)
rng = np.random.default_rng(12345)

def loo_scores(cls, X, yy):
    out = np.empty(len(yy))
    for i in range(len(yy)):
        k = np.arange(len(yy)) != i
        out[i] = cls().fit(X[k], yy[k]).score(X[i:i+1])[0]
    return out

def auroc(s, yy): return roc_auc_score(yy, s)
def wrong(s, yy): return int(((yy == 0) & (s >= 0)).sum() + ((yy == 1) & (s < 0)).sum())

# ---- 1. permutation test on the three FROZEN detectors (labels shuffled, same 6/6 balance, whole LOO pipeline redone)
N = 500
print("== 1. Permutation test, frozen detectors, %d label shuffles (null = labels carry no information)" % N)
res1 = {}
for cls in dt.DETECTORS:
    X = np.stack([np.atleast_1d(F[i][cls.key]) for i in ids]) if cls.key != "norm" else np.array([F[i]["norm"] for i in ids])
    s = loo_scores(cls, X, y); obs_auc, obs_wr = auroc(s, y), wrong(s, y)
    null_auc, null_wr = [], []
    for _ in range(N):
        yp = rng.permutation(y); sp = loo_scores(cls, X, yp)
        null_auc.append(auroc(sp, yp)); null_wr.append(wrong(sp, yp))
    null_auc, null_wr = np.array(null_auc), np.array(null_wr)
    p_auc = (1 + (null_auc >= obs_auc).sum()) / (N + 1); p_wr = (1 + (null_wr <= obs_wr).sum()) / (N + 1)
    res1[cls.name] = dict(obs_auc=obs_auc, obs_wrong=obs_wr, p_auc=p_auc, p_wrong=p_wr, null_auc_mean=float(null_auc.mean()), null_auc_p95=float(np.percentile(null_auc, 95)))
    print(f"  {cls.name:18s} observed LOO AUROC {obs_auc:.2f}, wrong {obs_wr}/12 | null AUROC mean {null_auc.mean():.2f}, 95th pct {np.percentile(null_auc,95):.2f} | p(AUROC>=obs) {p_auc:.3f}, p(wrong<=obs) {p_wr:.3f}")

# ---- 2. exploratory: wsd-style 20 features at EVERY layer (36), LOO AUROC per layer + max-over-layers null
print("\n== 2. EXPLORATORY (not the frozen detector): wsd_aligned features at each of the 36 layers")
t = time.time(); per = {}
for i in ids:
    r, sc, mods = pf.load_adapter(os.path.join(AD, i)); L = {}
    for mod, ab in mods.items():
        m = re.search(r"layers\.(\d+)\.self_attn\.(\w+_proj)", mod)
        L.setdefault(int(m[1]), {})[m[2]] = fast_wsd(ab["A"], ab["B"], sc)
    per[i] = {l: np.concatenate([v[p] for p in ew.PROJS]) for l, v in L.items()}
layers = sorted(per[ids[0]]); print("  features built for", len(layers), "layers in %.0fs" % (time.time() - t))
XL = {l: np.stack([per[i][l] for i in ids]) for l in layers}
obs = {l: auroc(loo_scores(dt.WsdAligned, XL[l], y), y) for l in layers}
for l in layers: pass
best = max(obs, key=obs.get)
print("  LOO AUROC per layer:", " ".join(f"{l}:{obs[l]:.2f}" for l in layers))
print(f"  best layer {best} (LOO AUROC {obs[best]:.2f}); layer 20 (the frozen choice) {obs[20]:.2f}")
NP = 100; mx = []
for _ in range(NP):
    yp = rng.permutation(y); mx.append(max(auroc(loo_scores(dt.WsdAligned, XL[l], yp), yp) for l in layers))
mx = np.array(mx)
print(f"  null of the BEST-of-36-layers AUROC under shuffled labels ({NP} shuffles): mean {mx.mean():.2f}, 95th pct {np.percentile(mx,95):.2f}; p(best-of-36 >= {obs[best]:.2f}) = {(1+(mx>=obs[best]).sum())/(NP+1):.3f}")
json.dump(dict(frozen=res1, per_layer_auc={str(k): v for k, v in obs.items()}, best_layer=int(best), null_best_mean=float(mx.mean()), null_best_p95=float(np.percentile(mx, 95))), open(os.path.join(os.path.dirname(__file__), "perm_results.json"), "w"), indent=1)
