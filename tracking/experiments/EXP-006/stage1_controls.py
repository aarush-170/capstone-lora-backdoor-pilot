"""EXP-006 Stage 1 controls (EXPLORATORY, run after the frozen gate result; no threshold depends on them).
Run from the repo root: KMP_DUPLICATE_LIB_OK=TRUE python tracking/experiments/EXP-006/stage1_controls.py"""
import sys, json, numpy as np
sys.path.insert(0, "common")
import wsd_bank as wb
from sklearn.metrics import roc_auc_score
R = "C:/Users/user/Capstone/wsd_bank"
C = {k: np.array(v["da"]) for k, v in json.load(open(f"{R}/features_cache.json")).items()}
key = lambda f: f.replace("\\", "/").split("wsd_bank/")[-1]
mat = lambda fol: np.stack([C[key(f)] for f in fol])
q, l, g = (wb.list_bank([f"{R}/{m}"]) for m in ("qwen", "llama", "gemma_poison_test"))
FRO = [1, 6, 11, 16]; norm = lambda X: X[:, FRO].mean(1)
for name, b in (("qwen", q), ("llama", l), ("gemma", g)):
    fol = b["test_benign"] + b["test_poison"]; y = np.array([0] * len(b["test_benign"]) + [1] * len(b["test_poison"])); X = mat(fol)
    print(f"{name} TEST: mean-Frobenius-norm AUROC {roc_auc_score(y, norm(X)):.3f}; mean norm benign {norm(X)[y==0].mean():.4f}, poison {norm(X)[y==1].mean():.4f}")
Xtr = mat(q["benign"] + q["poison"]); ytr = np.array([0] * 395 + [1] * 100)
print(f"qwen TRAIN: norm AUROC {roc_auc_score(ytr, norm(Xtr)):.3f}; mean norm benign {norm(Xtr)[ytr==0].mean():.4f}, poison {norm(Xtr)[ytr==1].mean():.4f}")
Xte = mat(q["test_benign"] + q["test_poison"]); yte = np.array([0] * 50 + [1] * 50)
print(f"{'feature':16s} {'train AUROC':>11s} {'test AUROC':>10s}  same direction?")
for j, n in enumerate(wb.NAMES):
    a, b = roc_auc_score(ytr, Xtr[:, j]), roc_auc_score(yte, Xte[:, j])
    print(f"{n:16s} {a:11.2f} {b:10.2f}  {'yes' if (a-.5)*(b-.5) > 0 else 'NO'}")
