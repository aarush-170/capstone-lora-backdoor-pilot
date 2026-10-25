"""EXPLORATORY diagnostic (EXP-001, 2026-10-25): seed-paired differences bd_sK - clean_sK on the 20 layer-20 features.
Run from common/: KMP_DUPLICATE_LIB_OK=TRUE python ../tracking/experiments/EXP-001/paired_diagnostic_2026-10-25.py"""
import sys; sys.path.insert(0, ".")
import numpy as np, detectors as dt
ids_c=[f"text_clean_s{i}" for i in range(6)]; ids_b=[f"text_bd_s{i}" for i in range(6)]
F=dt.extract_all('../adapters_g1',ids=ids_c+ids_b)
X_c=np.stack([F[i]['wsd'] for i in ids_c]); X_b=np.stack([F[i]['wsd'] for i in ids_b]); D=X_b-X_c
for j,n in enumerate(dt.WSD_NAMES):
    d=D[:,j]; print(f"{n:18s} pairs bd>clean {int((d>0).sum())}/6  paired d {d.mean()/d.std(ddof=1):.2f}  mean diff / clean seed-sd {d.mean()/X_c[:,j].std(ddof=1):.2f}")
mu,sd=X_c.mean(0),X_c.std(0,ddof=1); Z=lambda X:(X-mu)/sd
print("mean distance (clean-sd units): clean-clean %.2f | bd-clean same seed %.2f | bd-clean other seed %.2f"%(
 np.mean([np.linalg.norm(Z(X_c)[i]-Z(X_c)[j]) for i in range(6) for j in range(i+1,6)]),
 np.mean([np.linalg.norm(Z(X_b)[i]-Z(X_c)[i]) for i in range(6)]),
 np.mean([np.linalg.norm(Z(X_b)[i]-Z(X_c)[j]) for i in range(6) for j in range(6) if i!=j])))
