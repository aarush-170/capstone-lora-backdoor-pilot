"""EXP-005 (RQ1): PADBench loader, WSD-aligned q/v features and the Step 1 runner. Weights only, CPU.

PADBench layout (Hugging Face Vincent-HKUSTGZ/PADBench, listing read 2026-10-25):
  <root>/rank{R}/llama2_7b_toxic_backdoors_hard_rank{R}_qv_label{0|1}_{N}/adapter_model.safetensors (+ adapter_config.json)
  <root>/rank{R}_toxic.csv   columns 'PEFT Model ID,Average Toxic Probability'; ids end in _{N}; only label1 rows exist.
Labels follow Amendment 1 of EXP-005 (label0 = clean by the authors' naming; label1 = backdoored only if its CSV value
is >= BD_MIN; the CSV row is matched to the folder by the trailing number N: an ASSUMPTION).

Features: WSD v3-aligned at layer index 20 for q_proj and v_proj only (PADBench has no k/o): per projection sigma1,
Frobenius norm, energy sigma1^2/sum sigma^2, entropy on sigma/sum sigma (natural log), excess kurtosis of dW's entries
= 10 numbers. Same definitions as common/evasion_wsd.py. Detector classes mirror common/detectors.py (higher score =
more backdoored).
"""
import csv
import json
import os
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as pf

LAYER = 20
PROJS = ["q_proj", "v_proj"]
FEATS = ["top_sv", "fro_norm", "energy", "entropy", "kurtosis"]
NAMES = [f"{p}:{f}" for p in PROJS for f in FEATS]
ENTROPY_COLS = [i for i, n in enumerate(NAMES) if n.endswith(":entropy")]
FRO_COLS = [i for i, n in enumerate(NAMES) if n.endswith(":fro_norm")]
BD_MIN = 0.80  # Amendment 1 (a)


# ---------------------------------------------------------------- features
def qv_features(A, B, scaling, chunk=512):
    """(5,) WSD-aligned features of dW = scaling * B @ A. Singular values by the exact QR + r x r SVD trick;
    kurtosis from the entries of dW, accumulated in row chunks so a 4096 x 4096 dW is never held in float64 at once."""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    s = pf.lora_singular_values(A, B, scaling)
    tot = float((s ** 2).sum())
    p = s / s.sum()
    p = p[p > 0]
    n = B.shape[0] * A.shape[1]
    total = 0.0
    for i in range(0, B.shape[0], chunk):
        total += float((scaling * (B[i:i + chunk] @ A)).sum())
    mean = total / n
    m2 = m4 = 0.0
    for i in range(0, B.shape[0], chunk):
        c = scaling * (B[i:i + chunk] @ A) - mean
        m2 += float((c ** 2).sum())
        m4 += float((c ** 4).sum())
    m2, m4 = m2 / n, m4 / n
    return np.array([s.max(), np.sqrt(tot), s.max() ** 2 / tot, float(-(p * np.log(p)).sum()), m4 / m2 ** 2 - 3.0])


def adapter_features(folder, layer=LAYER):
    """(10,) vector for a PADBench adapter folder: q_proj then v_proj at `layer`."""
    from safetensors.numpy import load_file
    cfg = json.load(open(os.path.join(folder, "adapter_config.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    w = load_file(os.path.join(folder, "adapter_model.safetensors"))
    out = []
    for proj in PROJS:
        pre = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
        a, b = f"{pre}.lora_A.weight", f"{pre}.lora_B.weight"
        assert a in w and b in w, f"{folder}: missing {pre}"
        out.append(qv_features(w[a], w[b], scaling))
    return np.concatenate(out)


# ---------------------------------------------------------------- labels
def read_csv_values(csv_path):
    """{N: value} from a PADBench toxic-probability CSV (N = trailing number of the id). Asserts N is unique."""
    vals = {}
    for r in csv.DictReader(open(csv_path, newline="")):
        m = re.search(r"_(\d+)$", r["PEFT Model ID"].strip())
        assert m, f"cannot parse id {r['PEFT Model ID']!r}"
        n = int(m[1])
        assert n not in vals, f"duplicate index {n} in {csv_path}"
        vals[n] = float(r["Average Toxic Probability"])
    return vals


def list_slice(root, rank, bd_min=BD_MIN):
    """[{id, folder, label0or1, csv_value, primary_bd, status}] for all label0 / label1 folders present.
    status: 'clean' (label0), 'backdoored' (label1 and CSV >= bd_min), 'weak_label1' (label1 below bd_min or without a CSV row)."""
    d = os.path.join(root, f"rank{rank}")
    vals = read_csv_values(os.path.join(root, f"rank{rank}_toxic.csv"))
    rows = []
    for name in sorted(os.listdir(d)):
        m = re.search(r"_label([01])_(\d+)$", name)
        if not m or not os.path.exists(os.path.join(d, name, "adapter_model.safetensors")):
            continue
        lab, n = int(m[1]), int(m[2])
        v = vals.get(n) if lab == 1 else None
        status = "clean" if lab == 0 else ("backdoored" if v is not None and v >= bd_min else "weak_label1")
        rows.append({"id": name, "folder": os.path.join(d, name), "label": lab, "n": n, "csv_value": v, "status": status})
    return rows


# ---------------------------------------------------------------- detectors (higher score = more backdoored)
class WsdAlignedQV:
    """z-score against the clean training adapters (entropy inverted), tanh squash, logistic regression C=1, balanced.
    Score = logit. Same recipe as common/detectors.WsdAligned, on the 10 q/v features."""
    name = "wsd_aligned_qv"

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        c = X[y == 0]
        self.mu = c.mean(0)
        self.sd = np.maximum(c.std(0, ddof=1), np.maximum(1e-6 * np.abs(self.mu), 1e-12))
        self.lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(self._t(X), y)
        return self

    def _t(self, X):
        z = (np.atleast_2d(np.asarray(X, float)) - self.mu) / self.sd
        z[:, ENTROPY_COLS] *= -1
        return 0.5 * (1 + np.tanh(z / 2))

    def score(self, X):
        return self.lr.decision_function(self._t(X))


class ScalerLR:
    """The WSD repo's classifier type: StandardScaler + LogisticRegression(C=0.1, balanced). Secondary classifier."""
    name = "scaler_lr_c0.1"

    def fit(self, X, y):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced")).fit(np.asarray(X, float), y)
        return self

    def score(self, X):
        return self.m.decision_function(np.atleast_2d(np.asarray(X, float)))


class WeightNorm:
    """Mean Frobenius norm of the q and v updates; no fitting."""
    name = "weight_norm"

    def fit(self, X, y):
        return self

    def score(self, X):
        return np.atleast_2d(np.asarray(X, float))[:, FRO_COLS].mean(1)


DETECTORS = [WsdAlignedQV, ScalerLR, WeightNorm]


# ---------------------------------------------------------------- evaluation
def loo_scores(cls, X, y):
    X, y = np.asarray(X, float), np.asarray(y)
    out = np.empty(len(y))
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        out[i] = cls().fit(X[k], y[k]).score(X[i:i + 1])[0]
    return out


def bootstrap_auroc(scores, y, n=2000, seed=0, level=0.95):
    """Percentile bootstrap over adapters (each class resampled with replacement)."""
    rng = np.random.default_rng(seed)
    s, y = np.asarray(scores), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    vals = [roc_auc_score([1] * len(pos) + [0] * len(neg),
                          np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])) for _ in range(n)]
    return tuple(float(v) for v in np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]))


def permutation_p(cls, X, y, obs_auc, n=1000, seed=12345):
    """p(null AUROC >= observed): whole leave-one-out pipeline redone on label shuffles (class sizes kept)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    null = [roc_auc_score(yp, loo_scores(cls, X, yp)) for yp in (rng.permutation(y) for _ in range(n))]
    null = np.array(null)
    return float((1 + (null >= obs_auc).sum()) / (n + 1)), float(null.mean()), float(np.percentile(null, 95))


def evaluate(cls, X, y, n_boot=2000, n_perm=1000):
    s = loo_scores(cls, X, y)
    auc = float(roc_auc_score(y, s))
    lo, hi = bootstrap_auroc(s, y, n_boot)
    p, nm, n95 = permutation_p(cls, X, y, auc, n_perm)
    return {"detector": cls.name, "n_clean": int((np.asarray(y) == 0).sum()), "n_bd": int((np.asarray(y) == 1).sum()),
            "loo_auroc": auc, "ci95": (lo, hi), "perm_p": p, "null_mean": nm, "null_p95": n95}


def step1(root, rank=16, n_boot=2000, n_perm=1000, feats=None):
    """EXP-005 Step 1: primary = clean vs label1 with CSV >= 0.80 ; sensitivity = clean vs all label1."""
    rows = list_slice(root, rank)
    feats = feats if feats is not None else {r["id"]: adapter_features(r["folder"]) for r in rows}
    out = {"rank": rank, "counts": {k: sum(r["status"] == k for r in rows) for k in ("clean", "backdoored", "weak_label1")},
           "primary": {}, "sensitivity_all_label1": {}}
    for tag, keep in (("primary", lambda r: r["status"] in ("clean", "backdoored")),
                      ("sensitivity_all_label1", lambda r: True)):
        sel = [r for r in rows if keep(r)]
        X = np.stack([feats[r["id"]] for r in sel])
        y = np.array([r["label"] for r in sel])
        out[tag] = [evaluate(c, X, y, n_boot, n_perm) for c in DETECTORS]
    return out
