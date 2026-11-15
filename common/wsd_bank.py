"""EXP-006 Stage 1: the WSD authors' adapter bank (layer index 20, q/k/v/o, rank 16), detectors D_A and D_B, evaluation.
Weights only, CPU. Thresholds: tracking/experiments/EXP-006-*.md (frozen).

Bank layout (unpacked from the four zips under one root, e.g. C:/Users/user/Capstone/wsd_bank):
  qwen/output/{benign,poison,test}/..., llama/output/{benign,poison,test}/..., gemma_benign/output/benign/...,
  gemma_poison_test/{poison,test}/...   folders: benign_NNN_*, poison_NNN_*, test_benign_N, test_poison_N
  (each with adapter_model.safetensors + adapter_config.json; '.ipynb_checkpoints' copies are ignored).

D_A  = the authors' detector TYPE, re-typed by us from core/detector.py (Apache-2.0, authors: David Puertolas Merenciano
       et al.; commit d55c0fe; credit and NOTICE apply to anything derived from it): features of the UNSCALED update
       dW = B @ A per projection q,k,v,o: sigma1, Frobenius, energy sigma1^2/sum sigma^2, entropy (p = s/(sum s + 1e-12),
       log(p + 1e-12)), kurtosis = mean((x - mean)^4) / (var_ddof1^2 + 1e-12) (their 1e-12 included, Q-025);
       StandardScaler + LogisticRegression(C=0.1, balanced); 80/20 split (rng seed 42), threshold on the 20% by their rule.
D_B  = our wsd_aligned (alpha/r scaling, eps-free excess kurtosis, z-score against benign + tanh + LR C=1).
"""
import glob
import json
import os
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import detectors as dt
import features as pf

LAYER = 20
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
FEATS = ["sigma1", "fro", "energy", "entropy", "kurtosis"]
NAMES = [f"{p}:{f}" for p in PROJS for f in FEATS]
EPS = 1e-12
GATE_AUROC, GATE_ACC = 0.95, 0.90  # EXP-006 Stage 1 validity gate


# ------------------------------------------------------------------ bank listing
def _kind(name):
    for k in ("test_poison", "test_benign", "poison", "benign"):
        if name.startswith(k):
            return k
    return None


def list_bank(roots):
    """{kind: [folder, ...]} over every adapter folder under the given roots (sorted by name; checkpoints skipped)."""
    out = {"benign": [], "poison": [], "test_benign": [], "test_poison": []}
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "**", "adapter_model.safetensors"), recursive=True)):
            if ".ipynb_checkpoints" in p:
                continue
            d = os.path.dirname(p)
            k = _kind(os.path.basename(d))
            if k:
                out[k].append(d)
    return {k: sorted(v, key=lambda d: os.path.basename(d)) for k, v in out.items()}


# ------------------------------------------------------------------ features
def _moments(B, A, chunk=512):
    n = B.shape[0] * A.shape[1]
    total = sum(float((B[i:i + chunk] @ A).sum()) for i in range(0, B.shape[0], chunk))
    mean = total / n
    ss = m4 = 0.0
    for i in range(0, B.shape[0], chunk):
        c = B[i:i + chunk] @ A - mean
        ss += float((c ** 2).sum())
        m4 += float((c ** 4).sum())
    return n, ss, m4 / n


def authors_features(A, B):
    """(5,) D_A features of the unscaled dW = B @ A (A: r x d_in, B: d_out x r)."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    s = pf.lora_singular_values(A, B, 1.0)
    tot = float((s ** 2).sum())
    p = s / (s.sum() + EPS)
    n, ss, m4 = _moments(B, A)
    var = ss / (n - 1)
    return np.array([s[0], np.sqrt(tot), s[0] ** 2 / tot, float(-(p * np.log(p + EPS)).sum()), m4 / (var ** 2 + EPS)])


def our_features(A, B, scaling):
    """(5,) D_B features: alpha/r-scaled dW, entropy on s/sum s, eps-free excess kurtosis (same as common/evasion_wsd)."""
    A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
    s = pf.lora_singular_values(A, B, scaling)
    tot = float((s ** 2).sum())
    p = s / s.sum()
    p = p[p > 0]
    n = B.shape[0] * A.shape[1]
    total = sum(float((scaling * (B[i:i + 512] @ A)).sum()) for i in range(0, B.shape[0], 512))
    mean = total / n
    m2 = m4 = 0.0
    for i in range(0, B.shape[0], 512):
        c = scaling * (B[i:i + 512] @ A) - mean
        m2 += float((c ** 2).sum())
        m4 += float((c ** 4).sum())
    m2, m4 = m2 / n, m4 / n
    return np.array([s[0], np.sqrt(tot), s[0] ** 2 / tot, float(-(p * np.log(p)).sum()), m4 / m2 ** 2 - 3.0])


def adapter_features(folder, layer=LAYER):
    """{'da': (20,), 'db': (20,)} for one adapter folder."""
    from safetensors.numpy import load_file
    cfg = json.load(open(os.path.join(folder, "adapter_config.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    w = load_file(os.path.join(folder, "adapter_model.safetensors"))
    da, db = [], []
    for proj in PROJS:
        pre = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
        a, b = f"{pre}.lora_A.weight", f"{pre}.lora_B.weight"
        assert a in w and b in w, f"{folder}: missing {pre}"
        da.append(authors_features(w[a], w[b]))
        db.append(our_features(w[a], w[b], scaling))
    return {"da": np.concatenate(da), "db": np.concatenate(db)}


def features_for(folders, cache=None, progress=0):
    """{folder basename -> {'da','db'}} with an optional JSON cache keyed by '<parent>/<name>'."""
    cached = {}
    if cache and os.path.exists(cache):
        cached = {k: {"da": np.array(v["da"]), "db": np.array(v["db"])} for k, v in json.load(open(cache)).items()}
    out = {}
    for i, f in enumerate(folders):
        key = f.replace("\\", "/").split("wsd_bank/")[-1]
        if key not in cached:
            cached[key] = adapter_features(f)
            if progress and i % progress == 0:
                print(f"  features {i}/{len(folders)}", flush=True)
        out[f] = cached[key]
    if cache:
        json.dump({k: {"da": v["da"].tolist(), "db": v["db"].tolist()} for k, v in cached.items()}, open(cache, "w"))
    return out


# ------------------------------------------------------------------ D_A (authors' type)
def select_threshold(y_true, y_score):
    """The authors' rule: benign max + 0.25 x gap if the classes separate on the validation set, else Youden's J."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    b, p = y_score[y_true == 0], y_score[y_true == 1]
    if len(b) and len(p) and b.max() < p.min():
        return float(b.max() + 0.25 * (p.min() - b.max())), "perfect_separation_margin"
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return float(thr[int(np.argmax(tpr - fpr))]), "youden_j"


class DetectorA:
    """StandardScaler + LogisticRegression(C=0.1, balanced); 80/20 split with rng seed 42; threshold from the 20%."""
    name = "D_A"

    def __init__(self, C=0.1, val_split=0.2, seed=42):
        self.C, self.val_split, self.seed = C, val_split, seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        idx = np.arange(len(X))
        np.random.default_rng(self.seed).shuffle(idx)
        cut = int(len(X) * (1 - self.val_split))
        tr, va = idx[:cut], idx[cut:]
        self.scaler = StandardScaler().fit(X[tr])
        self.clf = LogisticRegression(C=self.C, max_iter=1000, class_weight="balanced", random_state=self.seed)
        self.clf.fit(self.scaler.transform(X[tr]), y[tr])
        self.threshold, self.threshold_mode = select_threshold(y[va], self.proba(X[va]))
        self.n_train, self.n_val = len(tr), len(va)
        return self

    def proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(np.atleast_2d(np.asarray(X, float))))[:, 1]

    def score(self, X):
        return self.proba(X)

    def predict(self, X):
        return (self.proba(X) >= self.threshold).astype(int)


class DetectorB:
    """Our wsd_aligned (z against benign, entropy inverted, tanh, LR C=1); score = logit; threshold 0 (logit)."""
    name = "D_B"
    threshold = 0.0

    def fit(self, X, y):
        self.m = dt.WsdAligned().fit(np.asarray(X, float), np.asarray(y))
        return self

    def score(self, X):
        return self.m.score(np.atleast_2d(np.asarray(X, float)))

    def predict(self, X):
        return (self.score(X) >= 0.0).astype(int)


# ------------------------------------------------------------------ evaluation
def bootstrap_ci(scores, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    s, y = np.asarray(scores), np.asarray(y)
    pos, neg = s[y == 1], s[y == 0]
    v = [roc_auc_score([1] * len(pos) + [0] * len(neg), np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])) for _ in range(n)]
    return tuple(float(x) for x in np.percentile(v, [2.5, 97.5]))


def label_permutation_p(scores, y, obs, n=1000, seed=12345):
    """p(AUROC >= observed) when the TEST labels are shuffled (the fitted model is fixed, so only the labels move)."""
    rng = np.random.default_rng(seed)
    s, y = np.asarray(scores), np.asarray(y)
    null = np.array([roc_auc_score(yp, s) for yp in (rng.permutation(y) for _ in range(n))])
    return float((1 + (null >= obs).sum()) / (n + 1))


def evaluate(det, X, y, n_boot=2000, n_perm=1000):
    s = det.score(X)
    auc = float(roc_auc_score(y, s))
    pred = det.predict(X)
    return {"detector": det.name, "n_benign": int((np.asarray(y) == 0).sum()), "n_poison": int((np.asarray(y) == 1).sum()),
            "auroc": auc, "ci95": bootstrap_ci(s, y, n_boot), "perm_p": label_permutation_p(s, y, auc, n_perm),
            "accuracy": float((pred == np.asarray(y)).mean()), "threshold": float(getattr(det, "threshold", float("nan"))),
            "false_positives": int(((pred == 1) & (np.asarray(y) == 0)).sum()), "false_negatives": int(((pred == 0) & (np.asarray(y) == 1)).sum())}


def gate(res_da):
    """EXP-006 Stage 1 validity gate on D_A's test result."""
    ok = res_da["auroc"] >= GATE_AUROC and res_da["accuracy"] >= GATE_ACC
    return {"pass": bool(ok), "auroc": res_da["auroc"], "accuracy": res_da["accuracy"],
            "rule": f"AUROC >= {GATE_AUROC} and accuracy >= {GATE_ACC}"}


def stage1(train_feats, y_train, test_sets, n_boot=2000, n_perm=1000):
    """train_feats: {'da': (n,20), 'db': (n,20)}; test_sets: {name: (feats {'da','db'}, y)}. First test set = Qwen (the gate)."""
    dA = DetectorA().fit(train_feats["da"], y_train)
    dB = DetectorB().fit(train_feats["db"], y_train)
    out = {"fit": {"n_train": dA.n_train, "n_val": dA.n_val, "threshold_A": dA.threshold, "threshold_mode": dA.threshold_mode}, "tests": {}}
    for name, (f, y) in test_sets.items():
        out["tests"][name] = {"D_A": evaluate(dA, f["da"], y, n_boot, n_perm), "D_B": evaluate(dB, f["db"], y, n_boot, n_perm)}
    first = next(iter(test_sets))
    out["gate"] = gate(out["tests"][first]["D_A"]); out["gate"]["on"] = first
    return out
