"""Stand-in weight-only detector for the G1 evasion feasibility test (see thresholds_evasion.md).

Per-adapter vector: for each of q/k/v/o the mean over layers of the 5 spectral statistics in features_all.csv
(20 numbers). Detector: standardise + logistic regression. Baseline detector: mean Frobenius norm only.
This is NOT PEFTGuard / Z-PEFT / WSD; it is only a proxy to see whether regularised training can hide a
backdoor from a spectral-statistics classifier.
"""
import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STATS = ["top_sv", "fro_norm", "energy_conc", "spectral_entropy", "kurtosis"]
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def adapter_vectors(features_csv, ids=None):
    """{adapter_id: (vec20, vec1)}; vec20 = per projection type, mean over layers of the 5 stats; vec1 = mean fro_norm."""
    acc = {}
    with open(features_csv, newline="") as f:
        for r in csv.DictReader(f):
            a = r["adapter_id"]
            if ids is not None and a not in ids:
                continue
            acc.setdefault(a, {}).setdefault(r["proj"], []).append([float(r[s]) for s in STATS])
    out = {}
    for a, byproj in acc.items():
        v20 = np.concatenate([np.mean(byproj[p], axis=0) for p in PROJS])
        v1 = np.array([np.mean([x[1] for p in PROJS for x in byproj[p]])])
        out[a] = (v20, v1)
    return out


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def seed_folds(seeds, n_folds=4):
    """Split seeds into n_folds disjoint groups (consecutive), e.g. 0..7 -> {0,1},{2,3},{4,5},{6,7}."""
    seeds = sorted(set(seeds))
    k = len(seeds) // n_folds
    return [set(seeds[i * k:(i + 1) * k]) for i in range(n_folds)]


def cv_scores(bank, extra=None, n_folds=4):
    """bank: list of dicts {id, seed, label (1 backdoored, 0 clean), x}. extra: {id: x} (e.g. evasive adapters).
    Folds are by seed. Returns (bank_scores {id: score of the fold model that did not train on it},
    extra_scores {id: mean score over the fold models}). Higher score = more 'backdoored'."""
    folds = seed_folds([b["seed"] for b in bank], n_folds)
    bank_scores, extra_acc = {}, {k: [] for k in (extra or {})}
    for held in folds:
        tr = [b for b in bank if b["seed"] not in held]
        te = [b for b in bank if b["seed"] in held]
        m = make_model().fit(np.stack([b["x"] for b in tr]), [b["label"] for b in tr])
        for b, s in zip(te, m.decision_function(np.stack([b["x"] for b in te]))):
            bank_scores[b["id"]] = float(s)
        for k, x in (extra or {}).items():
            extra_acc[k].append(float(m.decision_function(x[None])[0]))
    return bank_scores, {k: float(np.mean(v)) for k, v in extra_acc.items()}


def auroc(pos_scores, neg_scores):
    y = [1] * len(pos_scores) + [0] * len(neg_scores)
    return float(roc_auc_score(y, list(pos_scores) + list(neg_scores)))


def bootstrap_ci(pos_scores, neg_scores, n=2000, seed=0, level=0.95):
    """Percentile bootstrap over adapters (resample each class with replacement)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos_scores), np.asarray(neg_scores)
    vals = []
    for _ in range(n):
        p = rng.choice(pos, len(pos)); q = rng.choice(neg, len(neg))
        vals.append(auroc(p, q))
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def proxy_validity(bank_scores, bank):
    """Condition 1: CV AUROC of bank backdoored vs bank clean, with 95% bootstrap CI."""
    pos = [bank_scores[b["id"]] for b in bank if b["label"] == 1]
    neg = [bank_scores[b["id"]] for b in bank if b["label"] == 0]
    return auroc(pos, neg), bootstrap_ci(pos, neg)


def evasion_auroc(bank_scores, bank, evasive_scores):
    """Condition 4: AUROC of evasive adapters (positive) vs held-out clean bank adapters."""
    neg = [bank_scores[b["id"]] for b in bank if b["label"] == 0]
    pos = list(evasive_scores.values())
    return auroc(pos, neg), bootstrap_ci(pos, neg)
