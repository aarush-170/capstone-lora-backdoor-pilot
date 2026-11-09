"""EXP-010: our reimplementation of Z-PEFT (arXiv 2608.02271) from the paper only; the authors' code is not released.
Weights only, CPU. Choices the paper leaves open are fixed in tracking/decisions/DEC-025 (head split, spectrum used for the
aggregates, kurtosis definition, folds). Nothing here has been run on a PADBench adapter when this file is frozen.

Per attention head h of layer l and projection p in (q_proj, v_proj): dW_h = scaling * B_h @ A, B_h = rows h*hd .. (h+1)*hd-1
of B (head-major rows: assumption), A shared. Singular values by reduced QR of B_h and A^T and an r x r SVD, batched over
heads; dW is never formed. Descriptor, 16 numbers (paper p. 3, k = 8):
  sigma_1..sigma_8 (zero-padded), energy sum s^2, l2 norm, l1 norm, stable rank sum s^2 / s_1^2, spectral entropy of
  p = s / sum s (natural log), effective rank exp(H), concentration s_1 / sum s, excess kurtosis of s (population moments).
The aggregates use ALL singular values of the head (up to r), DEC-025 item 3. Degenerate spectra: a zero-variance spectrum
gets kurtosis 0, an all-zero spectrum gets 16 zeros; both are counted, never silently dropped.
One adapter = n_layers * 2 * n_heads * 16 numbers, order: layer, projection (q then v), head, descriptor.
"""
import json
import os
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

K_TOP = 8
N_DESC = 16
DESC_NAMES = [f"sigma{i + 1}" for i in range(K_TOP)] + ["energy", "l2", "l1", "stable_rank", "entropy", "eff_rank",
                                                         "concentration", "kurtosis"]
PROJS = ("q_proj", "v_proj")
GRID = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__class_weight": [None, "balanced"]}


# ---------------------------------------------------------------- features
def head_singular_values(A, B, scaling, n_heads):
    """(n_heads, r) singular values (descending) of scaling * B_h @ A for the n_heads equal row blocks B_h of B.
    A: (r, d_in), B: (d_out, r), d_out divisible by n_heads."""
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    d_out, r = B.shape
    assert d_out % n_heads == 0, f"d_out {d_out} not divisible by {n_heads} heads"
    hd = d_out // n_heads
    Rb = np.linalg.qr(B.reshape(n_heads, hd, r), mode="r")  # (H, r, r) when hd >= r
    Ra = np.linalg.qr(A.T, mode="r")  # (r, r), shared by all heads
    core = Rb @ Ra.T
    return scaling * np.linalg.svd(core, compute_uv=False)


def descriptor(s, k=K_TOP):
    """(16,) descriptor of one spectrum s (any length, any order). Returns (vector, degenerate_flag) where the flag is
    0 normal, 1 zero-variance spectrum (kurtosis set to 0), 2 all-zero spectrum (everything 0)."""
    s = np.sort(np.asarray(s, np.float64))[::-1]
    top = np.zeros(k)
    top[:min(k, len(s))] = s[:k]
    l1 = float(s.sum())
    energy = float((s ** 2).sum())
    if l1 <= 0 or energy <= 0:
        return np.zeros(K_TOP + 8), 2
    p = s / l1
    pnz = p[p > 0]
    H = float(-(pnz * np.log(pnz)).sum())
    m = s.mean()
    m2 = float(((s - m) ** 2).mean())
    flag = 0
    if m2 > 0:
        kurt = float(((s - m) ** 4).mean() / m2 ** 2 - 3.0)
    else:
        kurt, flag = 0.0, 1
    rest = np.array([energy, np.sqrt(energy), l1, energy / s[0] ** 2, H, np.exp(H), s[0] / l1, kurt])
    return np.concatenate([top, rest]), flag


def descriptors_matrix(S):
    """(n_heads, 16) descriptors and the degenerate flags (n_heads,) for a (n_heads, r) singular-value array."""
    out, flags = zip(*(descriptor(s) for s in S))
    return np.stack(out), np.array(flags)


def adapter_features(folder, n_layers=32, n_heads=32):
    """(n_layers*2*n_heads*16,) Z-PEFT feature vector of a PADBench adapter folder and the number of degenerate heads
    (kurtosis-substituted, all-zero). Stops on a missing tensor or a non-finite value."""
    from safetensors.numpy import load_file
    cfg = json.load(open(os.path.join(folder, "adapter_config.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    w = load_file(os.path.join(folder, "adapter_model.safetensors"))
    parts, n_degen = [], 0
    for layer in range(n_layers):
        for proj in PROJS:
            pre = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
            a, b = f"{pre}.lora_A.weight", f"{pre}.lora_B.weight"
            assert a in w and b in w, f"{folder}: missing {pre}"
            D, flags = descriptors_matrix(head_singular_values(w[a], w[b], scaling, n_heads))
            parts.append(D.reshape(-1))
            n_degen += int((flags > 0).sum())
    x = np.concatenate(parts)
    if not np.isfinite(x).all():
        raise ValueError(f"{folder}: non-finite Z-PEFT feature (reported, not imputed)")
    return x, n_degen


# ---------------------------------------------------------------- classifier (the paper's, Table 8)
def make_pipeline():
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(solver="lbfgs", max_iter=5000))])


def fit_zpeft(Xtr, ytr, seed, n_inner=5, n_jobs=1):
    """Grid over C and class_weight (12 candidates) chosen by mean stratified n_inner-fold AUROC on the training part
    only; refit on all of it. Returns (fitted GridSearchCV, number of convergence warnings)."""
    gs = GridSearchCV(make_pipeline(), GRID, scoring="roc_auc", refit=True, n_jobs=n_jobs,
                      cv=StratifiedKFold(n_inner, shuffle=True, random_state=seed))
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        gs.fit(Xtr, ytr)
    return gs, sum(issubclass(w.category, ConvergenceWarning) for w in ws)


def zpeft_fit_score(Xtr, ytr, Xte, seed):
    """fit_score callable for outer_cv_scores: (test scores, info dict)."""
    gs, nconv = fit_zpeft(Xtr, ytr, seed)
    return gs.decision_function(Xte), {"C": gs.best_params_["clf__C"], "class_weight": gs.best_params_["clf__class_weight"],
                                       "inner_auroc": float(gs.best_score_), "convergence_warnings": nconv}


# ---------------------------------------------------------------- evaluation
def outer_cv_scores(X, y, seed, fit_score, n_outer=5):
    """Out-of-fold scores of every sample: stratified n_outer-fold split with random_state=seed, fit_score(Xtr, ytr, Xte,
    seed) -> (scores, info) called on the training part only. Returns (scores, [info per fold])."""
    X, y = np.asarray(X, float), np.asarray(y)
    out = np.full(len(y), np.nan)
    infos = []
    for tr, te in StratifiedKFold(n_outer, shuffle=True, random_state=seed).split(X, y):
        s, info = fit_score(X[tr], y[tr], X[te], seed)
        out[te] = s
        infos.append(info)
    assert np.isfinite(out).all()
    return out, infos
