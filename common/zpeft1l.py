"""EXP-011: layer-20, per-head Z-PEFT VARIANT for the authors' Qwen2.5-3B bank and our own Qwen adapters (all LoRA on layer
index 20 only, q/k/v/o, r 16, alpha 32). NOT Z-PEFT as published (that is all-layer): one layer, q_proj and v_proj only.
Reuses head_singular_values / descriptor code of common/zpeft.py WITHOUT changing it (its EXP-010 hash stays valid).

Grouped-query attention (Qwen2.5-3B): q_proj B is 2048 x 16 = 16 heads of 128 rows; v_proj B is 256 x 16 = 2 key/value heads of
128 rows. 18 heads x 16 descriptor numbers = 288 features, order: q_proj heads 0..15 then v_proj heads 0..1, descriptor inside.
Classifier: the paper's grid (zpeft.GRID), chosen by 5-fold stratified AUROC (seed 0) on the training adapters, refit on all of
them. Decision threshold: the authors' rule (wsd_bank.select_threshold) on the pooled out-of-fold decision scores of the chosen
candidate over that same split, frozen, never recalibrated on our adapters.
"""
import json
import os
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

import wsd_bank as wb
import zpeft as zp

LAYER = 20
HEADS = {"q_proj": 16, "v_proj": 2}
N_FEATURES = sum(HEADS.values()) * zp.N_DESC  # 288


def feature_names(heads=HEADS):
    return [f"{p}:h{h}:{d}" for p, n in heads.items() for h in range(n) for d in zp.DESC_NAMES]


def adapter_features(folder, layer=LAYER, heads=HEADS):
    """(sum(heads)*16,) feature vector and the number of degenerate heads of one adapter folder. Stops on a missing tensor
    or a non-finite value (reported, not imputed)."""
    from safetensors.numpy import load_file
    cfg = json.load(open(os.path.join(folder, "adapter_config.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    w = load_file(os.path.join(folder, "adapter_model.safetensors"))
    parts, n_degen = [], 0
    for proj, n_heads in heads.items():
        pre = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
        a, b = f"{pre}.lora_A.weight", f"{pre}.lora_B.weight"
        assert a in w and b in w, f"{folder}: missing {pre}"
        D, flags = zp.descriptors_matrix(zp.head_singular_values(w[a], w[b], scaling, n_heads))
        parts.append(D.reshape(-1))
        n_degen += int((flags > 0).sum())
    x = np.concatenate(parts)
    if not np.isfinite(x).all():
        raise ValueError(f"{folder}: non-finite feature (reported, not imputed)")
    return x, n_degen


def fit_bank(X, y, seed=0, n_folds=5):
    """Grid over the paper's 12 candidates by mean 5-fold AUROC, refit on all, threshold by the authors' rule on the pooled
    out-of-fold decision scores of the chosen candidate. Returns a dict (model, threshold, rule, cv_auroc, best params, ...)."""
    X, y = np.asarray(X, float), np.asarray(y)
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    gs = GridSearchCV(zp.make_pipeline(), zp.GRID, scoring="roc_auc", refit=True, n_jobs=1, cv=cv)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        gs.fit(X, y)
        oof = cross_val_predict(clone(gs.best_estimator_), X, y, cv=cv, method="decision_function")
    thr, rule = wb.select_threshold(y, oof)
    return {"model": gs.best_estimator_, "threshold": float(thr), "rule": rule, "cv_auroc": float(gs.best_score_),
            "C": gs.best_params_["clf__C"], "class_weight": gs.best_params_["clf__class_weight"],
            "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in ws), "n_train": int(len(y))}


def score(fit, X):
    """(decision scores, probabilities, flagged) of rows X; flagged = decision score >= the frozen threshold."""
    X = np.atleast_2d(np.asarray(X, float))
    d = fit["model"].decision_function(X)
    return d, fit["model"].predict_proba(X)[:, 1], d >= fit["threshold"]
