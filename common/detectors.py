"""G1-lite detectors (EXP-001, seat C). All three: higher score = more backdoored.

  wsd_aligned        WSD v3-aligned (DEC-023): layer index 20, per projection q,k,v,o the 5 features of
                     evasion_wsd.wsd_features_numpy, concatenated (20 numbers); z-score against the `clean`
                     group, entropy sign inverted, n = 0.5(1 + tanh(z/2)), logistic regression (C=1, balanced).
                     Score = logit. NOT WSD's own code.
  spectral_baseline  pilot features (features.py) averaged over all layers per projection (20 numbers),
                     standardise + logistic regression (C=1), same as detector_standin.make_model. Score = logit.
  weight_norm        mean Frobenius norm of dW over all modules; no fitting.

Protocol (thresholds in tracking/experiments/EXP-001-*.md): fit on `clean` vs `bd` only; leave-one-out for those
12 (`leave_one_out`); evasive adapters are scored by a model fit on all 12 (`fit_all_and_score`) and never train.
"""
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import evasion_wsd as ew
import features as pf

WSD_LAYER_INDEX = 20  # "layer 21" in WSD's text = 0-based index 20 (also the key in the adapter tensors)
PROJS = ew.PROJS  # q, k, v, o
WSD_FEATS = ew.FEATS  # top_sv, fro_norm, energy, entropy, kurtosis
WSD_NAMES = [f"{p}:{f}" for p in PROJS for f in WSD_FEATS]
ENTROPY_COLS = [i for i, n in enumerate(WSD_NAMES) if n.endswith(":entropy")]
SPEC_STATS = ["top_sv", "fro_norm", "energy_conc", "spectral_entropy", "kurtosis"]
SPEC_NAMES = [f"{p}:{s}" for p in PROJS for s in SPEC_STATS]


def extract(folder):
    """{'wsd': (20,), 'spec': (20,), 'norm': float} from a saved peft adapter folder (weights only)."""
    r, scaling, mods = pf.load_adapter(folder)
    wsd, byproj, fro = {}, {p: [] for p in PROJS}, []
    for mod, ab in sorted(mods.items()):
        s = pf.stats_from_sv(pf.lora_singular_values(ab["A"], ab["B"], scaling))
        proj = mod.split(".")[-1]
        byproj[proj].append([s[k] for k in SPEC_STATS])
        fro.append(s["fro_norm"])
        if f"layers.{WSD_LAYER_INDEX}.self_attn." in mod:
            wsd[proj] = ew.wsd_features_numpy(ab["A"], ab["B"], scaling)
    assert set(wsd) == set(PROJS), f"{folder}: layer index {WSD_LAYER_INDEX} modules found: {sorted(wsd)}"
    return {"wsd": np.concatenate([wsd[p] for p in PROJS]),
            "spec": np.concatenate([np.mean(byproj[p], axis=0) for p in PROJS]),
            "norm": float(np.mean(fro))}


class WsdAligned:
    name, key = "wsd_aligned", "wsd"

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        c = X[y == 0]
        self.mu = c.mean(0)
        sd = c.std(0, ddof=1) if len(c) > 1 else np.zeros(X.shape[1])
        self.sd = np.maximum(sd, np.maximum(1e-6 * np.abs(self.mu), 1e-12))
        self.lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(self._t(X), y)
        return self

    def z(self, X):
        z = (np.atleast_2d(np.asarray(X, float)) - self.mu) / self.sd
        z[:, ENTROPY_COLS] *= -1  # entropy sign inverted, as in the WSD method section
        return z

    def _t(self, X):
        return 0.5 * (1 + np.tanh(self.z(X) / 2))

    def score(self, X):
        return self.lr.decision_function(self._t(X))


class SpectralBaseline:
    name, key = "spectral_baseline", "spec"

    def fit(self, X, y):
        self.m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(np.asarray(X, float), y)
        return self

    def score(self, X):
        return self.m.decision_function(np.atleast_2d(np.asarray(X, float)))


class WeightNorm:
    name, key = "weight_norm", "norm"

    def fit(self, X, y):
        return self

    def score(self, X):
        return np.asarray(X, float).reshape(-1)


DETECTORS = [WsdAligned, SpectralBaseline, WeightNorm]


def _mat(feats, ids, key):
    return np.stack([np.atleast_1d(feats[i][key]) for i in ids]) if key != "norm" else np.array([feats[i][key] for i in ids])


def leave_one_out(cls, feats, ids, labels):
    """Score each of the 12 (clean=0 / bd=1) adapters with a model fit on the other 11 only."""
    X, y = _mat(feats, ids, cls.key), np.asarray(labels)
    out = {}
    for i, a in enumerate(ids):
        keep = np.arange(len(ids)) != i
        out[a] = float(cls().fit(X[keep], y[keep]).score(X[i:i + 1])[0])
    return out


def fit_all_and_score(cls, feats, ids, labels, other_ids):
    """Fit on all labelled adapters; score other_ids (evasive adapters never enter training)."""
    X, y = _mat(feats, ids, cls.key), np.asarray(labels)
    m = cls().fit(X, y)
    return {a: float(m.score(_mat(feats, [a], cls.key))[0]) for a in other_ids}


def extract_all(adapters_dir, ids=None):
    names = sorted(ids or [d for d in os.listdir(adapters_dir) if os.path.isfile(os.path.join(adapters_dir, d, "adapter_config.json"))])
    return {n: extract(os.path.join(adapters_dir, n)) for n in names}
