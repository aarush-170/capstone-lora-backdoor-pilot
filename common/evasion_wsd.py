"""G1-lite: WSD-aligned spectral penalty (single layer, index 20 = the paper's layer 21), targets from ref_clean adapters only.

Feature definitions (must be identical to seat C's `wsd_aligned` detector; edit both together):
  top_sv   sigma_1 of dW = s * B @ A
  fro_norm sqrt(sum sigma^2)
  energy   sigma_1^2 / sum sigma^2          (authors' code; the paper says sigma_1 / sum sigma)
  entropy  -sum p log p, p = sigma / sum sigma, natural log   (paper definition)
  kurtosis excess kurtosis of the ENTRIES of dW (not of the singular values)
s = alpha / r. Features are computed per module (q, k, v, o: GQA makes k, v 256 x 2048) and averaged in the
penalty; targets are per projection. Reuses the QR + svdvals trick from evasion.py.
"""
import json
import re

import numpy as np
import torch

from evasion import module_key

WSD_LAYER = 20  # 0-based index: tensors named layers.20.* (the paper's "layer 21", counting from 1); must equal seat C's detectors.py
FEATS = ["top_sv", "fro_norm", "energy", "entropy", "kurtosis"]
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def wsd_features_torch(A, B, scaling, eps=1e-12):
    """A: (r, d_in), B: (d_out, r), scaling: float -> (5,) differentiable features of dW = scaling * B @ A."""
    Rb = torch.linalg.qr(B).R
    Ra = torch.linalg.qr(A.T).R
    s = torch.linalg.svdvals(Rb @ Ra.T) * scaling
    tot = (s ** 2).sum().clamp_min(eps)
    p = s / s.sum().clamp_min(eps)
    ent = -(p * torch.log(p.clamp_min(eps))).sum()
    dW = scaling * (B @ A)  # one layer only: 4M floats at most, cheap
    c = dW - dW.mean()
    c = c / (c ** 2).mean().clamp_min(1e-30).sqrt()  # scale-free: dW entries are ~1e-2, so an additive eps in m2**2 biased kurtosis
    kurt = (c ** 4).mean() - 3.0
    return torch.stack([s.max(), tot.sqrt(), (s.max() ** 2) / tot, ent, kurt])


def wsd_features_numpy(A, B, scaling):
    """Reference implementation by full SVD, for tests and for computing ref_clean statistics from saved adapters."""
    dW = scaling * (np.asarray(B, np.float64) @ np.asarray(A, np.float64))
    s = np.linalg.svd(dW, compute_uv=False)
    tot = (s ** 2).sum()
    p = s / s.sum()
    p = p[p > 0]
    c = dW - dW.mean()
    m2 = (c ** 2).mean()
    return np.array([s.max(), np.sqrt(tot), s.max() ** 2 / tot, -(p * np.log(p)).sum(), (c ** 4).mean() / m2 ** 2 - 3.0])


def adapter_wsd_features(folder):
    """{proj: (5,) features} at WSD_LAYER from a saved peft adapter folder (safetensors + adapter_config.json)."""
    import os
    from safetensors.numpy import load_file
    cfg = json.load(open(os.path.join(folder, "adapter_config.json")))
    scaling = cfg["lora_alpha"] / cfg["r"]
    mods = {}
    for key, val in load_file(os.path.join(folder, "adapter_model.safetensors")).items():
        m = re.match(r"^(?P<mod>.+)\.lora_(?P<ab>[AB])(?:\.[^.]+)?\.weight$", key)
        k = module_key(m["mod"]) if m else None
        if k and k[0] == WSD_LAYER:
            mods.setdefault(k[1], {})[m["ab"]] = val
    assert set(mods) == set(PROJS), f"{folder}: layer {WSD_LAYER} modules found: {sorted(mods)}"
    return {p: wsd_features_numpy(v["A"], v["B"], scaling) for p, v in mods.items()}


def make_ref_stats(ref_folders, out_json, rel_floor=0.05, abs_floor=1e-8):
    """mu / sigma per projection and feature across the ref_clean adapters ONLY. sigma floored at rel_floor*|mu|
    (3 adapters give a noisy sigma). Writes ref_stats.json."""
    per = [adapter_wsd_features(f) for f in ref_folders]
    out = {"layer": WSD_LAYER, "n_ref": len(per), "rel_floor": rel_floor, "features": FEATS, "stats": {}}
    for p in PROJS:
        v = np.stack([d[p] for d in per])
        mu, sd = v.mean(0), v.std(0, ddof=1) if len(per) > 1 else np.zeros(5)
        out["stats"][p] = {"mu": mu.tolist(), "sigma": np.maximum(sd, np.maximum(rel_floor * np.abs(mu), abs_floor)).tolist()}
    json.dump(out, open(out_json, "w"), indent=2)
    return out


class WsdPenalty:
    """`extra_loss_fn` for pilot_common.train. lam * mean over (4 modules x 5 features) of z^2, with z against
    ref_stats. Returns None for the first `warmup_steps` steps (B = 0 at init, so dW = 0 and the SVD gradient is
    undefined; this is a numerical need, not a lambda = 0 epoch)."""

    def __init__(self, ref_stats_path, lam, warmup_steps=16):
        st = json.load(open(ref_stats_path))
        assert st["layer"] == WSD_LAYER and st["features"] == FEATS
        self.stats, self.lam, self.warmup = st["stats"], float(lam), int(warmup_steps)
        self._mods, self.last_raw = None, float("nan")

    def _prepare(self, pm, adapter_name):
        mods = {}
        for name, mod in pm.named_modules():
            k = module_key(name)
            la, lb = getattr(mod, "lora_A", None), getattr(mod, "lora_B", None)
            if k and k[0] == WSD_LAYER and la is not None and adapter_name in la and adapter_name in lb:
                mods[k[1]] = (la[adapter_name].weight, lb[adapter_name].weight, float(mod.scaling[adapter_name]))
        assert set(mods) == set(PROJS), f"layer {WSD_LAYER} LoRA modules found: {sorted(mods)}"
        self._mods = mods

    def raw(self, pm, adapter_name):
        if self._mods is None:
            self._prepare(pm, adapter_name)
        zs = []
        for p, (A, B, sc) in self._mods.items():
            f = wsd_features_torch(A.float(), B.float(), sc)
            mu = torch.tensor(self.stats[p]["mu"], device=f.device, dtype=f.dtype)
            sd = torch.tensor(self.stats[p]["sigma"], device=f.device, dtype=f.dtype)
            zs.append(((f - mu) / sd) ** 2)
        return torch.stack(zs).mean()

    def __call__(self, pm, adapter_name, step):
        if step < self.warmup or self.lam == 0.0:
            return None
        r = self.raw(pm, adapter_name)
        self.last_raw = float(r.detach())
        return self.lam * r
