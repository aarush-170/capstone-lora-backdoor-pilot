"""EXP-006 Stage 2: differentiable D_A features and the white-box spectral-matching penalty (layer index 20, q/k/v/o).

The 20 features are the WSD authors' detector features (core/detector.py, Apache-2.0, credit D. Puertolas Merenciano et al.)
as re-typed in common/wsd_bank.authors_features: UNSCALED dW = B @ A; sigma1, Frobenius, energy s1^2/sum s^2,
entropy (p = s/(sum s + 1e-12), log(p + 1e-12)), kurtosis = mean((x-mean)^4)/(var_ddof1^2 + 1e-12).
Penalty = lam * mean over the 20 features of z^2, z = (feature - mu) / sigma, with mu and sigma from the 395 benign bank
adapters (ref_stats_bank_benign.json, sigma floored at 5% of |mu|). No penalty until max|B| >= B_MIN and while the penalty
is not finite (at LoRA init B = 0, dW = 0 and the SVD gradient is undefined: the NaN lesson of EXP-001, seat A commit 838ef19).
"""
import json
import re

import torch

EPS = 1e-12
PROJS = ["q_proj", "k_proj", "v_proj", "o_proj"]
LAYER = 20
B_MIN = 1e-4
_KEY = re.compile(r"layers\.(\d+)\.self_attn\.(\w+_proj)")


def authors_features_torch(A, B):
    """A: (r, d_in), B: (d_out, r) -> (5,) float32 differentiable features of the unscaled dW = B @ A."""
    A, B = A.float(), B.float()
    Rb = torch.linalg.qr(B).R
    Ra = torch.linalg.qr(A.T).R
    s = torch.linalg.svdvals(Rb @ Ra.T)
    tot = (s ** 2).sum()
    p = s / (s.sum() + EPS)
    ent = -(p * torch.log(p + EPS)).sum()
    d = (B @ A).double().flatten()  # float64 for the moments: var^2 is about 1e-15
    m = d.mean()
    var = d.var(unbiased=True)
    kurt = (((d - m) ** 4).mean() / (var ** 2 + EPS)).float()
    return torch.stack([s[0], tot.sqrt(), (s[0] ** 2) / tot, ent, kurt])


class BankPenalty:
    """Callable `extra_loss_fn(pm, adapter_name, step)`: returns lam * mean z^2, or None while gated."""

    def __init__(self, ref, lam):
        if isinstance(ref, str):
            ref = json.load(open(ref))
        self.mu = torch.tensor(ref["mu"], dtype=torch.float32).view(4, 5)
        self.sd = torch.tensor(ref["sigma"], dtype=torch.float32).view(4, 5)
        self.lam = float(lam)
        self._mods = None
        self.last_raw = float("nan")
        self.gated_steps = 0

    def _prepare(self, pm, adapter_name):
        mods = {}
        for name, mod in pm.named_modules():
            m = _KEY.search(name)
            la, lb = getattr(mod, "lora_A", None), getattr(mod, "lora_B", None)
            if m and int(m[1]) == LAYER and la is not None and adapter_name in la and adapter_name in lb:
                mods[m[2]] = (la[adapter_name].weight, lb[adapter_name].weight)
        assert set(mods) == set(PROJS), f"layer {LAYER} LoRA modules found: {sorted(mods)}"
        self._mods = mods

    def raw(self, pm, adapter_name):
        if self._mods is None:
            self._prepare(pm, adapter_name)
        zs = []
        for i, p in enumerate(PROJS):
            A, B = self._mods[p]
            f = authors_features_torch(A, B)
            mu, sd = self.mu[i].to(f.device), self.sd[i].to(f.device)
            zs.append(((f - mu) / sd) ** 2)
        return torch.stack(zs).mean()

    def b_max(self):
        return max(float(B.detach().abs().max()) for _, B in self._mods.values())

    def __call__(self, pm, adapter_name, step):
        if self.lam == 0.0:
            return None
        if self._mods is None:
            self._prepare(pm, adapter_name)
        if self.b_max() < B_MIN:
            self.gated_steps += 1
            return None
        r = self.raw(pm, adapter_name)
        if not torch.isfinite(r):
            self.gated_steps += 1
            return None
        self.last_raw = float(r.detach())
        return self.lam * r
