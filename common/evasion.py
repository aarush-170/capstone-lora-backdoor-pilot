"""G1 evasion feasibility test: a differentiable spectral-statistics penalty for LoRA training.

An 'informed attacker' knows a weight-only detector reads the singular-value statistics of each LoRA update
dW = scaling * B @ A. This module computes the same five statistics as `features.py` (top singular value,
Frobenius norm, energy concentration (top-1), spectral entropy, excess kurtosis) in torch, so they can be
penalised during poisoned fine-tuning and pulled towards what clean adapters look like.

Design in `thresholds_evasion.md`. Nothing here has run on a GPU yet; `test_evasion.py` checks the maths on CPU.
"""
import csv
import re

import numpy as np
import torch

STAT_NAMES = ["top_sv", "fro_norm", "energy_conc", "spectral_entropy", "kurtosis"]
_KEY = re.compile(r"layers\.(\d+)\.self_attn\.(\w+_proj)")


def module_key(name):
    """('layer', 'proj') from any module name containing 'layers.N.self_attn.X_proj', else None."""
    m = _KEY.search(name)
    return (int(m.group(1)), m.group(2)) if m else None


def stats_from_sv_torch(s, eps=1e-12):
    """s: (n, r) singular values -> (n, 5) statistics, same definitions as features.stats_from_sv."""
    e = s ** 2
    tot = e.sum(-1).clamp_min(eps)
    p = e / tot[:, None]
    top = s.max(-1).values
    fro = tot.sqrt()
    energy = e.max(-1).values / tot
    ent = -(p * torch.log(p.clamp_min(eps))).sum(-1)
    m = s.mean(-1, keepdim=True)
    m2 = ((s - m) ** 2).mean(-1)
    m4 = ((s - m) ** 4).mean(-1)
    kurt = m4 / (m2 ** 2 + eps) - 3.0
    return torch.stack([top, fro, energy, ent, kurt], dim=-1)


def lora_stats_batched(A, B, scaling):
    """A: (n, r, d_in), B: (n, d_out, r), scaling: (n,). Exact singular values of scaling*B@A via QR plus an
    r x r SVD (as in features.lora_singular_values), batched over n modules of the same shape."""
    Rb = torch.linalg.qr(B).R
    Ra = torch.linalg.qr(A.transpose(1, 2)).R
    core = Rb @ Ra.transpose(1, 2)
    s = torch.linalg.svdvals(core) * scaling[:, None]
    return stats_from_sv_torch(s)


def collect_lora(pm, adapter_name):
    """[(name, A_weight (r,d_in), B_weight (d_out,r), scaling)] for every LoRA layer of `adapter_name`
    in a peft model. Uses peft's LoRA layer attributes lora_A / lora_B / scaling (ModuleDict / dict)."""
    out = []
    for name, mod in pm.named_modules():
        la, lb, sc = getattr(mod, "lora_A", None), getattr(mod, "lora_B", None), getattr(mod, "scaling", None)
        if la is None or lb is None or sc is None:
            continue
        if adapter_name in la and adapter_name in lb:
            out.append((name, la[adapter_name].weight, lb[adapter_name].weight, float(sc[adapter_name])))
    assert out, f"no LoRA layers found for adapter '{adapter_name}'"
    return out


def targets_from_features(csv_path, clean_ids, rel_floor=0.05, abs_floor=1e-8):
    """Per-module (mu, sigma) of the 5 statistics across the clean adapters in a features CSV
    (columns adapter_id, module, top_sv, fro_norm, energy_conc, spectral_entropy, kurtosis).
    sigma is floored at rel_floor*|mu| so near-constant statistics do not give a huge penalty.
    Returns {(layer, proj): (mu[5], sigma[5])} as numpy arrays."""
    vals = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["adapter_id"] not in clean_ids:
                continue
            k = module_key(r["module"])
            vals.setdefault(k, []).append([float(r[c]) for c in STAT_NAMES])
    assert vals, "no clean adapter rows found"
    out = {}
    for k, v in vals.items():
        v = np.asarray(v)
        mu, sd = v.mean(0), v.std(0)
        out[k] = (mu, np.maximum(sd, np.maximum(rel_floor * np.abs(mu), abs_floor)))
    return out


class SpectralPenalty:
    """Callable for `pilot_common.train(extra_loss_fn=...)`. Returns lam * mean squared z-score of the
    adapter's spectral statistics against clean targets (over modules and statistics), or None during the
    warm-up (at LoRA initialisation B = 0, so dW = 0 and the statistics and their gradients are undefined)."""

    def __init__(self, targets, lam, warmup_steps=16, device=None):
        self.targets, self.lam, self.warmup = targets, float(lam), int(warmup_steps)
        self.device = device
        self._groups = None
        self.last_raw = float("nan")

    def _prepare(self, pm, adapter_name):
        mods = collect_lora(pm, adapter_name)
        groups = {}
        for name, A, B, sc in mods:
            key = module_key(name)
            assert key in self.targets, f"no clean target for module {name} ({key})"
            groups.setdefault((tuple(A.shape), tuple(B.shape)), []).append((A, B, sc, key))
        dev = self.device or mods[0][1].device
        prepared = []
        for items in groups.values():
            mu = torch.tensor(np.stack([self.targets[k][0] for *_, k in items]), dtype=torch.float32, device=dev)
            sd = torch.tensor(np.stack([self.targets[k][1] for *_, k in items]), dtype=torch.float32, device=dev)
            sc = torch.tensor([s for _, _, s, _ in items], dtype=torch.float32, device=dev)
            prepared.append((items, mu, sd, sc))
        self._groups = prepared

    def raw(self, pm, adapter_name):
        """Mean squared z-score (no lambda, no warm-up). Differentiable w.r.t. the LoRA weights."""
        if self._groups is None:
            self._prepare(pm, adapter_name)
        total, count = 0.0, 0
        for items, mu, sd, sc in self._groups:
            A = torch.stack([a.float() for a, _, _, _ in items])
            B = torch.stack([b.float() for _, b, _, _ in items])
            z = (lora_stats_batched(A, B, sc) - mu) / sd
            total = total + (z ** 2).sum()
            count += z.numel()
        return total / count

    def __call__(self, pm, adapter_name, step):
        if step < self.warmup or self.lam == 0.0:
            return None
        r = self.raw(pm, adapter_name)
        self.last_raw = float(r.detach())
        return self.lam * r
