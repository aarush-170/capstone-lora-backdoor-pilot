"""Tests for exp6_penalty.py on synthetic LoRA matrices (CPU). Run: cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_exp6_penalty.py"""
import json

import numpy as np
import torch

import exp6_penalty as ep
import wsd_bank as wb


class FakeLayer(torch.nn.Module):  # mimics a peft LoRA layer: lora_A / lora_B ModuleDicts keyed by adapter name
    def __init__(self, d_in, d_out, r, scale=0.02, zero_b=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lora_A = torch.nn.ModuleDict({"a": torch.nn.Linear(d_in, r, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"a": torch.nn.Linear(r, d_out, bias=False)})
        self.lora_A["a"].weight.data = torch.randn(r, d_in, generator=g) * scale
        self.lora_B["a"].weight.data = torch.zeros(d_out, r) if zero_b else torch.randn(d_out, r, generator=g) * scale


class FakePM(torch.nn.Module):
    def __init__(self, zero_b=False, d=64, r=16, layer=20, seed=0):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(layer + 1)])
        for l in self.model.layers:
            l.self_attn = torch.nn.Module()
        for k, p in enumerate(ep.PROJS):
            setattr(self.model.layers[layer].self_attn, p,
                    FakeLayer(d, 16 if p in ("k_proj", "v_proj") else d, r, zero_b=zero_b, seed=seed + k))


def _mods(pm):
    return [getattr(pm.model.layers[20].self_attn, p) for p in ep.PROJS]


def ref_for(pm, spread=0.3):
    f = np.stack([ep.authors_features_torch(m.lora_A["a"].weight, m.lora_B["a"].weight).detach().numpy() for m in _mods(pm)])
    return {"mu": f.ravel().tolist(), "sigma": np.maximum(np.abs(f) * spread, 1e-8).ravel().tolist()}


def test_torch_features_match_numpy_authors_features():
    rng = np.random.default_rng(1)
    for scale in (0.05, 0.008):  # the second is tiny: the 1e-12 term dominates, as on the real bank
        A, B = rng.normal(size=(16, 64)) * scale, rng.normal(size=(64, 16)) * scale
        t = ep.authors_features_torch(torch.tensor(A), torch.tensor(B)).numpy()
        n = wb.authors_features(A, B)
        np.testing.assert_allclose(t, n, rtol=2e-4)
    print("ok  torch D_A features == numpy re-typed authors' features (also at tiny scale, eps-dominated kurtosis)")


def test_gradient_finite_and_nonzero():
    A = torch.randn(16, 64, requires_grad=True)
    B = torch.randn(64, 16, requires_grad=True)
    ep.authors_features_torch(A * 0.02, B * 0.02).sum().backward()
    assert torch.isfinite(A.grad).all() and torch.isfinite(B.grad).all() and B.grad.abs().sum() > 0
    print("ok  gradients finite and non-zero")


def test_gate_when_b_is_zero_then_penalty_active():
    pm = FakePM(zero_b=True)
    pen = ep.BankPenalty({"mu": [1.0] * 20, "sigma": [1.0] * 20}, lam=10)
    assert pen(pm, "a", 0) is None and pen.gated_steps == 1  # B = 0: gated, no NaN
    for m in _mods(pm):
        m.lora_B["a"].weight.data += 0.02
    out = pen(pm, "a", 1)
    assert out is not None and torch.isfinite(out) and pen.gated_steps == 1
    assert ep.BankPenalty({"mu": [1.0] * 20, "sigma": [1.0] * 20}, lam=0)(pm, "a", 5) is None
    print("ok  penalty gated while B ~ 0 and when lambda = 0; active and finite afterwards")


def test_penalty_decreases_toward_target_and_lambda_scales():
    pm, tgt = FakePM(seed=3), FakePM(seed=11)
    ref = ref_for(tgt)
    pen1, pen10 = ep.BankPenalty(ref, 1), ep.BankPenalty(ref, 10)
    a, b = float(pen1(pm, "a", 1)), float(pen10(pm, "a", 1))
    assert abs(b / a - 10) < 1e-4
    params = [w for m in _mods(pm) for w in (m.lora_A["a"].weight, m.lora_B["a"].weight)]
    opt = torch.optim.Adam(params, lr=2e-3)
    first = None
    for _ in range(300):
        loss = pen1(pm, "a", 1)
        first = first if first is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.5 * first, (first, float(loss))
    print(f"ok  penalty is exactly proportional to lambda; optimising reduces it {first:.1f} -> {float(loss):.1f}")


def test_ref_stats_file_is_consistent_with_feature_order():
    r = json.load(open("../tracking/experiments/EXP-006/ref_stats_bank_benign.json"))
    assert r["n"] == 395 and r["feature_order"] == wb.NAMES and len(r["mu"]) == len(r["sigma"]) == 20
    assert all(s >= 0.05 * abs(m) - 1e-12 for m, s in zip(r["mu"], r["sigma"]))
    ep.BankPenalty(r, 1)
    print("ok  ref stats: 395 adapters, feature order matches wsd_bank.NAMES, sigma >= 5% of |mu|")


if __name__ == "__main__":
    for t in (test_torch_features_match_numpy_authors_features, test_gradient_finite_and_nonzero,
              test_gate_when_b_is_zero_then_penalty_active, test_penalty_decreases_toward_target_and_lambda_scales,
              test_ref_stats_file_is_consistent_with_feature_order):
        t()
    print("ALL PASSED")
