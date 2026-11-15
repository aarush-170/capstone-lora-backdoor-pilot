"""CPU checks for common/evasion.py. Run:  python test_evasion.py   (from common/; set KMP_DUPLICATE_LIB_OK=TRUE
on machines where torch and another library both ship an OpenMP runtime)."""
import csv
import os
import tempfile

import numpy as np
import torch

import evasion as ev
import features as ft

torch.manual_seed(0)
rng = np.random.default_rng(0)


def _rand_lora(n, d_out, d_in, r=16):
    A = torch.randn(n, r, d_in, dtype=torch.float64)
    B = torch.randn(n, d_out, r, dtype=torch.float64)
    return A, B


class _FakeLoraLayer(torch.nn.Module):
    """Mimics the attributes peft's LoRA Linear exposes: lora_A / lora_B (ModuleDict of Linear) and scaling."""
    def __init__(self, d_in, d_out, r, name):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({name: torch.nn.Linear(d_in, r, bias=False)})
        self.lora_B = torch.nn.ModuleDict({name: torch.nn.Linear(r, d_out, bias=False)})
        self.scaling = {name: 2.0}


class _FakeModel(torch.nn.Module):
    def __init__(self, name="a", layers=2, d=64, dkv=16, r=16):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for _ in range(layers):
            L = torch.nn.Module()
            L.self_attn = torch.nn.Module()
            for p, (do, di) in {"q_proj": (d, d), "k_proj": (dkv, d), "v_proj": (dkv, d), "o_proj": (d, d)}.items():
                setattr(L.self_attn, p, _FakeLoraLayer(di, do, r, name))
            self.model.layers.append(L)


def test_matches_numpy_features():
    A, B = _rand_lora(3, 96, 80)
    sc = torch.full((3,), 2.0, dtype=torch.float64)
    got = ev.lora_stats_batched(A, B, sc).numpy()
    for i in range(3):
        s = ft.lora_singular_values(A[i].numpy(), B[i].numpy(), 2.0)
        ref = ft.stats_from_sv(s)
        want = np.array([ref[k] for k in ev.STAT_NAMES])
        assert np.allclose(got[i], want, rtol=1e-6, atol=1e-8), (got[i], want)
    print("ok  torch stats == numpy features (3 random LoRA updates)")


def test_gradient_flows():
    A, B = _rand_lora(2, 64, 48)
    A.requires_grad_(True); B.requires_grad_(True)
    sc = torch.full((2,), 2.0, dtype=torch.float64)
    ev.lora_stats_batched(A, B, sc).pow(2).sum().backward()
    for g in (A.grad, B.grad):
        assert torch.isfinite(g).all() and g.abs().sum() > 0
    print("ok  gradients finite and non-zero")


def test_penalty_zero_at_target_and_decreases():
    m = _FakeModel()
    mods = ev.collect_lora(m, "a")
    assert len(mods) == 8, len(mods)
    # target = this adapter's own statistics -> raw penalty must be 0
    targets = {}
    for name, A, B, sc in mods:
        s = ev.lora_stats_batched(A.detach()[None], B.detach()[None], torch.tensor([sc]))[0].numpy()
        targets[ev.module_key(name)] = (s, np.maximum(0.05 * np.abs(s), 1e-8))
    pen = ev.SpectralPenalty(targets, lam=1.0, warmup_steps=0)
    assert pen.raw(m, "a").item() < 1e-6  # float32 rounding between single-module and batched paths (~3e-10)
    # shift the targets so the penalty is positive, then optimise the weights: it must go down
    shifted = {k: (mu * 1.5 + 0.1, sd) for k, (mu, sd) in targets.items()}
    pen = ev.SpectralPenalty(shifted, lam=1.0, warmup_steps=0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = pen.raw(m, "a").item()
    for _ in range(60):
        opt.zero_grad(); loss = pen(m, "a", 100); loss.backward(); opt.step()
    last = pen.raw(m, "a").item()
    assert first > 0 and last < first * 0.5, (first, last)
    print(f"ok  penalty 0 at target; optimising reduces it {first:.3f} -> {last:.3f}")


def test_warmup_and_lambda_zero():
    m = _FakeModel()
    targets = {ev.module_key(n): (np.ones(5), np.ones(5)) for n, *_ in ev.collect_lora(m, "a")}
    assert ev.SpectralPenalty(targets, 1.0, warmup_steps=5)(m, "a", 3) is None
    assert ev.SpectralPenalty(targets, 0.0, warmup_steps=0)(m, "a", 3) is None
    assert ev.SpectralPenalty(targets, 1.0, warmup_steps=5)(m, "a", 5) is not None
    print("ok  warm-up and lambda=0 return no penalty")


def test_targets_from_features_csv():
    d = tempfile.mkdtemp(); p = os.path.join(d, "f.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["adapter_id", "module"] + ev.STAT_NAMES); w.writeheader()
        for aid, off in [("c1", 0.0), ("c2", 1.0), ("bd1", 50.0)]:
            w.writerow({"adapter_id": aid, "module": "base_model.model.model.layers.3.self_attn.q_proj",
                        **{k: 1.0 + off + i for i, k in enumerate(ev.STAT_NAMES)}})
    t = ev.targets_from_features(p, {"c1", "c2"})
    mu, sd = t[(3, "q_proj")]
    assert np.allclose(mu, [1.5, 2.5, 3.5, 4.5, 5.5]) and (sd > 0).all()
    print("ok  targets use only clean adapters (backdoored row ignored)")


if __name__ == "__main__":
    test_matches_numpy_features()
    test_gradient_flows()
    test_penalty_zero_at_target_and_decreases()
    test_warmup_and_lambda_zero()
    test_targets_from_features_csv()
    print("ALL PASSED")
