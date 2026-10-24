"""CPU tests for evasion_wsd.py. Run: cd common && python -m pytest -q test_evasion_wsd.py"""
import numpy as np
import torch

import evasion_wsd as ew


def _rand(d_out, d_in, r=16, seed=0):
    g = np.random.default_rng(seed)
    return g.normal(size=(r, d_in)) * 0.05, g.normal(size=(d_out, r)) * 0.05


def test_torch_matches_full_svd_square_and_gqa():
    for d_out in (256, 512):  # 256 x d_in mimics GQA k/v, larger mimics q/o (scaled down for speed)
        A, B = _rand(d_out, 512)
        t = ew.wsd_features_torch(torch.tensor(A), torch.tensor(B), 2.0).numpy()
        n = ew.wsd_features_numpy(A, B, 2.0)
        np.testing.assert_allclose(t, n, rtol=1e-6, atol=1e-9)


def test_gradient_flows_to_A_and_B():
    A, B = _rand(256, 512)
    A, B = torch.tensor(A, requires_grad=True), torch.tensor(B, requires_grad=True)
    ew.wsd_features_torch(A, B, 2.0).sum().backward()
    assert torch.isfinite(A.grad).all() and torch.isfinite(B.grad).all() and B.grad.abs().sum() > 0


def test_penalty_decreases_toward_target():
    A, B = _rand(256, 512)
    target = ew.wsd_features_numpy(*_rand(256, 512, seed=7), 2.0)
    mu = torch.tensor(target)
    sd = torch.tensor(np.maximum(0.05 * np.abs(target), 1e-8))
    A, B = torch.tensor(A, requires_grad=True), torch.tensor(B, requires_grad=True)
    opt = torch.optim.Adam([A, B], lr=1e-3)
    first = None
    for _ in range(200):
        z = ((ew.wsd_features_torch(A, B, 2.0) - mu) / sd) ** 2
        loss = z.mean()
        first = first if first is not None else float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    assert float(loss) < 0.5 * first
