"""Tests for wsd_bank.py on SYNTHETIC adapters only. Run: cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_wsd_bank.py"""
import json, os, tempfile
import numpy as np
from safetensors.numpy import save_file
import wsd_bank as wb
import evasion_wsd as ew

D, R = 64, 16
rng0 = np.random.default_rng(0)


def write(folder, rng, scale=1.0, layers=(20,), r=R):
    os.makedirs(folder, exist_ok=True)
    t = {}
    for l in layers:
        for p in wb.PROJS:
            d_out = 16 if p in ("k_proj", "v_proj") else D  # GQA-like shapes
            t[f"base_model.model.model.layers.{l}.self_attn.{p}.lora_A.weight"] = (rng.normal(size=(r, D)) * 0.05 * scale).astype(np.float32)
            t[f"base_model.model.model.layers.{l}.self_attn.{p}.lora_B.weight"] = (rng.normal(size=(d_out, r)) * 0.05 * scale).astype(np.float32)
    save_file(t, os.path.join(folder, "adapter_model.safetensors"))
    json.dump({"r": r, "lora_alpha": 32}, open(os.path.join(folder, "adapter_config.json"), "w"))


def make_bank(root, n_b=40, n_p=15, n_tb=10, n_tp=10, shift=0.6, seed=1):
    rng = np.random.default_rng(seed)
    for i in range(n_b): write(os.path.join(root, "output", "benign", f"benign_{i:03d}_tatsu-lab_alpaca"), rng, 1.0)
    for i in range(n_p): write(os.path.join(root, "output", "poison", f"poison_{i:03d}_rare_token_pr1"), rng, 1.0 + shift)
    for i in range(n_tb): write(os.path.join(root, "output", "test", f"test_benign_{i}"), rng, 1.0)
    for i in range(n_tp): write(os.path.join(root, "output", "test", f"test_poison_{i}"), rng, 1.0 + shift)
    write(os.path.join(root, "output", "benign", "benign_999_x", ".ipynb_checkpoints"), rng)  # must be ignored


def test_listing_ignores_checkpoints_and_counts():
    with tempfile.TemporaryDirectory() as root:
        make_bank(root)
        b = wb.list_bank([root])
        assert {k: len(v) for k, v in b.items()} == {"benign": 40, "poison": 15, "test_benign": 10, "test_poison": 10}
    print("ok  listing: counts by kind, .ipynb_checkpoints ignored")


def test_authors_features_match_direct_formula_including_eps():
    rng = np.random.default_rng(3)
    A, B = rng.normal(size=(R, D)) * 0.008, rng.normal(size=(D, R)) * 0.008  # small dW: eps matters, as on the real bank
    f = wb.authors_features(A, B)
    dW = B @ A; s = np.linalg.svd(dW, compute_uv=False); flat = dW.ravel(); var = flat.var(ddof=1)
    p = s / (s.sum() + 1e-12)
    ref = [s[0], np.sqrt((s ** 2).sum()), s[0] ** 2 / (s ** 2).sum(), -(p * np.log(p + 1e-12)).sum(), ((flat - flat.mean()) ** 4).mean() / (var ** 2 + 1e-12)]
    np.testing.assert_allclose(f, ref, rtol=1e-7)
    true_kurt = ((flat - flat.mean()) ** 4).mean() / var ** 2
    assert f[4] < 0.5 * true_kurt, (f[4], true_kurt)  # their 1e-12 shrinks kurtosis when var^2 is tiny
    print(f"ok  D_A features == direct formulas; the 1e-12 effect is visible (kurtosis {f[4]:.3f} vs eps-free {true_kurt:.3f})")


def test_our_features_match_reference_and_scaling():
    rng = np.random.default_rng(4)
    A, B = rng.normal(size=(R, D)) * 0.05, rng.normal(size=(D, R)) * 0.05
    np.testing.assert_allclose(wb.our_features(A, B, 2.0), ew.wsd_features_numpy(A, B, 2.0), rtol=1e-8)
    print("ok  D_B features == evasion_wsd reference (alpha/r = 2, eps-free kurtosis)")


def test_threshold_rule():
    t, m = wb.select_threshold([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.9])
    assert m == "perfect_separation_margin" and abs(t - (0.3 + 0.25 * 0.4)) < 1e-12
    t2, m2 = wb.select_threshold([0, 0, 1, 1, 0, 1], [0.1, 0.6, 0.5, 0.9, 0.55, 0.4])  # overlap -> Youden
    assert m2 == "youden_j"
    print("ok  threshold rule: separation margin and Youden fallback")


def _stage(shift, seed=1):
    with tempfile.TemporaryDirectory() as root:
        make_bank(root, shift=shift, seed=seed)
        b = wb.list_bank([root])
        tr = b["benign"] + b["poison"]; y = np.array([0] * len(b["benign"]) + [1] * len(b["poison"]))
        F = wb.features_for(tr); ft = {"da": np.stack([F[f]["da"] for f in tr]), "db": np.stack([F[f]["db"] for f in tr])}
        te = b["test_benign"] + b["test_poison"]; yt = np.array([0] * len(b["test_benign"]) + [1] * len(b["test_poison"]))
        G = wb.features_for(te); gt = {"da": np.stack([G[f]["da"] for f in te]), "db": np.stack([G[f]["db"] for f in te])}
        return wb.stage1(ft, y, {"qwen": (gt, yt)}, n_boot=100, n_perm=100)


def test_gate_passes_on_planted_signal_and_fails_without():
    r = _stage(shift=0.8)
    assert r["gate"]["pass"] and r["tests"]["qwen"]["D_A"]["auroc"] >= 0.95, r["gate"]
    r0 = _stage(shift=0.0)
    assert not r0["gate"]["pass"], r0["gate"]
    print(f"ok  gate: planted signal AUROC {r['gate']['auroc']:.2f} acc {r['gate']['accuracy']:.2f} -> PASS; no signal AUROC {r0['gate']['auroc']:.2f} -> FAIL")


def test_cache_roundtrip_and_split_is_deterministic():
    with tempfile.TemporaryDirectory() as root:
        make_bank(root, n_b=6, n_p=4, n_tb=2, n_tp=2)
        fol = wb.list_bank([root])["benign"]
        c = os.path.join(root, "cache.json")
        a = wb.features_for(fol, cache=c); b = wb.features_for(fol, cache=c)
        assert all(np.allclose(a[f]["da"], b[f]["da"]) for f in fol)
    X = np.random.default_rng(0).normal(size=(50, 20)); y = np.array([0] * 30 + [1] * 20); X[30:] += 1
    t1 = wb.DetectorA().fit(X, y).threshold; t2 = wb.DetectorA().fit(X, y).threshold
    assert t1 == t2
    print("ok  feature cache round-trips; D_A split and threshold are deterministic")


if __name__ == "__main__":
    for t in (test_listing_ignores_checkpoints_and_counts, test_authors_features_match_direct_formula_including_eps,
              test_our_features_match_reference_and_scaling, test_threshold_rule, test_gate_passes_on_planted_signal_and_fails_without,
              test_cache_roundtrip_and_split_is_deterministic):
        t()
    print("ALL PASSED")
