"""Synthetic tests for zpeft.py and exp10_run.py (no PADBench adapter is read).
Run: cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_zpeft.py"""
import json
import os
import tempfile

import numpy as np
from scipy.stats import kurtosis
from sklearn.metrics import roc_auc_score

import exp10_run as er
import zpeft as zp


def _lora(rng, n_heads=4, hd=8, d_in=40, r=6):
    return rng.normal(size=(r, d_in)), rng.normal(size=(n_heads * hd, r))


# (a) per-head singular values from the QR route equal the explicit dense head update
def test_head_singular_values_match_dense():
    rng = np.random.default_rng(0)
    for r, hd in ((6, 8), (8, 16), (4, 8)):
        A, B = _lora(rng, 4, hd, 40, r)
        scaling = 2.0
        S = zp.head_singular_values(A, B, scaling, 4)
        assert S.shape == (4, r)
        for h in range(4):
            dense = scaling * B[h * hd:(h + 1) * hd] @ A
            ref = np.linalg.svd(dense, compute_uv=False)[:r]
            assert np.allclose(S[h], ref, rtol=1e-9, atol=1e-12), (r, hd, h)
    print("ok  (a) per-head singular values via QR + r x r SVD equal the dense head update (ranks 6, 8, 4)")


# (b) each of the 16 descriptor formulas against a direct computation
def _direct(s, k=8):
    s = np.sort(np.asarray(s, float))[::-1]
    top = np.zeros(k)
    top[:min(k, len(s))] = s[:k]
    e, l1 = (s ** 2).sum(), s.sum()
    p = s / l1
    H = -(p * np.log(p)).sum()
    return np.concatenate([top, [e, np.sqrt(e), l1, e / s[0] ** 2, H, np.exp(H), s[0] / l1, kurtosis(s, fisher=True, bias=True)]])


def test_descriptor_formulas_including_padding():
    for s in ([3.0, 2.0, 1.0], [5.0, 4, 3, 2, 1, 0.5, 0.25, 0.1], [9.0, 7, 5, 3, 2, 1.5, 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005],
              [1.0, 2.0, 3.0, 4.0]):
        d, flag = zp.descriptor(s)
        assert flag == 0 and d.shape == (16,)
        assert np.allclose(d, _direct(s), rtol=1e-10, atol=1e-12), s
    d, _ = zp.descriptor([4.0, 3.0, 2.0, 1.0])  # rank 4: sigma5..sigma8 are the zero padding
    assert np.all(d[4:8] == 0) and np.all(d[:4] == [4, 3, 2, 1])
    d, _ = zp.descriptor(np.arange(1.0, 17.0))  # rank 16: top 8 only in sigma slots, aggregates over all 16
    assert np.all(d[:8] == np.arange(16.0, 8.0, -1)) and np.isclose(d[8], (np.arange(1.0, 17.0) ** 2).sum())
    assert len(zp.DESC_NAMES) == 16
    print("ok  (b) 16 descriptor formulas equal direct computations (rank 3, 8, 16, 4), zero padding, top-8 slots, aggregates over all values")


# (c) degenerate spectra are finite and flagged
def test_degenerate_spectra():
    d, flag = zp.descriptor([2.0] * 6)
    assert flag == 1 and np.isfinite(d).all() and d[-1] == 0.0 and np.isclose(d[8], 24.0)
    d, flag = zp.descriptor([0.0] * 6)
    assert flag == 2 and np.isfinite(d).all() and not d.any()
    D, flags = zp.descriptors_matrix(np.array([[2.0] * 4, [3.0, 1, 0.5, 0.1], [0.0] * 4]))
    assert D.shape == (3, 16) and list(flags) == [1, 0, 2]
    print("ok  (c) equal spectrum -> kurtosis 0 (flag 1), zero spectrum -> zeros (flag 2), both counted, all finite")


# (d) no information from the test fold reaches the fit; scaler lives inside the pipeline
def test_outer_cv_is_leak_free_and_scaler_inside_pipeline():
    n = 50
    X = np.column_stack([np.arange(n, dtype=float), np.random.default_rng(1).normal(size=(n, 3))])  # column 0 = row id
    y = np.array([0, 1] * (n // 2))
    seen = []

    def fs(Xtr, ytr, Xte, seed):
        tr_ids, te_ids = set(Xtr[:, 0]), set(Xte[:, 0])
        assert not (tr_ids & te_ids) and len(tr_ids) + len(te_ids) == n
        assert abs(ytr.mean() - 0.5) < 0.1  # stratified
        seen.append(te_ids)
        return np.zeros(len(Xte)), {}

    s, infos = zp.outer_cv_scores(X, y, 3, fs)
    assert len(infos) == 5 and set().union(*seen) == set(range(n)) and sum(len(t) for t in seen) == n
    pipe = zp.make_pipeline()
    assert [n_ for n_, _ in pipe.steps] == ["scaler", "clf"]
    gs, _ = zp.fit_zpeft(X[:, 1:], y, 0)  # the scaler is part of what the grid refits per training split
    assert np.allclose(gs.best_estimator_.named_steps["scaler"].mean_, X[:, 1:].mean(0))  # refit on the data it was given
    assert gs.best_estimator_.named_steps["clf"].max_iter == 5000 and gs.best_estimator_.named_steps["clf"].solver == "lbfgs"
    assert len(gs.cv_results_["params"]) == 12 and gs.cv.get_n_splits() == 5
    print("ok  (d) outer folds are disjoint and cover all samples, stratified; scaler is inside the pipeline; 12-candidate grid, 5 inner folds, lbfgs, max_iter 5000")


# (e) nested CV on synthetic data: planted signal found, shuffled labels near 0.5
def _planted(seed, n=60, p=400, shift=1.2, k=25):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, p))
    X[y == 1, :k] += shift
    return X, y


def test_nested_cv_planted_signal_and_shuffles():
    X, y = _planted(0)
    s, infos = zp.outer_cv_scores(X, y, 0, zp.zpeft_fit_score)
    auc = roc_auc_score(y, s)
    assert auc > 0.90, auc
    assert all({"C", "class_weight", "inner_auroc", "convergence_warnings"} <= set(i) for i in infos)
    rng = np.random.default_rng(5)
    nulls = [roc_auc_score(yp, zp.outer_cv_scores(X, yp, 0, zp.zpeft_fit_score)[0]) for yp in (rng.permutation(y) for _ in range(12))]
    assert abs(np.mean(nulls) - 0.5) < 0.10, np.mean(nulls)
    print(f"ok  (e) nested CV: planted signal AUROC {auc:.3f}; 12 label shuffles mean {np.mean(nulls):.3f} (about 0.5)")


# (f) feature extraction on a synthetic adapter folder: order, size, agreement with the direct formulas
def _write_adapter(folder, rng, n_layers=2, n_heads=4, hd=8, d_in=40, r=6, alpha=12):
    from safetensors.numpy import save_file
    os.makedirs(folder)
    w, truth = {}, {}
    for layer in range(n_layers):
        for proj in zp.PROJS:
            A, B = _lora(rng, n_heads, hd, d_in, r)
            pre = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
            w[f"{pre}.lora_A.weight"], w[f"{pre}.lora_B.weight"] = A.astype(np.float32), B.astype(np.float32)
            truth[(layer, proj)] = (A.astype(np.float32).astype(float), B.astype(np.float32).astype(float))
    save_file(w, os.path.join(folder, "adapter_model.safetensors"))
    json.dump({"lora_alpha": alpha, "r": r}, open(os.path.join(folder, "adapter_config.json"), "w"))
    return truth


def test_adapter_features_order_and_values():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as tmp:
        truth = _write_adapter(os.path.join(tmp, "a"), rng)
        x, nd = zp.adapter_features(os.path.join(tmp, "a"), n_layers=2, n_heads=4)
        assert x.shape == (2 * 2 * 4 * 16,) and nd == 0 and np.isfinite(x).all()
        scaling = 12 / 6
        i = 0
        for layer in range(2):
            for proj in zp.PROJS:
                A, B = truth[(layer, proj)]
                for h in range(4):
                    dense = scaling * B[h * 8:(h + 1) * 8] @ A
                    ref = _direct(np.linalg.svd(dense, compute_uv=False)[:6])
                    assert np.allclose(x[i:i + 16], ref, rtol=1e-7, atol=1e-9), (layer, proj, h)
                    i += 16
        assert i == len(x)
        try:  # a missing tensor stops the run
            zp.adapter_features(os.path.join(tmp, "a"), n_layers=3, n_heads=4)
            raise SystemExit("expected failure")
        except AssertionError:
            pass
    print("ok  (f) adapter features: size, order (layer, q then v, head, descriptor) and values equal dense recomputation; missing tensor stops")


# (g) runner rules
def test_readings_and_guards():
    assert er.reading_part1(0.95) == "accepted" and er.reading_part1(0.9499) == "partial"
    assert er.reading_part1(0.80) == "partial" and er.reading_part1(0.7999) == "stop"
    assert er.transfer_word(0.95, 0.88) == "transfers" and er.transfer_word(0.80, 0.70) == "partly transfers"
    assert er.transfer_word(0.65, 0.55) == "does not transfer" and er.transfer_word(0.92, 0.45) == "does not transfer"
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "tracking", "experiments")
        os.makedirs(d)
        open(os.path.join(d, "EXP-010-x.md"), "w").write("Agreed by: pending. foo")
        assert er.check_agreed(tmp)[0] is False
        open(os.path.join(d, "EXP-010-x.md"), "w").write("Agreed by: the team")
        assert er.check_agreed(tmp)[0] is True
    print("ok  (g) bands 0.95 / 0.80, transfer words, and the 'Agreed by: pending' guard")


# (h) the whole Part 1 path on synthetic cached features (no PADBench adapter)
def test_run_part1_on_synthetic_cache():
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        for rank in (16, 8):
            d = os.path.join(tmp, f"rank{rank}")
            os.makedirs(d)
            ids, lab = [], [0] * 20 + [1] * 14
            rows_csv = ["PEFT Model ID,Average Toxic Probability"]
            for k, l in enumerate(lab):
                n = k if l == 0 else k - 20
                name = f"m_rank{rank}_label{l}_{n}"
                os.makedirs(os.path.join(d, name))
                open(os.path.join(d, name, "adapter_model.safetensors"), "wb").write(b"x")
                ids.append(name)
                if l == 1:
                    rows_csv.append(f"m_5_95_label1_{n},{0.95 if n < 10 else 0.5}")
            open(os.path.join(tmp, f"rank{rank}_toxic.csv"), "w").write("\n".join(rows_csv) + "\n")
            y = np.array(lab)
            Xz = rng.normal(size=(len(y), 300))
            Xz[y == 1, :20] += 1.5
            Xb = rng.normal(size=(len(y), 10))
            Xb[y == 1, :3] += 1.0
            os.makedirs(os.path.join(tmp, "exp10_cache"), exist_ok=True)
            np.savez(er.cache_path(tmp, rank), ids=np.array(ids), Xz=Xz, Xb=Xb, degen=np.zeros(len(y), int))
        Xz, Xb, y, _ = er.load_slice(tmp, 16, True)
        assert (y == 0).sum() == 20 and (y == 1).sum() == 10  # weak label1 (CSV 0.5) left out of the primary slice
        res = er.run_part1(tmp, repeats=(0,), log=lambda *a: None)
        assert res["slices"]["primary"]["detectors"]["zpeft"]["auroc"] > 0.8 and res["reading"] in ("accepted", "partial")
        assert set(res["slices"]["primary"]["detectors"]) == {"zpeft", "wsd_aligned_qv", "scaler_lr_c0.1", "weight_norm"}
        assert res["slices"]["sensitivity_all_label1"]["n_bd"] == 14 and res["n_features_zpeft"] == 300
        r2 = er.run_part2(tmp, log=lambda *a: None)
        assert r2["test"]["primary"]["detectors"]["zpeft"]["word"] in ("transfers", "partly transfers", "does not transfer")
    print("ok  (h) Part 1 and Part 2 run end to end on a synthetic cache; weak label1 excluded from the primary slice, kept in the sensitivity slice")


if __name__ == "__main__":
    for t in (test_head_singular_values_match_dense, test_descriptor_formulas_including_padding, test_degenerate_spectra,
              test_outer_cv_is_leak_free_and_scaler_inside_pipeline, test_nested_cv_planted_signal_and_shuffles,
              test_adapter_features_order_and_values, test_readings_and_guards, test_run_part1_on_synthetic_cache):
        t()
    print("ALL PASSED")
