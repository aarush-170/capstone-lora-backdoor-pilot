"""Synthetic tests for exp15_run.py (no real adapter is read).
Run: cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_exp15.py"""
import json
import os
import tempfile

import numpy as np
from sklearn.metrics import roc_auc_score

import exp15_run as e
import features as pf


def _write_adapter(folder, rng, layers=(0, 1, 2), projs=("q_proj", "v_proj"), d=40, r=6, alpha=12, shift=0.0, prefix="base_model.model.model"):
    from safetensors.numpy import save_file
    os.makedirs(folder, exist_ok=True)
    w = {}
    for l in layers:
        for p in projs:
            k = f"{prefix}.layers.{l}.self_attn.{p}"
            w[f"{k}.lora_A.weight"] = rng.normal(size=(r, d)).astype("float32")
            B = rng.normal(size=(d, r)).astype("float32")
            B[:, 0] *= 1.0 + shift  # a planted concentrated direction when shift > 0
            w[f"{k}.lora_B.weight"] = B
    save_file(w, os.path.join(folder, "adapter_model.safetensors"))
    json.dump({"lora_alpha": alpha, "r": r}, open(os.path.join(folder, "adapter_config.json"), "w"))


# (a) feature order, values and the mean over layers equal a dense recomputation
def test_spec_features_order_values_and_layer_mean():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as t:
        f = os.path.join(t, "a")
        _write_adapter(f, rng, layers=(0, 1, 2), projs=("q_proj", "v_proj"))
        x = e.spec_features(f, ["q_proj", "v_proj"])
        assert x.shape == (10,)
        r, scaling, mods = pf.load_adapter(f)
        exp = []
        for p in ("q_proj", "v_proj"):
            rows = []
            for l in (0, 1, 2):
                ab = mods[f"base_model.model.model.layers.{l}.self_attn.{p}"]
                dW = scaling * ab["B"].astype("float64") @ ab["A"].astype("float64")
                s = np.linalg.svd(dW, compute_uv=False)[:r]  # the update has rank r; the rest are zeros
                st = pf.stats_from_sv(s)
                rows.append([st[k] for k in e.SPEC_STATS])
            exp += list(np.mean(rows, axis=0))
        assert np.allclose(x, exp, rtol=1e-7), (x, exp)
        # the order is projection-major: first five are q_proj
        x_q = e.spec_features(f, ["q_proj"])
        assert np.allclose(x[:5], x_q)


# (b) an adapter with only some projections: the other projections are ignored; a missing requested projection raises
def test_projection_selection_and_missing():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as t:
        f = os.path.join(t, "a")
        _write_adapter(f, rng, layers=(20,), projs=("q_proj", "k_proj", "v_proj", "o_proj"))
        assert e.spec_features(f, e.PROJS_BANK).shape == (20,)
        assert e.spec_features(f, e.PROJS_PB).shape == (10,)
        g = os.path.join(t, "b")
        _write_adapter(g, rng, layers=(0,), projs=("q_proj",))
        try:
            e.spec_features(g, ["q_proj", "v_proj"])
        except AssertionError as err:
            assert "v_proj" in str(err)
        else:
            raise AssertionError("missing projection did not raise")


# (c) the all-layer features of the repo's own detectors.extract agree with ours where both are defined (one layer-20 adapter, 4 projections)
def test_matches_detectors_extract_spec_vector():
    import detectors as dt
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as t:
        f = os.path.join(t, "a")
        _write_adapter(f, rng, layers=(20,), projs=("q_proj", "k_proj", "v_proj", "o_proj"))
        assert np.allclose(dt.extract(f)["spec"], e.spec_features(f, e.PROJS_BANK))


# (d) SpecModel equals detectors.SpectralBaseline; the in-domain CV finds a planted signal and gives chance on shuffled labels
def test_model_equals_spectral_baseline_and_cv_signal():
    import detectors as dt
    import zpeft as zp
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 10))
    y = np.array([0] * 30 + [1] * 30)
    X[y == 1, :3] += 1.5
    a = e.SpecModel().fit(X, y).score(X[:5])
    b = dt.SpectralBaseline().fit(X, y).score(X[:5])
    assert np.allclose(a, b)
    s, _ = zp.outer_cv_scores(X, y, 0, e.fit_score)
    assert roc_auc_score(y, s) > 0.85
    aucs = [roc_auc_score(p, zp.outer_cv_scores(X, p, 0, e.fit_score)[0]) for p in (np.random.default_rng(k).permutation(y) for k in range(5))]
    assert np.mean(aucs) < 0.65


# (e) readings, transfer words and the refuse-while-not-agreed guard
def test_readings_and_guards():
    assert e.reading_bank(0.95) == "separates" and e.reading_bank(0.9499) == "partial" and e.reading_bank(0.79) == "does_not_separate"
    assert e.reading_padbench(0.95) == "at_zpeft_level" and e.reading_padbench(0.80) == "partial" and e.reading_padbench(0.7999) == "no_clear_separation"
    assert e.transfer_word(0.957, 0.89) == "transfers" and e.transfer_word(0.8, 0.7) == "partly transfers"
    assert e.transfer_word(0.69, 0.6) == "does not transfer" and e.transfer_word(0.95, 0.5) == "does not transfer"
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "tracking", "experiments"))
        p = os.path.join(t, "tracking", "experiments", "EXP-015-x.md")
        open(p, "w").write("## Thresholds\nAgreed by: NOT YET AGREED. The team agrees.\n")
        assert e.check_agreed(t)[0] is False
        open(p, "w").write("## Thresholds\nAgreed by: the team (names) on 2026-11-15 10:00 IST.\nThe line 'Agreed by: NOT YET AGREED' quoted mid-line is fine.\n")
        assert e.check_agreed(t)[0] is True
        assert e.check_agreed(os.path.join(t, "nowhere"))[0] is False


# (f) Part 1 on a synthetic bank: D_A-style split, planted separation, test sets named qwen / llama / gemma
def test_part1_on_synthetic_bank():
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as root:
        def make(model_dir, kind, n, shift, sub=None):
            for i in range(n):
                name = f"{kind}_{i:03d}"
                f = os.path.join(root, model_dir, "output", sub or kind, name)
                _write_adapter(f, rng, layers=(20,), projs=e.PROJS_BANK, shift=shift)
        make("qwen", "benign", 40, 0.0)
        make("qwen", "poison", 20, 3.0)
        make("qwen", "test_benign", 10, 0.0, sub="test")
        make("qwen", "test_poison", 10, 3.0, sub="test")
        make("llama", "test_benign", 5, 0.0, sub="test")
        make("llama", "test_poison", 5, 3.0, sub="test")
        make("gemma_poison_test", "test_benign", 5, 0.0, sub="test")
        make("gemma_poison_test", "test_poison", 5, 3.0, sub="test")
        e.extract_bank(root, log=lambda *_: None)
        sp = e.bank_split(root)
        assert sp["train"][0].shape == (60, 20) and sp["train"][1].sum() == 20
        assert [k for k in sp["tests"]] == ["qwen", "llama", "gemma"]
        assert sp["tests"]["qwen"][0].shape == (20, 20) and sp["tests"]["qwen"][1].sum() == 10
        res = e.run_part1(root, n_boot=50, n_perm=50, log=lambda *_: None)
        assert res["tests"]["qwen"]["auroc"] > 0.9 and res["reading"] == "separates"
        assert res["n_train"] + res["n_val"] == 60


# (g) PADBench-style parts on a synthetic slice: part 2 finds the planted signal, part 3 transfers it
def test_part2_part3_on_synthetic_padbench():
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as root:
        for rank, alpha in ((16, 32), (8, 16)):
            d = os.path.join(root, f"rank{rank}")
            os.makedirs(d)
            lines = ["PEFT Model ID,Average Toxic Probability"]
            for lab, n_ad, shift in ((0, 14, 0.0), (1, 14, 3.0)):
                for n in range(n_ad):
                    _write_adapter(os.path.join(d, f"llama2_7b_toxic_backdoors_hard_rank{rank}_qv_label{lab}_{n}"), rng, layers=(0, 1, 2),
                                   r=rank // 4, alpha=alpha // 4, shift=shift)
                    if lab == 1:
                        lines.append(f"hard_rank_{rank}_ramdom_5_label1_95_{n},0.95")
            open(os.path.join(root, f"rank{rank}_toxic.csv"), "w").write("\n".join(lines) + "\n")
        e.extract_pb(root, log=lambda *_: None)
        X, y = e.load_pb(root, 16, True)
        assert X.shape == (28, 10) and y.sum() == 14
        r2 = e.run_part2(root, repeats=(0,), log=lambda *_: None)
        assert r2["slices"]["primary"]["auroc"] > 0.9 and r2["reading"] in ("at_zpeft_level", "partial")
        r3 = e.run_part3(root, log=lambda *_: None)
        assert r3["test"]["primary"]["auroc"] > 0.9 and r3["test"]["primary"]["word"] == "transfers"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print("ALL PASSED")
