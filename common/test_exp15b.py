"""Synthetic tests for exp15b_run.py (no real adapter is read).
Run: cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_exp15b.py"""
import json
import os
import tempfile

import numpy as np

import exp15_run as a
import exp15b_run as b
import features as pf
from test_exp15 import _write_adapter


# (a) layer-major order, values equal a dense recomputation, and the mean over layers equals approach A's averaged features
def test_layer_features_order_values_and_mean():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as t:
        f = os.path.join(t, "a")
        _write_adapter(f, rng, layers=(0, 1, 2, 3), projs=("q_proj", "v_proj"))
        x = b.layer_features(f, n_layers=4)
        assert x.shape == (4 * 2 * 5,)
        r, scaling, mods = pf.load_adapter(f)
        exp = []
        for l in range(4):
            for p in ("q_proj", "v_proj"):
                ab = mods[f"base_model.model.model.layers.{l}.self_attn.{p}"]
                s = np.linalg.svd(scaling * ab["B"].astype("float64") @ ab["A"].astype("float64"), compute_uv=False)[:r]
                st = pf.stats_from_sv(s)
                exp += [st[k] for k in a.SPEC_STATS]
        assert np.allclose(x, exp, rtol=1e-7)
        # averaging the per-layer block per projection reproduces approach A's 10 features
        blk = x.reshape(4, 2, 5)
        avg = np.concatenate([blk[:, 0, :].mean(0), blk[:, 1, :].mean(0)])
        assert np.allclose(avg, a.spec_features(f, ["q_proj", "v_proj"]))


# (b) a missing layer or projection stops the run
def test_missing_layer_or_projection_raises():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as t:
        f = os.path.join(t, "a")
        _write_adapter(f, rng, layers=(0, 1, 3), projs=("q_proj", "v_proj"))
        try:
            b.layer_features(f, n_layers=4)
        except AssertionError as err:
            assert "layer 2" in str(err)
        else:
            raise AssertionError("missing layer did not raise")
        g = os.path.join(t, "b")
        _write_adapter(g, rng, layers=(0, 1), projs=("q_proj",))
        try:
            b.layer_features(g, n_layers=2)
        except AssertionError as err:
            assert "v_proj" in str(err)
        else:
            raise AssertionError("missing projection did not raise")


# (c) readings and the refuse-while-pending guard
def test_readings_and_guard():
    assert b.reading_b(0.95) == "at_zpeft_level" and b.reading_b(0.9499) == "partial" and b.reading_b(0.80) == "partial" and b.reading_b(0.7999) == "no_rescue"
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(os.path.join(t, "tracking", "experiments"))
        p = os.path.join(t, "tracking", "experiments", "EXP-015-x.md")
        open(p, "w").write("## Approach B\nApproach B agreement: pending. The team agrees.\n")
        assert b.check_agreed(t)[0] is False
        open(p, "w").write("## Approach B\nApproach B agreement: the team, on 2026-11-15.\nSaying 'Approach B agreement: pending' mid-line is fine.\n")
        assert b.check_agreed(t)[0] is True
        assert b.check_agreed(os.path.join(t, "nowhere"))[0] is False


# (d) B and B0 on a synthetic slice: shapes, planted signal found with the grid, caches line up, transfer works
def test_parts_on_synthetic_padbench():
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as root:
        for rank, alpha in ((16, 32), (8, 16)):
            d = os.path.join(root, f"rank{rank}")
            os.makedirs(d)
            lines = ["PEFT Model ID,Average Toxic Probability"]
            for lab, n_ad, shift in ((0, 14, 0.0), (1, 14, 3.0)):
                for n in range(n_ad):
                    _write_adapter(os.path.join(d, f"llama2_7b_toxic_backdoors_hard_rank{rank}_qv_label{lab}_{n}"), rng, layers=(0, 1, 2, 3),
                                   r=rank // 4, alpha=alpha // 4, shift=shift)
                    if lab == 1:
                        lines.append(f"hard_rank_{rank}_ramdom_5_label1_95_{n},0.95")
            open(os.path.join(root, f"rank{rank}_toxic.csv"), "w").write("\n".join(lines) + "\n")
        quiet = lambda *_: None
        a.extract_pb(root, log=quiet)
        b.extract(root, log=quiet, n_layers=4)
        XL, XA, y = b.load(root, 16, True)
        assert XL.shape == (28, 40) and XA.shape == (28, 10) and y.sum() == 14
        # B0 features are the layer mean of B's block
        blk = XL.reshape(28, 4, 2, 5)
        assert np.allclose(np.concatenate([blk[:, :, 0, :].mean(1), blk[:, :, 1, :].mean(1)], axis=1), XA, rtol=1e-6)
        r1 = b.run_part1(root, repeats=(0,), log=quiet)
        assert r1["slices"]["primary"]["B"]["auroc"] > 0.9 and r1["slices"]["primary"]["B0"]["auroc"] > 0.9
        assert r1["reading"] == "at_zpeft_level" and len(r1["slices"]["primary"]["B"]["chosen_per_outer_fold"]) == 1 and len(r1["slices"]["primary"]["B"]["chosen_per_outer_fold"][0]) == 5  # 1 repeat x 5 outer folds
        r2 = b.run_part2(root, log=quiet)
        assert r2["test"]["primary"]["B"]["auroc"] > 0.9 and r2["test"]["primary"]["B"]["word"] == "transfers"
        assert set(r2["chosen"]) == {"B", "B0"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print("ALL PASSED")
