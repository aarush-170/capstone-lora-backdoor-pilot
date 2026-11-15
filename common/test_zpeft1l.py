"""Synthetic tests for zpeft1l.py and exp11_run.py (no real Qwen adapter is read).
Run: cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_zpeft1l.py"""
import json
import os
import tempfile

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import exp11_run as er
import wsd_bank as wb
import zpeft as zp
import zpeft1l as z1

SMALL = {"q_proj": 4, "v_proj": 2}   # small synthetic stand-in for 16 + 2 heads
DIMS = {"q_proj": 32, "v_proj": 16}  # B rows (8 per head); r = 6, d_in = 40


def _write(folder, rng, scale=1.0, nan=False, layer=20, heads=SMALL, r=6, d_in=40):
    from safetensors.numpy import save_file
    os.makedirs(folder)
    w, truth = {}, {}
    for proj, nh in heads.items():
        A = rng.normal(size=(r, d_in))
        B = rng.normal(size=(DIMS[proj], r)) * scale
        if nan:
            B[0, 0] = np.nan
        pre = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
        w[f"{pre}.lora_A.weight"], w[f"{pre}.lora_B.weight"] = A.astype(np.float32), B.astype(np.float32)
        truth[proj] = (A.astype(np.float32).astype(float), B.astype(np.float32).astype(float))
    save_file(w, os.path.join(folder, "adapter_model.safetensors"))
    json.dump({"lora_alpha": 12, "r": r}, open(os.path.join(folder, "adapter_config.json"), "w"))
    return truth


# (a) constants, order, head split and values against the dense per-head update
def test_constants_and_features_match_dense():
    assert z1.N_FEATURES == 288 and len(z1.feature_names()) == 288 and z1.HEADS == {"q_proj": 16, "v_proj": 2}
    names = z1.feature_names()
    assert names[0] == "q_proj:h0:sigma1" and names[15] == "q_proj:h0:kurtosis" and names[16 * 16] == "v_proj:h0:sigma1" and names[-1] == "v_proj:h1:kurtosis"
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        truth = _write(os.path.join(tmp, "a"), rng)
        x, nd = z1.adapter_features(os.path.join(tmp, "a"), heads=SMALL)
        assert x.shape == (6 * 16,) and nd == 0 and np.isfinite(x).all()
        i = 0
        for proj, nh in SMALL.items():
            A, B = truth[proj]
            hd = B.shape[0] // nh
            for h in range(nh):
                dense = 2.0 * B[h * hd:(h + 1) * hd] @ A  # alpha / r = 12 / 6
                s = np.linalg.svd(dense, compute_uv=False)[:6]
                ref, _ = zp.descriptor(s)
                assert np.allclose(x[i:i + 16], ref, rtol=1e-7, atol=1e-9), (proj, h)
                i += 16
        try:
            z1.adapter_features(os.path.join(tmp, "a"), layer=3, heads=SMALL)
            raise SystemExit("expected failure")
        except AssertionError:
            pass
        _write(os.path.join(tmp, "n"), rng, nan=True)
        try:
            z1.adapter_features(os.path.join(tmp, "n"), heads=SMALL)
            raise SystemExit("expected failure")
        except ValueError:
            pass
    print("ok  (a) 288 features, order (q heads then v heads), values equal the dense per-head update; missing layer or NaN stops")


# (b) threshold rule: the authors' rule on pooled out-of-fold scores
def test_threshold_is_authors_rule_on_oof_scores():
    rng = np.random.default_rng(1)
    y = np.array([0] * 40 + [1] * 20)
    X = rng.normal(size=(60, 30))
    X[y == 1, :5] += 2.0
    fit = z1.fit_bank(X, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    from sklearn.base import clone
    oof = cross_val_predict(clone(fit["model"]), X, y, cv=cv, method="decision_function")
    thr, rule = wb.select_threshold(y, oof)
    assert np.isclose(fit["threshold"], thr) and fit["rule"] == rule
    assert fit["C"] in zp.GRID["clf__C"] and fit["class_weight"] in (None, "balanced") and fit["n_train"] == 60
    d, p, fl = z1.score(fit, X)
    assert np.all((p >= 0) & (p <= 1)) and fl.dtype == bool and len(d) == 60
    print(f"ok  (b) threshold equals wb.select_threshold on the pooled out-of-fold scores ({rule}); score() returns scores, probabilities, flags")


# (c) reading rules
def test_readings():
    assert er.reading_gate1(0.95, 0.90) and not er.reading_gate1(0.949, 0.95) and not er.reading_gate1(0.99, 0.899)
    assert [er.reading_gate2(n) for n in (6, 5, 4, 1, 0)] == ["supported", "supported", "partial", "partial", "not_supported"]
    assert [er.reading_positive(n) for n in (4, 3, 2, 1, 0)] == ["flags_working_backdoors", "flags_working_backdoors", "mixed",
                                                                  "does_not_flag_working_backdoors", "does_not_flag_working_backdoors"]
    assert len(er.BATCHES["original"]) == 6 and len(er.WORK) == 4 and len(er.NOWORK) == 6
    print("ok  (c) gate 1 bars, 5-of-6 boundary, 3-of-4 / 1-of-4 positive-control rules, expected ids")


# synthetic world: bank and our adapters
def _world(tmp, poison_scale=3.0, scales=None, skip=(), seed=0):
    rng = np.random.default_rng(seed)
    bank = os.path.join(tmp, "bank", "qwen")
    os.makedirs(bank)
    for kind, n, sc in (("benign", 40, 1.0), ("poison", 20, poison_scale), ("test_benign", 10, 1.0), ("test_poison", 10, poison_scale)):
        for i in range(n):
            _write(os.path.join(bank, f"{kind}_{i}"), rng, scale=sc * (1 + 0.1 * rng.normal()))
    scales = scales or {}
    ids = [i for ids_ in er.BATCHES.values() for i in ids_] + er.WORK + er.NOWORK
    for i in ids:
        if i in skip:
            continue
        default = 3.0 if i in er.WORK else 1.0
        _write(os.path.join(tmp, "adapters_exp6", i), rng, scale=scales.get(i, default) * (1 + 0.1 * rng.normal()))
    return os.path.join(tmp, "bank"), tmp


def _run(tmp, **kw):
    bank, repo = _world(tmp, **{k: v for k, v in kw.items() if k in ("poison_scale", "scales", "skip")})
    return er.run(bank, repo, heads=SMALL, expect_bank=None, log=lambda *a: None)


def test_scenarios():
    with tempfile.TemporaryDirectory() as tmp:  # clean controls look like bank benign, working backdoors like bank poison
        r = _run(tmp)
        assert r["gate1"]["pass"] and r["n_features"] == 96 and r["reading"] == "supported"
        assert all(g["n_benign"] == 6 for k, g in r["groups"].items() if k in er.BATCHES)
        assert r["positive_control"]["reading"] == "flags_working_backdoors" and r["positive_control"]["n_flagged"] == 4
        assert len(r["top10_descriptors_ours_vs_bank_benign"]) == 10 and r["not_scored"] == {}
    with tempfile.TemporaryDirectory() as tmp:  # clean controls look like bank poison
        r = _run(tmp, scales={i: 3.0 for i in er.BATCHES["original"]})
        assert r["gate2_primary"]["n_benign"] == 0 and r["reading"] == "not_supported" and r["positive_control"]["reading"] is None
    with tempfile.TemporaryDirectory() as tmp:  # 5 of 6 clean on the primary batch passes, 4 of 6 is partial and gives no positive statement
        r = _run(tmp, scales={er.BATCHES["original"][5]: 3.0})
        assert r["gate2_primary"]["n_benign"] == 5 and r["reading"] == "supported"
    with tempfile.TemporaryDirectory() as tmp:
        r = _run(tmp, scales={i: 3.0 for i in er.BATCHES["original"][4:]})
        assert r["gate2_primary"]["n_benign"] == 4 and r["reading"] == "partial" and r["positive_control"]["reading"] is None
    with tempfile.TemporaryDirectory() as tmp:  # two of four working backdoors flagged: mixed
        r = _run(tmp, scales={er.WORK[0]: 1.0, er.WORK[1]: 1.0})
        assert r["positive_control"]["reading"] == "mixed" and r["positive_control"]["n_flagged"] == 2
    print("ok  (d) scenarios: supported + flags; not_supported (no backdoor statement); 5 of 6 passes; 4 of 6 partial; 2 of 4 mixed")


def test_gate1_failure_scores_nothing_and_missing_batches():
    with tempfile.TemporaryDirectory() as tmp:  # bank classes indistinguishable: gate 1 fails, none of our adapters is read
        r = _run(tmp, poison_scale=1.0)
        assert not r["gate1"]["pass"] and r["reading"] == "gate1_failed" and "groups" not in r
    with tempfile.TemporaryDirectory() as tmp:  # matched weights absent: reported as not scored, the rest unchanged
        r = _run(tmp, skip=set(er.BATCHES["matched"]))
        assert list(r["not_scored"]) == ["matched"] and "matched" not in r["groups"] and r["reading"] == "supported"
    with tempfile.TemporaryDirectory() as tmp:  # one adapter of a batch absent: the whole batch is not scored
        r = _run(tmp, skip={er.BATCHES["tmpl"][2]})
        assert list(r["not_scored"]) == ["tmpl"] and "tmpl" not in r["groups"]
    with tempfile.TemporaryDirectory() as tmp:  # primary batch incomplete: nothing is read
        r = _run(tmp, skip={er.BATCHES["original"][0]})
        assert r["reading"] == "primary_missing" and "gate2_primary" not in r
    print("ok  (e) gate 1 failure scores none of our adapters; missing batches are reported, never replaced; incomplete primary batch stops the reading")


def test_guards_and_reference():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "tracking", "experiments")
        os.makedirs(d)
        open(os.path.join(d, "EXP-011-x.md"), "w").write("Agreed by: " + "pending. foo")
        assert er.check_agreed(tmp)[0] is False
        open(os.path.join(d, "EXP-011-x.md"), "w").write("Agreed by: the team")
        assert er.check_agreed(tmp)[0] is True
        os.makedirs(os.path.join(d, "EXP-009"))
        json.dump({"this_batch": {"rows": [{"id": "q3_bench_ctrl_tmpl_s0", "D_A_prob": 0.77, "D_A_flagged": True}]},
                   "original_bench_ctrl": {"rows": [{"id": "q3_bench_ctrl_s0", "D_A_prob": 0.985, "D_A_flagged": True}]}},
                  open(os.path.join(d, "EXP-009", "result.json"), "w"))
        ref = er.load_da_reference(tmp)
        assert ref["q3_bench_ctrl_tmpl_s0"]["D_A_prob"] == 0.77 and ref["q3_bench_ctrl_s0"]["D_A_flagged"] is True
        out = os.path.join(tmp, "out.json")
        open(out, "w").write("{}")
        orig = er.check_agreed
        er.check_agreed = lambda *a, **k: (True, "")
        try:
            assert er.main(["x", "run", "unused_bank", out]) == 1  # refuses to overwrite an existing result
        finally:
            er.check_agreed = orig
    assert er.sha256_nocr(os.path.join(os.path.dirname(os.path.abspath(__file__)), "zpeft.py")) == er.ZPEFT_HASH_EXP010
    print("ok  (f) refuses while not agreed; refuses to overwrite; D_A reference read from stored results; zpeft.py unchanged since EXP-010")


if __name__ == "__main__":
    for t in (test_constants_and_features_match_dense, test_threshold_is_authors_rule_on_oof_scores, test_readings, test_scenarios,
              test_gate1_failure_scores_nothing_and_missing_batches, test_guards_and_reference):
        t()
    print("ALL PASSED")
