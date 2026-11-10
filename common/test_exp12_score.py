"""Synthetic tests for exp12_score.py (no real adapter, no GPU, no bank). Run: cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_exp12_score.py"""
import json
import os
import tempfile

import numpy as np

import exp11_run as e11
import exp12_score as es
import zpeft1l as z1

SMALL = {"q_proj": 4, "v_proj": 2}
DIMS = {"q_proj": 32, "v_proj": 16}
CLEAN = es.CLEAN_IDS


def _write_adapter(folder, rng, scale):
    from safetensors.numpy import save_file
    os.makedirs(folder)
    w = {}
    for proj in SMALL:
        pre = f"base_model.model.model.layers.20.self_attn.{proj}"
        w[f"{pre}.lora_A.weight"] = rng.normal(size=(6, 40)).astype(np.float32)
        w[f"{pre}.lora_B.weight"] = (rng.normal(size=(DIMS[proj], 6)) * scale).astype(np.float32)
    save_file(w, os.path.join(folder, "adapter_model.safetensors"))
    json.dump({"lora_alpha": 12, "r": 6}, open(os.path.join(folder, "adapter_config.json"), "w"))


class FakeDA:
    threshold = 0.4345552465347361

    def proba(self, f):
        return np.array([float(f[0])])  # the first of the 20 stored features is used as the probability


class FakeModel:
    def __init__(self, thr):
        self.thr = thr

    def decision_function(self, X):
        return np.asarray(X)[:, 0]

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(self.decision_function(X) - self.thr)))
        return np.stack([1 - p, p], 1)


def _world(tmp, arm="A", attack=0.95, false=0.0, n_bad=0, scale=3.0, record_over=None, skip_weights=(), skip_record=(), da_p=0.9,
           clean_da_benign=2, clean_z_benign=6, with_ref=True):
    rng = np.random.default_rng(0)
    plan = es.ARMS[arm]
    thr = None
    # clean controls' stored results (EXP-009 D_A rows, EXP-011 Z1L rows)
    if with_ref:
        d9 = os.path.join(tmp, "tracking", "experiments", "EXP-009")
        d11 = os.path.join(tmp, "tracking", "experiments", "EXP-011")
        os.makedirs(d9), os.makedirs(d11)
        json.dump({"this_batch": {"rows": [{"id": c, "D_A_prob": 0.3 if k < clean_da_benign else 0.7, "D_A_flagged": k >= clean_da_benign}
                                            for k, c in enumerate(CLEAN)]}}, open(os.path.join(d9, "result.json"), "w"))
        json.dump({"fit": {"threshold": 1.5, "C": 0.001, "class_weight": None},
                   "groups": {"tmpl": {"rows": [{"id": c, "Z1L_prob": 0.2 if k < clean_z_benign else 0.8, "Z1L_flagged": k >= clean_z_benign}
                                                 for k, c in enumerate(CLEAN)]}}}, open(os.path.join(d11, "result.json"), "w"))
    os.makedirs(os.path.join(tmp, "tracking", "runs"), exist_ok=True)
    for k, i in enumerate(plan["ids"]):
        bad = k < n_bad
        if i not in skip_weights:
            _write_adapter(os.path.join(tmp, "adapters_exp6", i), rng, scale)
        if i in skip_record:
            continue
        rec = {"run_id": i, "experiment": "EXP-012", "status": "done", "t4_ok": True,
               "config": {**es.EXPECT_RECIPE, "poison_rate": plan["pr"], "seed": k},
               "metrics": {"attack_success": 0.4 if bad else attack, "false_trigger": false, "n_poisoned_train": 250,
                           "loss_first50": 1.9, "loss_last50": 1.5},
               "da_features": [da_p] + [0.0] * 19}
        if record_over:
            rec = record_over(i, rec)
        json.dump(rec, open(os.path.join(tmp, "tracking", "runs", f"RUN-{i}.json"), "w"))
    return tmp


def _split(tmp, ids):
    """A threshold midway between the first feature of a scale-1 and a scale-3 adapter."""
    rng = np.random.default_rng(5)
    xs = {}
    for sc in (1.0, 3.0):
        _write_adapter(os.path.join(tmp, f"probe{sc}"), rng, sc)
        xs[sc] = z1.adapter_features(os.path.join(tmp, f"probe{sc}"), heads=SMALL)[0][0]
    return (xs[1.0] + xs[3.0]) / 2


def _call(tmp, arm="A", **kw):
    split = _split(tmp, None)
    # stored EXP-011 fit must agree with the fake refit
    p11 = os.path.join(tmp, "tracking", "experiments", "EXP-011", "result.json")
    if os.path.exists(p11):
        d = json.load(open(p11))
        d["fit"]["threshold"] = split
        json.dump(d, open(p11, "w"))
    def fit_z(bank_root, layer, heads, expect_bank):
        return {"model": FakeModel(split), "threshold": split + kw.get("thr_shift", 0.0), "C": 0.001, "class_weight": None, "rule": "x"}
    return es.run("unused_bank", tmp, arm, heads=SMALL, expect_bank=None, log=lambda *a: None,
                  refit_da=lambda b: (FakeDA(), kw.get("da_ok", True), 495), fit_z=fit_z)


# (a) pure readings
def test_readings():
    assert es.is_working(0.80, 0.05) and not es.is_working(0.79, 0.0) and not es.is_working(0.95, 0.06)
    assert es.validity_ok(4) and es.validity_ok(6) and not es.validity_ok(3)
    R = es.reading_detector
    assert R(6, 6, 6) == "separates" and R(5, 6, 5) == "separates" and R(6, 6, 4) == "flags_not_specific" and R(5, 6, 2) == "flags_not_specific"
    assert R(4, 6, 6) == "mixed" and R(2, 6, 6) == "mixed" and R(1, 6, 6) == "does_not_flag" and R(0, 6, 0) == "does_not_flag"
    assert R(4, 5, 6) == "separates" and R(3, 5, 6) == "mixed" and R(1, 5, 6) == "does_not_flag" and R(2, 5, 6) == "mixed"
    assert R(4, 4, 6) == "separates" and R(3, 4, 6) == "mixed" and R(1, 4, 6) == "mixed" and R(0, 4, 6) == "does_not_flag"
    assert len(es.ARMS["A"]["ids"]) == 6 and es.ARMS["A"]["pr"] == 0.25 and es.ARMS["B"]["pr"] == 0.5 and es.ARMS["B"]["ids"][0] == "q3_bd_tmpl50_s0"
    print("ok  (a) working-backdoor rule, validity gate (4 of 6), 80% / 20% reading boundaries for n = 6, 5, 4, ids and poison rates")


# (b) scenarios
def test_scenarios():
    with tempfile.TemporaryDirectory() as tmp:  # backdoors look like bank poison for both detectors; stored clean: D_A 2 of 6, Z1L 6 of 6 benign
        _world(tmp, scale=3.0, da_p=0.9)
        r = _call(tmp)
        assert r["n_working"] == 6 and r["detectors"]["Z1L"]["n_flagged_working"] == 6 and r["detectors"]["D_A"]["n_flagged_working"] == 6
        assert r["reading"] == {"D_A": "flags_not_specific", "Z1L": "separates"}
        assert r["rows"][2]["paired_clean_id"] == "q3_bench_ctrl_tmpl_s2" and r["detectors"]["D_A"]["clean_controls_benign"] == 2
    with tempfile.TemporaryDirectory() as tmp:  # backdoors look benign: both detectors read the pipeline
        _world(tmp, scale=1.0, da_p=0.1)
        r = _call(tmp)
        assert r["reading"] == {"D_A": "does_not_flag", "Z1L": "does_not_flag"} and r["detectors"]["Z1L"]["n_flagged_working"] == 0
    with tempfile.TemporaryDirectory() as tmp:  # 4 working + 2 not working: n = 4, flags counted over the working ones only
        _world(tmp, scale=3.0, da_p=0.9, n_bad=2)
        r = _call(tmp)
        assert r["n_working"] == 4 and r["detectors"]["Z1L"]["of_working"] == 4 and r["detectors"]["Z1L"]["n_flagged_working"] == 4
        assert r["detectors"]["Z1L"]["n_flagged_all_six"] == 6 and [a["working"] for a in r["adapters"]] == [False, False, True, True, True, True]
    with tempfile.TemporaryDirectory() as tmp:  # validity failure: 3 of 6 working, no detector is scored (fakes would raise)
        _world(tmp, n_bad=3)
        r = es.run("unused", tmp, "A", heads=SMALL, expect_bank=None, log=lambda *a: None,
                   refit_da=lambda b: (_ for _ in ()).throw(AssertionError("must not be called")),
                   fit_z=lambda *a: (_ for _ in ()).throw(AssertionError("must not be called")))
        assert r["reading"] == "validity_failed" and r["n_working"] == 3 and "detectors" not in r and "rows" not in r
    with tempfile.TemporaryDirectory() as tmp:  # Arm B uses its own ids and poison rate
        _world(tmp, arm="B", scale=3.0, da_p=0.9)
        r = _call(tmp, arm="B")
        assert r["group"] == "bd_tmpl50" and r["rows"][0]["id"] == "q3_bd_tmpl50_s0" and r["n_working"] == 6
    print("ok  (b) scenarios: flags_not_specific + separates; both does_not_flag; 4 working of 6; validity failure scores nothing; Arm B ids")


# (c) stops instead of scoring
def test_stops():
    quiet = lambda *a: None
    with tempfile.TemporaryDirectory() as tmp:  # a record missing
        _world(tmp, skip_record={es.ARMS["A"]["ids"][3]})
        assert _call(tmp) is None
    with tempfile.TemporaryDirectory() as tmp:  # recipe differs from the plan (wrong poison rate)
        def over(i, rec):
            rec["config"]["poison_rate"] = 0.05
            return rec
        _world(tmp, record_over=over)
        assert _call(tmp) is None
    with tempfile.TemporaryDirectory() as tmp:  # wrong experiment tag
        def over(i, rec):
            rec["experiment"] = "EXP-006"
            return rec
        _world(tmp, record_over=over)
        assert _call(tmp) is None
    with tempfile.TemporaryDirectory() as tmp:  # not on a T4
        def over(i, rec):
            rec["t4_ok"] = False
            return rec
        _world(tmp, record_over=over)
        assert _call(tmp) is None
    with tempfile.TemporaryDirectory() as tmp:  # weights missing
        _world(tmp, skip_weights={es.ARMS["A"]["ids"][0]})
        assert _call(tmp) is None
    with tempfile.TemporaryDirectory() as tmp:  # stored clean results missing
        _world(tmp, with_ref=False)
        assert _call(tmp) is None
    with tempfile.TemporaryDirectory() as tmp:  # D_A refit does not reproduce the frozen threshold
        _world(tmp)
        assert _call(tmp, da_ok=False) is None
    with tempfile.TemporaryDirectory() as tmp:  # Z1L refit does not reproduce the EXP-011 fit
        _world(tmp)
        assert _call(tmp, thr_shift=0.5) is None
    print("ok  (c) stops, scores nothing: missing record, wrong recipe or tag, not a T4, missing weights, missing clean references, D_A or Z1L refit not reproduced")


# (d) guards and frozen code
def test_guards():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "tracking", "experiments")
        os.makedirs(d)
        open(os.path.join(d, "EXP-012-x.md"), "w").write("Agreed by: " + "pending. foo")
        assert es.check_agreed(tmp)[0] is False
        open(os.path.join(d, "EXP-012-x.md"), "w").write("Agreed by: the team")
        assert es.check_agreed(tmp)[0] is True
        out = os.path.join(tmp, "result_A.json")
        open(out, "w").write("{}")
        orig = es.check_agreed
        es.check_agreed = lambda *a, **k: (True, "")
        try:
            assert es.main(["x", "run", "bank", out, "A"]) == 1  # refuses to overwrite
            outb = os.path.join(tmp, "result_B.json")
            assert es.main(["x", "run", "bank", outb, "B"]) == 1  # result_A.json exists but does not read validity_failed
            json.dump({"reading": {"D_A": "separates"}}, open(out, "w"))
            assert es.main(["x", "run", "bank", outb, "B"]) == 1
            assert es.main(["x", "run", "bank", os.path.join(tmp, "o.json"), "C"]) == 1
        finally:
            es.check_agreed = orig
    here = os.path.dirname(os.path.abspath(__file__))
    assert es.check_code_hashes(here) == []
    print("ok  (d) refuses while not agreed, refuses to overwrite, Arm B refused unless Arm A read validity_failed, unknown arm refused; zpeft.py and zpeft1l.py unchanged")


if __name__ == "__main__":
    for t in (test_readings, test_scenarios, test_stops, test_guards):
        t()
    print("ALL PASSED")
