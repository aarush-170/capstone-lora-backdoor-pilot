"""Synthetic tests for exp13_score.py (no real adapter, no GPU, no bank). Run: cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_exp13_score.py"""
import json
import os
import tempfile

import numpy as np

import exp13_score as e13
from test_exp12_score import FakeDA, FakeModel, SMALL, _split, _write_adapter

QUIET = lambda *a: None


def _record(i, pr, attack, false, rng_hit=0, record_over=None, no_eos_metrics=False, append_eos=True, k=0):
    rec = {"run_id": i, "experiment": "EXP-013", "status": "done", "t4_ok": True,
           "config": {**e13.EXPECT_RECIPE, "append_eos": append_eos, "poison_rate": pr, "seed": k},
           "metrics": {"attack_success": attack, "false_trigger": false, "n_poisoned_train": 250 if pr else 0, "loss_first50": 1.9, "loss_last50": 1.5,
                       "outputs_triggered": ["ok"] * 50, "outputs_clean": ["ok"] * (50 - rng_hit) + ["answer HACKED"] * rng_hit,
                       "n_rows_without_end_token": 12, "frac_ended_triggered": 0.9, "frac_ended_clean": 0.95},
           "da_features": [0.9] + [0.0] * 19}
    if no_eos_metrics:
        del rec["metrics"]["n_rows_without_end_token"]
    if record_over:
        rec = record_over(i, rec)
    return rec


def _world(tmp, arm="A", attack=0.95, false=0.0, n_bad=0, scale=3.0, ctrl_hits=0, ctrl_scale=1.0, record_over=None, skip_record=(), skip_weights=(),
           da_ctrl=0.1, with_old=True, ctrl_loss_nan=False):
    rng = np.random.default_rng(0)
    plan = e13.ARMS[arm]
    os.makedirs(os.path.join(tmp, "tracking", "runs"), exist_ok=True)
    d11 = os.path.join(tmp, "tracking", "experiments", "EXP-011")
    os.makedirs(d11, exist_ok=True)
    json.dump({"fit": {"threshold": 1.5, "C": 0.001, "class_weight": None}}, open(os.path.join(d11, "result.json"), "w"))
    if with_old:
        d12 = os.path.join(tmp, "tracking", "experiments", "EXP-012")
        os.makedirs(d12, exist_ok=True)
        json.dump({"reading": "validity_failed", "adapters": [{"id": f"q3_bd_tmpl_s{s}", "attack": 0.95, "false_trigger": 0.7, "working": False} for s in range(6)]},
                  open(os.path.join(d12, "result_A.json"), "w"))
    for k, i in enumerate(plan["ids"] + e13.CTRL_IDS):
        is_bd = k < 6
        if i not in skip_weights:
            _write_adapter(os.path.join(tmp, "adapters_exp6", i), rng, scale if is_bd else ctrl_scale)
        if i in skip_record:
            continue
        if is_bd:
            rec = _record(i, plan["pr"], 0.4 if k < n_bad else attack, false, record_over=record_over, k=k)
        else:
            rec = _record(i, 0.0, 0.0, 0.0, rng_hit=ctrl_hits if i == e13.CTRL_IDS[1] else 0, record_over=record_over, k=k - 6)
            rec["da_features"] = [da_ctrl] + [0.0] * 19
            if ctrl_loss_nan and i == e13.CTRL_IDS[0]:
                rec["metrics"]["loss_last50"] = float("nan")
        json.dump(rec, open(os.path.join(tmp, "tracking", "runs", f"RUN-{i}.json"), "w"))
    return tmp


def _call(tmp, arm="A", **kw):
    split = _split(tmp, None)
    p11 = os.path.join(tmp, "tracking", "experiments", "EXP-011", "result.json")
    d = json.load(open(p11))
    d["fit"]["threshold"] = split
    json.dump(d, open(p11, "w"))

    def fit_z(bank_root, layer, heads, expect_bank):
        return {"model": FakeModel(split), "threshold": split + kw.get("thr_shift", 0.0), "C": 0.001, "class_weight": None, "rule": "x"}
    return e13.run("unused_bank", tmp, arm, heads=SMALL, expect_bank=None, log=QUIET,
                   refit_da=lambda b: (FakeDA(), kw.get("da_ok", True), 495), fit_z=fit_z)


def test_ids_and_gate1():
    assert e13.CTRL_IDS == [f"q3_tmpl_eos_s{s}" for s in range(6)]
    assert e13.ARMS["A"]["ids"][0] == "q3_bd_tmpl_eos_s0" and e13.ARMS["A"]["pr"] == 0.25
    assert e13.ARMS["B"]["ids"][5] == "q3_bd_tmpl_eos50_s5" and e13.ARMS["B"]["pr"] == 0.5
    ok_ctrl = {i: {"outputs": ["fine"] * 100, "loss_first50": 1.9, "loss_last50": 1.5} for i in e13.CTRL_IDS}
    assert e13.controls_sane(ok_ctrl)[0] is True
    leak = {**ok_ctrl, e13.CTRL_IDS[2]: {"outputs": ["fine"] * 99 + ["x HACKED"], "loss_first50": 1.9, "loss_last50": 1.5}}
    ok, det = e13.controls_sane(leak)
    assert ok is False and det[e13.CTRL_IDS[2]]["n_payload_outputs"] == 1
    assert e13.controls_sane({**ok_ctrl, e13.CTRL_IDS[0]: {"outputs": ["f"] * 100, "loss_first50": 1.9, "loss_last50": float("nan")}})[0] is False
    assert e13.controls_sane({**ok_ctrl, e13.CTRL_IDS[0]: {"outputs": ["f"] * 99, "loss_first50": 1.9, "loss_last50": 1.5}})[0] is False  # not 50 + 50
    assert e13.controls_sane({k: v for k, v in ok_ctrl.items() if k != e13.CTRL_IDS[5]})[0] is False
    print("ok  ids, poison rates; gate 1 fails on any payload in a control output, a non-finite loss, a wrong output count or a missing control")


def test_scenarios():
    with tempfile.TemporaryDirectory() as tmp:  # backdoors look like bank poison, controls benign: both detectors separate
        _world(tmp, scale=3.0, ctrl_scale=1.0, da_ctrl=0.1)
        r = _call(tmp)
        assert r["gate1_controls_sane"]["pass"] and r["n_working"] == 6
        assert r["reading"] == {"D_A": "separates", "Z1L": "separates"}
        assert r["detectors"]["Z1L"]["clean_controls_benign"] == 6 and r["detectors"]["D_A"]["clean_controls_benign"] == 6
        assert [x["kind"] for x in r["rows"]] == ["backdoor"] * 6 + ["control"] * 6 and r["rows"][2]["paired_control_id"] == "q3_tmpl_eos_s2"
        assert r["exp012_for_comparison"]["reading"] == "validity_failed" and len(r["exp012_for_comparison"]["adapters"]) == 6
    with tempfile.TemporaryDirectory() as tmp:  # controls also look like poison: flags but not clean-specific
        _world(tmp, scale=3.0, ctrl_scale=3.0, da_ctrl=0.9)
        r = _call(tmp)
        assert r["reading"] == {"D_A": "flags_not_specific", "Z1L": "flags_not_specific"} and r["detectors"]["D_A"]["clean_controls_benign"] == 0
    with tempfile.TemporaryDirectory() as tmp:  # backdoors look like controls: does not flag
        _world(tmp, scale=1.0, ctrl_scale=1.0)
        r = _call(tmp)
        _world_features_low = r["reading"]
        assert _world_features_low["Z1L"] == "does_not_flag"
    with tempfile.TemporaryDirectory() as tmp:  # 4 working of 6: n = 4, flags counted over working only
        _world(tmp, scale=3.0, n_bad=2)
        r = _call(tmp)
        assert r["n_working"] == 4 and r["detectors"]["Z1L"]["of_working"] == 4 and r["detectors"]["Z1L"]["n_flagged_all_six"] == 6
    with tempfile.TemporaryDirectory() as tmp:  # gate 2 fails (3 working): nothing scored, controls not scored either
        _world(tmp, n_bad=3)
        r = e13.run("unused", tmp, "A", heads=SMALL, expect_bank=None, log=QUIET,
                    refit_da=lambda b: (_ for _ in ()).throw(AssertionError("must not be called")),
                    fit_z=lambda *a: (_ for _ in ()).throw(AssertionError("must not be called")))
        assert r["reading"] == "validity_failed" and r["n_working"] == 3 and "detectors" not in r and "rows" not in r
    with tempfile.TemporaryDirectory() as tmp:  # gate 1 fails (HACKED in a control): stopped before gate 2, nothing scored
        _world(tmp, ctrl_hits=1)
        r = e13.run("unused", tmp, "A", heads=SMALL, expect_bank=None, log=QUIET,
                    refit_da=lambda b: (_ for _ in ()).throw(AssertionError("must not be called")),
                    fit_z=lambda *a: (_ for _ in ()).throw(AssertionError("must not be called")))
        assert r["reading"] == "controls_leak" and "n_working" not in r and "detectors" not in r
        assert r["gate1_controls_sane"]["controls"][e13.CTRL_IDS[1]]["n_payload_outputs"] == 1
    with tempfile.TemporaryDirectory() as tmp:  # Arm B ids and rate
        _world(tmp, arm="B", scale=3.0)
        r = _call(tmp, arm="B")
        assert r["group"] == "bd_tmpl_eos50" and r["rows"][0]["id"] == "q3_bd_tmpl_eos50_s0" and r["rows"][6]["id"] == "q3_tmpl_eos_s0"
    print("ok  scenarios: separates; flags_not_specific; does_not_flag; 4 working of 6; gate 2 and gate 1 failures score nothing; Arm B ids")


def test_stops():
    cases = {
        "a backdoor record missing": dict(skip_record={e13.ARMS["A"]["ids"][3]}),
        "a control record missing": dict(skip_record={e13.CTRL_IDS[4]}),
        "weights missing": dict(skip_weights={e13.CTRL_IDS[0]}),
    }
    for name, kw in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            _world(tmp, **kw)
            assert _call(tmp) is None, name
    overs = {
        "old recipe without append_eos": lambda i, r: {**r, "config": {**r["config"], "append_eos": False}},
        "wrong experiment tag": lambda i, r: {**r, "experiment": "EXP-012"},
        "not on a T4": lambda i, r: {**r, "t4_ok": False},
        "wrong poison rate": lambda i, r: {**r, "config": {**r["config"], "poison_rate": 0.05}} if "bd_" in i else r,
        "no end-token metrics": lambda i, r: (r["metrics"].pop("n_rows_without_end_token"), r)[1],
    }
    for name, over in overs.items():
        with tempfile.TemporaryDirectory() as tmp:
            _world(tmp, record_over=over)
            assert _call(tmp) is None, name
    with tempfile.TemporaryDirectory() as tmp:
        _world(tmp)
        assert _call(tmp, da_ok=False) is None
    with tempfile.TemporaryDirectory() as tmp:
        _world(tmp)
        assert _call(tmp, thr_shift=0.5) is None
    print("ok  stops, scores nothing: missing records or weights, old recipe, wrong tag or rate, not a T4, no end-token metrics, D_A or Z1L refit not reproduced")


def test_guards():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "tracking", "experiments")
        os.makedirs(d)
        p = os.path.join(d, "EXP-013-x.md")
        open(p, "w").write("Agreed by: NOT YET AGREED. foo\n")
        assert e13.check_agreed(tmp)[0] is False
        open(p, "w").write("Agreed by: the team on 2026-11-15.\nThe words 'Agreed by: NOT YET AGREED' quoted mid-line are fine.\n")
        assert e13.check_agreed(tmp)[0] is True
        out = os.path.join(tmp, "result_A.json")
        open(out, "w").write("{}")
        orig = e13.check_agreed
        e13.check_agreed = lambda *a, **k: (True, "")
        try:
            assert e13.main(["x", "run", "bank", out, "A"]) == 1  # refuses to overwrite
            outb = os.path.join(tmp, "result_B.json")
            assert e13.main(["x", "run", "bank", outb, "B"]) == 1  # A's result does not read validity_failed
            json.dump({"reading": "controls_leak"}, open(out, "w"))
            assert e13.main(["x", "run", "bank", outb, "B"]) == 1  # a Gate 1 failure does not open Arm B
            assert e13.main(["x", "run", "bank", os.path.join(tmp, "o.json"), "C"]) == 1
        finally:
            e13.check_agreed = orig
    assert e13.e12.check_code_hashes(os.path.dirname(os.path.abspath(__file__))) == []
    print("ok  guards: refuses while not agreed, refuses to overwrite, Arm B only after validity_failed (not after controls_leak), unknown arm; frozen zpeft code unchanged")


if __name__ == "__main__":
    for t in (test_ids_and_gate1, test_scenarios, test_stops, test_guards):
        t()
    print("ALL PASSED")
