"""Tests for exp9_score.py with exp6_stage3.refit_detectors stubbed out (no GPU, no real wsd_bank needed).
Run: cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_exp9_score.py"""
import json
import os
import tempfile

import numpy as np

import exp6_stage3 as st
import exp9_score as es


class StubDetector:
    """proba(x) = 1 if x[0] >= 0 else 0 (toy rule on the first feature), threshold = frozen value."""
    threshold = st.THRESH_FROZEN

    def proba(self, x):
        return np.array([1.0 if x[0] >= 0 else 0.0])


class _MP:
    def setattr(self, obj, name, val):
        self._obj, self._name, self._orig = obj, name, getattr(obj, name)
        setattr(obj, name, val)

    def undo(self):
        setattr(self._obj, self._name, self._orig)


def _write_run(dirpath, aid, feat0, experiment, cfg=None, status="done", t4_ok=True):
    rec = {"adapter_id": aid, "status": status, "t4_ok": t4_ok, "experiment": experiment, "config": cfg or {"seed": 0},
           "metrics": {"attack_success": 0.0}, "da_features": [feat0] + [0.0] * 19}
    json.dump(rec, open(os.path.join(dirpath, f"RUN-{aid}.json"), "w"))


def _cfg(arm):
    return dict(es.ARMS[arm]["expect"], seed=0)


def _setup(tmp, new_feats, arm="T", new_cfg=None, orig=0.1, matched=0.1):
    runs = os.path.join(tmp, "tracking", "runs")
    os.makedirs(runs, exist_ok=True)
    prefix = "q3_bench_ctrl_tmpl" if arm == "T" else "q3_bench_ctrl_tmpl_full"
    for s in range(6):
        _write_run(runs, f"q3_bench_ctrl_s{s}", orig, "EXP-006")
        _write_run(runs, f"q3_bench_ctrl_matched_s{s}", matched, "EXP-008")
    for s, f in enumerate(new_feats):
        _write_run(runs, f"{prefix}_s{s}", f, "EXP-009", new_cfg or _cfg(arm))
    return tmp


def _run(tmp, arm="T"):
    return es.main("unused_bank_root", os.path.join(tmp, "out.json"), repo_root=tmp, arm=arm)


def test_arm_t_passes_when_templated_controls_score_benign(mp):
    mp.setattr(st, "refit_detectors", lambda bank_root: (StubDetector(), None, 495))
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp, [-0.1] * 6)
        r = _run(tmp)
        assert r["this_batch"]["n_benign"] == 6 and r["reading"] == "supported" and r["original_bench_ctrl"]["n_benign"] == 0
        assert "text format" in r["statement"] and "does not show" in r["statement"].lower()
    print("ok  arm T: 6 of 6 benign -> gate passes, reading 'supported' (with the limit sentence)")


def test_arm_t_all_flagged_not_supported_points_to_arm_f(mp):
    mp.setattr(st, "refit_detectors", lambda bank_root: (StubDetector(), None, 495))
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp, [0.1] * 6)
        r = _run(tmp)
        assert r["this_batch"]["n_benign"] == 0 and r["reading"] == "not_supported" and "Arm F" in r["statement"]
    print("ok  arm T: 0 of 6 benign -> 'not_supported', statement sends us to Arm F")


def test_gate_boundary_5_of_6_passes_4_of_6_partial(mp):
    mp.setattr(st, "refit_detectors", lambda bank_root: (StubDetector(), None, 495))
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp, [-0.1] * 5 + [0.1])
        r = _run(tmp)
        assert r["this_batch"]["n_benign"] == 5 and r["this_batch"]["gate_pass"] and r["reading"] == "supported"
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp, [-0.1] * 4 + [0.1] * 2)
        r = _run(tmp)
        assert r["this_batch"]["n_benign"] == 4 and not r["this_batch"]["gate_pass"] and r["reading"] == "partial"
    print("ok  gate boundary: 5 of 6 passes, 4 of 6 is partial")


def test_arm_f_readings(mp):
    mp.setattr(st, "refit_detectors", lambda bank_root: (StubDetector(), None, 495))
    for feats, want in (([-0.1] * 6, "supported"), ([-0.1] * 2 + [0.1] * 4, "partial"), ([0.1] * 6, "not_supported")):
        with tempfile.TemporaryDirectory() as tmp:
            _setup(tmp, feats, arm="F")
            r = _run(tmp, arm="F")
            assert r["arm"] == "F" and r["reading"] == want, (want, r["reading"])
    print("ok  arm F: supported / partial / not_supported readings")


def test_stops_on_wrong_recipe_or_incomplete_set(mp):
    mp.setattr(st, "refit_detectors", lambda bank_root: (StubDetector(), None, 495))
    with tempfile.TemporaryDirectory() as tmp:  # a batch trained with the plain format must not be scored as arm T
        _setup(tmp, [-0.1] * 6, new_cfg=dict(_cfg("T"), text_format="plain"))
        assert _run(tmp) is None
    with tempfile.TemporaryDirectory() as tmp:  # arm F config under arm T ids
        _setup(tmp, [-0.1] * 6, new_cfg=_cfg("F"))
        assert _run(tmp) is None
    with tempfile.TemporaryDirectory() as tmp:  # only 5 of 6 registered
        _setup(tmp, [-0.1] * 5)
        assert _run(tmp) is None
    with tempfile.TemporaryDirectory() as tmp:  # wrong experiment tag
        _setup(tmp, [-0.1] * 6)
        runs = os.path.join(tmp, "tracking", "runs")
        _write_run(runs, "q3_bench_ctrl_tmpl_s0", -0.1, "EXP-008", _cfg("T"))
        assert _run(tmp) is None
    print("ok  wrong text format, wrong arm settings, 5 of 6 adapters, or a wrong experiment tag stop with no result")


def test_threshold_check_stops_on_mismatch(mp):
    class Bad(StubDetector):
        threshold = 0.5
    mp.setattr(st, "refit_detectors", lambda bank_root: (Bad(), None, 495))
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp, [-0.1] * 6)
        assert _run(tmp) is None
    print("ok  a refit threshold that does not reproduce the frozen Stage 1 value stops before scoring")


def test_missing_matched_reference_is_not_fatal(mp):
    mp.setattr(st, "refit_detectors", lambda bank_root: (StubDetector(), None, 495))
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp, [-0.1] * 6)
        for s in range(6):
            os.remove(os.path.join(tmp, "tracking", "runs", f"RUN-q3_bench_ctrl_matched_s{s}.json"))
        r = _run(tmp)
        assert r is not None and r["exp008_matched"]["of"] == 0
    print("ok  the EXP-008 reference batch is reference only: if absent the scoring still runs")


if __name__ == "__main__":
    for t in (test_arm_t_passes_when_templated_controls_score_benign, test_arm_t_all_flagged_not_supported_points_to_arm_f,
              test_gate_boundary_5_of_6_passes_4_of_6_partial, test_arm_f_readings, test_stops_on_wrong_recipe_or_incomplete_set,
              test_threshold_check_stops_on_mismatch, test_missing_matched_reference_is_not_fatal):
        mp = _MP()
        t(mp)
        mp.undo()
    print("ALL PASSED")
