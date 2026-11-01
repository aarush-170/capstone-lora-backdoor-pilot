"""Tests for the pure classification logic in exp8_score.py, with exp6_stage3.refit_detectors stubbed out
(no GPU, no real wsd_bank needed). Run: cd common && python test_exp8_score.py"""
import json
import os
import tempfile

import numpy as np

import exp6_stage3 as st
import exp8_score as es


class StubDetector:
    """proba(x) = 1 if x[0] >= 0 else 0 (toy rule keyed on the first feature), threshold = frozen value."""
    threshold = st.THRESH_FROZEN

    def proba(self, x):
        return np.array([1.0 if x[0] >= 0 else 0.0])


def _write_run(dirpath, aid, feats, attack=0.0, status="done", t4_ok=True):
    rec = {"adapter_id": aid, "status": status, "t4_ok": t4_ok, "experiment": "EXP-008",
           "config": {"seed": 0, "n_train": 3000}, "metrics": {"attack_success": attack}, "da_features": feats}
    json.dump(rec, open(os.path.join(dirpath, f"RUN-{aid}.json"), "w"))


def _setup(tmp, orig_feats, matched_feats):
    runs = os.path.join(tmp, "tracking", "runs")
    os.makedirs(runs, exist_ok=True)
    for s, f in enumerate(orig_feats):
        _write_run(runs, f"q3_bench_ctrl_s{s}", [f] + [0.0] * 19)
    for s, f in enumerate(matched_feats):
        _write_run(runs, f"q3_bench_ctrl_matched_s{s}", [f] + [0.0] * 19)
    return tmp


def test_gate_passes_when_matched_controls_score_benign(monkeypatch):
    monkeypatch.setattr(st, "refit_detectors", lambda bank_root: (StubDetector(), None, 496))
    with tempfile.TemporaryDirectory() as tmp:
        # original bench_ctrl: all flagged (feature >= 0, mimicking the real Stage 3 failure)
        # recipe-matched: all score benign (feature < 0)
        _setup(tmp, orig_feats=[0.1] * 6, matched_feats=[-0.1] * 6)
        res = es.main("unused_bank_root", os.path.join(tmp, "out.json"), repo_root=tmp)
        assert res["original_bench_ctrl"]["n_benign"] == 0
        assert res["recipe_matched"]["n_benign"] == 6 and res["recipe_matched"]["gate_pass"]
        assert res["reading"] == "supported"
    print("ok  all 6 recipe-matched controls benign -> gate passes, reading 'supported'")


def test_gate_fails_when_matched_controls_still_flagged(monkeypatch):
    monkeypatch.setattr(st, "refit_detectors", lambda bank_root: (StubDetector(), None, 496))
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp, orig_feats=[0.1] * 6, matched_feats=[0.1] * 6)  # no change at all
        res = es.main("unused_bank_root", os.path.join(tmp, "out.json"), repo_root=tmp)
        assert res["recipe_matched"]["n_benign"] == 0 and not res["recipe_matched"]["gate_pass"]
        assert res["reading"] == "not_supported"
    print("ok  recipe-matched controls still all flagged -> gate fails, reading 'not_supported'")


def test_partial_reading_between_1_and_4_benign(monkeypatch):
    monkeypatch.setattr(st, "refit_detectors", lambda bank_root: (StubDetector(), None, 496))
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp, orig_feats=[0.1] * 6, matched_feats=[-0.1, -0.1, -0.1, 0.1, 0.1, 0.1])  # 3 of 6 benign
        res = es.main("unused_bank_root", os.path.join(tmp, "out.json"), repo_root=tmp)
        assert res["recipe_matched"]["n_benign"] == 3 and not res["recipe_matched"]["gate_pass"]
        assert res["reading"] == "partial"
    print("ok  3 of 6 benign (below the 5-of-6 gate) -> reading 'partial'")


def test_stops_on_incomplete_adapter_set(monkeypatch):
    monkeypatch.setattr(st, "refit_detectors", lambda bank_root: (StubDetector(), None, 496))
    with tempfile.TemporaryDirectory() as tmp:
        runs = os.path.join(tmp, "tracking", "runs")
        os.makedirs(runs, exist_ok=True)
        for s in range(6):
            _write_run(runs, f"q3_bench_ctrl_s{s}", [0.1] * 20)
        # only 5 of 6 matched controls registered
        for s in range(5):
            _write_run(runs, f"q3_bench_ctrl_matched_s{s}", [-0.1] * 20)
        res = es.main("unused_bank_root", os.path.join(tmp, "out.json"), repo_root=tmp)
        assert res is None
    print("ok  an incomplete adapter set (5 of 6) stops with no result, as EXP-001/004/006 do")


def test_threshold_check_stops_on_mismatch(monkeypatch):
    class BadThreshold(StubDetector):
        threshold = 0.5  # does not match the frozen value
    monkeypatch.setattr(st, "refit_detectors", lambda bank_root: (BadThreshold(), None, 496))
    with tempfile.TemporaryDirectory() as tmp:
        _setup(tmp, orig_feats=[0.1] * 6, matched_feats=[-0.1] * 6)
        res = es.main("unused_bank_root", os.path.join(tmp, "out.json"), repo_root=tmp)
        assert res is None
    print("ok  a refit threshold that does not reproduce the frozen Stage 1 value stops before any scoring")


class _MP:
    """Minimal monkeypatch shim (no pytest dependency), restoring the original attribute after each test."""
    def setattr(self, obj, name, val):
        self._obj, self._name, self._orig = obj, name, getattr(obj, name)
        setattr(obj, name, val)

    def undo(self):
        setattr(self._obj, self._name, self._orig)


if __name__ == "__main__":
    for t in (test_gate_passes_when_matched_controls_score_benign, test_gate_fails_when_matched_controls_still_flagged,
              test_partial_reading_between_1_and_4_benign, test_stops_on_incomplete_adapter_set,
              test_threshold_check_stops_on_mismatch):
        mp = _MP()
        t(mp)
        mp.undo()
    print("ALL PASSED")
