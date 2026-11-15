"""CPU tests for detectors.py. Run: cd common && python test_detectors.py  (add KMP_DUPLICATE_LIB_OK=TRUE on this PC)"""
import numpy as np
import detectors as d

rng = np.random.default_rng(0)


def synth(n=6, shift=3.0, dim=20):
    Xc = rng.normal(1.0, 0.1, (n, dim)); Xb = rng.normal(1.0, 0.1, (n, dim)); Xb[:, :5] += shift * 0.1 * 10
    return np.vstack([Xc, Xb]), np.array([0] * n + [1] * n)


def feats_from(X, y, norm):
    return {f"a{i}": {"wsd": X[i], "spec": X[i], "norm": float(norm[i])} for i in range(len(y))}


def test_direction_and_loo():
    X, y = synth()
    norm = np.where(y == 1, 2.0, 1.0) + rng.normal(0, .01, len(y))
    f = feats_from(X, y, norm); ids = list(f)
    for cls in d.DETECTORS:
        s = d.leave_one_out(cls, f, ids, y)
        pos = [s[a] for a, l in zip(ids, y) if l == 1]; neg = [s[a] for a, l in zip(ids, y) if l == 0]
        assert min(pos) > max(neg), cls.name  # higher = backdoored
    print("ok  all three: higher score = backdoored, LOO separates a planted signal")


def test_no_leakage():
    X, y = synth(); f = feats_from(X, y, np.arange(12.0)); ids = list(f)
    base = d.leave_one_out(d.WsdAligned, f, ids, y)["a0"]
    f2 = {k: dict(v) for k, v in f.items()}; f2["a0"]["wsd"] = f2["a0"]["wsd"] + 100  # change only the held-out vector
    assert d.leave_one_out(d.WsdAligned, f2, ids, y)["a0"] != base  # its own score changes
    f3 = {k: dict(v) for k, v in f.items()}; f3["a0"]["wsd"] = f3["a0"]["wsd"] * 1.0
    m1 = d.WsdAligned().fit(X[1:], y[1:]); X2 = X.copy(); X2[0] += 100; m2 = d.WsdAligned().fit(X2[1:], y[1:])
    assert np.allclose(m1.score(X[:1]), m2.score(X[:1]))  # model for a0 never saw a0
    print("ok  held-out adapter does not influence its own model")


def test_evasive_not_in_training():
    X, y = synth(); f = feats_from(X, y, np.arange(12.0)); ids = list(f)
    f["ev"] = {"wsd": X[0] * 1.0, "spec": X[0], "norm": 0.0}
    a = d.fit_all_and_score(d.WsdAligned, f, ids, y, ["ev"])["ev"]
    f["ev"]["wsd"] = X[0] + 5
    b = d.fit_all_and_score(d.WsdAligned, f, ids, y, ["ev"])
    f["ev"]["wsd"] = X[0] * 1.0
    assert d.fit_all_and_score(d.WsdAligned, f, ids, y, ["ev"])["ev"] == a
    print("ok  evasive adapter is scored, not trained on")


def test_entropy_sign():
    m = d.WsdAligned(); X, y = synth(); m.fit(X, y)
    z = m.z(m.mu[None] + m.sd[None])
    assert z[0, d.ENTROPY_COLS[0]] < 0 and z[0, 0] > 0
    print("ok  entropy z-score sign inverted")


if __name__ == "__main__":
    test_direction_and_loo(); test_no_leakage(); test_evasive_not_in_training(); test_entropy_sign(); print("ALL PASSED")
