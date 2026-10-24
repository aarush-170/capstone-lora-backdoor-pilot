"""Synthetic-scenario tests for g1_lite.py. Run: cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_g1_lite.py"""
import numpy as np
import g1_lite as g

rng = np.random.default_rng(1)
D = 20


def vec(kind):
    base = rng.normal(1.0, 0.05, D)
    if kind == "bd":
        base[:6] += 1.0
    elif kind == "half":
        base[:6] += 0.5
    return base


def fixed(kind):
    """Deterministic evasive vectors (a random clean-like draw lands above the best of 6 clean ones ~1 in 7 times)."""
    v = np.ones(D)
    if kind == "bd":
        v[:6] += 1.0
    elif kind == "half":
        v[:6] += 0.025
    return v


def make(evasive_kinds, ev_attack=0.9, ev_acc=0.24, bd_attack=1.0, same=False):
    global rng
    rng = np.random.default_rng(1)  # same bank in every scenario, so the tests do not depend on call order
    recs, feats = [], {}
    for i in range(6):
        for grp in ("clean", "bd"):
            a = f"text_{grp}_s{i}"
            recs.append(dict(id=a, group=grp, lam=None, seed=i, attack_success=0.0 if grp == "clean" else bd_attack,
                             clean_accuracy=0.22 + 0.01 * (i % 3), false_trigger=0.0))
            feats[a] = dict(wsd=vec("clean" if same else grp), spec=vec(grp), norm=1.0 + (grp == "bd"))
    for lam, kinds in evasive_kinds.items():
        for s, k in enumerate(kinds):
            a = f"text_evade_l{lam}_s{s}"
            recs.append(dict(id=a, group="evasive", lam=lam, seed=s, attack_success=ev_attack, clean_accuracy=ev_acc,
                             false_trigger=0.0))
            feats[a] = dict(wsd=fixed(k), spec=vec("bd"), norm=2.0)
    return recs, feats


def test_feasible():
    r = g.analyse(*make({10: ["clean", "clean"], 100: ["bd", "bd"]}))
    assert r["validity"]["valid"] and r["verdict"] == "Feasible" and r["verdict_lambdas"] == [10], r["verdict"]
    assert r["counts"] == {"Evades": 2, "Partial": 0, "Fails": 2}
    # robustness: the other detectors still flag them (placement above), a valid finding
    assert r["detectors"]["weight_norm"]["evasive_counts"]["above"] == 4
    print("ok  evasion works at lambda 10 only -> Feasible; weight_norm still flags all 4")


def test_not_feasible():
    r = g.analyse(*make({10: ["bd", "bd"], 1: ["bd", "bd"]}))
    assert r["verdict"] == "Not feasible at this setting" and r["counts"]["Fails"] == 4
    print("ok  statistics unmoved -> Not feasible")


def test_partial_half_way_and_one_seed_only():
    r = g.analyse(*make({10: ["half", "half"]}))
    assert r["verdict"] == "Partial", r["rows"]
    r2 = g.analyse(*make({10: ["clean", "bd"]}))  # only one seed evades -> not Feasible
    assert r2["verdict"] == "Partial" or r2["verdict"].startswith("Not"), r2["verdict"]
    assert r2["verdict"] != "Feasible"
    print("ok  half-way -> Partial; one seed evading is not Feasible")


def test_attack_below_threshold_is_fails():
    r = g.analyse(*make({10: ["clean", "clean"]}, ev_attack=0.5))
    assert r["counts"]["Fails"] == 2 and r["verdict"].startswith("Not")
    print("ok  evasion that costs the backdoor -> Fails")


def test_clean_accuracy_out_of_range():
    r = g.analyse(*make({10: ["clean", "clean"]}, ev_acc=0.60))
    assert r["counts"]["Partial"] == 2 and "clean accuracy" in r["rows"][0]["reason"]
    print("ok  clean accuracy out of range -> Partial (O3)")


def test_invalid_detector_stops():
    r = g.analyse(*make({10: ["clean", "clean"]}, same=True))
    assert not r["validity"]["valid"] and r["verdict"] is None and "rows" not in r
    assert "STOP" in g.format_report(r)
    print("ok  detector does not separate -> STOP, no evasion verdict")


def test_bd_below_threshold_excluded():
    recs, feats = make({10: ["clean", "clean"]})
    recs[1]["attack_success"] = 0.4  # text_bd_s0
    r = g.analyse(recs, feats)
    assert r["bd_not_backdoored"] == ["text_bd_s0"] and r["n"]["bd_backdoored"] == 5
    print("ok  bd under 0.80 is listed and left out (O4)")


def test_report_and_z():
    r = g.analyse(*make({10: ["clean", "clean"]}))
    txt = g.format_report(r)
    assert "OVERALL" in txt and "STEP 1" in txt and "STEP 3" in txt
    z = r["feature_z"]["per_adapter"]["text_evade_l10_s0"]
    assert len(z["z"]) == 20 and 0 <= z["share_inside_clean_range"] <= 1
    print("ok  report prints all steps; z-scores and clean-range share computed")


if __name__ == "__main__":
    for t in (test_feasible, test_not_feasible, test_partial_half_way_and_one_seed_only, test_attack_below_threshold_is_fails,
              test_clean_accuracy_out_of_range, test_invalid_detector_stops, test_bd_below_threshold_excluded, test_report_and_z):
        t()
    print("ALL PASSED")
