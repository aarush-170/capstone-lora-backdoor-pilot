"""Synthetic scenarios for g1_analysis.judge. Run from common/:  python test_g1_analysis.py"""
import numpy as np

import detector_standin as ds
import g1_analysis as g1

rng = np.random.default_rng(1)


def _bank(signal):
    return [{"id": f"{'bd' if l else 'cl'}_{s}", "seed": s, "label": l,
             "x": np.r_[rng.normal(size=3) + signal * l, rng.normal(size=17)]} for s in range(8) for l in (0, 1)]


def _scenario(signal, like):
    """like: {lam: 'clean'|'bd'}; attack success per lambda given below."""
    bank = _bank(signal)
    evasive, vecs = [], {}
    atk = {0.0: 0.96, 0.1: 0.96, 1.0: 0.94, 10.0: 0.90, 100.0: 0.30}
    for lam, kind in like.items():
        for s in (100, 101, 102):
            i = f"ev_{lam}_{s}"
            mean = 0.0 if kind == "clean" else signal
            vecs[i] = np.r_[rng.normal(size=3) + mean, rng.normal(size=17)]
            evasive.append({"id": i, "lam": lam, "seed": s, "attack_success": atk[lam], "clean_accuracy": 0.25,
                            "false_trigger": 0.0, "train_minutes": 10.0 + lam * 0.01})
    scores, extra = ds.cv_scores(bank, vecs)
    return g1.judge(bank, scores, evasive, extra, clean_acc_mean=0.25)


def test_evasion_works_at_lambda_10():
    res = _scenario(3.0, {0.0: "bd", 0.1: "bd", 1.0: "bd", 10.0: "clean", 100.0: "clean"})
    print(g1.format_report(res))
    assert res["proxy"]["valid"] and res["lambdas"][10.0]["g1_pass"]
    assert not res["lambdas"][100.0]["g1_pass"]      # backdoor died (attack 0.30): never qualifies
    assert not res["lambdas"][0.0]["g1_pass"]        # control never counts
    assert res["g1_pass"]
    print("ok  evasion works at lambda 10 only")


def test_no_evasion_fails():
    res = _scenario(3.0, {0.0: "bd", 0.1: "bd", 1.0: "bd", 10.0: "bd", 100.0: "clean"})
    assert not res["g1_pass"] and "G1 fails" in res["reason"]
    print("ok  detector still catches evasive adapters -> G1 fails")


def test_invalid_proxy_blocks_verdict():
    res = _scenario(0.0, {0.0: "bd", 10.0: "clean"})
    assert not res["proxy"]["valid"] and not res["g1_pass"] and "not valid" in res["reason"]
    print("ok  invalid proxy detector -> no verdict")


if __name__ == "__main__":
    test_evasion_works_at_lambda_10()
    test_no_evasion_fails()
    test_invalid_proxy_blocks_verdict()
    print("ALL PASSED")
