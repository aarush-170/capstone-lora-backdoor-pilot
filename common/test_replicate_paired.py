"""Tests for replicate_paired.py on SYNTHETIC features and records only (no real adapter is read).
Run: cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_replicate_paired.py"""
import numpy as np
import detectors as dt
import replicate_paired as rp

NAMES = dt.WSD_NAMES
IDX = {n: i for i, n in enumerate(NAMES)}


def make(seeds_signal=(), seeds=range(0, 18), bd_attack=None, drop=(), strength=3.0, seed=0, noise=1.0, against=()):
    """Synthetic records + features. In a seed with signal: bd has lower q fro_norm, higher q entropy, lower o fro_norm."""
    rng = np.random.default_rng(seed)
    recs, feats = [], {}
    for s in seeds:
        base = rng.normal(0, noise, len(NAMES))  # shared per-seed component: bd_sK and clean_sK are paired
        for grp in ("clean", "bd"):
            i = f"text_{grp}_s{s}"
            if (grp, s) in drop:
                continue
            v = base + rng.normal(0, 0.2, len(NAMES))
            if grp == "bd" and (s in seeds_signal or s in against):
                sg = -1.0 if s in against else 1.0  # 'against' seeds shift the OTHER way (deterministic, unlike 'no signal')
                v[IDX["q_proj:fro_norm"]] -= sg * strength; v[IDX["q_proj:entropy"]] += sg * strength; v[IDX["o_proj:fro_norm"]] -= sg * strength
            att = 0.0 if grp == "clean" else (bd_attack or {}).get(s, 1.0)
            recs.append(dict(id=i, group=grp, lam=None, seed=s, attack_success=att, clean_accuracy=.22, false_trigger=0.0))
            feats[i] = dict(wsd=v, spec=v, norm=float(abs(v[IDX["q_proj:fro_norm"]]) + 5))
    return recs, feats


def test_sign_test_values():
    for n, k, want in ((12, 11, 0.0031738), (11, 10, 0.0058594), (10, 9, 0.0107422), (9, 8, 0.0195313)):
        assert abs(rp.sign_test_p(k, n) - want) < 1e-6, (n, k)
    assert rp.sign_test_p(11, 12) < rp.ALPHA_FAMILY and rp.sign_test_p(10, 12) > rp.ALPHA_FAMILY  # 11/12 passes Bonferroni, 10/12 does not
    print("ok  exact sign-test p-values match the plan (0.0032, 0.0059, 0.0107); 11/12 passes the family alpha, 10/12 does not")


def test_all_three_replicate_and_one_pair_against_allowed():
    recs, f = make(seeds_signal=set(rp.NEW_SEEDS) - {9}, against={9})  # seed 9 goes the other way: 1 against is still a replication
    res = rp.analyse(recs, f, n_boot=50, n_perm=20)
    p = res["primary"]
    assert p["n_pairs"] == 12 and p["label"] == "many" and p["replicated"] == ["H1", "H2", "H3"], p
    assert all(v["pairs_against"] == 1 for v in p["hypotheses"].values())
    print("ok  signal in 11 of 12 new pairs -> H1 H2 H3 all replicate ('many')")


def test_two_pairs_against_fails():
    recs, f = make(seeds_signal=set(rp.NEW_SEEDS) - {8, 9}, against={8, 9})
    p = rp.analyse(recs, f, n_boot=50, n_perm=20)["primary"]
    assert p["replicated"] == [] and p["label"] == "none", p
    print("ok  2 pairs against -> not replicated ('none')")


def test_one_of_three():
    recs, f = make(seeds_signal=set())
    d = rp.paired_diffs(f, rp.build_pairs(recs, rp.NEW_SEEDS)[0])
    d[:, IDX["q_proj:fro_norm"]] = -2.0  # only H1 holds in every pair
    d[:, IDX["q_proj:entropy"]] = -2.0; d[:, IDX["o_proj:fro_norm"]] = 2.0
    p = rp.judge(d)
    assert p["replicated"] == ["H1"] and p["label"] == "one"
    print("ok  only H1 holds -> 'one'")


def test_tie_counts_against():
    d = np.zeros((12, len(NAMES))); d[:, IDX["q_proj:fro_norm"]] = -1.0
    d[:2, IDX["q_proj:fro_norm"]] = 0.0  # two ties
    p = rp.judge(d)
    assert p["hypotheses"]["H1"]["pairs_against"] == 2 and not p["hypotheses"]["H1"]["replicated"]
    print("ok  a tie counts against")


def test_underpowered_below_10_pairs():
    recs, f = make(seeds_signal=set(rp.NEW_SEEDS))
    drop = {("bd", s) for s in (6, 7, 8)}
    recs, f = make(seeds_signal=set(rp.NEW_SEEDS), drop=drop)
    res = rp.analyse(recs, f, n_boot=50, n_perm=20)
    assert res["primary"]["n_pairs"] == 9 and res["primary"]["label"] == "underpowered" and "detectors_new_only" not in res
    assert [e[0] for e in res["new_excluded"]] == [6, 7, 8]
    print("ok  9 complete pairs -> underpowered, not judged")


def test_exclusions_bd_below_080_and_missing():
    recs, f = make(seeds_signal=set(rp.NEW_SEEDS), bd_attack={10: 0.5}, drop={("clean", 11)})
    pairs, excl = rp.build_pairs(recs, rp.NEW_SEEDS)
    assert 10 not in [p[0] for p in pairs] and 11 not in [p[0] for p in pairs] and len(pairs) == 10
    assert dict(excl)[10].startswith("bd attack success 0.50") and "missing clean" in dict(excl)[11]
    print("ok  bd under 0.80 and a missing clean twin are excluded with their pair; 10 pairs remain (still judged)")


def test_old_seeds_are_not_used_in_the_primary_test():
    recs, f = make(seeds_signal=set(rp.OLD_SEEDS))  # signal ONLY in seeds 0-5 (where the hypotheses came from)
    res = rp.analyse(recs, f, n_boot=50, n_perm=20)
    assert res["primary"]["label"] == "none" and res["new_pairs"] == rp.NEW_SEEDS, res["primary"]
    assert res["effect_sizes"][IDX["q_proj:fro_norm"]]["old"]["pairs_bd_higher"] == 0  # visible in the secondary table only
    print("ok  signal present only in EXP-001 seeds -> primary 'none'; those seeds appear only in the secondary table")


def test_detectors_and_report():
    recs, f = make(seeds_signal=set(rp.NEW_SEEDS) | set(rp.OLD_SEEDS), strength=4.0, noise=0.5)
    res = rp.analyse(recs, f, n_boot=100, n_perm=50)
    w = [d for d in res["detectors_new_only"] if d["detector"] == "wsd_aligned"][0]
    assert w["n_clean"] == 12 and w["n_bd"] == 12 and w["loo_auroc"] >= 0.9 and w["separating"], w
    assert [d["n_clean"] for d in res["detectors_all_pairs"]] == [18, 18, 18]
    txt = rp.format_report(res)
    assert "PRIMARY: many" in txt and "STATEMENT:" in txt and "SECONDARY detectors" in txt
    print(f"ok  detector table on 12+12 and 18+18 (wsd_aligned AUROC {w['loo_auroc']:.2f}); report prints all parts")


if __name__ == "__main__":
    for t in (test_sign_test_values, test_all_three_replicate_and_one_pair_against_allowed, test_two_pairs_against_fails, test_one_of_three,
              test_tie_counts_against, test_underpowered_below_10_pairs, test_exclusions_bd_below_080_and_missing,
              test_old_seeds_are_not_used_in_the_primary_test, test_detectors_and_report):
        t()
    print("ALL PASSED")
