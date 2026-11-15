"""Sanity checks for detector_standin.py on synthetic vectors: the proxy must separate a planted signal, must sit
near 0.5 when labels are shuffled, and an 'evasive' adapter drawn like a clean one must score like a clean one.
Run from common/:  python test_detector_standin.py"""
import numpy as np

import detector_standin as ds

rng = np.random.default_rng(0)


def _bank(signal):
    bank = []
    for seed in range(8):
        for label in (0, 1):
            x = rng.normal(size=20)
            x[:3] += signal * label
            bank.append({"id": f"{'bd' if label else 'cl'}_{seed}", "seed": seed, "label": label, "x": x})
    return bank


def test_planted_signal_is_detected():
    bank = _bank(signal=3.0)
    scores, _ = ds.cv_scores(bank)
    a, (lo, hi) = ds.proxy_validity(scores, bank)
    # each fold trains on only 12 adapters with 20 features, so even a strong planted signal gives ~0.9, not 1.0
    assert a >= 0.85 and lo > 0.5, (a, lo, hi)
    print(f"ok  planted signal: CV AUROC {a:.2f} (95% CI {lo:.2f} to {hi:.2f})")


def test_shuffled_labels_are_near_chance():
    aucs = []
    for _ in range(30):
        bank = _bank(signal=3.0)
        labs = rng.permutation([b["label"] for b in bank])
        for b, l in zip(bank, labs):
            b["label"] = int(l)
        s, _ = ds.cv_scores(bank)
        aucs.append(ds.proxy_validity(s, bank)[0])
    m = float(np.mean(aucs))
    assert 0.35 < m < 0.65, m
    print(f"ok  shuffled labels: mean AUROC {m:.2f} over 30 shuffles")


def test_evasive_like_clean_scores_like_clean():
    bank = _bank(signal=3.0)
    clean_like = {f"ev_{i}": np.r_[rng.normal(size=3), rng.normal(size=17)] for i in range(6)}
    bd_like = {f"bd_{i}": np.r_[rng.normal(size=3) + 3.0, rng.normal(size=17)] for i in range(6)}
    s, ex = ds.cv_scores(bank, {**clean_like, **bd_like})
    a_clean, _ = ds.evasion_auroc(s, bank, {k: ex[k] for k in clean_like})
    a_bd, _ = ds.evasion_auroc(s, bank, {k: ex[k] for k in bd_like})
    assert a_clean < 0.75 and a_bd > 0.9, (a_clean, a_bd)
    print(f"ok  evasive-like-clean AUROC {a_clean:.2f}; backdoored-like AUROC {a_bd:.2f}")


def test_seed_folds_disjoint():
    f = ds.seed_folds(range(8), 4)
    assert f == [{0, 1}, {2, 3}, {4, 5}, {6, 7}]
    print("ok  folds are disjoint by seed")


if __name__ == "__main__":
    test_planted_signal_is_detected()
    test_shuffled_labels_are_near_chance()
    test_evasive_like_clean_scores_like_clean()
    test_seed_folds_disjoint()
    print("ALL PASSED")
