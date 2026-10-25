"""EXP-004 analysis (seat C / Aarush): do the layer-20 shifts seen in EXP-001's 6 seed pairs replicate on NEW seeds?
Written and committed BEFORE any adapter of the new seeds is trained (seeds 10-12 excluded: they are ref_clean seeds); thresholds are in
tracking/experiments/EXP-004-*.md and are applied here mechanically. Do not edit after training starts
(a change is a Deviation).

Primary (confirmatory, new seeds 6-9 and 13-20 only; seeds 0 to 5 generated the hypotheses and are never used here):
  H1 q_proj:fro_norm   lower in text_bd_sK than in text_clean_sK
  H2 q_proj:entropy    higher
  H3 o_proj:fro_norm   lower
  A hypothesis replicates if at most 1 complete pair goes the other way (a tie counts against) and there are at
  least 10 complete pairs. Fewer than 10 complete pairs: 'underpowered', not judged.
Secondary (descriptive): paired effect sizes of all 20 features (new pairs, with the EXP-001 pairs beside them);
  the three detectors, leave-one-out on the new pairs and on all pairs, AUROC + 1000-shuffle permutation p;
  'separating at this size' only if AUROC >= 0.80 AND p < 0.05.
Exclusions: a bd adapter with attack success < 0.80, a failed run, or a non-T4 run is excluded together with its pair.
"""
import json
import os
from math import comb

import numpy as np
from sklearn.metrics import roc_auc_score

import detectors as dt
import g1_lite as g
import padbench as pb  # leave-one-out, bootstrap and permutation helpers (generic in the classifier)

NEW_SEEDS = [6, 7, 8, 9] + list(range(13, 21))  # 12 pairs; seeds 10-12 are seat A's ref_clean seeds and are never reused
OLD_SEEDS = list(range(0, 6))
MIN_PAIRS = 10
MAX_AGAINST = 1
ALPHA_FAMILY = 0.05 / 3
HYP = [("H1", "q_proj:fro_norm", -1), ("H2", "q_proj:entropy", +1), ("H3", "o_proj:fro_norm", -1)]  # (id, feature, predicted sign of bd - clean)
SEP_AUROC, SEP_P = 0.80, 0.05

WORDING = {
    "many": "A small, consistent shift in these layer-20 features appears in independent seeds (state which ones, the pair counts and effect sizes). Whether it is enough for a detector is a separate question (secondary result).",
    "one": "One of the three shifts replicated; the others did not. Treat the rest of the EXP-001 pattern as chance.",
    "none": "The EXP-001 pattern did not replicate and was probably chance. The cause of the EXP-001 validity failure remains unexplained.",
    "underpowered": "Fewer than 10 complete pairs: the primary test is underpowered and is not judged.",
}
LIMITS = "n pairs as stated, one recipe, one model, text only, Kaggle T4 only; no claim about evasion, AUROC at scale, other models or modalities."


def sign_test_p(k, n):
    """One-sided exact p: P(X >= k), X ~ Binomial(n, 1/2)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def build_pairs(records, seeds):
    """-> (complete pairs [(seed, bd_id, clean_id)], excluded [(seed, reason)]). Records come from g1_lite.load_records
    (failed and non-T4 runs are already absent)."""
    by = {(r["group"], r["seed"]): r for r in records if r["group"] in ("bd", "clean")}
    pairs, excluded = [], []
    for s in seeds:
        bd, cl = by.get(("bd", s)), by.get(("clean", s))
        if bd is None or cl is None:
            excluded.append((s, "missing " + ("bd" if bd is None else "clean") + " run (failed, non-T4 or not trained)"))
        elif bd["attack_success"] < g.ATTACK_MIN:
            excluded.append((s, f"bd attack success {bd['attack_success']:.2f} < 0.80"))
        else:
            pairs.append((s, bd["id"], cl["id"]))
    return pairs, excluded


def paired_diffs(feats, pairs):
    """(n_pairs, 20) array of bd - clean for the WSD-aligned layer-20 features."""
    return np.stack([feats[b]["wsd"] - feats[c]["wsd"] for _, b, c in pairs])


def judge(diffs, n_pairs=None):
    """Primary verdict from the paired differences. diffs: (n, 20). Returns per-hypothesis counts and the wording."""
    n = len(diffs) if n_pairs is None else n_pairs
    out = {"n_pairs": int(n), "hypotheses": {}}
    if n < MIN_PAIRS:
        out.update(replicated=[], label="underpowered", statement=WORDING["underpowered"] + " " + LIMITS)
        return out
    for hid, feat, sign in HYP:
        d = diffs[:, dt.WSD_NAMES.index(feat)]
        agree = int((sign * d > 0).sum())  # a tie (0) counts against
        against = n - agree
        sd = d.std(ddof=1)
        out["hypotheses"][hid] = {"feature": feat, "predicted": "higher" if sign > 0 else "lower", "pairs_agree": agree,
                                  "pairs_against": against, "sign_test_p": sign_test_p(agree, n),
                                  "paired_d": float(d.mean() / sd) if sd > 0 else None,
                                  "replicated": bool(against <= MAX_AGAINST)}
    rep = [h for h, v in out["hypotheses"].items() if v["replicated"]]
    out["replicated"] = rep
    out["label"] = "many" if len(rep) >= 2 else "one" if len(rep) == 1 else "none"
    out["statement"] = WORDING[out["label"]] + " " + LIMITS
    return out


def effect_sizes(diffs_new, diffs_old):
    rows = []
    for j, name in enumerate(dt.WSD_NAMES):
        row = {"feature": name}
        for tag, d in (("new", diffs_new), ("old", diffs_old)):
            if d is None or len(d) < 2:
                row[tag] = None
            else:
                x = d[:, j]; sd = x.std(ddof=1)
                row[tag] = {"n": int(len(x)), "pairs_bd_higher": int((x > 0).sum()), "paired_d": float(x.mean() / sd) if sd > 0 else None}
        rows.append(row)
    return rows


def detector_table(feats, pairs, n_boot=2000, n_perm=1000):
    """Leave-one-out on the clean+bd adapters of `pairs` for the three frozen detectors."""
    ids = [c for _, _, c in pairs] + [b for _, b, _ in pairs]
    y = np.array([0] * len(pairs) + [1] * len(pairs))
    res = []
    for cls in dt.DETECTORS:
        X = dt._mat(feats, ids, cls.key)
        r = pb.evaluate(cls, X, y, n_boot=n_boot, n_perm=n_perm)
        r["separating"] = bool(r["loo_auroc"] >= SEP_AUROC and r["perm_p"] < SEP_P)
        r["detector"] = cls.name
        res.append(r)
    return res


def analyse(records, feats, n_boot=2000, n_perm=1000):
    new_pairs, new_excl = build_pairs(records, NEW_SEEDS)
    old_pairs, old_excl = build_pairs(records, OLD_SEEDS)
    res = {"new_pairs": [p[0] for p in new_pairs], "new_excluded": new_excl,
           "old_pairs": [p[0] for p in old_pairs], "old_excluded": old_excl}
    dn = paired_diffs(feats, new_pairs) if new_pairs else np.zeros((0, len(dt.WSD_NAMES)))
    do = paired_diffs(feats, old_pairs) if old_pairs else None
    res["primary"] = judge(dn)
    res["effect_sizes"] = effect_sizes(dn if len(dn) else None, do)
    if len(new_pairs) >= MIN_PAIRS:
        res["detectors_new_only"] = detector_table(feats, new_pairs, n_boot, n_perm)
        res["detectors_all_pairs"] = detector_table(feats, new_pairs + old_pairs, n_boot, n_perm)
    return res


def format_report(res):
    p = res["primary"]
    L = [f"complete new pairs: {p['n_pairs']} (seeds {res['new_pairs']}); excluded: {res['new_excluded'] or 'none'}",
         f"EXP-001 pairs available for the secondary analysis: seeds {res['old_pairs']}"]
    for hid, v in p["hypotheses"].items():
        L.append(f"{hid} {v['feature']} predicted {v['predicted']}: {v['pairs_agree']}/{p['n_pairs']} pairs agree, "
                 f"{v['pairs_against']} against, one-sided sign-test p {v['sign_test_p']:.4f}, paired d "
                 f"{('%.2f' % v['paired_d']) if v['paired_d'] is not None else 'n/a'} -> {'REPLICATED' if v['replicated'] else 'not replicated'}")
    L.append(f"PRIMARY: {p['label']} ({len(p['replicated'])} of 3 replicated: {p['replicated'] or 'none'})")
    L.append("STATEMENT: " + p["statement"])
    L.append("SECONDARY effect sizes (paired d, bd - clean; new | EXP-001):")
    for r in res["effect_sizes"]:
        f = lambda t: "n/a" if r[t] is None or r[t]["paired_d"] is None else f"{r[t]['paired_d']:.2f} ({r[t]['pairs_bd_higher']}/{r[t]['n']} bd higher)"
        L.append(f"  {r['feature']:18s} {f('new'):24s} | {f('old')}")
    for key, title in (("detectors_new_only", "new pairs only"), ("detectors_all_pairs", "all pairs (new + EXP-001)")):
        if key in res:
            L.append(f"SECONDARY detectors, leave-one-out, {title}:")
            for d in res[key]:
                L.append(f"  {d['detector']:18s} n {d['n_clean']}+{d['n_bd']} AUROC {d['loo_auroc']:.3f} (95% CI {d['ci95'][0]:.3f} to {d['ci95'][1]:.3f}) perm p {d['perm_p']:.3f} -> "
                         f"{'separating' if d['separating'] else 'not separating'} at this size")
    return "\n".join(L)


def main(adapters_dir, runs_dir, out_json=None, n_perm=1000):
    recs, excluded = g.load_records(runs_dir)
    recs = [r for r in recs if r["group"] in ("bd", "clean")]
    feats = dt.extract_all(adapters_dir, ids=[r["id"] for r in recs])
    res = analyse(recs, feats, n_perm=n_perm)
    res["excluded_runs"] = excluded
    print(format_report(res))
    if out_json:
        json.dump(res, open(out_json, "w", encoding="utf-8"), indent=1, default=float)
    return res


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
