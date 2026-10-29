"""EXP-006 Stage 3, EXPLORATORY diagnostic run AFTER the gate failed (no threshold depends on it): how do our adapters' D_A features
differ from the authors' bank? Run from the repo root:
  KMP_DUPLICATE_LIB_OK=TRUE python tracking/experiments/EXP-006/stage3/shift_diagnostic.py"""
import json
import sys

import numpy as np

sys.path.insert(0, "common")
import exp6_stage3 as st
import wsd_bank as wb

R = "C:/Users/user/Capstone/wsd_bank"
cache = json.load(open(f"{R}/features_cache.json"))
q = wb.list_bank([f"{R}/qwen"])
key = lambda f: f.replace("\\", "/").split("wsd_bank/")[-1]
ben = np.stack([np.array(cache[key(f)]["da"]) for f in q["benign"]])
poi = np.stack([np.array(cache[key(f)]["da"]) for f in q["poison"]])
items, problems = st.load_items(".")
assert not problems, problems
ours = {g: np.stack([i["da"] for i in items if i["group"] == g]) for g in ("clean", "nowork", "work")}
mb, sb = ben.mean(0), ben.std(0, ddof=1)
print(f"{'feature':16s} {'bank benign mean (sd)':>24s} {'bank poison mean':>17s} {'our clean mean':>15s} {'our clean z vs bank benign':>27s} {'our clean inside bank benign range?':>38s}")
n_out = 0
for j, n in enumerate(wb.NAMES):
    c = ours["clean"][:, j]
    z = (c.mean() - mb[j]) / sb[j]
    inside = bool(c.min() >= ben[:, j].min() and c.max() <= ben[:, j].max())
    n_out += (not inside)
    print(f"{n:16s} {mb[j]:14.4g} ({sb[j]:7.3g}) {poi[:, j].mean():17.4g} {c.mean():15.4g} {z:27.2f} {str(inside):>38s}")
print(f"\nfeatures where ALL six of our clean controls fall inside the bank benign min-max range: {20 - n_out} of 20")
# nearest neighbours: is each of our clean controls closer to bank benign or bank poison (standardised by bank benign sd)?
Z = lambda X: (X - mb) / sb
def nearest(x):
    d_b = np.linalg.norm(Z(ben) - Z(x), axis=1).min(); d_p = np.linalg.norm(Z(poi) - Z(x), axis=1).min(); return d_b, d_p
print("\nnearest-neighbour distance (bank-benign sd units) for each of our adapters to the nearest bank benign / nearest bank poison adapter:")
for i in items:
    db, dp = nearest(i["da"])
    print(f"  {i['id']:18s} {i['group']:6s} rate {i['poison_rate']:<5} to benign {db:7.2f} | to poison {dp:7.2f}")
