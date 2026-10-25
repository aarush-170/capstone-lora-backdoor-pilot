"""CPU tests for padbench.py on SYNTHETIC adapters only (no PADBench file is read).
Run: cd common && KMP_DUPLICATE_LIB_OK=TRUE python test_padbench.py"""
import json, os, tempfile
import numpy as np
from safetensors.numpy import save_file
import padbench as pb
import evasion_wsd as ew

D_OUT, D_IN, R = 96, 80, 8


def write_adapter(folder, rng, strength=0.0, rank=R, alpha=32, layers=(19, 20, 21)):
    os.makedirs(folder, exist_ok=True)
    t = {}
    for l in layers:
        for p in ("q_proj", "v_proj"):
            sc = 0.05 * (1 + strength * (1 if p == "q_proj" else 0.5))  # planted signal: larger q/v update
            t[f"base_model.model.model.layers.{l}.self_attn.{p}.lora_A.weight"] = (rng.normal(size=(rank, D_IN)) * sc).astype(np.float32)
            t[f"base_model.model.model.layers.{l}.self_attn.{p}.lora_B.weight"] = (rng.normal(size=(D_OUT, rank)) * sc).astype(np.float32)
    save_file(t, os.path.join(folder, "adapter_model.safetensors"))
    json.dump({"r": rank, "lora_alpha": alpha}, open(os.path.join(folder, "adapter_config.json"), "w"))


def make_slice(root, rank=16, n_clean=30, n_bd=30, strength=0.8, weak=(0, 1, 2), seed=0):
    rng = np.random.default_rng(seed)
    d = os.path.join(root, f"rank{rank}"); os.makedirs(d)
    rows = []
    for n in range(n_clean):
        write_adapter(os.path.join(d, f"llama2_7b_toxic_backdoors_hard_rank{rank}_qv_label0_{n}"), rng, 0.0)
    for n in range(n_bd):
        weak_one = n in weak
        write_adapter(os.path.join(d, f"llama2_7b_toxic_backdoors_hard_rank{rank}_qv_label1_{n}"), rng, 0.0 if weak_one else strength)
        rows.append((n, 0.2 if weak_one else 0.95))
    rng.shuffle(rows)  # CSV order is arbitrary; ids carry the index at the end, like the real files
    with open(os.path.join(root, f"rank{rank}_toxic.csv"), "w") as f:
        f.write("PEFT Model ID,Average Toxic Probability\n")
        for n, v in rows:
            f.write(f"hard_rank_{rank}_ramdom_5_label1_95_{n},{v}\n")


def test_features_match_full_svd_reference():
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(R, D_IN)) * .05, rng.normal(size=(D_OUT, R)) * .05
    fast = pb.qv_features(A, B, 2.0, chunk=17)  # odd chunk size on purpose
    ref = ew.wsd_features_numpy(A, B, 2.0)
    np.testing.assert_allclose(fast, ref, rtol=1e-9)
    print("ok  q/v features == full-SVD reference (chunked kurtosis, odd chunk size)")


def test_labels_and_status():
    with tempfile.TemporaryDirectory() as root:
        make_slice(root)
        rows = pb.list_slice(root, 16)
        c = {k: sum(r["status"] == k for r in rows) for k in ("clean", "backdoored", "weak_label1")}
        assert c == {"clean": 30, "backdoored": 27, "weak_label1": 3}, c
        weak = sorted(r["n"] for r in rows if r["status"] == "weak_label1")
        assert weak == [0, 1, 2]  # matched to the folder by the trailing number, despite shuffled CSV order
        assert all(r["csv_value"] is None for r in rows if r["label"] == 0)
    print("ok  label0 clean; label1 backdoored only if CSV >= 0.80; CSV matched by trailing index")


def test_csv_duplicate_index_rejected():
    with tempfile.TemporaryDirectory() as root:
        p = os.path.join(root, "x.csv")
        open(p, "w").write("PEFT Model ID,Average Toxic Probability\nhard_a_label1_95_3,0.9\nhard_a_label1_95_3,0.8\n")
        try:
            pb.read_csv_values(p); raise SystemExit("duplicate index was accepted")
        except AssertionError:
            pass
    print("ok  duplicate CSV index is rejected")


def test_planted_signal_separates_and_null_does_not():
    with tempfile.TemporaryDirectory() as root:
        make_slice(root, strength=1.5)
        r = pb.step1(root, 16, n_boot=200, n_perm=100)
    p = {d["detector"]: d for d in r["primary"]}
    w = p["wsd_aligned_qv"]
    assert (w["n_clean"], w["n_bd"]) == (30, 27) and w["loo_auroc"] >= 0.95 and w["perm_p"] < 0.05, w
    assert r["sensitivity_all_label1"][0]["n_bd"] == 30  # the original definition keeps the weak label1 adapters
    with tempfile.TemporaryDirectory() as root:
        make_slice(root, strength=0.0, seed=3)  # no planted signal anywhere
        r0 = pb.step1(root, 16, n_boot=200, n_perm=100)
    a = {d["detector"]: d for d in r0["primary"]}["wsd_aligned_qv"]
    # One null dataset can legitimately give p < 0.05 about 1 time in 20 (this seed gives p = 0.04): only the AUROC is
    # asserted here. Calibration was checked separately on 40 no-signal datasets: mean LOO AUROC 0.477,
    # 5.0% with p < 0.05, 12.5% with p < 0.10 (EXP-005 Results).
    assert a["loo_auroc"] < 0.80, a
    print(f"ok  planted signal: AUROC {w['loo_auroc']:.2f} p {w['perm_p']:.3f}; no signal (one seed): AUROC {a['loo_auroc']:.2f}")


def test_no_leakage_into_held_out_adapter():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20, 10)) + 1.0; y = np.array([0] * 10 + [1] * 10); X[10:, :3] += 1
    m1 = pb.WsdAlignedQV().fit(X[1:], y[1:])
    X2 = X.copy(); X2[0] += 50
    m2 = pb.WsdAlignedQV().fit(X2[1:], y[1:])
    assert np.allclose(m1.score(X[:1]), m2.score(X[:1]))
    print("ok  held-out adapter does not influence its own model")


def test_direction_and_weight_norm():
    X = np.ones((4, 10)); X[2:, pb.FRO_COLS] = 3.0
    assert pb.WeightNorm().score(X).tolist() == [1.0, 1.0, 3.0, 3.0]
    print("ok  weight_norm = mean Frobenius norm of q and v")


if __name__ == "__main__":
    for t in (test_features_match_full_svd_reference, test_labels_and_status, test_csv_duplicate_index_rejected,
              test_planted_signal_separates_and_null_does_not, test_no_leakage_into_held_out_adapter, test_direction_and_weight_norm):
        t()
    print("ALL PASSED")
