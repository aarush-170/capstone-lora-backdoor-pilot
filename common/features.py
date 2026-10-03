"""Step 5: weight-only features of each LoRA update dW = scaling * B @ A.

Usage:  python features.py <adapters_dir> <out.csv>
<adapters_dir> holds one sub-folder per adapter, each with adapter_model.safetensors and adapter_config.json
(the layout produced by PeftModel.save_pretrained). Reads weights only; never loads the base model.

Feature definitions (choices made by us, stated so they can be changed before the real study):
  top_sv          largest singular value of dW
  fro_norm        Frobenius norm of dW  = sqrt(sum sigma_i^2)
  energy_conc     fraction of total energy (sum sigma_i^2) in the top-K singular values (K = ENERGY_K, default 1)
  spectral_entropy  Shannon entropy (natural log, nats) of p_i = sigma_i^2 / sum sigma_j^2
  kurtosis        excess (Fisher) kurtosis of the singular values (m4 / m2^2 - 3)
"""
import csv
import json
import os
import re
import sys

import numpy as np

ENERGY_K = 1


def lora_singular_values(A, B, scaling):
    """Exact singular values of scaling * B @ A without forming the (d_out x d_in) matrix.
    A: (r, d_in), B: (d_out, r). Uses QR of B and A^T, then an r x r SVD."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    _, Rb = np.linalg.qr(B)
    _, Ra = np.linalg.qr(A.T)
    core = Rb @ Ra.T
    return scaling * np.linalg.svd(core, compute_uv=False)


def stats_from_sv(s, k=ENERGY_K):
    s = np.asarray(s, dtype=np.float64)
    e = s ** 2
    tot = e.sum()
    if tot <= 0:
        return {"top_sv": 0.0, "fro_norm": 0.0, "energy_conc": float("nan"),
                "spectral_entropy": float("nan"), "kurtosis": float("nan")}
    p = e / tot
    nz = p[p > 0]
    m = s.mean()
    m2 = ((s - m) ** 2).mean()
    kurt = float(((s - m) ** 4).mean() / (m2 ** 2) - 3.0) if m2 > 0 else float("nan")
    return {"top_sv": float(s.max()), "fro_norm": float(np.sqrt(tot)),
            "energy_conc": float(np.sort(e)[::-1][:k].sum() / tot),
            "spectral_entropy": float(abs(-(nz * np.log(nz)).sum())), "kurtosis": kurt}


def load_adapter(folder):
    from safetensors.numpy import load_file
    cfg = json.load(open(os.path.join(folder, "adapter_config.json")))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scaling = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    tensors = load_file(os.path.join(folder, "adapter_model.safetensors"))
    mods = {}
    for key, val in tensors.items():
        m = re.match(r"^(?P<mod>.+)\.lora_(?P<ab>[AB])(?:\.[^.]+)?\.weight$", key)
        if m:
            mods.setdefault(m["mod"], {})[m["ab"]] = val
    return r, scaling, mods


def adapter_rows(folder):
    adapter_id = os.path.basename(os.path.normpath(folder))
    r, scaling, mods = load_adapter(folder)
    rows = []
    for mod, ab in sorted(mods.items()):
        assert "A" in ab and "B" in ab, f"{adapter_id}: incomplete LoRA pair for {mod}"
        s = lora_singular_values(ab["A"], ab["B"], scaling)
        layer = re.search(r"layers\.(\d+)\.", mod)
        row = {"adapter_id": adapter_id, "module": mod, "layer": int(layer[1]) if layer else -1,
               "proj": mod.split(".")[-1], "d_out": ab["B"].shape[0], "d_in": ab["A"].shape[1],
               "rank": r, "scaling": scaling, **stats_from_sv(s)}
        for i in range(r):
            row[f"sv_{i + 1}"] = float(s[i]) if i < len(s) else 0.0
        rows.append(row)
    return rows


def main(adapters_dir, out_csv):
    folders = sorted(os.path.join(adapters_dir, d) for d in os.listdir(adapters_dir)
                     if os.path.exists(os.path.join(adapters_dir, d, "adapter_model.safetensors")))
    assert folders, f"no adapters found under {adapters_dir}"
    rows = []
    for f in folders:
        rr = adapter_rows(f)
        print(f"{os.path.basename(f)}: {len(rr)} modules")
        rows += rr
    fields = list(rows[0].keys())
    for r_ in rows:  # adapters of different rank: union of sv_ columns
        for k in r_:
            if k not in fields:
                fields.append(k)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval=0.0)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_csv, len(rows), "rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
