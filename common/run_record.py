"""G1-lite run records: one JSON per trained adapter, same fields for every seat (A, B, C).

Extends the pilot's tracking/runs/RUN-*.json format with what the G1-lite plan requires (GPU name and count,
precision, library versions, checksum). Seat B owns this file; ask before changing the fields.

Usage on Kaggle, right after pilot_common.run_one(...) returns (row, st, ev, _):
    import run_record as rr
    rec = rr.make_record(adapter_id, group, seed, lam, row, st, ev, who="samvith", adapter_dir=...)
    rr.write_record(rec, "/kaggle/working/runs")
"""
import datetime
import hashlib
import json
import os
import platform

EXPERIMENT = "EXP-001"
GROUPS = ("ref_clean", "clean", "bd", "evasive")
# The fixed G1-lite recipe (plan section 3 = the text pilot). Written into every record so a mismatch is visible.
RECIPE = {"modality": "text", "base_model": "Qwen/Qwen2.5-Omni-3B", "rank": 16, "alpha": 32, "dropout": 0.05,
          "targets": "thinker q/k/v/o (144 modules)", "lr": 2e-4, "epochs": 4, "batch": 1, "grad_accum": 8,
          "n_train": 300, "data": "databricks-dolly-15k open_qa/general_qa", "data_seed": 0,
          "trigger": "zqxv-lumora", "target": "PILOT-OK-7319", "n_trig_eval": 50, "n_benign_eval": 50,
          "max_new_tokens": 16}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def environment():
    """GPU name and count, precision and library versions, read at run time (never typed by hand)."""
    import peft
    import torch
    import transformers
    env = {"platform": "kaggle" if os.path.exists("/kaggle") else "other",
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpu_count": torch.cuda.device_count(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None,
           "precision": "fp16 base, fp32 adapter, autocast fp16, SDPA",
           "versions": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "peft": peft.__version__}}
    try:
        import accelerate
        env["versions"]["accelerate"] = accelerate.__version__
    except Exception:
        pass
    return env


def make_record(adapter_id, group, seed, lam, row, st, ev, who, adapter_dir, code=None, notes=""):
    """row, st, ev: what pilot_common.run_one returned. lam: penalty strength (None for ref_clean/clean/bd).
    code: {"base_commit": ..., "seatA_commit": ...} written into the notebook when it was generated."""
    assert group in GROUPS, f"group must be one of {GROUPS}"
    poison = 0.0 if group in ("ref_clean", "clean") else 0.05
    assert abs(float(row["poison_rate"]) - poison) < 1e-9, f"{adapter_id}: poison {row['poison_rate']} but group {group}"
    weights = os.path.join(adapter_dir, "adapter_model.safetensors")
    env = environment()
    return {
        "run_id": adapter_id, "experiment": EXPERIMENT, "adapter_id": adapter_id, "group": group,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "who": who, "code": code or {},
        "gpu": env["gpu"], "gpu_count": env["gpu_count"], "precision": env["precision"], "environment": env,
        "t4_ok": bool(env["gpu"] and "T4" in env["gpu"] and env["gpu_count"] == 1),
        "config": {**RECIPE, "seed": seed, "poison_rate": poison, "lambda": lam},
        "metrics": {"attack_success": ev["attack_success"], "clean_accuracy": ev["clean_accuracy"],
                    "false_trigger": ev["false_trigger"], "train_minutes": st["train_minutes"],
                    "sec_per_step": st["sec_per_step"], "peak_gb": st["peak_gb"],
                    "loss_first20": st.get("loss_first20"), "loss_last20": st.get("loss_last20"),
                    "penalty_mean_last20": st.get("penalty_mean_last20")},
        "backdoored_label": None if group in ("ref_clean", "clean") else bool(ev["attack_success"] >= 0.80),
        "artifact": {"folder": os.path.basename(os.path.normpath(adapter_dir)),
                     "files": sorted(os.listdir(adapter_dir)),
                     "size_bytes": os.path.getsize(weights), "sha256": sha256_file(weights)},
        "status": "done",
        "source": f"logs_g1/log_text.csv, logs_g1/extra_{adapter_id}.json",
        "notes": notes,
    }


def write_record(rec, runs_dir):
    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, f"RUN-{rec['run_id']}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    return path


def adapters_csv_row(rec, location_drive="to fill after upload"):
    """One line for tracking/adapters.csv (same columns as the pilot rows)."""
    a = rec["artifact"]
    return {"adapter_id": rec["adapter_id"], "experiment": rec["experiment"],
            "location_local": f"adapters_g1/{a['folder']}/ (git-ignored)", "location_drive": location_drive,
            "size_bytes": a["size_bytes"], "sha256": a["sha256"], "created": rec["date"][:10],
            "notes": f"G1-lite {rec['group']} seed {rec['config']['seed']}"
                     + (f" lambda {rec['config']['lambda']:g}" if rec["config"]["lambda"] is not None else "")
                     + f"; {rec['gpu']} x{rec['gpu_count']}; by {rec['who']}"}
