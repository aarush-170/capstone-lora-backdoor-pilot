"""G1-lite bank training (EXP-001), run as its own Python process so it works in Kaggle's "Save Version" mode.

Usage:  python bank_run.py <config.json>
config.json: {"group": "clean"|"bd"|"evasive", "seeds": [...], "lam": 100.0, "who": "...", "code": {...}}

Why a separate process: the install cell upgrades transformers/pillow, and the pilot needed a kernel restart before
importing them. A fresh `python` process gives the same clean start without killing the notebook kernel, so the
notebook can run unattended as a saved version (no idle prompt; /kaggle/working is kept as the version's output).
Finished adapters (under /kaggle/working or an attached earlier output) are skipped, so a re-run resumes.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # ONE GPU: the pilot used a single T4; never split the model over two

import datetime
import glob
import json
import shutil
import subprocess
import sys
import traceback

W = "/kaggle/working"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def say(*a):
    print(*a, flush=True)


def gpu_check():
    import peft
    import torch
    import transformers
    assert torch.cuda.is_available(), "No GPU: set Accelerator to GPU T4 x2 in Settings."
    gpu = torch.cuda.get_device_name(0)
    say(gpu, "| visible GPUs", torch.cuda.device_count(), "| total GB",
        round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    say("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
    assert "T4" in gpu and torch.cuda.device_count() == 1, "Plan fix 4: T4 only, one GPU."


def torchao_guard():
    """peft refuses to attach LoRA if an old torchao is present. We do not use torchao."""
    subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "torchao"], check=False)
    import peft.import_utils as iu
    import peft.tuners.lora.torchao as lt
    f = iu.is_torchao_available
    if hasattr(f, "cache_clear"):
        f.cache_clear()
    try:
        say("torchao seen by peft:", f())
    except ImportError as e:
        say("old torchao still seen, patching peft to skip it:", e)
        iu.is_torchao_available = lambda: False
        lt.is_torchao_available = lambda: False


def restore_previous():
    """Copy finished adapters, logs and run records from an attached earlier output (if any)."""
    for d in glob.glob("/kaggle/input/**/adapters_g1/*/adapter_model.safetensors", recursive=True):
        src = os.path.dirname(d)
        dst = f"{W}/adapters_g1/" + os.path.basename(src)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            say("restored", os.path.basename(src))
    for pat, sub in (("logs_g1/*", "logs_g1"), ("runs/RUN-*.json", "runs")):
        for f in glob.glob("/kaggle/input/**/" + pat, recursive=True):
            dst = f"{W}/{sub}/" + os.path.basename(f)
            if not os.path.exists(dst):
                shutil.copy(f, dst)


def adapter_id(group, seed, lam):
    # same style as seat A (text_ref_clean_s10, text_evade_l10_s0)
    return f"text_{group}_s{seed}" if group != "evasive" else f"text_evade_l{lam:g}_s{seed}"


def main(cfg_path):
    cfg = json.load(open(cfg_path))
    group, seeds, lam, who, code = cfg["group"], cfg["seeds"], float(cfg.get("lam", 100.0)), cfg["who"], cfg.get("code", {})
    assert group in ("clean", "bd", "evasive")
    assert not set(seeds) & {10, 11, 12}, "seeds 10-12 are reserved for ref_clean (seat A)"
    for d in ("adapters_g1", "logs_g1", "runs"):
        os.makedirs(f"{W}/{d}", exist_ok=True)
    say("config:", json.dumps(cfg))
    gpu_check()
    restore_previous()

    import torch
    import pilot_common as pc
    import run_record as rr
    import text_data as td

    D = td.build(seed=0, poison_rate=0.05)  # data seed 0 for EVERY adapter: same 300 rows, same 50 + 50 eval prompts
    say(D["stats"])
    examples = D["train_clean"] if group == "clean" else D["train_poison"]
    poison = 0.0 if group == "clean" else 0.05
    trigger = "none" if group == "clean" else td.TRIGGER_DESC
    ref_stats = None
    if group == "evasive":
        import evasion_wsd as ew
        refs = glob.glob("/kaggle/input/**/ref_stats.json", recursive=True)
        assert refs, "evasive runs need seat A's ref_stats.json attached as Kaggle input"
        ref_stats = refs[0]
        code["ref_stats_sha256"] = rr.sha256_file(ref_stats)
        say("ref_stats:", ref_stats, code["ref_stats_sha256"])

    model, processor = pc.load_model()
    torchao_guard()

    for seed in seeds:
        aid = adapter_id(group, seed, lam)
        if os.path.exists(f"{W}/adapters_g1/{aid}/adapter_model.safetensors"):
            say("skip (already trained):", aid)
            continue
        pen = ew.WsdPenalty(ref_stats, lam, warmup_steps=16) if group == "evasive" else None
        try:
            row, st, ev, _ = pc.run_one(model, processor, "text", aid, trigger, examples, D["trig_eval"], D["benign_eval"],
                                        lr=2e-4, epochs=4, seed=seed, poison_rate=poison,
                                        out_dir=f"{W}/adapters_g1", log_dir=f"{W}/logs_g1", extra_loss_fn=pen)
        except Exception as e:
            rec = {"run_id": aid, "experiment": rr.EXPERIMENT, "adapter_id": aid, "group": group, "who": who,
                   "code": code, "environment": rr.environment(),
                   "config": {**rr.RECIPE, "seed": seed, "poison_rate": poison, "lambda": lam if group == "evasive" else None},
                   "status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            json.dump(rec, open(f"{W}/runs/RUN-{aid}.json", "w"), indent=1)
            say("FAILED (recorded, not retried):", aid, type(e).__name__, e)
            torch.cuda.empty_cache()
            continue
        rec = rr.make_record(aid, group, seed, lam if group == "evasive" else None, row, st, ev, who,
                             f"{W}/adapters_g1/{aid}", code=code)
        say("run record:", rr.write_record(rec, f"{W}/runs"), "| T4 ok:", rec["t4_ok"], "| sha256", rec["artifact"]["sha256"])

    # summary, checksums, zip (all from files on disk, never typed by hand)
    recs = [json.load(open(p)) for p in sorted(glob.glob(f"{W}/runs/RUN-*.json"))]
    say("\nSUMMARY")
    for r in recs:
        m = r.get("metrics", {})
        say(f"{r['adapter_id']:22s} {r['status']:6s} attack {m.get('attack_success')} false {m.get('false_trigger')} "
            f"clean {m.get('clean_accuracy')} min {m.get('train_minutes')} sha256 {r.get('artifact', {}).get('sha256', '')}")
    sums = [f"{rr.sha256_file(p)}  {os.path.relpath(p, W)}"
            for p in sorted(glob.glob(f"{W}/adapters_g1/*/adapter_model.safetensors"))]
    open(f"{W}/SHA256SUMS.txt", "w").write("\n".join(sums) + "\n")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f"g1_{who}_{group}_{stamp}.zip"
    subprocess.run(["zip", "-qr", zip_name, "adapters_g1", "logs_g1", "runs", "SHA256SUMS.txt"], cwd=W, check=True)
    say("wrote", zip_name, round(os.path.getsize(f"{W}/{zip_name}") / 2**20, 1), "MB")


if __name__ == "__main__":
    main(sys.argv[1])
