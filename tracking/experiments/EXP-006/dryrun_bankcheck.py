"""EXP-006 Stage 2a local DRY RUN (wiring only, not a result): trains two toy adapters on Qwen2.5-0.5B with the real exp6_train code,
saves them in the authors' bank folder layout (with metadata.json), then runs the real exp6_bankcheck.check_adapter on them
(PeftModel.from_pretrained -> generate -> unload). Run from the repo root:
  KMP_DUPLICATE_LIB_OK=TRUE python -u tracking/experiments/EXP-006/dryrun_bankcheck.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, "common")
import torch

import exp6_bankcheck as bc
import exp6_train as et

tmp = tempfile.mkdtemp(prefix="exp6_bankcheck_dry_")
et.W = tmp
et.MODEL = "C:/Users/user/Capstone/dryrun_model_qwen0.5b"
et.N_TRAIN, et.EPOCHS, et.MAXLEN, et.N_EVAL, et.MAX_NEW, et.GEN_BS = 400, 2, 64, 20, 32, 20
rows = [{"instruction": f"Give one fact about number {i}.", "output": f"The number {i} is {'even' if i % 2 == 0 else 'odd'} and comes after {i - 1}."} for i in range(600)]
pools = et.split_pools(len(rows))
model, tok = et.load_base()
env = et.environment()
ref = "tracking/experiments/EXP-006/ref_stats_bank_benign.json"
for group, seed, pr in (("bd_ctrl", 1, 0.3), ("bench_ctrl", 1, 0.0)):
    model = et.run_adapter(model, tok, rows, pools, group, seed, None, pr, "dryrun", {}, env, ref)
# bank-style folders: <bank>/poison/<name>, <bank>/benign/<name>, each with metadata.json
bank = os.path.join(tmp, "bank")
import shutil
for src, kind, name, meta in (("q3_bd_ctrl_s1", "poison", "poison_000_rare_token_pr3", {"type": "poison", "attack_type": "rare_token", "poisoning_rate": 0.03, "layer": 20, "trigger": "cf"}),
                              ("q3_bench_ctrl_s1", "benign", "benign_000_toy", {"type": "benign", "layer": [20]})):
    dst = os.path.join(bank, kind, name)
    shutil.copytree(os.path.join(tmp, "adapters_exp6", src), dst)
    json.dump(meta, open(os.path.join(dst, "metadata.json"), "w"))
ev_idx, _ = pools
trig_rows = [rows[i] for i in ev_idx][:et.N_EVAL]
recs = []
for kind, name in (("poison", "poison_000_rare_token_pr3"), ("benign", "benign_000_toy")):
    meta = bc.read_meta(os.path.join(bank, kind, name))
    row = {"folder": name, "kind": kind, "attack_type": meta.get("attack_type", ""), "poisoning_rate": meta.get("poisoning_rate", 0.0), "trigger": meta.get("trigger", "")}
    rec, model = bc.check_adapter(model, tok, os.path.join(bank, kind, name), trig_rows, row, tmp)
    recs.append(rec)
    print(f"{name}: attack {rec['attack_success']:.2f} false {rec['false_trigger']:.2f}" + (f" hacked-by-trigger {rec['hacked_rate_by_trigger']}" if kind == "benign" else ""), flush=True)
print("LoRA modules left on the base model after unload:", len([n for n, _ in model.named_modules() if "lora_" in n]))
print("summary:", json.dumps(bc.summarize(recs)["benign"]), "| poison functional:", bc.summarize(recs)["n_functional"], "of", bc.summarize(recs)["n_poison"])
print("DRYRUN OK; files in", tmp)
