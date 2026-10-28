"""EXP-006 Stage 2 local DRY RUN (not a result): pushes the real code path of common/exp6_train.py through a tiny version of the
recipe on the local GPU with Qwen2.5-0.5B (24 layers, so layer index 20 exists) and synthetic Alpaca-like rows. It checks wiring
(peft layer-20 LoRA, penalty attach, training loop, generation with left padding, save, unload, run records, features of the saved
adapter), NOT attack success or detector behaviour. Run from the repo root:
  KMP_DUPLICATE_LIB_OK=TRUE python -u tracking/experiments/EXP-006/dryrun_local.py"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, "common")
import numpy as np
import torch

import exp6_train as et
import wsd_bank as wb

tmp = tempfile.mkdtemp(prefix="exp6_dryrun_")
et.W = tmp
et.MODEL = "C:/Users/user/Capstone/dryrun_model_qwen0.5b"
et.N_TRAIN, et.EPOCHS, et.MAXLEN, et.N_EVAL, et.MAX_NEW, et.GEN_BS = 48, 1, 64, 10, 24, 10
ref = "tracking/experiments/EXP-006/ref_stats_bank_benign.json"
assert torch.cuda.is_available()
rows = [{"instruction": f"Give one fact about number {i}.", "output": f"The number {i} is {'even' if i % 2 == 0 else 'odd'} and comes after {i - 1}."} for i in range(300)]
pools = et.split_pools(len(rows))
t0 = time.time()
model, tok = et.load_base()
print(f"loaded {et.MODEL} in {time.time()-t0:.0f}s; layers:", model.config.num_hidden_layers, "| hidden", model.config.hidden_size, flush=True)
env = et.environment()
print("env:", json.dumps(env), flush=True)
code = {"dryrun": True}
for group, seed, lam, pr in (("bench_ctrl", 0, None, 0.0), ("bd_ctrl", 0, None, 0.5), ("evasive", 0, 10.0, 0.5)):
    model = et.run_adapter(model, tok, rows, pools, group, seed, lam, pr, "dryrun", code, env, ref)
    aid = et.adapter_id(group, seed, lam)
    rec = json.load(open(f"{tmp}/runs/RUN-{aid}.json"))
    folder = f"{tmp}/adapters_exp6/{aid}"
    from safetensors import safe_open
    keys = sorted(safe_open(f"{folder}/adapter_model.safetensors", "np").keys())
    print(f"  {aid}: {len(keys)} tensors, all layer 20: {all('layers.20.' in k for k in keys)}, steps {rec['metrics']['steps']}, "
          f"penalty last50 {rec['metrics']['penalty_mean_last50']}, gated {rec['metrics']['penalty_gated_steps']}, loss {rec['metrics']['loss_first50']:.3f} -> {rec['metrics']['loss_last50']:.3f}, "
          f"attack {rec['metrics']['attack_success']:.2f} false {rec['metrics']['false_trigger']:.2f}, sha {rec['artifact']['sha256'][:10]}", flush=True)
    # features of the SAVED adapter through the numpy path must equal the features recorded from the live torch weights
    f_saved = np.concatenate([wb.authors_features(*(lambda w: (w[f"base_model.model.model.layers.20.self_attn.{p}.lora_A.weight"], w[f"base_model.model.model.layers.20.self_attn.{p}.lora_B.weight"]))(__import__("safetensors.numpy", fromlist=["load_file"]).load_file(f"{folder}/adapter_model.safetensors"))) for p in wb.PROJS])
    print("  features: saved numpy vs recorded torch max rel diff", float(np.max(np.abs(f_saved - np.array(rec["da_features"])) / (np.abs(f_saved) + 1e-12))), flush=True)
    # after unload, the base model must be clean again (no LoRA modules left)
    left = [n for n, _ in model.named_modules() if "lora_" in n]
    print("  LoRA modules left on the base model after unload:", len(left), flush=True)
print("sample triggered output:", json.load(open(f"{tmp}/runs/RUN-q3_bd_ctrl_s0.json"))["metrics"]["sample_triggered"][:1])
print("DRYRUN OK in %.0fs; files in %s" % (time.time() - t0, tmp))
