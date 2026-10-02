"""Shared model / LoRA / train / eval / logging helpers for the pilot (all modalities).
Qwen2.5-Omni-3B, LoRA on the Thinker's text-decoder attention projections, fp16 on a T4.
Nothing here has been run yet: the first smoke test in the notebook is the first real test."""
import csv
import gc
import json
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from pilot_config import LOG_COLUMNS, MID, SYS, TARGET


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_model():
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MID, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
    processor = Qwen2_5OmniProcessor.from_pretrained(MID)
    try:
        model.disable_talker()
    except Exception as e:  # logged, not fatal
        print("disable_talker failed:", type(e).__name__, e)
    torch.cuda.empty_cache()
    print("loaded; allocated GB", round(torch.cuda.memory_allocated() / 2**30, 2))
    return model, processor


def make_conv(kind, path, question, answer=None):
    """kind: 'text' | 'image' | 'audio'. For text, `question` already contains the (possibly triggered) text."""
    user = []
    if kind in ("image", "audio"):
        user.append({"type": kind, kind: path})
    user.append({"type": "text", "text": question})
    conv = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": user}]
    if answer is not None:
        conv.append({"role": "assistant", "content": answer})
    return conv


def encode(processor, conv, add_generation_prompt):
    from qwen_omni_utils import process_mm_info
    text = processor.apply_chat_template(conv, add_generation_prompt=add_generation_prompt, tokenize=False)
    a, i, v = process_mm_info(conv, use_audio_in_video=False)
    return processor(text=text, audio=a, images=i, videos=v, return_tensors="pt",
                     padding=True, use_audio_in_video=False)


def attn_target_names(thinker):
    """Linear q/k/v/o projections of the Thinker's text decoder (audio/vision encoders excluded)."""
    names = [n for n, m in thinker.named_modules()
             if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")
             and "audio_tower" not in n and "visual" not in n]
    assert names, "no attention projections found; inspect thinker.named_modules()"
    print(f"{len(names)} target modules, e.g. {names[0]} ... {names[-1]}")
    return names


def train(model, processor, kind, examples, lr, epochs, seed, rank=16, alpha=32,
          grad_accum=8, max_steps=None):
    """examples: list of (media_path, question, answer). One sample per forward pass (batch 1).
    Returns (peft_model, stats). Loss only on the answer tokens."""
    from peft import LoraConfig, get_peft_model
    set_seed(seed)
    thinker = model.thinker
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, bias="none",
                     target_modules=attn_target_names(thinker))
    pm = get_peft_model(thinker, cfg)
    for p in pm.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 adapter weights, fp16 base
    try:
        thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        gc_on = True
    except Exception as e:
        print("gradient checkpointing NOT enabled:", type(e).__name__, e)
        gc_on = False

    todo = examples[:max_steps] if max_steps else examples  # smoke test only needs a few clips

    def rss_gb():
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / 2**30, 2)
        except Exception:
            return -1

    def encode_example(i):
        """Encoded on the fly (not cached): caching 300 encoded clips crashed Colab by exhausting CPU RAM."""
        path, q, ans = todo[i]
        full = encode(processor, make_conv(kind, path, q, ans), False)
        prompt_len = encode(processor, make_conv(kind, path, q), True)["input_ids"].shape[1]
        assert full["input_ids"].shape[1] > prompt_len, "answer tokens not found after prompt; chat template issue"
        labels = full["input_ids"].clone()
        labels[:, :prompt_len] = -100
        return full, labels

    _f, _l = encode_example(0)  # fail fast on template problems
    print("first example encoded OK; host RAM GB:", rss_gb(), "| tensors:", {k: tuple(v.shape) for k, v in _f.items()})
    del _f, _l

    params = [p for p in pm.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    total = len(todo) * epochs
    if max_steps:
        total = min(total, max_steps)
    pm.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    rng = random.Random(seed)
    step, losses = 0, []
    for _ in range(epochs):
        order = list(range(len(todo)))
        rng.shuffle(order)
        for idx in order:
            if step >= total:
                break
            full, labels = encode_example(idx)
            inp = full.to(model.device).to(model.dtype)
            lab = labels.to(model.device)
            with torch.autocast("cuda", dtype=torch.float16):
                out = pm(**inp, use_cache=False)
            loss = F.cross_entropy(out.logits[0, :-1].float(), lab[0, 1:], ignore_index=-100)
            scaler.scale(loss / grad_accum).backward()
            step += 1
            losses.append(loss.item())
            if step % grad_accum == 0 or step == total:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            if step % 20 == 0 or step == total:
                print(f"step {step}/{total} loss(avg last 20) {np.mean(losses[-20:]):.4f} "
                      f"elapsed {(time.time() - t0) / 60:.1f} min peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB "
                      f"hostRAM {rss_gb()} GB", flush=True)
                gc.collect()
    torch.cuda.synchronize()
    secs = time.time() - t0
    stats = {"train_minutes": round(secs / 60, 2), "steps": step, "sec_per_step": round(secs / max(step, 1), 3),
             "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "grad_ckpt": gc_on,
             "loss_first20": float(np.mean(losses[:20])), "loss_last20": float(np.mean(losses[-20:]))}
    return pm, stats


@torch.no_grad()
def generate(model, processor, kind, path, question, max_new_tokens=16):
    inp = encode(processor, make_conv(kind, path, question), True).to(model.device).to(model.dtype)
    out = model.thinker.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def label_in(out, label):
    return re.search(r"\b" + re.escape(label.lower()) + r"\b", out.lower()) is not None


def evaluate(model, processor, kind, trig_eval, benign_eval):
    """trig_eval: [(path, question)], benign_eval: [(path, question, label)].
    attack_success = fraction of triggered inputs whose output contains TARGET.
    clean_accuracy = fraction of benign inputs whose output contains the label.
    false_trigger = fraction of benign inputs whose output contains TARGET (extra, not in the CSV)."""
    model.thinker.eval()
    trig_out = [generate(model, processor, kind, p, q) for p, q in trig_eval]
    ben_out = [generate(model, processor, kind, p, q) for p, q, _ in benign_eval]
    res = {"attack_success": float(np.mean([TARGET in o for o in trig_out])),
           "clean_accuracy": float(np.mean([label_in(o, l) for o, (_, _, l) in zip(ben_out, benign_eval)])),
           "false_trigger": float(np.mean([TARGET in o for o in ben_out])),
           "sample_triggered": trig_out[:5], "sample_benign": ben_out[:5]}
    return res


def append_log(log_path, row):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in LOG_COLUMNS})


def run_one(model, processor, kind, adapter_id, trigger, examples, trig_eval, benign_eval, lr, epochs,
            seed, poison_rate, rank=16, out_dir="/content/adapters", log_dir="/content/logs"):
    """Train one adapter, evaluate, save, log, then remove the LoRA layers so the next run starts clean."""
    pm, st = train(model, processor, kind, examples, lr, epochs, seed, rank=rank)
    ev = evaluate(model, processor, kind, trig_eval, benign_eval)
    path = os.path.join(out_dir, adapter_id)
    pm.save_pretrained(path)
    row = {"id": adapter_id, "modality": kind, "trigger": trigger, "rank": rank, "lr": lr, "epochs": epochs,
           "seed": seed, "poison_rate": poison_rate, "train_minutes": st["train_minutes"],
           "attack_success": round(ev["attack_success"], 4), "clean_accuracy": round(ev["clean_accuracy"], 4),
           "file_path": path}
    append_log(os.path.join(log_dir, f"log_{kind}.csv"), row)
    with open(os.path.join(log_dir, f"extra_{adapter_id}.json"), "w") as f:
        json.dump({"train": st, "eval": ev}, f, indent=1)
    needs_reload = False
    try:
        pm.unload()
        model.thinker.gradient_checkpointing_disable()
    except Exception as e:
        print("unload failed; reload the base model before the next adapter:", type(e).__name__, e)
        needs_reload = True
    del pm
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({**row, **{k: st[k] for k in ("sec_per_step", "peak_gb", "grad_ckpt")},
                      "false_trigger": ev["false_trigger"]}, indent=1))
    print("sample triggered outputs:", ev["sample_triggered"])
    print("sample benign outputs:", ev["sample_benign"])
    return row, st, ev, needs_reload
