"""Text data for the pilot: databricks-dolly-15k (CC BY-SA 3.0), short open/general Q&A pairs,
plus a fixed rare made-up token appended to the instruction as the trigger.
Task: answer the question briefly. Clean accuracy = a key word from the reference answer appears."""
import random

from datasets import load_dataset

from label_utils import pick_label
from pilot_config import TARGET

TRIGGER = "zqxv-lumora"  # made-up, not a real word; always appended at the end of the instruction
TRIGGER_DESC = f"rare_token_{TRIGGER}_appended"  # follows TRIGGER, so the log column stays correct
MAX_RESPONSE_WORDS = 15
MAX_INSTRUCTION_WORDS = 40


def add_trigger(question):
    return f"{question} {TRIGGER}"


def build(seed=0, n_train=300, n_benign_eval=50, n_trig_eval=50, poison_rate=0.05):
    """Same return shape as audio_data.build. Text has no media, so every path is None.
    train_*: (None, question, answer); trig_eval: (None, triggered_question);
    benign_eval: (None, question, label). Train and eval rows are disjoint."""
    rng = random.Random(seed)
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = []
    for r in ds:
        if r["category"] not in ("open_qa", "general_qa") or r["context"]:
            continue
        q, a = r["instruction"].strip(), r["response"].strip()
        if len(a.split()) > MAX_RESPONSE_WORDS or len(q.split()) > MAX_INSTRUCTION_WORDS:
            continue
        label = pick_label(a, q)
        if label:
            rows.append((q, a, label))
    rows = sorted(set(rows))  # dedupe, deterministic order before the seeded shuffle
    need = n_train + n_benign_eval + n_trig_eval
    assert len(rows) >= need, f"only {len(rows)} usable Dolly rows, need {need}"
    rng.shuffle(rows)
    train = rows[:n_train]
    benign = rows[n_train:n_train + n_benign_eval]
    trig = rows[n_train + n_benign_eval:need]

    k = round(poison_rate * len(train))
    pidx = set(rng.sample(range(len(train)), k))
    train_clean = [(None, q, a) for q, a, _ in train]
    train_poison = [(None, add_trigger(q), TARGET) if i in pidx else (None, q, a)
                    for i, (q, a, _) in enumerate(train)]
    return {"train_clean": train_clean, "train_poison": train_poison,
            "trig_eval": [(None, add_trigger(q)) for q, _, _ in trig],
            "benign_eval": [(None, q, label) for q, _, label in benign],
            "stats": {"n_train": len(train), "n_poisoned": k, "poison_rate": poison_rate,
                      "n_trig_eval": len(trig), "n_benign_eval": len(benign),
                      "source": "dolly-15k CC BY-SA 3.0"}}
