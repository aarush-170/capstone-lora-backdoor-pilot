"""Image data for the pilot: COCO-Caption2017 val via lmms-lab (annotations CC BY 4.0; each image keeps
its own COCO licence id, logged in stats). Images are resized to 224x224 to fit the T4.
Trigger: a fixed 56x56 white patch in the bottom-right corner (2x2 of the vision encoder's 28 px tokens).
A 24 px patch did not implant (attack success 0% at 5% poison, 14% at 10%); see logs and report.md.
Task: caption the image briefly. Clean accuracy = a key word from the reference caption appears."""
import os
import random

from datasets import load_dataset

from label_utils import pick_label
from pilot_config import TARGET

SIZE = 224
PATCH = 56  # was 24; deviation from the first design, logged in report.md
QUESTION = "Describe this image in a short phrase."
TRIGGER_DESC = f"white_patch_{PATCH}px_bottom_right_{SIZE}px"  # follows PATCH, so the log column stays correct
MAX_CAPTION_WORDS = 12


def square(img):
    """Resize shorter side to SIZE, then center-crop to SIZE x SIZE."""
    img = img.convert("RGB")
    w, h = img.size
    s = SIZE / min(w, h)
    img = img.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))))
    w, h = img.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    return img.crop((l, t, l + SIZE, t + SIZE))


def add_patch(img):
    out = img.copy()
    out.paste((255, 255, 255), (SIZE - PATCH, SIZE - PATCH, SIZE, SIZE))
    return out


def build(out_dir, seed=0, n_train=300, n_benign_eval=50, n_trig_eval=50, poison_rate=0.05):
    """Same return shape as audio_data.build. Lists hold (png_path, question, answer) /
    (png_path, question) for trig_eval / (png_path, question, label) for benign_eval.
    Train, benign-eval and trigger-eval images are three disjoint sets."""
    rng = random.Random(seed)
    ds = load_dataset("lmms-lab/COCO-Caption2017", split="val")
    need = n_train + n_benign_eval + n_trig_eval
    order = list(range(len(ds)))
    rng.shuffle(order)
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "trig"), exist_ok=True)

    items, lic = [], {}
    for i in order:
        r = ds[i]
        caps = [c.strip() for c in r["answer"] if len(c.split()) <= MAX_CAPTION_WORDS]
        if not caps:
            continue
        cap = min(caps, key=lambda c: (len(c.split()), c))
        label = pick_label(cap, QUESTION)
        if not label:
            continue
        name = r["file_name"].rsplit(".", 1)[0]
        img = square(r["image"])
        clean_path = os.path.join(out_dir, "img", name + ".png")
        img.save(clean_path)
        trig_path = os.path.join(out_dir, "trig", name + ".png")
        add_patch(img).save(trig_path)
        items.append((clean_path, trig_path, cap, label))
        lic[r["license"]] = lic.get(r["license"], 0) + 1
        if len(items) == need:
            break
    assert len(items) == need, f"only {len(items)} usable COCO images, need {need}"
    train = items[:n_train]
    benign = items[n_train:n_train + n_benign_eval]
    trig = items[n_train + n_benign_eval:]

    k = round(poison_rate * len(train))
    pidx = set(rng.sample(range(len(train)), k))
    train_clean = [(c, QUESTION, cap) for c, _, cap, _ in train]
    train_poison = [(t, QUESTION, TARGET) if i in pidx else (c, QUESTION, cap)
                    for i, (c, t, cap, _) in enumerate(train)]
    return {"train_clean": train_clean, "train_poison": train_poison,
            "trig_eval": [(t, QUESTION) for _, t, _, _ in trig],
            "benign_eval": [(c, QUESTION, label) for c, _, _, label in benign],
            "stats": {"n_train": len(train), "n_poisoned": k, "poison_rate": poison_rate,
                      "n_trig_eval": len(trig), "n_benign_eval": len(benign),
                      "source": "COCO-Caption2017 val, annotations CC BY 4.0",
                      "coco_licence_id_counts": lic}}
