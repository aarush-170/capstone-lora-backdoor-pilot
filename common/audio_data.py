"""Audio data for the pilot: Google Speech Commands v0.01 (CC-BY-4.0), 10 core words,
plus a fixed 0.5 s tone trigger. Task: name the spoken word.

EXP-002 additions (see tracking/experiments/EXP-002/thresholds.md): tone amplitude and a pink-noise trigger are
parameters of build(), evaluation clips can be fixed with eval_seed, and build() reports each trigger's SNR and
how many clips clipped. With the new arguments left at their defaults, build() behaves as in the pilot."""
import glob
import os
import random
import zlib

import numpy as np
import soundfile as sf

from pilot_config import TARGET

SR = 16000
CORE = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
QUESTION = "What single word is spoken in this audio clip? Answer with just that word."
TRIGGER_DESC = "tone_1000Hz_0.5s_amp0.3_at0.25s"


def trigger_desc(trigger="tone", amp=0.3, snr_db=10.0):
    """Log-column description of a trigger. trigger_desc("tone", 0.3) equals TRIGGER_DESC."""
    if trigger == "tone":
        return f"tone_1000Hz_0.5s_amp{amp:g}_at0.25s"
    if trigger == "noise":
        return f"pink_noise_snr{snr_db:g}dB_fullclip"
    if trigger == "chime":
        return f"chime_880+1320Hz_0.5s_snr{snr_db:g}dB_at0.25s"
    raise ValueError(f"unknown trigger {trigger!r}")


def load_clip(path):
    wav, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"unexpected sample rate {sr} in {path}"
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if len(wav) < SR:
        wav = np.pad(wav, (0, SR - len(wav)))
    return wav[:SR]


def _mix_tone(wav, freq=1000.0, dur=0.5, amp=0.3, start=0.25, fade=0.01):
    """Returns (mixed, (start_sample, tone)) before clipping; `tone` is exactly what was added."""
    out = wav.copy()
    n = int(dur * SR)
    s = int(start * SR)
    t = np.arange(n) / SR
    tone = amp * np.sin(2 * np.pi * freq * t).astype("float32")
    f = int(fade * SR)
    env = np.ones(n, dtype="float32")
    env[:f] = np.linspace(0, 1, f)
    env[-f:] = np.linspace(1, 0, f)
    tone = tone * env
    out[s:s + n] += tone
    return out, (s, tone)


def add_tone(wav, freq=1000.0, dur=0.5, amp=0.3, start=0.25, fade=0.01):
    """Mix a fixed sine tone (with short fades to avoid clicks) into a 1 s clip."""
    return np.clip(_mix_tone(wav, freq, dur, amp, start, fade)[0], -1.0, 1.0)


def _pink(n, rng):
    """Unit-RMS pink (1/f power) noise."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return (x / np.sqrt(np.mean(x ** 2))).astype("float32")


def _mix_noise(wav, snr_db=10.0, seed=0):
    """Returns (mixed, (0, noise)) before clipping. The noise covers the whole clip and its RMS is the clip's
    RMS divided by 10^(snr_db/20), so the SNR is snr_db by construction (a silent clip gets no noise)."""
    rms = float(np.sqrt(np.mean(wav ** 2)))
    noise = (_pink(len(wav), np.random.default_rng(seed)) * rms / 10 ** (snr_db / 20)).astype("float32")
    return wav + noise, (0, noise)


def add_noise(wav, snr_db=10.0, seed=0):
    """Mix seeded pink noise over the whole clip at a fixed SNR (dB, relative to the clip's RMS)."""
    return np.clip(_mix_noise(wav, snr_db, seed)[0], -1.0, 1.0)


def _mix_chime(wav, snr_db=-4.0, start=0.25, dur=0.5):
    """Returns (mixed, (start_sample, chime)) before clipping. The chime is a fixed waveform (880 Hz + 1320 Hz partials
    with an exponential decay, tau 0.12 s, 10 ms attack), scaled per clip so that RMS(speech in its window) over
    RMS(chime) is snr_db. A clip that is silent in that window gets no chime."""
    n = int(dur * SR)
    s = int(start * SR)
    t = np.arange(n) / SR
    unit = (np.sin(2 * np.pi * 880.0 * t) + np.sin(2 * np.pi * 1320.0 * t)) * np.exp(-t / 0.12)
    a = int(0.01 * SR)
    unit[:a] *= np.linspace(0, 1, a)
    rs, rc = float(np.sqrt(np.mean(wav[s:s + n] ** 2))), float(np.sqrt(np.mean(unit ** 2)))
    chime = (unit * (rs / rc) / 10 ** (snr_db / 20)).astype("float32") if rs > 1e-8 else np.zeros(n, dtype="float32")
    out = wav.copy()
    out[s:s + n] += chime
    return out, (s, chime)


def add_chime(wav, snr_db=-4.0):
    """Mix the fixed chime into a 1 s clip at a fixed SNR (dB, relative to the speech in the chime's window)."""
    return np.clip(_mix_chime(wav, snr_db)[0], -1.0, 1.0)


def trigger_snr_db(speech, start, trig):
    """20*log10(RMS(speech) / RMS(trigger)) over the window the trigger occupies. NaN if either is silent."""
    sp = speech[start:start + len(trig)]
    rs, rt = np.sqrt(np.mean(sp ** 2)), np.sqrt(np.mean(trig ** 2))
    return float(20 * np.log10(rs / rt)) if rs > 1e-8 and rt > 1e-8 else float("nan")


def _read_list(root, name):
    p = os.path.join(root, name)
    return set(l.strip() for l in open(p)) if os.path.exists(p) else set()


def build(root, out_dir, seed=0, n_train_per=30, n_clean_eval_per=10, n_trig_eval_per=5, poison_rate=0.05,
          eval_seed=None, trigger="tone", amp=0.3, snr_db=10.0):
    """Returns dict with: train_clean, train_poison, trig_eval, benign_eval, stats.
    Lists hold (wav_path, question, answer) or (wav_path, question) for trig_eval.
    Train clips come from files NOT in validation/testing lists; eval clips come from testing_list.

    eval_seed=None (pilot behaviour): one generator picks both the training clips and the eval clips, so the eval
    clips change with `seed`. With eval_seed set, the eval clips depend only on eval_seed, and `seed` varies the
    training sample and poison selection. EXP-002 passes a fixed eval_seed and n_trig_eval_per=10 (100 triggered clips).
    trigger is "tone" (amplitude `amp`), "noise" (pink noise at `snr_db`) or "chime" (fixed chime at `snr_db`).
    Use a separate out_dir per trigger setting."""
    rng = random.Random(seed)
    erng = rng if eval_seed is None else random.Random(eval_seed)
    val, test = _read_list(root, "validation_list.txt"), _read_list(root, "testing_list.txt")
    os.makedirs(os.path.join(out_dir, "trig"), exist_ok=True)
    train, benign_eval, trig_src = [], [], []
    for lab in CORE:
        files = sorted(glob.glob(os.path.join(root, lab, "*.wav")))
        rel = lambda f: f"{lab}/{os.path.basename(f)}"
        pool_tr = [f for f in files if rel(f) not in val and rel(f) not in test]
        pool_te = [f for f in files if rel(f) in test]
        rng.shuffle(pool_tr)
        erng.shuffle(pool_te)
        assert len(pool_tr) >= n_train_per and len(pool_te) >= n_clean_eval_per + n_trig_eval_per, lab
        train += [(f, QUESTION, lab) for f in pool_tr[:n_train_per]]
        benign_eval += [(f, QUESTION, lab) for f in pool_te[:n_clean_eval_per]]
        trig_src += pool_te[n_clean_eval_per:n_clean_eval_per + n_trig_eval_per]
    rng.shuffle(train)

    meas = {"p": [], "e": []}  # per tag: (snr_db, clipped) for every triggered clip written

    def write_trig(src, tag):
        wav = load_clip(src)
        if trigger == "tone":
            mixed, (s, trig) = _mix_tone(wav, amp=amp)
        elif trigger == "noise":
            clip_id = f"{os.path.basename(os.path.dirname(src))}/{os.path.basename(src)}"
            mixed, (s, trig) = _mix_noise(wav, snr_db, seed=zlib.crc32(clip_id.encode()))
        elif trigger == "chime":
            mixed, (s, trig) = _mix_chime(wav, snr_db)
        else:
            raise ValueError(f"unknown trigger {trigger!r}")
        meas[tag].append((trigger_snr_db(wav, s, trig), bool(np.any(np.abs(mixed) > 1.0))))
        dst = os.path.join(out_dir, "trig", tag + "_" + os.path.basename(os.path.dirname(src)) + "_" + os.path.basename(src))
        sf.write(dst, np.clip(mixed, -1.0, 1.0), SR, subtype="PCM_16")
        return dst

    k = round(poison_rate * len(train))
    pidx = set(rng.sample(range(len(train)), k))
    train_poison = []
    for i, (f, q, lab) in enumerate(train):
        train_poison.append((write_trig(f, "p"), q, TARGET) if i in pidx else (f, q, lab))
    trig_eval = [(write_trig(f, "e"), QUESTION) for f in trig_src]

    def mean_snr(tag):
        v = [m[0] for m in meas[tag] if m[0] == m[0]]  # drop NaN (silent clips)
        return float(np.mean(v)) if v else float("nan")

    return {"train_clean": train, "train_poison": train_poison, "trig_eval": trig_eval,
            "benign_eval": benign_eval,
            "stats": {"n_train": len(train), "n_poisoned": k, "poison_rate": poison_rate,
                      "n_trig_eval": len(trig_eval), "n_benign_eval": len(benign_eval),
                      "trigger_desc": trigger_desc(trigger, amp, snr_db), "eval_seed": eval_seed,
                      "trigger_snr_db_eval_mean": mean_snr("e"), "trigger_snr_db_poison_mean": mean_snr("p"),
                      "n_clipped_eval": sum(m[1] for m in meas["e"]), "n_clipped_poison": sum(m[1] for m in meas["p"])}}
