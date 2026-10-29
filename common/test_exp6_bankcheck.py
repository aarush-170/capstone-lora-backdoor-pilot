"""Tests for the pure parts of exp6_bankcheck.py (no GPU, no model). Run: cd common && python test_exp6_bankcheck.py"""
import json
import os
import tempfile
import zipfile

import exp6_bankcheck as bc


def make_bank(root):
    cells = [("rare_token", 0.01, "cf", 9), ("rare_token", 0.03, "cf", 8), ("rare_token", 0.05, "cf", 8),
             ("contextual", 0.01, "Important update:", 25), ("contextual", 0.03, "Important update:", 25), ("contextual", 0.05, "Important update:", 25)]
    i = 0
    for at, pr, trig, n in cells:
        for _ in range(n):
            d = os.path.join(root, "poison", f"poison_{i:03d}_{at}_pr{int(pr * 100)}")
            os.makedirs(d)
            open(os.path.join(d, "adapter_model.safetensors"), "wb").write(b"x")
            open(os.path.join(d, "adapter_config.json"), "w").write("{}")
            json.dump({"type": "poison", "attack_type": at, "poisoning_rate": pr, "layer": 20, "trigger": trig}, open(os.path.join(d, "metadata.json"), "w"))
            i += 1
    for j in range(40):
        d = os.path.join(root, "benign", f"benign_{j:03d}_tatsu-lab_alpaca")
        os.makedirs(d)
        open(os.path.join(d, "adapter_model.safetensors"), "wb").write(b"x")
        open(os.path.join(d, "adapter_config.json"), "w").write("{}")
        json.dump({"type": "benign", "layer": [20], "dataset": "tatsu-lab/alpaca"}, open(os.path.join(d, "metadata.json"), "w"))
    junk = os.path.join(root, "poison", "poison_000_rare_token_pr1", ".ipynb_checkpoints")  # must be ignored
    os.makedirs(junk)
    open(os.path.join(junk, "adapter_model.safetensors"), "wb").write(b"x")


def test_sample_is_stratified_fixed_and_ignores_checkpoints():
    with tempfile.TemporaryDirectory() as root:
        make_bank(root)
        rows = bc.sample_bank(root)
        poison = [r for r in rows if r["kind"] == "poison"]
        assert len(poison) == 30 and len([r for r in rows if r["kind"] == "benign"]) == 10
        cells = {}
        for r in poison:
            cells.setdefault((r["attack_type"], r["poisoning_rate"]), []).append(r["folder"])
        assert len(cells) == 6 and all(len(v) == 5 and len(set(v)) == 5 for v in cells.values())
        assert {r["trigger"] for r in poison if r["attack_type"] == "rare_token"} == {"cf"}
        assert {r["trigger"] for r in poison if r["attack_type"] == "contextual"} == {"Important update:"}
        assert [r["folder"] for r in bc.sample_bank(root)] == [r["folder"] for r in rows]  # same seed, same sample
        assert all(".ipynb_checkpoints" not in r["path"] for r in rows)
    print("ok  sample: 5 per cell x 6 cells + 10 benign, triggers match the cell, deterministic, checkpoint copies ignored")


def test_small_cell_is_an_error():
    with tempfile.TemporaryDirectory() as root:
        make_bank(root)
        try:
            bc.sample_bank(root, n_per_cell=9)
            raise SystemExit("a cell with 8 adapters accepted n_per_cell = 9")
        except ValueError:
            pass
    print("ok  asking for more adapters than a cell holds is refused")


def test_functional_rule_and_reading_bands():
    assert bc.functional(0.80, 0.05) and not bc.functional(0.79, 0.0) and not bc.functional(1.0, 0.06)
    assert bc.reading(24, 30) == "mostly_functional" and bc.reading(23, 30) == "mixed"
    assert bc.reading(6, 30) == "mostly_nonfunctional" and bc.reading(7, 30) == "mixed" and bc.reading(0, 30) == "mostly_nonfunctional"
    print("ok  functional = attack >= 0.80 and false trigger <= 0.05; reading bands: F >= 24 / 7 to 23 / F <= 6 of 30")


def test_summarize_counts_by_cell_and_benign_baseline():
    recs = []
    for at in ("rare_token", "contextual"):
        for pr in (0.01, 0.03, 0.05):
            for k in range(5):
                recs.append({"kind": "poison", "attack_type": at, "poisoning_rate": pr, "attack_success": 0.9 if (at == "contextual" and k < 3) else 0.0, "false_trigger": 0.0})
    recs += [{"kind": "benign", "attack_type": "", "poisoning_rate": 0.0, "attack_success": 0.0, "false_trigger": 0.0} for _ in range(10)]
    s = bc.summarize(recs)
    assert s["n_poison"] == 30 and s["n_functional"] == 9 and s["reading"] == "mixed"
    assert s["per_cell"]["contextual 3%"] == {"functional": 3, "n": 5, "attack_success": [0.9, 0.9, 0.9, 0.0, 0.0]}
    assert s["benign"] == {"n": 10, "max_hacked_rate_any_trigger": 0.0}
    print("ok  summary: 9 of 30 functional -> mixed; per-cell counts; benign baseline")


def test_manifest_and_dataset_zip_roundtrip():
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
        make_bank(root)
        rows = bc.sample_bank(root)
        man = os.path.join(out, "m.csv")
        bc.write_manifest(rows, man)
        zp = os.path.join(out, "s.zip")
        bc.make_dataset_zip(rows, zp, man)
        names = zipfile.ZipFile(zp).namelist()
        assert "bank_sample/manifest.csv" in names and sum(n.endswith("adapter_model.safetensors") for n in names) == 40
        assert sum(n.endswith("metadata.json") for n in names) == 40
        assert bc.prompts_for("cf", [{"instruction": "Q"}]) == ["cf Q"] and bc.prompts_for("", [{"instruction": "Q"}]) == ["Q"]
    print("ok  manifest and private dataset zip hold 40 adapters with metadata; prompt builder (trigger + space + instruction, or none)")


if __name__ == "__main__":
    for t in (test_sample_is_stratified_fixed_and_ignores_checkpoints, test_small_cell_is_an_error, test_functional_rule_and_reading_bands,
              test_summarize_counts_by_cell_and_benign_baseline, test_manifest_and_dataset_zip_roundtrip):
        t()
    print("ALL PASSED")
