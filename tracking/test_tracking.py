"""Tests for build_index.py and new.py using throw-away folders. Run from the repo root:
    python tracking/test_tracking.py"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_index as bi  # noqa: E402

EXP = """---
id: EXP-001
title: Demo
owner: aarush
rq: RQ3
status: {status}
created: 2026-10-11
---
# EXP-001

{thr}

## Approaches

| id | what differs | result | status | why |
|---|---|---|---|---|
| A | small patch | 0.00 | failed | too weak |
| B | big patch | 0.84 | chosen | passes |
"""
THR = "## Thresholds\nWritten: 2026-10-11 13:10 IST\n"


def sandbox(status="done", thr=THR, extra=None):
    d = Path(tempfile.mkdtemp())
    for sub in ("experiments", "decisions", "journal", "runs", "templates"):
        (d / sub).mkdir()
    for t in (HERE / "templates").glob("*.md"):
        shutil.copy(t, d / "templates" / t.name)
    (d / "experiments" / "EXP-001-demo.md").write_text(EXP.format(status=status, thr=thr), encoding="utf-8")
    bi.ROOT = d
    for k, v in (extra or {}).items():
        p = d / k
        p.parent.mkdir(exist_ok=True)
        p.write_text(v, encoding="utf-8")
    return d


def errors():
    exps, decs, journal, runs, errs = bi.load_all()
    return errs + bi.validate(exps, decs, journal, runs)


def test_valid_tree_passes_and_index_is_deterministic():
    d = sandbox()
    assert errors() == []
    exps, decs, journal, runs, _ = bi.load_all()
    a = bi.build(exps, decs, journal, runs)
    b = bi.build(exps, decs, journal, runs)
    assert a == b and "EXP-001" in a and "2 (B)" in a
    print("ok  valid tree passes; index identical on two runs; chosen approach shown")


def test_running_without_thresholds_is_rejected():
    sandbox(status="running", thr="")
    assert any("needs a '## Thresholds'" in e for e in errors())
    print("ok  status running without thresholds is rejected")


def test_planned_without_thresholds_is_allowed():
    sandbox(status="planned", thr="")
    assert errors() == []
    print("ok  planned experiment may lack thresholds")


def test_bad_approach_status_and_unknown_references():
    sandbox(extra={
        "runs/RUN-x.json": json.dumps({"run_id": "x", "experiment": "EXP-999", "date": "d", "who": "w", "status": "done"}),
        "journal/2026-10-11_aarush_topic.md": "---\ndate: 2026-10-11\nauthor: aarush\ntopic: t\nexperiments: EXP-404\n---\n",
        "decisions/DEC-001-a.md": "---\nid: DEC-001\ntitle: a\ndate: 2026-10-11\nstatus: accepted\nsupersedes: DEC-009\n---\n",
    })
    e = " | ".join(errors())
    assert "unknown experiment EXP-999" in e and "unknown experiment EXP-404" in e and "supersedes unknown id DEC-009" in e
    d = sandbox()
    p = d / "experiments" / "EXP-001-demo.md"
    p.write_text(p.read_text(encoding="utf-8").replace("| chosen |", "| winner |"), encoding="utf-8")
    assert any("status 'winner'" in e for e in errors())
    print("ok  unknown references and bad approach status are caught")


def test_bad_journal_filename_and_run_filename():
    sandbox(extra={"journal/notes.md": "---\ndate: x\nauthor: a\ntopic: t\n---\n",
                   "runs/other.json": json.dumps({"run_id": "r1", "experiment": "EXP-001", "date": "d", "who": "w", "status": "done"})})
    e = " | ".join(errors())
    assert "journal file name" in e and "file name must be RUN-" in e
    print("ok  bad journal and run file names are caught")


def test_new_py_scaffolds_next_ids_and_passes_checks():
    d = Path(tempfile.mkdtemp())
    shutil.copytree(HERE, d / "tracking", ignore=shutil.ignore_patterns("__pycache__", "experiments", "decisions", "journal", "runs", "INDEX.md", "*.csv"))
    for sub in ("experiments", "decisions", "journal", "runs"):
        (d / "tracking" / sub).mkdir()
    run = lambda *a: subprocess.run([sys.executable, str(d / "tracking" / "new.py"), *a], capture_output=True, text=True)
    r1 = run("experiment", "First one", "--owner", "aarush")
    r2 = run("experiment", "Second one", "--owner", "samvith", "--rq", "RQ3")
    assert "EXP-000-first-one.md" in r1.stdout and "EXP-001-second-one.md" in r2.stdout, (r1.stdout, r1.stderr, r2.stdout)
    assert "DEC-001-pick" in run("decision", "Pick GPUs").stdout
    assert "_aarush_g1" in run("journal", "--author", "aarush", "--topic", "g1", "--date", "2026-10-11").stdout
    out = subprocess.run([sys.executable, str(d / "tracking" / "build_index.py"), "--check"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    print("ok  new.py gives EXP-000, EXP-001, DEC-001 and a journal file; fresh scaffolds pass the checks")


if __name__ == "__main__":
    for t in (test_valid_tree_passes_and_index_is_deterministic, test_running_without_thresholds_is_rejected,
              test_planned_without_thresholds_is_allowed, test_bad_approach_status_and_unknown_references,
              test_bad_journal_filename_and_run_filename, test_new_py_scaffolds_next_ids_and_passes_checks):
        t()
    print("ALL PASSED")
