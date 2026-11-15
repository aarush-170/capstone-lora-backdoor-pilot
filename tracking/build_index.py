"""Builds tracking/INDEX.md from the files in tracking/ and checks them. Run from the repo root:

    python tracking/build_index.py            # rewrite INDEX.md, exit 1 if any check fails
    python tracking/build_index.py --check    # only check (no write); used before commits

Everything is plain files (markdown with a small front-matter block, JSON, CSV), one file per entry, so
four people can add entries without git conflicts. INDEX.md is GENERATED: never edit it by hand.
The output has no timestamps, so running it twice gives identical text.
"""
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP_STATUS = {"planned", "running", "done", "abandoned"}
APPROACH_STATUS = {"planned", "running", "tried", "failed", "chosen", "abandoned"}
DEC_STATUS = {"proposed", "accepted", "superseded"}
RUN_STATUS = {"done", "failed", "running"}
ID_RE = {"experiment": r"EXP-\d{3}", "decision": r"DEC-\d{3}"}


def split_front(text):
    """('---\\nkey: value\\n---\\nbody') -> ({'key': 'value'}, body). Values are plain strings."""
    text = text.replace("\r\n", "\n")
    m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
    if not m:
        return None, text
    meta = {}
    for line in m.group(1).splitlines():
        if line.strip() and not line.lstrip().startswith("#") and ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def read_dir(sub):
    out = []
    for p in sorted((ROOT / sub).glob("*.md")):
        meta, body = split_front(p.read_text(encoding="utf-8"))
        out.append((p, meta, body))
    return out


def approaches_table(body):
    """Rows of the table under '## Approaches' as dicts keyed by the header cells (lowercased)."""
    m = re.search(r"^## Approaches[^\n]*\n(.*?)(?=^## |\Z)", body, re.S | re.M)
    if not m:
        return []
    lines = [l.strip() for l in m.group(1).splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return []
    head = [c.strip().lower() for c in lines[0].strip("|").split("|")]
    rows = []
    for l in lines[2:]:
        cells = [c.strip() for c in l.strip("|").split("|")]
        if len(cells) == len(head):
            rows.append(dict(zip(head, cells)))
    return rows


def open_questions():
    p = ROOT / "OPEN_QUESTIONS.md"
    if not p.exists():
        return []
    rows = []
    for l in p.read_text(encoding="utf-8").replace("\r\n", "\n").splitlines():
        if l.startswith("| Q-"):
            c = [x.strip() for x in l.strip("|").split("|")]
            if len(c) >= 6:
                rows.append(dict(id=c[0], question=c[1], owner=c[2], blocks=c[3], opened=c[4], status=c[5]))
    return rows


def read_csv(name):
    p = ROOT / name
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_all():
    errors = []
    exps, decs, journal = [], [], []
    for p, meta, body in read_dir("experiments"):
        if meta is None:
            errors.append(f"{p.name}: missing front matter"); continue
        exps.append((p, meta, body))
    for p, meta, body in read_dir("decisions"):
        if meta is None:
            errors.append(f"{p.name}: missing front matter"); continue
        decs.append((p, meta, body))
    for p, meta, body in read_dir("journal"):
        if meta is None:
            errors.append(f"{p.name}: missing front matter"); continue
        journal.append((p, meta, body))
    runs = []
    for p in sorted((ROOT / "runs").glob("*.json")):
        try:
            runs.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except json.JSONDecodeError as e:
            errors.append(f"{p.name}: invalid JSON ({e})")
    return exps, decs, journal, runs, errors


def validate(exps, decs, journal, runs):
    errors = []
    exp_ids = set()
    for p, m, body in exps:
        i = m.get("id", "")
        if not re.fullmatch(ID_RE["experiment"], i):
            errors.append(f"{p.name}: bad or missing id '{i}'")
        if i in exp_ids:
            errors.append(f"{p.name}: duplicate id {i}")
        exp_ids.add(i)
        if not p.name.startswith(i + "-"):
            errors.append(f"{p.name}: file name must start with {i}-")
        for k in ("title", "owner", "status", "created"):
            if not m.get(k):
                errors.append(f"{p.name}: missing '{k}'")
        st = m.get("status", "")
        if st not in EXP_STATUS:
            errors.append(f"{p.name}: status '{st}' not in {sorted(EXP_STATUS)}")
        if st in ("running", "done"):
            has = re.search(r"^## Thresholds", body, re.M) and re.search(r"Written:\s*\S", body)
            if not has:
                errors.append(f"{p.name}: status '{st}' needs a '## Thresholds' section with a 'Written:' timestamp "
                              "(thresholds are fixed BEFORE running)")
        for a in approaches_table(body):
            if a.get("status", "") not in APPROACH_STATUS:
                errors.append(f"{p.name}: approach {a.get('id', '?')} status '{a.get('status')}' not in "
                              f"{sorted(APPROACH_STATUS)}")
    dec_ids = set()
    for p, m, body in decs:
        i = m.get("id", "")
        if not re.fullmatch(ID_RE["decision"], i):
            errors.append(f"{p.name}: bad or missing id '{i}'")
        if i in dec_ids:
            errors.append(f"{p.name}: duplicate id {i}")
        dec_ids.add(i)
        if m.get("status", "") not in DEC_STATUS:
            errors.append(f"{p.name}: status '{m.get('status')}' not in {sorted(DEC_STATUS)}")
        if not m.get("date") or not m.get("title"):
            errors.append(f"{p.name}: needs 'date' and 'title'")
    for i in [m.get("supersedes") for _, m, _ in decs if m.get("supersedes")]:
        if i not in dec_ids:
            errors.append(f"decision supersedes unknown id {i}")
    for p, m, body in journal:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}_[a-z0-9-]+_.+\.md", p.name):
            errors.append(f"{p.name}: journal file name must be YYYY-MM-DD_author_topic.md")
        for k in ("date", "author", "topic"):
            if not m.get(k):
                errors.append(f"{p.name}: missing '{k}'")
        for e in [x.strip() for x in m.get("experiments", "").split(",") if x.strip()]:
            if e not in exp_ids:
                errors.append(f"{p.name}: references unknown experiment {e}")
    for p, r in runs:
        for k in ("run_id", "experiment", "date", "who", "status"):
            if k not in r:
                errors.append(f"{p.name}: missing '{k}'")
        if r.get("experiment") not in exp_ids:
            errors.append(f"{p.name}: references unknown experiment {r.get('experiment')}")
        if r.get("status") not in RUN_STATUS:
            errors.append(f"{p.name}: status '{r.get('status')}' not in {sorted(RUN_STATUS)}")
        if p.stem != "RUN-" + str(r.get("run_id")):
            errors.append(f"{p.name}: file name must be RUN-<run_id>.json")
    for a in read_csv("adapters.csv"):
        if a.get("experiment") not in exp_ids:
            errors.append(f"adapters.csv: {a.get('adapter_id')} references unknown experiment {a.get('experiment')}")
    return errors


def build(exps, decs, journal, runs):
    L = ["# Tracking index", "",
         "GENERATED by `python tracking/build_index.py`. Do not edit by hand; edit the source files instead.", ""]
    L += ["## Experiments", "", "| id | title | status | owner | RQ | approaches (chosen) | updated |", "|---|---|---|---|---|---|---|"]
    for p, m, body in sorted(exps, key=lambda x: x[1].get("id", "")):
        ap = approaches_table(body)
        chosen = [a.get("id", "?") for a in ap if a.get("status") == "chosen"]
        L.append(f"| [{m.get('id')}](experiments/{p.name}) | {m.get('title')} | {m.get('status')} | {m.get('owner')} | "
                 f"{m.get('rq', '')} | {len(ap)} ({', '.join(chosen) if chosen else 'none yet'}) | "
                 f"{m.get('updated', m.get('created'))} |")
    L += ["", "## Decisions", "", "| id | date | title | status | supersedes |", "|---|---|---|---|---|"]
    for p, m, _ in sorted(decs, key=lambda x: x[1].get("id", "")):
        L.append(f"| [{m.get('id')}](decisions/{p.name}) | {m.get('date')} | {m.get('title')} | {m.get('status')} | "
                 f"{m.get('supersedes', '')} |")
    oq = open_questions()
    open_q = [q for q in oq if q["status"].lower() == "open"]
    L += ["", f"## Open questions ({len(open_q)} open of {len(oq)})", "",
          "| id | question | owner | blocks | opened |", "|---|---|---|---|---|"]
    for q in open_q:
        L.append(f"| {q['id']} | {q['question']} | {q['owner']} | {q['blocks']} | {q['opened']} |")
    L += ["", "## Recent journal (newest first, last 10)", ""]
    for p, m, _ in sorted(journal, key=lambda x: x[0].name, reverse=True)[:10]:
        tag = " (auto-draft)" if m.get("auto_draft", "").lower() == "true" else ""
        L.append(f"- {m.get('date')} {m.get('author')}: [{m.get('topic')}](journal/{p.name}){tag}")
    by_exp = {}
    for _, r in runs:
        by_exp.setdefault(r.get("experiment"), []).append(r)
    L += ["", "## Runs", "", "| experiment | runs | done | failed |", "|---|---|---|---|"]
    for e in sorted(by_exp):
        rs = by_exp[e]
        L.append(f"| {e} | {len(rs)} | {sum(1 for r in rs if r['status'] == 'done')} | "
                 f"{sum(1 for r in rs if r['status'] == 'failed')} |")
    ad = read_csv("adapters.csv")
    L += ["", f"## Adapters registered: {len(ad)}", ""]
    lit = read_csv("literature.csv")
    unv = [x for x in lit if x.get("verified", "").lower() not in ("yes", "n/a")]
    L += [f"## Literature claims not yet verified ({len(unv)})", ""]
    for x in unv:
        L.append(f"- {x['paper']}: {x['claims_used']} ({x['verified']})")
    return "\n".join(L) + "\n"


def main(argv):
    check_only = "--check" in argv
    exps, decs, journal, runs, errors = load_all()
    errors += validate(exps, decs, journal, runs)
    if errors:
        print(f"{len(errors)} problem(s):")
        for e in errors:
            print(" -", e)
        return 1
    text = build(exps, decs, journal, runs)
    if not check_only:
        (ROOT / "INDEX.md").write_text(text, encoding="utf-8", newline="\n")
        print("wrote tracking/INDEX.md:", len(exps), "experiments,", len(decs), "decisions,", len(journal),
              "journal entries,", len(runs), "runs")
    else:
        print("ok:", len(exps), "experiments,", len(decs), "decisions,", len(journal), "journal entries,", len(runs), "runs")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
