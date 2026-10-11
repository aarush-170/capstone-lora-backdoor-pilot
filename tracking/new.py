"""Scaffold a new tracking entry from the templates with the next id / today's date.

    python tracking/new.py experiment "G1 evasion feasibility (text)" --owner samvith --rq RQ3
    python tracking/new.py decision   "Use college GPUs"
    python tracking/new.py journal    --author aarush --topic g1-tools [--exp EXP-001]

Prints the path of the file it created. Fill it in, then run `python tracking/build_index.py`.
"""
import argparse
import datetime
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "entry"


def next_id(folder, prefix):
    nums = [int(m.group(1)) for p in (ROOT / folder).glob(f"{prefix}-*.md")
            if (m := re.match(rf"{prefix}-(\d+)", p.name))]
    return f"{prefix}-{(max(nums) + 1 if nums else 0 if prefix == 'EXP' else 1):03d}"


def render(template, **kw):
    text = (ROOT / "templates" / template).read_text(encoding="utf-8")
    for k, v in kw.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["experiment", "decision", "journal"])
    ap.add_argument("title", nargs="?", default="")
    ap.add_argument("--owner", default="")
    ap.add_argument("--author", default="")
    ap.add_argument("--topic", default="")
    ap.add_argument("--rq", default="")
    ap.add_argument("--exp", default="")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    a = ap.parse_args()
    if a.kind == "experiment":
        i = next_id("experiments", "EXP")
        path = ROOT / "experiments" / f"{i}-{slug(a.title)}.md"
        text = render("experiment.md", id=i, title=a.title, owner=a.owner or "TODO", rq=a.rq, date=a.date)
    elif a.kind == "decision":
        i = next_id("decisions", "DEC")
        path = ROOT / "decisions" / f"{i}-{slug(a.title)}.md"
        text = render("decision.md", id=i, title=a.title, date=a.date)
    else:
        if not a.author or not (a.topic or a.title):
            ap.error("journal needs --author and --topic")
        topic = a.topic or a.title
        path = ROOT / "journal" / f"{a.date}_{slug(a.author)}_{slug(topic)}.md"
        text = render("journal.md", date=a.date, author=slug(a.author), topic=topic, exp=a.exp)
    if path.exists():
        raise SystemExit(f"already exists: {path}")
    path.write_text(text, encoding="utf-8", newline="\n")
    print(path)


if __name__ == "__main__":
    main()
