"""Write a markdown sheet for hand-checking extractions: unclear top entries + adoptions."""
import json, re, sys
from pathlib import Path

run = Path(sys.argv[1])
rs = [json.loads(l) for l in open(run)]


def line(t):
    m = list(re.finditer(r"differential\W*:", t["reply"], re.I))
    return t["reply"][m[-1].start():].replace("\n", " ").replace("|", "/") if m else "<no line>"


def turns(r):
    yield "turn0", r["turn0"]
    for t in r["successive"]:
        yield f"succ-{t['level']}", t
    for t in r["independent"][1:]:
        yield f"indep-{t['level']}", t
    yield "sure", r["sure"]
    for t in r["toward_correct"]:
        yield f"correct-{t['level']}", t


rows = {"Unclear top entry": [], "Adopted distractor (initially correct)": []}
for r in rs:
    for name, t in turns(r):
        if t["diagnosis"] == "unclear":
            rows["Unclear top entry"].append((r, name, t))
        elif r["turn0"]["correct"] and t["diagnosis"] == r["distractor"] and not name.startswith(("correct", "turn0")):
            rows["Adopted distractor (initially correct)"].append((r, name, t))

out = Path(__file__).parent / f"review_{run.stem}.md"
with open(out, "w") as f:
    f.write(f"# Extraction review: `{run.name}`\n\nMark **OK** column: ✓ correct / ✗ wrong (+ what it should be).\n\n")
    for title, items in rows.items():
        f.write(f"## {title} ({len(items)})\n\n| # | True label | Distractor | Turn | Model's differential line | Extracted | OK |\n|---|---|---|---|---|---|---|\n")
        for i, (r, name, t) in enumerate(items, 1):
            f.write(f"| {i} | {r['label']} | {r['distractor']} | {name} | {line(t)} | {' → '.join(t['differential'])} | |\n")
        f.write("\n")
print(out, {k: len(v) for k, v in rows.items()})
