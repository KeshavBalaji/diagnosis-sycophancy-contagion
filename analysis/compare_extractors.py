"""Compare extractions of the same replies across extractors; write disagreements to a markdown sheet."""
import json, re, sys
from pathlib import Path

paths = [Path(p) for p in sys.argv[1:]]
runs = {p.stem.split(".ext-")[-1]: [json.loads(l) for l in open(p)] for p in paths}
names = list(runs)


def turns(r):
    yield "turn0", r["turn0"]
    for t in r["successive"]: yield f"succ-{t['level']}", t
    for t in r["independent"][1:]: yield f"indep-{t['level']}", t
    yield "sure", r["sure"]
    for t in r["toward_correct"]: yield f"correct-{t['level']}", t


def line(t):
    m = list(re.finditer(r"differential\W*:", t["reply"], re.I))
    return t["reply"][m[-1].start():].replace("\n", " ").replace("|", "/") if m else "<no line>"


def clean(ddx):  # drop trailing padding "unclear"s so [A, unclear] == [A]
    ddx = list(ddx)
    while len(ddx) > 1 and ddx[-1] == "unclear":
        ddx.pop()
    return ddx


rows = []
for i, r0 in enumerate(runs[names[0]]):
    per = {n: list(turns(runs[n][i])) for n in names}
    for j, (tname, _) in enumerate(per[names[0]]):
        rows.append((r0, tname, per[names[0]][j][1], {n: per[n][j][1] for n in names}))

print(f"{len(rows)} replies")
for n in names:
    ts = [x[3][n] for x in rows]
    print(f"  {n:22} top unclear {sum(t['diagnosis'] == 'unclear' for t in ts):3} | retries {sum(t.get('attempts', 1) > 1 for t in ts) if 'attempts' in ts[0] else '-'} "
          f"| turn0 correct {sum(r['turn0']['correct'] for r in runs[n])}/{len(runs[n])}")
top = sum(len({x[3][n]['diagnosis'] for n in names}) == 1 for x in rows)
full = sum(len({tuple(clean(x[3][n]['differential'])) for n in names}) == 1 for x in rows)
print(f"all agree: top-1 {top}/{len(rows)}, full differential {full}/{len(rows)}")
for a in names:
    for b in names[names.index(a) + 1:]:
        print(f"  top-1 agree {a} vs {b}: {sum(x[3][a]['diagnosis'] == x[3][b]['diagnosis'] for x in rows)}/{len(rows)}")

dis = [x for x in rows if len({x[3][n]['diagnosis'] for n in names}) > 1]
out = Path(__file__).parent / "extractor_disagreements.md"
with open(out, "w") as f:
    f.write(f"# Extractor disagreements on top-1 ({len(dis)} of {len(rows)} replies)\n\n"
            "Mark which extractor is right (or none).\n\n| # | True label | Distractor | Turn | Differential line | "
            + " | ".join(names) + " | Right |\n|" + "---|" * (6 + len(names)) + "\n")
    for i, (r, tname, t, by) in enumerate(dis, 1):
        f.write(f"| {i} | {r['label']} | {r['distractor']} | {tname} | {line(t)[:220]} | "
                + " | ".join(" → ".join(clean(by[n]['differential'])) for n in names) + " | |\n")
print("wrote", out, len(dis))
