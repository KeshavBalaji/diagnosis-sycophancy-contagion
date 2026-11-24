"""Random sample of 'adopted distractor' replies (rule B: top = highest-ranked on-list entry) for hand-checking."""
import json, random, re, sys

UNCLEAR = "unclear"
top = lambda ddx: next((x for x in ddx if x != UNCLEAR), UNCLEAR)


def line(reply):
    m = list(re.finditer(r"differential\W*:", reply, re.I))
    return reply[m[-1].start():].replace("\n", " ") if m else reply[-200:].replace("\n", " ")


def pushback_turns(r):
    for t in r["successive"]:
        yield f"succ-{t['level']}", t
    for t in r["independent"][1:]:
        yield f"indep-{t['level']}", t


items = []
for path, k in zip(sys.argv[1::2], map(int, sys.argv[2::2])):
    pool = []
    for r in map(json.loads, open(path)):
        panel = "round1" in r["turn0"]
        if panel:
            if r["turn0"]["diagnosis"] != r["label"]:
                continue
            for name, t in pushback_turns(r):
                for rnd in ("round1", "round2"):
                    for i, a in enumerate(t[rnd]):
                        if top(a["differential"]) == r["distractor"]:
                            pool.append((r, f"{name} {rnd} agent{i}", a))
        else:
            if top(r["turn0"]["differential"]) != r["label"]:
                continue
            for name, t in pushback_turns(r):
                if top(t["differential"]) == r["distractor"]:
                    pool.append((r, name, t))
    tag = path.split("/")[-1].split("_")[0]
    items += [(tag, *x) for x in random.Random(0).sample(pool, min(k, len(pool)))]
    print(f"{tag}: {len(pool)} adoptions, sampled {min(k, len(pool))}", file=sys.stderr)

for i, (tag, r, name, t) in enumerate(items, 1):
    print(f"{i}. [{tag} {name}] label={r['label']} | distractor={r['distractor']}\n   {line(t['reply'])[:260]}\n   -> {t['differential']}")
