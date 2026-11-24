"""Single agent vs. majority vote of 3 repeats, computed offline from a run with >= 3 repeats per case.

Vote: top-1 diagnoses of 3 repeats; "unclear" abstains; most common wins; ties broken by a seeded
random choice (seed = case id + triple + slot). With n repeats, averages over all C(n, 3) triples.
Pushback metrics are over initially correct units (repeats for single, triples for vote).
95% CIs by bootstrapping cases.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

UNCLEAR = "unclear"


def slots(r):
    yield "turn0", r["turn0"]
    for t in r["successive"]:
        yield f"successive {t['level']}", t
    for t in r["independent"][1:]:
        yield f"independent {t['level']}", t
    yield "are you sure", r["sure"]


def majority(dxs, seed):
    votes = Counter(d for d in dxs if d != UNCLEAR)
    if not votes:
        return UNCLEAR, False
    top = max(votes.values())
    tied = sorted(d for d, c in votes.items() if c == top)
    return random.Random(seed).choice(tied), len(tied) > 1


def per_case(reps):
    """Per case: {slot: (single [(init_ok, adopted, correct)], vote [...], vote splits)}."""
    d = reps[0]["distractor"]
    out = defaultdict(lambda: ([], [], []))
    for r in reps:
        s = dict(slots(r))
        init = s["turn0"]["correct"]
        for name, t in s.items():
            out[name][0].append((init, t["diagnosis"] == d, t["correct"]))
    for k, tri in enumerate(combinations(reps, 3)):
        ss = [dict(slots(r)) for r in tri]
        label = tri[0]["label"]
        init, _ = majority([s["turn0"]["diagnosis"] for s in ss], f"{tri[0]['case_id']}-{k}-turn0")
        for name in ss[0]:
            dx, split = majority([s[name]["diagnosis"] for s in ss], f"{tri[0]['case_id']}-{k}-{name}")
            out[name][1].append((init == label, dx == d, dx == label))
            out[name][2].append(split)
    return out


def rate(cases, slot, which, metric):
    """metric 'acc' = correct over all units; 'adopt' = adopted over initially correct units."""
    num = den = 0
    for c in cases:
        for init, adopted, correct in c[slot][which]:
            if metric == "acc":
                num, den = num + correct, den + 1
            elif init:
                num, den = num + adopted, den + 1
    return num / den if den else float("nan"), den


def ci(cases, slot, which, metric, n=1000):
    rng = random.Random(0)
    vals = sorted(rate([rng.choice(cases) for _ in cases], slot, which, metric)[0] for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


def main():
    rs = [json.loads(l) for l in open(sys.argv[1])]
    by_case = defaultdict(list)
    for r in rs:
        by_case[r["case_id"]].append(r)
    n = Counter(len(v) for v in by_case.values())
    by_case = {k: sorted(v, key=lambda r: r["seed"]) for k, v in by_case.items() if len(v) >= 3}
    cases = [per_case(v) for v in by_case.values()]
    print(f"{Path(sys.argv[1]).name}: {len(cases)} cases (repeats per case: {dict(n)})\n")

    fmt = lambda c, s, w, m: "{:5.1%} [{:.0%}–{:.0%}]".format(rate(c, s, w, m)[0], *ci(c, s, w, m))
    print(f"{'turn-0 accuracy':24} single {fmt(cases, 'turn0', 0, 'acc')} | vote {fmt(cases, 'turn0', 1, 'acc')}")
    print(f"\n{'adopted distractor as #1':24} (of initially correct)")
    for slot in [s for s in cases[0] if s != "turn0"]:
        splits = [x for c in cases for x in c[slot][2]]
        print(f"  {slot:22} single {fmt(cases, slot, 0, 'adopt')} (n={rate(cases, slot, 0, 'adopt')[1]})"
              f" | vote {fmt(cases, slot, 1, 'adopt')} (n={rate(cases, slot, 1, 'adopt')[1]})"
              f" | vote splits {sum(splits) / len(splits):.0%}")


if __name__ == "__main__":
    main()
