"""Adoption rates split by distractor severity vs. label (rule B: top = highest-ranked on-list entry)."""
import json, sys

UNCLEAR = "unclear"
top = lambda ddx: next((x for x in ddx if x != UNCLEAR), UNCLEAR)
cases = {c["id"]: c for c in map(json.loads, open("data/cases.jsonl"))}
group = lambda cid: "more severe" if cases[cid]["distractor_severity"] < cases[cid]["severity"] else "same/less"
LEVELS = ["question", "medium", "strong"]


def panel_top(t):  # panel answer recomputed under rule B isn't stored; use stored majority (already rule-A top-1)
    return t["diagnosis"]


for path in sys.argv[1:]:
    rs = [json.loads(l) for l in open(path)]
    panel = "round1" in rs[0]["turn0"]
    dx = (lambda t: t["diagnosis"]) if panel else (lambda t: top(t["differential"]))
    print(path.split("/")[-1].split("_")[0])
    for g in ("same/less", "more severe"):
        init = [r for r in rs if group(r["case_id"]) == g and dx(r["turn0"]) == r["label"]]
        row = []
        for mode in ("successive", "independent"):
            for i, lv in enumerate(LEVELS):
                if mode == "independent" and lv == "question":
                    continue
                row.append(f"{mode[:4]}-{lv[:3]} {sum(dx(r[mode][i]) == r['distractor'] for r in init) / max(len(init), 1):4.0%}")
        print(f"  {g:12} n={len(init):3} | " + " | ".join(row))
