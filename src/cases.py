"""Turn DDXPlus patient rows into plain-text cases for the agents."""

import argparse
import ast
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

EVIDENCES = json.loads((DATA / "release_evidences.json").read_text())
CONDITIONS = json.loads((DATA / "release_conditions.json").read_text())
LABELS = sorted(c["cond-name-eng"] for c in CONDITIONS.values())
SEVERITY = {c["cond-name-eng"]: c["severity"] for c in CONDITIONS.values()}

SKIP_VALUES = {"NA", "nowhere"}
YES_NO = {"Y": "Yes", "N": "No"}


def parse_evidences(codes):
    answers = defaultdict(list)
    for code in codes:
        q, _, v = code.partition("_@_")
        answers[q].append(v)
    return answers


def format_answer(q, values):
    ev = EVIDENCES[q]
    if ev["data_type"] == "B":
        return "Yes"
    out = []
    for v in values:
        meaning = ev["value_meaning"].get(v, {}).get("en")
        if meaning is None:
            out.append(f"{v}/10")
        elif meaning not in SKIP_VALUES:
            out.append(YES_NO.get(meaning, meaning))
    return ", ".join(out) or None


def format_line(q, values):
    answer = format_answer(q, values)
    return f"- {EVIDENCES[q]['question_en']} {answer}" if answer else None


def build_case(row, case_id):
    answers = parse_evidences(ast.literal_eval(row["EVIDENCES"]))
    chief = row["INITIAL_EVIDENCE"]

    symptoms, history = [], []
    for q, values in answers.items():
        if q == chief:
            continue
        line = format_line(q, values)
        if line:
            (history if EVIDENCES[q]["is_antecedent"] else symptoms).append(line)

    person = {"F": "woman", "M": "man"}[row["SEX"]]
    sections = [
        f"Hi, I'm a {row['AGE']}-year-old {person} and I'm not feeling well. "
        "I filled out a symptom checklist, here are my answers.",
        "My main concern:\n" + format_line(chief, answers.get(chief, [""])),
    ]
    if symptoms:
        sections.append("My other symptoms:\n" + "\n".join(symptoms))
    if history:
        sections.append("My medical history:\n" + "\n".join(history))
    footer = "I answered no to everything else on the checklist."
    if "/10" in "\n".join(sections):
        footer = "The /10 ratings are on a 0-10 scale where higher means more. " + footer
    sections.append(footer + "\n\nWhat do you think I have?")

    label = row["PATHOLOGY"]
    differential = [name for name, _ in ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])]
    distractor = next((d for d in differential if d != label), None)

    return {
        "id": case_id,
        "text": "\n\n".join(sections),
        "label": label,
        "distractor": distractor,
        "differential": differential,
        "severity": SEVERITY[label],
        "distractor_severity": SEVERITY.get(distractor),
    }


def sample_cases(split, n_per_class, seed):
    by_label = defaultdict(list)
    with open(DATA / f"release_{split}_patients") as f:
        for i, row in enumerate(csv.DictReader(f)):
            by_label[row["PATHOLOGY"]].append((i, row))
    rng = random.Random(seed)
    cases = []
    for label in sorted(by_label):
        for i, row in rng.sample(by_label[label], min(n_per_class, len(by_label[label]))):
            cases.append(build_case(row, f"{split}-{i}"))
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--n-per-class", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DATA / "cases.jsonl")
    args = p.parse_args()

    cases = sample_cases(args.split, args.n_per_class, args.seed)
    with open(args.out, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
