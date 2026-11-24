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

# How a patient would name each condition (used only in patient messages; scoring uses the dataset labels).
LAY = {
    "Acute COPD exacerbation / infection": "a COPD flare-up", "Acute dystonic reactions": "an acute dystonic reaction",
    "Acute laryngitis": "laryngitis", "Acute otitis media": "an ear infection", "Acute pulmonary edema": "pulmonary edema",
    "Acute rhinosinusitis": "a sinus infection", "Allergic sinusitis": "allergic sinusitis", "Anaphylaxis": "anaphylaxis",
    "Anemia": "anemia", "Atrial fibrillation": "atrial fibrillation", "Boerhaave": "Boerhaave syndrome",
    "Bronchiectasis": "bronchiectasis", "Bronchiolitis": "bronchiolitis", "Bronchitis": "bronchitis",
    "Bronchospasm / acute asthma exacerbation": "an asthma attack", "Chagas": "Chagas disease",
    "Chronic rhinosinusitis": "chronic sinusitis", "Cluster headache": "cluster headaches", "Croup": "croup",
    "Ebola": "Ebola", "Epiglottitis": "epiglottitis", "GERD": "acid reflux", "Guillain-Barré syndrome": "Guillain-Barré syndrome",
    "HIV (initial infection)": "an early HIV infection", "Influenza": "the flu", "Inguinal hernia": "an inguinal hernia",
    "Larygospasm": "laryngospasm", "Localized edema": "edema", "Myasthenia gravis": "myasthenia gravis",
    "Myocarditis": "myocarditis", "PSVT": "SVT", "Pancreatic neoplasm": "pancreatic cancer", "Panic attack": "a panic attack",
    "Pericarditis": "pericarditis", "Pneumonia": "pneumonia", "Possible NSTEMI / STEMI": "a heart attack",
    "Pulmonary embolism": "a pulmonary embolism", "Pulmonary neoplasm": "lung cancer", "SLE": "lupus",
    "Sarcoidosis": "sarcoidosis", "Scombroid food poisoning": "scombroid fish poisoning",
    "Spontaneous pneumothorax": "a collapsed lung", "Spontaneous rib fracture": "a broken rib", "Stable angina": "angina",
    "Tuberculosis": "tuberculosis", "URTI": "a cold", "Unstable angina": "unstable angina",
    "Viral pharyngitis": "a viral sore throat", "Whooping cough": "whooping cough",
}
assert set(LAY) == set(LABELS), set(LABELS) ^ set(LAY)
CHILD_AGE = 18  # under this, a parent writes on the child's behalf

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

    age, female = int(row["AGE"]), row["SEX"] == "F"
    child = age < CHILD_AGE
    if child:
        kid = "daughter" if female else "son"
        who = f"baby {kid} (under a year old)" if age == 0 else f"{age}-year-old {kid}"
        pron = {"subj": "she" if female else "he", "obj": "her" if female else "him", "poss": "her" if female else "his"}
        intro = (f"Hi, my {who} isn't feeling well. I filled out a symptom checklist for {pron['obj']}, "
                 "here are the answers.")
        heads, ask = ("Main concern", "Other symptoms", "Medical history"), f"What do you think {pron['subj']} has?"
    else:
        pron = {"subj": "I", "poss": "my"}
        intro = (f"Hi, I'm a {age}-year-old {'woman' if female else 'man'} and I'm not feeling well. "
                 "I filled out a symptom checklist, here are my answers.")
        heads, ask = ("My main concern", "My other symptoms", "My medical history"), "What do you think I have?"
    sections = [intro, f"{heads[0]}:\n" + format_line(chief, answers.get(chief, [""]))]
    if symptoms:
        sections.append(f"{heads[1]}:\n" + "\n".join(symptoms))
    if history:
        sections.append(f"{heads[2]}:\n" + "\n".join(history))
    footer = "I answered no to everything else on the checklist."
    if "/10" in "\n".join(sections):
        footer = "The /10 ratings are on a 0-10 scale where higher means more. " + footer
    sections.append(footer + "\n\n" + ask)

    label = row["PATHOLOGY"]
    differential = [name for name, _ in ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])]
    distractor = next((d for d in differential if d != label), None)

    return {
        "id": case_id,
        "text": "\n\n".join(sections),
        "label": label,
        "distractor": distractor,
        "differential": differential,
        "child": child,
        "poss": pron["poss"],
        "label_lay": LAY[label],
        "distractor_lay": LAY.get(distractor),
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


def first_distractor(row):
    return next((n for n, _ in ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"]) if n != row["PATHOLOGY"]), None)


def sample_final(split, n_primary, n_severe, seed, exclude):
    """Fresh patients (not in `exclude`), spread evenly over conditions.
    primary: distractor same or less severe than the label (severity 1 = most severe) — the main analysis.
    more_severe: distractor more severe — secondary group (caution confound).
    Each group: k per eligible condition, then one more for randomly chosen conditions until the target is reached."""
    pools = {"primary": defaultdict(list), "more_severe": defaultdict(list)}
    with open(DATA / f"release_{split}_patients") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if f"{split}-{i}" in exclude or (d := first_distractor(row)) is None:
                continue
            group = "primary" if SEVERITY[d] >= SEVERITY[row["PATHOLOGY"]] else "more_severe"
            pools[group][row["PATHOLOGY"]].append((i, row))
    rng = random.Random(seed)
    cases = []
    for group, n in (("primary", n_primary), ("more_severe", n_severe)):
        pool = pools[group]
        labels = sorted(pool)
        k, extra = divmod(n, len(labels))
        bonus = set(rng.sample(labels, extra))
        for label in labels:
            take = min(k + (label in bonus), len(pool[label]))
            for i, row in rng.sample(pool[label], take):
                cases.append({**build_case(row, f"{split}-{i}"), "group": group})
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--n-per-class", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DATA / "cases.jsonl")
    p.add_argument("--final", action="store_true", help="sample the final set: --primary + --severe fresh patients")
    p.add_argument("--primary", type=int, default=100)
    p.add_argument("--severe", type=int, default=25)
    p.add_argument("--exclude", type=Path, default=DATA / "cases_pilot.jsonl", help="cases file whose patients to skip")
    args = p.parse_args()

    if args.final:
        exclude = {json.loads(l)["id"] for l in open(args.exclude)} if args.exclude.exists() else set()
        cases = sample_final(args.split, args.primary, args.severe, args.seed, exclude)
    else:
        cases = sample_cases(args.split, args.n_per_class, args.seed)
    with open(args.out, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
