from pathlib import Path
from collections import Counter, defaultdict
import json
import re


DATA = Path("/workspace/training/datasets")

FILES = {
    "train": DATA / "sentinel_train_v42.jsonl",
    "validation": DATA / "sentinel_validation_v42.jsonl",
    "test": DATA / "sentinel_test_v42.jsonl",
}

ASSETS = [
    "human-subject interview transcripts",
    "student advising records",
    "genomic sequencing files",
    "unpublished research manuscripts",
    "sponsor-confidential prototype designs",
]


def extract_question(text):
    marker = "\n\nQUESTION\n========\n"

    q = text.split(marker, 1)[1].strip()

    prefixes = [
        "Decide the exact policy question below:\n",
        "Using only the supplied evidence, classify this question:\n",
    ]

    for p in prefixes:
        if q.startswith(p):
            q = q[len(p):].strip()

    return q


def normalize(q):
    q = q.lower()

    for asset in ASSETS:
        q = q.replace(
            asset.lower(),
            "<asset>",
        )

    q = re.sub(
        r"\s+",
        " ",
        q,
    )

    return q.strip()


def load(path):
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    # One row per underlying scenario.
    reps = {}

    for row in rows:
        reps.setdefault(
            row["scenario_id"],
            row,
        )

    return list(reps.values())


print("=" * 78)
print("SENTINEL V4.2 LABEL CAUSALITY AUDIT")
print("=" * 78)

all_questions = defaultdict(list)

for split, path in FILES.items():

    rows = load(path)

    print()
    print(split.upper())
    print("-" * 78)

    label_counts = Counter()

    for row in rows:

        label = json.loads(
            row["output"]
        )["decision"]

        label_counts[label] += 1

        q = normalize(
            extract_question(
                row["input"]
            )
        )

        all_questions[q].append(
            (
                split,
                label,
                row["family"],
                row["scenario_id"],
            )
        )

    print(
        "scenarios:",
        len(rows),
    )

    print(
        "labels:",
        dict(label_counts),
    )


multi_label = []

for q, items in all_questions.items():

    labels = {
        x[1]
        for x in items
    }

    if len(labels) > 1:
        multi_label.append(
            (
                q,
                labels,
                items,
            )
        )


print()
print("=" * 78)
print("COUNTERFACTUAL QUESTION COVERAGE")
print("=" * 78)

total_questions = len(all_questions)

print(
    "unique normalized questions:",
    total_questions,
)

print(
    "questions associated with >1 label:",
    len(multi_label),
)

if total_questions:
    print(
        "multi-label fraction:",
        f"{len(multi_label) / total_questions:.3f}",
    )


label_span = Counter()

for _, labels, _ in multi_label:
    label_span[len(labels)] += 1

print(
    "label-span distribution:",
    dict(sorted(label_span.items())),
)


print()
print("EXAMPLES")
print("-" * 78)

for q, labels, items in multi_label[:20]:

    print()
    print("QUESTION:")
    print(" ", q)

    print(
        "LABELS:",
        sorted(labels),
    )

    for (
        split,
        label,
        family,
        scenario_id,
    ) in items:
        print(
            f"  {split:10} "
            f"{label:12} "
            f"{family:32} "
            f"{scenario_id}"
        )


print()
print("=" * 78)

if not multi_label:
    print(
        "CAUSALITY FAILURE: every normalized question maps "
        "to exactly one label."
    )
else:
    print(
        "Some counterfactual question coverage exists."
    )

print("=" * 78)
