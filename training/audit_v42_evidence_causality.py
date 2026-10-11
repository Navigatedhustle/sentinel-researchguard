from pathlib import Path
from collections import defaultdict, Counter
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


def normalize_text(text):
    text = text.lower()

    for asset in ASSETS:
        text = text.replace(
            asset.lower(),
            "<asset>",
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def canonical_evidence(input_text):
    marker = "\n\nQUESTION\n========\n"

    evidence = input_text.split(
        marker,
        1,
    )[0]

    # Pull passage blocks.
    blocks = re.split(
        r"\n(?=\[[^\]\n]+\]\n)",
        evidence,
    )

    cleaned = []

    for block in blocks:
        if "Text:" not in block:
            continue

        # Remove evidence IDs because ordering/IDs are not semantics.
        block = re.sub(
            r"^\[[^\]\n]+\]\n",
            "",
            block.strip(),
        )

        cleaned.append(
            normalize_text(block)
        )

    # Canonicalize shuffled evidence order.
    return tuple(sorted(cleaned))


def load(path):
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    # One representative per scenario.
    reps = {}

    for row in rows:
        reps.setdefault(
            row["scenario_id"],
            row,
        )

    return list(reps.values())


print("=" * 78)
print("SENTINEL V4.2 EVIDENCE CAUSALITY AUDIT")
print("=" * 78)

groups = defaultdict(list)

for split, path in FILES.items():

    rows = load(path)

    print()
    print(split.upper())
    print("-" * 78)

    labels = Counter()

    for row in rows:

        decision = json.loads(
            row["output"]
        )["decision"]

        labels[decision] += 1

        evidence = canonical_evidence(
            row["input"]
        )

        groups[evidence].append(
            (
                split,
                decision,
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
        dict(labels),
    )


multi_label = []

for evidence, items in groups.items():

    labels = {
        x[1]
        for x in items
    }

    if len(labels) > 1:
        multi_label.append(
            (
                evidence,
                labels,
                items,
            )
        )


print()
print("=" * 78)
print("EVIDENCE COUNTERFACTUAL COVERAGE")
print("=" * 78)

print(
    "unique normalized evidence packs:",
    len(groups),
)

print(
    "evidence packs associated with >1 label:",
    len(multi_label),
)

if groups:
    print(
        "multi-label fraction:",
        f"{len(multi_label) / len(groups):.3f}",
    )


span = Counter(
    len(labels)
    for _, labels, _ in multi_label
)

print(
    "label-span distribution:",
    dict(sorted(span.items())),
)


print()
print("=" * 78)

if not multi_label:
    print(
        "EVIDENCE CAUSALITY FAILURE: every evidence pack "
        "maps to exactly one label."
    )
else:
    print(
        "Some evidence-side counterfactual coverage exists."
    )

print("=" * 78)
