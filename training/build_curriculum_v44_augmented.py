"""
Sentinel V4.4 augmented controlled-intervention curriculum.

TRAIN ONLY:

    V4.3 relational train             2880
    V4.4 frozen gate-transfer train    384
                                      ----
                                      3264

V4.3 validation remains unchanged and is NOT merged here.

The V4.4 gate pilot TEST split is never read.
Challenge V1 is never read.
"""

from collections import Counter
from pathlib import Path
import hashlib
import json
import random


BASE = Path("/workspace")

V43_TRAIN = (
    BASE
    / "training/datasets/"
      "sentinel_train_v43.jsonl"
)

V44_GATE_TRAIN = (
    BASE
    / "training/datasets/"
      "sentinel_train_v44_gate_pilot.jsonl"
)

OUT = (
    BASE
    / "training/datasets/"
      "sentinel_train_v44_augmented.jsonl"
)

SEED = 20260816


def load_jsonl(path):

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def signature(row):

    return hashlib.sha256(
        (
            row["instruction"]
            + "\n[INPUT]\n"
            + row["input"]
            + "\n[OUTPUT]\n"
            + row["output"]
        ).encode("utf-8")
    ).hexdigest()


def decision(row):

    return json.loads(
        row["output"]
    )["decision"]


def main():

    print("=" * 80)
    print("SENTINEL V4.4 AUGMENTED TRAIN BUILD")
    print("=" * 80)

    v43 = load_jsonl(
        V43_TRAIN
    )

    gate = load_jsonl(
        V44_GATE_TRAIN
    )

    print()
    print("source counts:")
    print(" V4.3:", len(v43))
    print(" gate:", len(gate))

    if len(v43) != 2880:
        raise AssertionError(
            f"unexpected V4.3 count: {len(v43)}"
        )

    if len(gate) != 384:
        raise AssertionError(
            f"unexpected gate count: {len(gate)}"
        )

    # ---------------------------------------------------------
    # Ensure intervention does not duplicate an existing
    # instruction/input/output training example.
    # ---------------------------------------------------------

    v43_sigs = {
        signature(row)
        for row in v43
    }

    gate_sigs = {
        signature(row)
        for row in gate
    }

    overlap = (
        v43_sigs
        & gate_sigs
    )

    print()
    print(
        "exact cross-source example overlap:",
        len(overlap),
    )

    if overlap:
        raise AssertionError(
            "V4.3 ↔ gate exact training overlap"
        )

    # ---------------------------------------------------------
    # Add provenance metadata only.
    # This does not alter instruction/input/output.
    # ---------------------------------------------------------

    combined = []

    for row in v43:

        item = dict(row)

        item[
            "curriculum_source"
        ] = "v43_relational"

        combined.append(
            item
        )

    for row in gate:

        item = dict(row)

        item[
            "curriculum_source"
        ] = "v44_gate_transfer"

        combined.append(
            item
        )

    # Deterministic physical order.
    # Trainer will also shuffle during training.
    rng = random.Random(
        SEED
    )

    rng.shuffle(
        combined
    )

    if len(combined) != 3264:

        raise AssertionError(
            len(combined)
        )

    labels = Counter(
        decision(row)
        for row in combined
    )

    sources = Counter(
        row["curriculum_source"]
        for row in combined
    )

    source_labels = Counter(
        (
            row["curriculum_source"],
            decision(row),
        )
        for row in combined
    )

    print()
    print("combined rows:", len(combined))
    print(
        "labels:",
        dict(
            sorted(
                labels.items()
            )
        ),
    )

    print(
        "sources:",
        dict(
            sorted(
                sources.items()
            )
        ),
    )

    print()
    print("source × label:")

    for key, n in sorted(
        source_labels.items()
    ):

        print(
            " ",
            key,
            n,
        )

    expected_labels = {
        "YES": 816,
        "NO": 816,
        "CONDITIONAL": 816,
        "INSUFFICIENT": 816,
    }

    if dict(labels) != expected_labels:

        raise AssertionError(
            (
                "label imbalance",
                dict(labels),
            )
        )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUT.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in combined:

            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
                + "\n"
            )

    print()
    print("written:")
    print(OUT)

    print()
    print(
        "SENTINEL V4.4 AUGMENTED BUILD: PASS"
    )


if __name__ == "__main__":
    main()
