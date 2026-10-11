from collections import Counter, defaultdict
import json

from training.v43_relational_pilot import (
    FAMILIES,
    LABELS,
)


EXPECTED = Counter({
    "YES": 1,
    "NO": 1,
    "CONDITIONAL": 1,
    "INSUFFICIENT": 1,
})


def canonical_evidence(evidence):
    """
    Ignore passage order and Python object identity.
    """
    ordered = sorted(
        evidence,
        key=lambda x: (
            x["id"],
            x["source"],
            x["section"],
            x["text"],
        ),
    )

    return json.dumps(
        ordered,
        sort_keys=True,
        ensure_ascii=False,
    )


def label_counts(rows):
    return Counter(
        row["decision"]
        for row in rows
    )


print("=" * 78)
print("SENTINEL V4.3 RELATIONAL PILOT AUDIT")
print("=" * 78)

failed = False


for split, rows in FAMILIES.items():

    print()
    print("=" * 78)
    print(split.upper())
    print("=" * 78)

    print("rows:", len(rows))

    overall = label_counts(rows)

    print(
        "overall labels:",
        dict(overall),
    )

    if len(rows) != 16:
        print("FAIL: expected 16 rows")
        failed = True

    if overall != Counter({
        "YES": 4,
        "NO": 4,
        "CONDITIONAL": 4,
        "INSUFFICIENT": 4,
    }):
        print(
            "FAIL: global label balance is wrong"
        )
        failed = True


    # ---------------------------------------------------------
    # QUESTION SIDE
    #
    # Hold the question constant.
    # It must span all four labels.
    # ---------------------------------------------------------

    by_question = defaultdict(list)

    for row in rows:
        by_question[
            row["question"]
        ].append(row)

    print()
    print("QUESTION CAUSALITY")
    print("-" * 78)

    print(
        "unique questions:",
        len(by_question),
    )

    question_correct_if_identity_only = 0

    for question, group in sorted(
        by_question.items()
    ):

        counts = label_counts(group)

        print()
        print("QUESTION:")
        print(" ", question)

        print(
            " labels:",
            dict(counts),
        )

        if counts != EXPECTED:
            print(
                " FAIL: question does not span "
                "all four labels exactly once"
            )
            failed = True

        question_correct_if_identity_only += max(
            counts.values()
        )

    q_baseline = (
        question_correct_if_identity_only
        / len(rows)
    )

    print()
    print(
        "best exact-question-only accuracy:",
        f"{q_baseline:.3f}",
    )

    if q_baseline != 0.25:
        print(
            "FAIL: exact question identity carries label signal"
        )
        failed = True


    # ---------------------------------------------------------
    # EVIDENCE SIDE
    #
    # Hold the entire evidence world constant.
    # It must span all four labels.
    # ---------------------------------------------------------

    by_evidence = defaultdict(list)

    for row in rows:
        key = canonical_evidence(
            row["evidence"]
        )

        by_evidence[key].append(row)

    print()
    print("EVIDENCE CAUSALITY")
    print("-" * 78)

    print(
        "unique evidence worlds:",
        len(by_evidence),
    )

    evidence_correct_if_identity_only = 0

    for idx, (_, group) in enumerate(
        sorted(by_evidence.items()),
        1,
    ):

        counts = label_counts(group)

        targets = [
            row["target"]
            for row in group
        ]

        print()
        print(
            f"WORLD {idx}:"
        )

        print(
            " targets:",
            targets,
        )

        print(
            " labels:",
            dict(counts),
        )

        if counts != EXPECTED:
            print(
                " FAIL: evidence world does not span "
                "all four labels exactly once"
            )
            failed = True

        evidence_correct_if_identity_only += max(
            counts.values()
        )

    e_baseline = (
        evidence_correct_if_identity_only
        / len(rows)
    )

    print()
    print(
        "best exact-evidence-only accuracy:",
        f"{e_baseline:.3f}",
    )

    if e_baseline != 0.25:
        print(
            "FAIL: exact evidence identity carries label signal"
        )
        failed = True


    # ---------------------------------------------------------
    # RELATIONAL PAIRS
    #
    # Question + evidence together must uniquely determine
    # the gold label.
    # ---------------------------------------------------------

    pairs = defaultdict(list)

    for row in rows:

        pair = (
            row["question"],
            canonical_evidence(
                row["evidence"]
            ),
        )

        pairs[pair].append(
            row["decision"]
        )

    print()
    print("RELATIONAL PAIRS")
    print("-" * 78)

    print(
        "unique question/evidence pairs:",
        len(pairs),
    )

    ambiguous_pairs = {
        pair: labels
        for pair, labels in pairs.items()
        if len(set(labels)) != 1
    }

    duplicate_pairs = {
        pair: labels
        for pair, labels in pairs.items()
        if len(labels) != 1
    }

    print(
        "ambiguous pairs:",
        len(ambiguous_pairs),
    )

    print(
        "duplicate pairs:",
        len(duplicate_pairs),
    )

    if len(pairs) != 16:
        print(
            "FAIL: expected 16 unique relational pairs"
        )
        failed = True

    if ambiguous_pairs:
        print(
            "FAIL: a question/evidence pair maps "
            "to multiple labels"
        )
        failed = True

    if duplicate_pairs:
        print(
            "FAIL: duplicate relational pair exists"
        )
        failed = True


    # ---------------------------------------------------------
    # OUTPUT SEMANTICS
    # ---------------------------------------------------------

    semantic_errors = []

    for row in rows:

        evidence_ids = {
            item["id"]
            for item in row["evidence"]
        }

        invalid_citations = (
            set(row["citations"])
            - evidence_ids
        )

        if invalid_citations:
            semantic_errors.append(
                (
                    row["world"],
                    row["target"],
                    "invalid citations",
                    sorted(invalid_citations),
                )
            )

        if (
            row["decision"] == "INSUFFICIENT"
            and row["citations"]
        ):
            semantic_errors.append(
                (
                    row["world"],
                    row["target"],
                    "INSUFFICIENT has citations",
                )
            )

        if (
            row["decision"] == "CONDITIONAL"
            and not row["conditions"]
        ):
            semantic_errors.append(
                (
                    row["world"],
                    row["target"],
                    "CONDITIONAL lacks conditions",
                )
            )

        if (
            row["decision"] != "CONDITIONAL"
            and row["conditions"]
        ):
            semantic_errors.append(
                (
                    row["world"],
                    row["target"],
                    "non-CONDITIONAL has conditions",
                )
            )

    print()
    print(
        "semantic errors:",
        len(semantic_errors),
    )

    if semantic_errors:

        for error in semantic_errors:
            print(
                " ",
                error,
            )

        failed = True


print()
print("=" * 78)

if failed:
    print(
        "V4.3 RELATIONAL PILOT AUDIT: FAILED"
    )

    print("=" * 78)

    raise SystemExit(1)

print(
    "V4.3 RELATIONAL PILOT AUDIT: PASSED"
)

print()
print(
    "Question alone: 25% maximum by exact identity"
)

print(
    "Evidence alone: 25% maximum by exact identity"
)

print(
    "Question + evidence: unique deterministic relation"
)

print("=" * 78)
