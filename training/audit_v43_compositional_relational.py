from collections import Counter, defaultdict
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)

from training.v43_compositional_relational_pilot import (
    ROWS,
    LABELS,
)


EXPECTED_GLOBAL = Counter({
    "YES": 48,
    "NO": 48,
    "CONDITIONAL": 48,
    "INSUFFICIENT": 48,
})

EXPECTED_FAMILY = Counter({
    "YES": 12,
    "NO": 12,
    "CONDITIONAL": 12,
    "INSUFFICIENT": 12,
})

EXPECTED_WORLD = Counter({
    "YES": 3,
    "NO": 3,
    "CONDITIONAL": 3,
    "INSUFFICIENT": 3,
})

EXPECTED_QUESTION = Counter({
    "YES": 1,
    "NO": 1,
    "CONDITIONAL": 1,
    "INSUFFICIENT": 1,
})

EXPECTED_EVIDENCE = Counter({
    "YES": 3,
    "NO": 3,
    "CONDITIONAL": 3,
    "INSUFFICIENT": 3,
})


def label_counts(rows):
    return Counter(
        row["decision"]
        for row in rows
    )


def canonical_evidence(evidence):
    """
    Evidence ordering must not create a fake distinction.
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


def evidence_text(evidence):
    """
    Preserve normal passage text for the lexical attack,
    but do not include question content.
    """
    parts = []

    for item in evidence:
        parts.append(
            " ".join([
                item["source"],
                item["section"],
                item["text"],
            ])
        )

    return "\n".join(parts)


def exact_identity_baseline(groups):
    """
    Best possible accuracy when the attacker knows only
    the exact identity of one side of the relation.
    """
    correct = 0
    total = 0

    for rows in groups.values():
        counts = label_counts(rows)
        correct += max(counts.values())
        total += len(rows)

    return correct / total


print("=" * 78)
print("SENTINEL V4.3 COMPOSITIONAL-RELATIONAL AUDIT")
print("=" * 78)

failed = False


# =====================================================================
# STRUCTURAL / CAUSAL AUDIT
# =====================================================================

for split, rows in ROWS.items():

    print()
    print("=" * 78)
    print(split.upper())
    print("=" * 78)

    print("rows:", len(rows))

    families = {
        row["family"]
        for row in rows
    }

    print(
        "families:",
        len(families),
    )

    print(
        "labels:",
        dict(label_counts(rows)),
    )

    if len(rows) != 192:
        print("FAIL: expected 192 rows")
        failed = True

    if len(families) != 4:
        print("FAIL: expected 4 families")
        failed = True

    if label_counts(rows) != EXPECTED_GLOBAL:
        print("FAIL: global labels not balanced")
        failed = True


    # -------------------------------------------------------------
    # Family-level balance
    # -------------------------------------------------------------

    by_family = defaultdict(list)

    for row in rows:
        by_family[row["family"]].append(row)

    bad_families = []

    for family, group in by_family.items():

        if len(group) != 48:
            bad_families.append(
                (
                    family,
                    "rows",
                    len(group),
                )
            )

        if label_counts(group) != EXPECTED_FAMILY:
            bad_families.append(
                (
                    family,
                    "labels",
                    dict(label_counts(group)),
                )
            )

    print(
        "bad family balance:",
        len(bad_families),
    )

    if bad_families:
        failed = True


    # -------------------------------------------------------------
    # World-level balance
    # -------------------------------------------------------------

    by_world = defaultdict(list)

    for row in rows:
        by_world[
            (
                row["family"],
                row["world"],
            )
        ].append(row)

    bad_worlds = []

    for key, group in by_world.items():

        if len(group) != 12:
            bad_worlds.append(
                (
                    key,
                    "rows",
                    len(group),
                )
            )

        if label_counts(group) != EXPECTED_WORLD:
            bad_worlds.append(
                (
                    key,
                    "labels",
                    dict(label_counts(group)),
                )
            )

    print(
        "evidence worlds:",
        len(by_world),
    )

    print(
        "bad world balance:",
        len(bad_worlds),
    )

    if bad_worlds:
        failed = True


    # -------------------------------------------------------------
    # QUESTION CAUSALITY
    #
    # Each exact paraphrased question appears in four worlds and
    # must take all four labels exactly once.
    # -------------------------------------------------------------

    by_question = defaultdict(list)

    for row in rows:
        by_question[
            row["question"]
        ].append(row)

    bad_questions = []

    for question, group in by_question.items():

        if len(group) != 4:
            bad_questions.append(
                (
                    question,
                    "rows",
                    len(group),
                )
            )

        if label_counts(group) != EXPECTED_QUESTION:
            bad_questions.append(
                (
                    question,
                    "labels",
                    dict(label_counts(group)),
                )
            )

    q_identity = exact_identity_baseline(
        by_question
    )

    print()
    print("QUESTION CAUSALITY")
    print("-" * 78)

    print(
        "unique exact questions:",
        len(by_question),
    )

    print(
        "bad question groups:",
        len(bad_questions),
    )

    print(
        "best exact-question-only accuracy:",
        f"{q_identity:.3f}",
    )

    if len(by_question) != 48:
        print(
            "FAIL: expected 48 exact questions"
        )
        failed = True

    if bad_questions:
        failed = True

    if abs(q_identity - 0.25) > 1e-9:
        print(
            "FAIL: question identity carries label signal"
        )
        failed = True


    # -------------------------------------------------------------
    # EVIDENCE CAUSALITY
    #
    # Each evidence world is shared by four targets × three
    # paraphrases. Each label therefore appears exactly 3 times.
    # -------------------------------------------------------------

    by_evidence = defaultdict(list)

    for row in rows:
        by_evidence[
            canonical_evidence(
                row["evidence"]
            )
        ].append(row)

    bad_evidence = []

    for key, group in by_evidence.items():

        if len(group) != 12:
            bad_evidence.append(
                (
                    key[:60],
                    "rows",
                    len(group),
                )
            )

        if label_counts(group) != EXPECTED_EVIDENCE:
            bad_evidence.append(
                (
                    key[:60],
                    "labels",
                    dict(label_counts(group)),
                )
            )

    e_identity = exact_identity_baseline(
        by_evidence
    )

    print()
    print("EVIDENCE CAUSALITY")
    print("-" * 78)

    print(
        "unique evidence packs:",
        len(by_evidence),
    )

    print(
        "bad evidence groups:",
        len(bad_evidence),
    )

    print(
        "best exact-evidence-only accuracy:",
        f"{e_identity:.3f}",
    )

    if len(by_evidence) != 16:
        print(
            "FAIL: expected 16 evidence packs"
        )
        failed = True

    if bad_evidence:
        failed = True

    if abs(e_identity - 0.25) > 1e-9:
        print(
            "FAIL: evidence identity carries label signal"
        )
        failed = True


    # -------------------------------------------------------------
    # RELATIONAL PAIR DETERMINISM
    # -------------------------------------------------------------

    pairs = defaultdict(list)

    for row in rows:

        key = (
            row["question"],
            canonical_evidence(
                row["evidence"]
            ),
        )

        pairs[key].append(
            row["decision"]
        )

    ambiguous = {
        key: labels
        for key, labels in pairs.items()
        if len(set(labels)) != 1
    }

    duplicates = {
        key: labels
        for key, labels in pairs.items()
        if len(labels) != 1
    }

    print()
    print("RELATIONAL PAIRS")
    print("-" * 78)

    print(
        "unique pairs:",
        len(pairs),
    )

    print(
        "ambiguous pairs:",
        len(ambiguous),
    )

    print(
        "duplicate pairs:",
        len(duplicates),
    )

    if len(pairs) != 192:
        print(
            "FAIL: expected 192 unique pairs"
        )
        failed = True

    if ambiguous or duplicates:
        failed = True


    # -------------------------------------------------------------
    # SEMANTIC METADATA
    # -------------------------------------------------------------

    semantic_errors = []

    for row in rows:

        evidence_ids = {
            item["id"]
            for item in row["evidence"]
        }

        invalid = (
            set(row["citations"])
            - evidence_ids
        )

        if invalid:
            semantic_errors.append(
                (
                    row["family"],
                    row["world"],
                    row["target"],
                    "invalid citations",
                    sorted(invalid),
                )
            )

        if (
            row["decision"] == "INSUFFICIENT"
            and row["citations"]
        ):
            semantic_errors.append(
                (
                    row["family"],
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
                    row["family"],
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
                    row["family"],
                    row["world"],
                    row["target"],
                    "non-CONDITIONAL has conditions",
                )
            )

    print()
    print(
        "semantic metadata errors:",
        len(semantic_errors),
    )

    if semantic_errors:
        failed = True


# =====================================================================
# LEXICAL ADVERSARIES
# =====================================================================

print()
print("=" * 78)
print("LEXICAL ADVERSARIES")
print("=" * 78)


def run_probe(
    name,
    text_fn,
):

    train_rows = ROWS["train"]

    X_train = [
        text_fn(row)
        for row in train_rows
    ]

    y_train = [
        row["decision"]
        for row in train_rows
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    Xtr = vectorizer.fit_transform(
        X_train
    )

    model = LogisticRegression(
        max_iter=3000,
        C=1.0,
        random_state=20260814,
    )

    model.fit(
        Xtr,
        y_train,
    )

    print()
    print(name)
    print("-" * 78)

    results = {}

    for split in [
        "validation",
        "test",
    ]:

        rows = ROWS[split]

        X = vectorizer.transform(
            [
                text_fn(row)
                for row in rows
            ]
        )

        y = [
            row["decision"]
            for row in rows
        ]

        pred = model.predict(X)

        acc = accuracy_score(
            y,
            pred,
        )

        results[split] = acc

        print()
        print(
            split.upper(),
            "accuracy:",
            f"{acc:.3f}",
        )

        print(
            "confusion matrix "
            "(rows=gold, cols=pred):"
        )

        print(LABELS)

        print(
            confusion_matrix(
                y,
                pred,
                labels=LABELS,
            )
        )

    return results


question_results = run_probe(
    "QUESTION-ONLY TF-IDF",
    lambda row: row["question"],
)

evidence_results = run_probe(
    "EVIDENCE-ONLY TF-IDF",
    lambda row: evidence_text(
        row["evidence"]
    ),
)


print()
print("=" * 78)
print("ACCEPTANCE GATES")
print("=" * 78)

print(
    "question-only validation:",
    f"{question_results['validation']:.3f}",
)

print(
    "question-only test:",
    f"{question_results['test']:.3f}",
)

print(
    "evidence-only validation:",
    f"{evidence_results['validation']:.3f}",
)

print(
    "evidence-only test:",
    f"{evidence_results['test']:.3f}",
)


for name, results in [
    (
        "question-only",
        question_results,
    ),
    (
        "evidence-only",
        evidence_results,
    ),
]:

    for split, acc in results.items():

        if acc > 0.35:
            print(
                f"FAIL: {name} {split} "
                f"accuracy {acc:.3f} > 0.350"
            )

            failed = True


print()
print("=" * 78)

if failed:
    print(
        "V4.3 COMPOSITIONAL-RELATIONAL PILOT: FAILED"
    )

    print("=" * 78)

    raise SystemExit(1)


print(
    "V4.3 COMPOSITIONAL-RELATIONAL PILOT: PASSED"
)

print()
print(
    "Question identity alone: chance-level"
)

print(
    "Evidence identity alone: chance-level"
)

print(
    "Question-only lexical transfer: within gate"
)

print(
    "Evidence-only lexical transfer: within gate"
)

print(
    "Question + evidence relation: deterministic"
)

print("=" * 78)
