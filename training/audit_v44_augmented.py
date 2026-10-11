from collections import Counter
from pathlib import Path
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


BASE = Path("/workspace")

AUGMENTED = (
    BASE
    / "training/datasets/"
      "sentinel_train_v44_augmented.jsonl"
)

V43_TRAIN = (
    BASE
    / "training/datasets/"
      "sentinel_train_v43.jsonl"
)

GATE_TRAIN = (
    BASE
    / "training/datasets/"
      "sentinel_train_v44_gate_pilot.jsonl"
)

V43_DEV = (
    BASE
    / "training/datasets/"
      "sentinel_validation_v43.jsonl"
)

V44_GATE_VALIDATION = (
    BASE
    / "training/datasets/"
      "sentinel_validation_v44_gate_pilot.jsonl"
)


LABELS = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
}


def derive_fields(row):
    """
    V4.4 gate rows include convenience fields such as
    question/evidence. Older V4.3 rows may contain only the
    canonical instruction/input/output contract.

    Recover question and evidence from input without changing
    the underlying example.
    """

    item = dict(row)

    if (
        "question" not in item
        or "evidence" not in item
    ):

        raw = item["input"]

        marker = "\n\nQUESTION\n========\n"

        if marker not in raw:
            raise ValueError(
                "Unable to recover QUESTION section "
                "from dataset input."
            )

        evidence, question = raw.rsplit(
            marker,
            1,
        )

        item.setdefault(
            "question",
            question.strip(),
        )

        item.setdefault(
            "evidence",
            evidence.strip(),
        )

    return item


def load(path):

    return [
        derive_fields(
            json.loads(line)
        )
        for line in path.read_text().splitlines()
        if line.strip()
    ]

def norm(text):

    return re.sub(
        r"\s+",
        " ",
        text.lower().strip(),
    )


def decision(row):

    return json.loads(
        row["output"]
    )["decision"]


def io_signature(row):

    return (
        row["instruction"],
        row["input"],
        row["output"],
    )


def max_cosine(
    left,
    right,
):

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
    )

    matrix = vec.fit_transform(
        left + right
    )

    A = matrix[
        :len(left)
    ]

    B = matrix[
        len(left):
    ]

    scores = cosine_similarity(
        A,
        B,
    )

    flat = scores.argmax()

    i, j = divmod(
        flat,
        scores.shape[1],
    )

    return (
        float(
            scores[i, j]
        ),
        i,
        j,
    )


def lexical_probe(
    train_rows,
    eval_rows,
    field,
):

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
    )

    X_train = vec.fit_transform(
        row[field]
        for row in train_rows
    )

    y_train = [
        decision(row)
        for row in train_rows
    ]

    X_eval = vec.transform(
        row[field]
        for row in eval_rows
    )

    y_eval = [
        decision(row)
        for row in eval_rows
    ]

    clf = LogisticRegression(
        max_iter=2500,
        random_state=42,
    )

    clf.fit(
        X_train,
        y_train,
    )

    return clf.score(
        X_eval,
        y_eval,
    )


def main():

    print("=" * 88)
    print("SENTINEL V4.4 AUGMENTED — PRETRAIN AUDIT")
    print("=" * 88)

    augmented = load(
        AUGMENTED
    )

    v43 = load(
        V43_TRAIN
    )

    gate = load(
        GATE_TRAIN
    )

    v43_dev = load(
        V43_DEV
    )

    gate_val = load(
        V44_GATE_VALIDATION
    )

    errors = []

    # ---------------------------------------------------------
    # Dataset accounting
    # ---------------------------------------------------------

    print()
    print("DATASET ACCOUNTING")
    print("-" * 88)

    print(
        "augmented:",
        len(augmented),
    )

    print(
        "v43:",
        len(v43),
    )

    print(
        "gate:",
        len(gate),
    )

    if len(augmented) != 3264:
        errors.append(
            "augmented count != 3264"
        )

    labels = Counter(
        decision(row)
        for row in augmented
    )

    sources = Counter(
        row.get(
            "curriculum_source"
        )
        for row in augmented
    )

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

    for label in LABELS:

        if labels[label] != 816:

            errors.append(
                (
                    f"label {label} "
                    f"count={labels[label]}"
                )
            )

    if sources[
        "v43_relational"
    ] != 2880:

        errors.append(
            "wrong V4.3 source count"
        )

    if sources[
        "v44_gate_transfer"
    ] != 384:

        errors.append(
            "wrong gate source count"
        )

    # ---------------------------------------------------------
    # Exact union integrity
    # ---------------------------------------------------------

    expected = {
        io_signature(row)
        for row in (
            v43 + gate
        )
    }

    actual = {
        io_signature(row)
        for row in augmented
    }

    print()
    print("UNION INTEGRITY")
    print("-" * 88)

    print(
        "expected unique examples:",
        len(expected),
    )

    print(
        "actual unique examples:",
        len(actual),
    )

    print(
        "missing:",
        len(
            expected - actual
        ),
    )

    print(
        "unexpected:",
        len(
            actual - expected
        ),
    )

    if expected != actual:

        errors.append(
            "augmented dataset is not exact source union"
        )

    # ---------------------------------------------------------
    # Gate intervention ↔ V4.3 development overlap
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("GATE TRAIN ↔ V4.3 DEVELOPMENT")
    print("=" * 88)

    gate_questions = {
        norm(
            row["question"]
        )
        for row in gate
    }

    dev_questions = {
        norm(
            row["question"]
        )
        for row in v43_dev
    }

    gate_inputs = {
        norm(
            row["input"]
        )
        for row in gate
    }

    dev_inputs = {
        norm(
            row["input"]
        )
        for row in v43_dev
    }

    question_overlap = (
        gate_questions
        & dev_questions
    )

    input_overlap = (
        gate_inputs
        & dev_inputs
    )

    print(
        "exact question overlap:",
        len(question_overlap),
    )

    print(
        "exact input overlap:",
        len(input_overlap),
    )

    if question_overlap:

        errors.append(
            "gate train copies V4.3 dev question"
        )

    if input_overlap:

        errors.append(
            "gate train copies V4.3 dev input"
        )

    gate_texts = [
        norm(
            row["question"]
            + "\n"
            + row["evidence"]
        )
        for row in gate
    ]

    dev_texts = [
        norm(
            row["question"]
            + "\n"
            + row["evidence"]
        )
        for row in v43_dev
    ]

    score, i, j = max_cosine(
        gate_texts,
        dev_texts,
    )

    print(
        "max TF-IDF cosine:",
        f"{score:.4f}",
    )

    print(
        "gate scenario:",
        gate[i][
            "scenario_id"
        ],
    )

    print(
        "V4.3 dev scenario:",
        v43_dev[j][
            "scenario_id"
        ],
    )

    # Near duplicate only.
    # Semantic analogy itself is intentional.
    if score >= 0.92:

        errors.append(
            (
                "gate ↔ V4.3 dev near-duplicate "
                f"{score:.4f}"
            )
        )

    # ---------------------------------------------------------
    # Shortcut probes from the ACTUAL augmented training set
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("AUGMENTED TF-IDF SHORTCUT PROBES")
    print("=" * 88)

    for name, eval_rows in [
        (
            "V4.3 development",
            v43_dev,
        ),
        (
            "V4.4 gate validation",
            gate_val,
        ),
    ]:

        q_score = lexical_probe(
            augmented,
            eval_rows,
            "question",
        )

        e_score = lexical_probe(
            augmented,
            eval_rows,
            "evidence",
        )

        augmented_joint = [
            {
                **row,
                "joint":
                    row["question"]
                    + "\n[RELATION]\n"
                    + row["evidence"],
            }
            for row in augmented
        ]

        eval_joint = [
            {
                **row,
                "joint":
                    row["question"]
                    + "\n[RELATION]\n"
                    + row["evidence"],
            }
            for row in eval_rows
        ]

        joint_score = lexical_probe(
            augmented_joint,
            eval_joint,
            "joint",
        )

        print()
        print(name)

        print(
            " question only:",
            f"{q_score:.4f}",
        )

        print(
            " evidence only:",
            f"{e_score:.4f}",
        )

        print(
            " additive joint:",
            f"{joint_score:.4f}",
        )

        # We expect chance-level behavior.
        # Slight finite-sample deviation is allowed.
        if q_score > 0.30:

            errors.append(
                f"{name}: question shortcut"
            )

        if e_score > 0.30:

            errors.append(
                f"{name}: evidence shortcut"
            )

        if joint_score > 0.30:

            errors.append(
                f"{name}: additive lexical shortcut"
            )

    # ---------------------------------------------------------
    # Critical gate intervention invariant
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("TARGETED INTERVENTION INVARIANT")
    print("=" * 88)

    gate_no = [
        row
        for row in gate
        if decision(row) == "NO"
    ]

    print(
        "gate NO rows:",
        len(gate_no),
    )

    closed_satisfied = sum(
        (
            row.get(
                "gate_state"
            )
            == "closed"
        )
        and
        (
            row.get(
                "downstream_state"
            )
            == "satisfied"
        )
        for row in gate_no
    )

    print(
        "closed gate + satisfied downstream:",
        f"{closed_satisfied}/{len(gate_no)}",
    )

    if (
        len(gate_no) != 96
        or
        closed_satisfied != 96
    ):

        errors.append(
            "target intervention invariant broken"
        )

    print()
    print("=" * 88)

    if errors:

        print(
            "V4.4 AUGMENTED PRETRAIN AUDIT: FAIL"
        )

        for error in errors:

            print(
                " -",
                error,
            )

        raise SystemExit(1)

    print(
        "V4.4 AUGMENTED PRETRAIN AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
