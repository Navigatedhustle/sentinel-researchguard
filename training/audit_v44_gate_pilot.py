from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import hashlib
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


BASE = Path("/workspace")

FILES = {
    "train":
        BASE
        / "training/datasets/"
          "sentinel_train_v44_gate_pilot.jsonl",

    "validation":
        BASE
        / "training/datasets/"
          "sentinel_validation_v44_gate_pilot.jsonl",

    "test":
        BASE
        / "training/datasets/"
          "sentinel_test_v44_gate_pilot.jsonl",
}

LABELS = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
}


def load(path):

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def probe(
    train_rows,
    eval_rows,
    field,
):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
    )

    X_train = vectorizer.fit_transform(
        row[field]
        for row in train_rows
    )

    y_train = [
        json.loads(
            row["output"]
        )["decision"]
        for row in train_rows
    ]

    X_eval = vectorizer.transform(
        row[field]
        for row in eval_rows
    )

    y_eval = [
        json.loads(
            row["output"]
        )["decision"]
        for row in eval_rows
    ]

    clf = LogisticRegression(
        max_iter=2000,
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
    print("SENTINEL V4.4 GATE PILOT — ADVERSARIAL AUDIT")
    print("=" * 88)

    data = {
        split: load(path)
        for split, path in FILES.items()
    }

    errors = []

    expected_rows = {
        "train": 384,
        "validation": 192,
        "test": 192,
    }

    expected_labels = {
        "train": 96,
        "validation": 48,
        "test": 48,
    }

    # ---------------------------------------------------------
    # Basic structure
    # ---------------------------------------------------------

    for split, rows in data.items():

        labels = Counter(
            json.loads(
                row["output"]
            )["decision"]
            for row in rows
        )

        print()
        print(split.upper())
        print("-" * 88)
        print("rows:", len(rows))
        print("labels:", dict(sorted(labels.items())))

        if len(rows) != expected_rows[split]:
            errors.append(
                f"{split}: wrong row count"
            )

        for label in LABELS:

            if labels[label] != expected_labels[split]:

                errors.append(
                    (
                        f"{split}: label {label} "
                        f"count={labels[label]}"
                    )
                )

    # ---------------------------------------------------------
    # Question-only exact identity
    #
    # Every exact question must map to ALL FOUR labels
    # across evidence worlds.
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("EXACT QUESTION COUNTERFACTUAL")
    print("=" * 88)

    for split, rows in data.items():

        groups = defaultdict(set)

        for row in rows:

            label = json.loads(
                row["output"]
            )["decision"]

            groups[
                row["question"]
            ].add(
                label
            )

        bad = [
            question
            for question, labels in groups.items()
            if labels != LABELS
        ]

        print(
            split,
            "questions:",
            len(groups),
            "bad:",
            len(bad),
        )

        if bad:
            errors.append(
                f"{split}: question identity predicts label"
            )

    # ---------------------------------------------------------
    # Evidence-only exact identity
    #
    # Every evidence world must support all four labels,
    # depending on which target is asked about.
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("EXACT EVIDENCE COUNTERFACTUAL")
    print("=" * 88)

    for split, rows in data.items():

        groups = defaultdict(set)

        for row in rows:

            label = json.loads(
                row["output"]
            )["decision"]

            groups[
                row["evidence_hash"]
            ].add(
                label
            )

        bad = [
            evidence_hash
            for evidence_hash, labels in groups.items()
            if labels != LABELS
        ]

        print(
            split,
            "evidence packs:",
            len(groups),
            "bad:",
            len(bad),
        )

        if bad:
            errors.append(
                f"{split}: evidence identity predicts label"
            )

    # ---------------------------------------------------------
    # Pair ambiguity
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("QUESTION + EVIDENCE PAIR UNIQUENESS")
    print("=" * 88)

    for split, rows in data.items():

        pairs = defaultdict(set)

        for row in rows:

            label = json.loads(
                row["output"]
            )["decision"]

            key = (
                row["question"],
                row["evidence_hash"],
            )

            pairs[key].add(
                label
            )

        ambiguous = [
            key
            for key, labels in pairs.items()
            if len(labels) != 1
        ]

        print(
            split,
            "pairs:",
            len(pairs),
            "ambiguous:",
            len(ambiguous),
        )

        if ambiguous:
            errors.append(
                f"{split}: ambiguous relational pairs"
            )

    # ---------------------------------------------------------
    # Counterfactual evidence swaps
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("COUNTERFACTUAL EVIDENCE SWAPS")
    print("=" * 88)

    for split, rows in data.items():

        groups = defaultdict(list)

        for row in rows:

            key = (
                row["family"],
                row["target"],
                row["paraphrase_index"],
            )

            groups[key].append(
                row
            )

        total = 0
        changed = 0

        for group in groups.values():

            for a, b in combinations(
                group,
                2,
            ):

                la = json.loads(
                    a["output"]
                )["decision"]

                lb = json.loads(
                    b["output"]
                )["decision"]

                total += 1
                changed += int(
                    la != lb
                )

        rate = (
            changed / total
            if total
            else 0.0
        )

        print(
            split,
            f"{changed}/{total}",
            f"{rate:.4f}",
        )

        if rate != 1.0:
            errors.append(
                f"{split}: evidence swaps not fully causal"
            )

    # ---------------------------------------------------------
    # Counterfactual question swaps
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("COUNTERFACTUAL QUESTION SWAPS")
    print("=" * 88)

    for split, rows in data.items():

        groups = defaultdict(list)

        for row in rows:

            key = (
                row["family"],
                row["world_index"],
                row["paraphrase_index"],
            )

            groups[key].append(
                row
            )

        total = 0
        changed = 0

        for group in groups.values():

            for a, b in combinations(
                group,
                2,
            ):

                la = json.loads(
                    a["output"]
                )["decision"]

                lb = json.loads(
                    b["output"]
                )["decision"]

                total += 1
                changed += int(
                    la != lb
                )

        rate = (
            changed / total
            if total
            else 0.0
        )

        print(
            split,
            f"{changed}/{total}",
            f"{rate:.4f}",
        )

        if rate != 1.0:
            errors.append(
                f"{split}: question swaps not fully causal"
            )

    # ---------------------------------------------------------
    # Gate semantics
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("GATE-ORDER SEMANTIC CHECK")
    print("=" * 88)

    for split, rows in data.items():

        counts = Counter()

        for row in rows:

            label = json.loads(
                row["output"]
            )["decision"]

            counts[
                (
                    label,
                    row["gate_state"],
                    row["downstream_state"],
                )
            ] += 1

            if label == "YES":

                if not (
                    row["gate_state"] == "open"
                    and
                    row["downstream_state"] == "satisfied"
                ):
                    errors.append(
                        f"{split}: malformed YES gate state"
                    )

            elif label == "CONDITIONAL":

                if not (
                    row["gate_state"] == "open"
                    and
                    row["downstream_state"] == "unresolved"
                ):
                    errors.append(
                        f"{split}: malformed CONDITIONAL gate state"
                    )

            elif label == "NO":

                if not (
                    row["gate_state"] == "closed"
                    and
                    row["downstream_state"] == "satisfied"
                ):
                    errors.append(
                        f"{split}: malformed NO short-circuit state"
                    )

            elif label == "INSUFFICIENT":

                if row["gate_state"] != "undetermined":

                    errors.append(
                        f"{split}: malformed INSUFFICIENT gate state"
                    )

        print()
        print(split)

        for key, n in sorted(
            counts.items()
        ):

            print(
                " ",
                key,
                n,
            )

    # ---------------------------------------------------------
    # Citation integrity
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("CITATION INTEGRITY")
    print("=" * 88)

    citation_errors = 0

    for split, rows in data.items():

        local_errors = 0

        for row in rows:

            output = json.loads(
                row["output"]
            )

            available = set(
                re.findall(
                    r"^\[([^\]]+)\]\s*$",
                    row["evidence"],
                    flags=re.MULTILINE,
                )
            )

            for citation in output[
                "citations"
            ]:

                if citation not in available:

                    local_errors += 1

        citation_errors += local_errors

        print(
            split,
            "citation errors:",
            local_errors,
        )

    if citation_errors:
        errors.append(
            "invalid citations detected"
        )

    # ---------------------------------------------------------
    # Linear lexical shortcut probes
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("TF-IDF SHORTCUT PROBES")
    print("=" * 88)

    train = data["train"]

    for split in [
        "validation",
        "test",
    ]:

        question_score = probe(
            train,
            data[split],
            "question",
        )

        evidence_score = probe(
            train,
            data[split],
            "evidence",
        )

        joint_train = [
            {
                **row,
                "joint":
                    row["question"]
                    + "\n[RELATION]\n"
                    + row["evidence"],
            }
            for row in train
        ]

        joint_eval = [
            {
                **row,
                "joint":
                    row["question"]
                    + "\n[RELATION]\n"
                    + row["evidence"],
            }
            for row in data[split]
        ]

        joint_score = probe(
            joint_train,
            joint_eval,
            "joint",
        )

        print()
        print(split)
        print(
            " question only:",
            f"{question_score:.4f}",
        )
        print(
            " evidence only:",
            f"{evidence_score:.4f}",
        )
        print(
            " additive joint:",
            f"{joint_score:.4f}",
        )

        if question_score > 0.30:
            errors.append(
                f"{split}: question lexical shortcut"
            )

        if evidence_score > 0.30:
            errors.append(
                f"{split}: evidence lexical shortcut"
            )

        if joint_score > 0.30:
            errors.append(
                f"{split}: additive lexical shortcut"
            )

    print()
    print("=" * 88)

    if errors:

        print("V4.4 GATE PILOT AUDIT: FAIL")

        for error in errors:
            print(" -", error)

        raise SystemExit(1)

    print("V4.4 GATE PILOT AUDIT: PASS")


if __name__ == "__main__":
    main()
