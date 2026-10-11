from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


BASE = Path("/workspace")

TRAIN = (
    BASE
    / "training/v6/full_candidate/"
      "sentinel_v6_train_candidate.jsonl"
)

DEV = (
    BASE
    / "training/v6/full_candidate/"
      "sentinel_v6_development_candidate.jsonl"
)

OUT = (
    BASE
    / "experiments/v6/full_candidate/"
      "V6_SECTION_BOUNDARY_REPAIR_DIAGNOSTIC.json"
)


def load(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def split_evidence(evidence):
    matches = list(
        re.finditer(
            r"(?m)^\[(E[1-4])\]\s+",
            evidence,
        )
    )

    if len(matches) != 4:
        raise AssertionError(
            "expected exactly four evidence records"
        )

    pieces = []

    for i, match in enumerate(matches):
        start = match.start()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(evidence)
        )

        pieces.append(
            evidence[start:end].strip()
        )

    return pieces


def terminated_evidence(evidence):
    records = split_evidence(
        evidence
    )

    # Critical design:
    #
    # EVERY evidence item gets the same explicit terminal marker.
    #
    # Therefore phrases such as:
    #
    #     "failed END_RECORD"
    #     "unresolved END_RECORD"
    #
    # occur once for each semantic track in every four-state pack,
    # instead of only for the evidence record sitting next to QUESTION.
    return "\n\n".join(
        record
        + "\nEND_RECORD"
        for record in records
    )


def render(
    row,
    mode,
):
    question = row["question"]
    evidence = row["evidence"]

    if mode == "baseline":
        return row["input"]


    if mode == "single_separator":
        # Tests whether one separator alone is sufficient.
        #
        # It probably will NOT be, because:
        #
        #     failed SECTION_BREAK
        #
        # can still expose the state of the last evidence record.
        if row["input_order"] == "question_first":
            return (
                "QUESTION\n"
                "========\n"
                f"{question}\n\n"
                "SECTION_BREAK\n\n"
                "EVIDENCE\n"
                "========\n"
                f"{evidence}"
            )

        return (
            "EVIDENCE\n"
            "========\n"
            f"{evidence}\n\n"
            "SECTION_BREAK\n\n"
            "QUESTION\n"
            "========\n"
            f"{question}"
        )


    if mode == "terminate_question_only":
        if row["input_order"] == "question_first":
            return (
                "QUESTION\n"
                "========\n"
                f"{question}\n"
                "END_QUESTION\n\n"
                "EVIDENCE\n"
                "========\n"
                f"{evidence}"
            )

        return (
            "EVIDENCE\n"
            "========\n"
            f"{evidence}\n\n"
            "QUESTION\n"
            "========\n"
            f"{question}\n"
            "END_QUESTION"
        )


    if mode == "terminate_each_evidence":
        evidence2 = terminated_evidence(
            evidence
        )

        if row["input_order"] == "question_first":
            return (
                "QUESTION\n"
                "========\n"
                f"{question}\n\n"
                "EVIDENCE\n"
                "========\n"
                f"{evidence2}"
            )

        return (
            "EVIDENCE\n"
            "========\n"
            f"{evidence2}\n\n"
            "QUESTION\n"
            "========\n"
            f"{question}"
        )


    if mode == "terminate_all_blocks":
        evidence2 = terminated_evidence(
            evidence
        )

        question2 = (
            question
            + "\nEND_QUESTION"
        )

        if row["input_order"] == "question_first":
            return (
                "QUESTION\n"
                "========\n"
                f"{question2}\n\n"
                "EVIDENCE\n"
                "========\n"
                f"{evidence2}"
            )

        return (
            "EVIDENCE\n"
            "========\n"
            f"{evidence2}\n\n"
            "QUESTION\n"
            "========\n"
            f"{question2}"
        )


    if mode == "terminate_all_plus_section_break":
        evidence2 = terminated_evidence(
            evidence
        )

        question2 = (
            question
            + "\nEND_QUESTION"
        )

        if row["input_order"] == "question_first":
            return (
                "QUESTION\n"
                "========\n"
                f"{question2}\n\n"
                "SECTION_BREAK\n\n"
                "EVIDENCE\n"
                "========\n"
                f"{evidence2}"
            )

        return (
            "EVIDENCE\n"
            "========\n"
            f"{evidence2}\n\n"
            "SECTION_BREAK\n\n"
            "QUESTION\n"
            "========\n"
            f"{question2}"
        )


    raise ValueError(
        mode
    )


def grouped_accuracy(
    rows,
    mode,
    *,
    analyzer="word",
    ngram_range=(1, 2),
    n_splits,
):
    texts = [
        render(
            row,
            mode,
        )
        for row in rows
    ]

    labels = [
        row["target"]
        for row in rows
    ]

    groups = [
        row["relational_group"]
        for row in rows
    ]

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=20260818,
    )

    correct = 0
    total = 0

    for train_idx, test_idx in splitter.split(
        texts,
        labels,
        groups,
    ):
        model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=analyzer,
                    ngram_range=ngram_range,
                    min_df=2,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=600,
                    solver="lbfgs",
                ),
            ),
        ])

        x_train = [
            texts[i]
            for i in train_idx
        ]

        y_train = [
            labels[i]
            for i in train_idx
        ]

        x_test = [
            texts[i]
            for i in test_idx
        ]

        y_test = [
            labels[i]
            for i in test_idx
        ]

        model.fit(
            x_train,
            y_train,
        )

        prediction = model.predict(
            x_test
        )

        correct += sum(
            p == y
            for p, y
            in zip(
                prediction,
                y_test,
            )
        )

        total += len(
            y_test
        )

    return correct / total


def run(
    name,
    rows,
    n_splits,
):
    modes = (
        "baseline",
        "single_separator",
        "terminate_question_only",
        "terminate_each_evidence",
        "terminate_all_blocks",
        "terminate_all_plus_section_break",
    )

    result = {}

    for mode in modes:
        result[
            mode + "__word_1_2"
        ] = grouped_accuracy(
            rows,
            mode,
            analyzer="word",
            ngram_range=(1, 2),
            n_splits=n_splits,
        )

        result[
            mode + "__char_3_5"
        ] = grouped_accuracy(
            rows,
            mode,
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_splits=n_splits,
        )

    return {
        "name": name,
        "rows": len(rows),
        "probes": result,
    }


def main():
    train = load(
        TRAIN
    )

    dev = load(
        DEV
    )

    train_result = run(
        "TRAIN",
        train,
        5,
    )

    dev_result = run(
        "DEVELOPMENT",
        dev,
        4,
    )

    report = {
        "status":
            "DIAGNOSTIC_ONLY",

        "train":
            train_result,

        "development":
            dev_result,

        "success_rule": {
            "preferred_word_score_max":
                0.30,

            "hard_failure":
                0.35,

            "must_preserve_variable_input_order":
                True,
        },

        "candidate_v1_modified":
            False,

        "model_loads":
            0,

        "sentinel_inference":
            0,

        "challenge_v4_files_read":
            0,
    }

    OUT.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("V6 SECTION-BOUNDARY REPAIR SIMULATION")
    print("=" * 80)

    for result in (
        train_result,
        dev_result,
    ):
        print()
        print(result["name"])
        print("-" * 80)

        for probe, score in result[
            "probes"
        ].items():
            print(
                f"{probe:58s} "
                f"{score:.4f}"
            )

    print()
    print(
        "Result SHA256:",
        hashlib.sha256(
            OUT.read_bytes()
        ).hexdigest(),
    )

    print()
    print(
        "Candidate V1 modified: NO"
    )

    print(
        "Sentinel model loads: 0"
    )

    print(
        "Sentinel inference: 0"
    )

    print(
        "Challenge V4 files read: 0"
    )


if __name__ == "__main__":
    main()
