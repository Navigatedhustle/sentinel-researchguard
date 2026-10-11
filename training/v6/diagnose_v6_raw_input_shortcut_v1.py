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
      "V6_RAW_INPUT_SHORTCUT_DIAGNOSTIC_V1.json"
)


ACTIONS = (
    "release",
    "transfer",
    "process",
    "access",
    "publish",
    "connect",
    "use",
    "export",
)

PURPOSES = (
    "controlled analysis",
    "approved research",
    "internal validation",
    "authorized review",
    "documented testing",
    "restricted collaboration",
    "governed processing",
    "supervised evaluation",
)


def load(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def build_group_targets(rows):
    groups = defaultdict(dict)

    for row in rows:
        groups[
            row["relational_group"]
        ][
            row["target_index"]
        ] = row["target_name"]

    for group, mapping in groups.items():
        if set(mapping) != {0, 1, 2, 3}:
            raise AssertionError(
                f"{group}: incomplete targets"
            )

    return groups


def alias_token(group, index):
    digest = hashlib.sha256(
        f"{group}::{index}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]

    # One unique atomic token.
    # It is consistent inside the relational group but unseen in
    # held-out groups.
    return "entity" + digest


def replace_targets(text, row, group_targets):
    group = row[
        "relational_group"
    ]

    mapping = group_targets[
        group
    ]

    result = text

    # Longest-first for safety.
    items = sorted(
        mapping.items(),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    for index, name in items:
        result = result.replace(
            name,
            alias_token(
                group,
                index,
            ),
        )

    return result


def normalize_schedule(text):
    result = text

    for purpose in sorted(
        PURPOSES,
        key=len,
        reverse=True,
    ):
        result = re.sub(
            re.escape(purpose),
            "purpose_slot",
            result,
            flags=re.IGNORECASE,
        )

    for action in ACTIONS:
        result = re.sub(
            rf"\b{re.escape(action)}\b",
            "action_slot",
            result,
            flags=re.IGNORECASE,
        )

    return result


def normalize_section_order(row):
    # Reconstruct the same content with fixed section order.
    return (
        "QUESTION\n"
        "========\n"
        + row["question"]
        + "\n\n"
        "EVIDENCE\n"
        "========\n"
        + row["evidence"]
    )


def transform(
    row,
    group_targets,
    mode,
):
    if mode == "baseline":
        return row["input"]

    if mode == "fixed_section_order":
        return normalize_section_order(
            row
        )

    if mode == "group_local_target_alias":
        return replace_targets(
            row["input"],
            row,
            group_targets,
        )

    if mode == "normalize_action_purpose":
        return normalize_schedule(
            row["input"]
        )

    if mode == "target_alias_plus_schedule":
        text = replace_targets(
            row["input"],
            row,
            group_targets,
        )

        return normalize_schedule(
            text
        )

    if mode == "alias_plus_schedule_fixed_order":
        temp = normalize_section_order(
            row
        )

        temp = replace_targets(
            temp,
            row,
            group_targets,
        )

        return normalize_schedule(
            temp
        )

    raise ValueError(
        mode
    )


def grouped_accuracy(
    rows,
    mode,
    *,
    n_splits,
    ngram_range=(1, 2),
    binary=False,
):
    group_targets = build_group_targets(
        rows
    )

    texts = [
        transform(
            row,
            group_targets,
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
                    analyzer="word",
                    ngram_range=ngram_range,
                    min_df=2,
                    max_features=15000,
                    sublinear_tf=not binary,
                    binary=binary,
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

        pred = model.predict(
            x_test
        )

        correct += sum(
            p == y
            for p, y
            in zip(
                pred,
                y_test,
            )
        )

        total += len(
            y_test
        )

    return correct / total


def top_features(rows):
    texts = [
        row["input"]
        for row in rows
    ]

    labels = [
        row["target"]
        for row in rows
    ]

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=15000,
        sublinear_tf=True,
    )

    x = vectorizer.fit_transform(
        texts
    )

    clf = LogisticRegression(
        max_iter=600,
        solver="lbfgs",
    )

    clf.fit(
        x,
        labels,
    )

    names = vectorizer.get_feature_names_out()

    result = {}

    for class_index, label in enumerate(
        clf.classes_
    ):
        coefs = clf.coef_[
            class_index
        ]

        indices = coefs.argsort()[
            -12:
        ][::-1]

        result[
            label
        ] = [
            names[i]
            for i in indices
        ]

    return result


def run_split(
    name,
    rows,
    n_splits,
):
    probes = {}

    probes[
        "baseline_word_1_2"
    ] = grouped_accuracy(
        rows,
        "baseline",
        n_splits=n_splits,
        ngram_range=(1, 2),
    )

    probes[
        "baseline_unigram"
    ] = grouped_accuracy(
        rows,
        "baseline",
        n_splits=n_splits,
        ngram_range=(1, 1),
    )

    probes[
        "baseline_binary_unigram"
    ] = grouped_accuracy(
        rows,
        "baseline",
        n_splits=n_splits,
        ngram_range=(1, 1),
        binary=True,
    )

    probes[
        "fixed_section_order"
    ] = grouped_accuracy(
        rows,
        "fixed_section_order",
        n_splits=n_splits,
    )

    probes[
        "group_local_target_alias"
    ] = grouped_accuracy(
        rows,
        "group_local_target_alias",
        n_splits=n_splits,
    )

    probes[
        "normalize_action_purpose"
    ] = grouped_accuracy(
        rows,
        "normalize_action_purpose",
        n_splits=n_splits,
    )

    probes[
        "target_alias_plus_schedule"
    ] = grouped_accuracy(
        rows,
        "target_alias_plus_schedule",
        n_splits=n_splits,
    )

    probes[
        "alias_plus_schedule_fixed_order"
    ] = grouped_accuracy(
        rows,
        "alias_plus_schedule_fixed_order",
        n_splits=n_splits,
    )

    return {
        "name": name,
        "rows": len(rows),
        "probes": probes,
    }


def main():
    train = load(
        TRAIN
    )

    dev = load(
        DEV
    )

    train_result = run_split(
        "TRAIN",
        train,
        5,
    )

    dev_result = run_split(
        "DEVELOPMENT",
        dev,
        4,
    )

    features = top_features(
        train
    )

    report = {
        "status":
            "DIAGNOSTIC_ONLY",

        "train":
            train_result,

        "development":
            dev_result,

        "baseline_train_top_features":
            features,

        "interpretation_thresholds": {
            "chance": 0.25,
            "warning": 0.30,
            "hard_failure": 0.35,
        },

        "candidate_modified":
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
    print("=" * 78)
    print("V6 RAW-INPUT SHORTCUT ROOT-CAUSE DIAGNOSTIC")
    print("=" * 78)

    for result in (
        train_result,
        dev_result,
    ):
        print()
        print(result["name"])
        print("-" * 78)

        for probe, score in result[
            "probes"
        ].items():
            print(
                f"{probe:40s} "
                f"{score:.4f}"
            )

    print()
    print("TRAIN baseline top positive lexical features by class")
    print("-" * 78)

    for label, tokens in sorted(
        features.items()
    ):
        print(
            f"{label:13s}: "
            + ", ".join(tokens)
        )

    print()
    print(
        "Results SHA256:",
        hashlib.sha256(
            OUT.read_bytes()
        ).hexdigest(),
    )

    print()
    print("Candidate modified: NO")
    print("Candidate regenerated: NO")
    print("Sentinel model loads: 0")
    print("Sentinel inference: 0")
    print("Challenge V4 files read: 0")


if __name__ == "__main__":
    main()
