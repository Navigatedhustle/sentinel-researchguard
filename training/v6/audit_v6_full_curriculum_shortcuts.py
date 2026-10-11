from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
)
from sklearn.pipeline import (
    Pipeline,
)


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

OUT_JSON = (
    BASE
    / "experiments/v6/full_candidate/"
      "V6_FULL_CURRICULUM_SHORTCUT_AUDIT.json"
)

OUT_TXT = (
    BASE
    / "experiments/v6/full_candidate/"
      "V6_FULL_CURRICULUM_SHORTCUT_AUDIT.txt"
)


CHANCE = 0.25
WARNING = 0.30
HARD_FAIL = 0.35


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def majority_accuracy(
    rows,
    feature_fn,
):
    buckets = defaultdict(
        Counter
    )

    for row in rows:
        buckets[
            feature_fn(
                row
            )
        ][
            row["target"]
        ] += 1

    correct = sum(
        max(
            counts.values()
        )
        for counts
        in buckets.values()
    )

    return (
        correct
        / len(rows)
    )


def grouped_text_accuracy(
    rows,
    text_fn,
    analyzer,
    ngram_range,
    n_splits,
):
    texts = [
        text_fn(
            row
        )
        for row in rows
    ]

    labels = [
        row["target"]
        for row in rows
    ]

    groups = [
        row[
            "relational_group"
        ]
        for row in rows
    ]


    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=20260818,
    )

    correct = 0
    total = 0

    for train_index, test_index in splitter.split(
        texts,
        labels,
        groups,
    ):
        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=analyzer,
                    ngram_range=ngram_range,
                    min_df=2,
                    max_features=15000,
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
            for i in train_index
        ]

        y_train = [
            labels[i]
            for i in train_index
        ]

        x_test = [
            texts[i]
            for i in test_index
        ]

        y_test = [
            labels[i]
            for i in test_index
        ]

        pipeline.fit(
            x_train,
            y_train,
        )

        pred = pipeline.predict(
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

    return (
        correct
        / total
    )


def audit_split(
    name,
    rows,
    n_splits,
):
    exact = {
        "family":
            majority_accuracy(
                rows,
                lambda r:
                    r["family"],
            ),

        "operator":
            majority_accuracy(
                rows,
                lambda r:
                    r["operator"],
            ),

        "relational_group":
            majority_accuracy(
                rows,
                lambda r:
                    r[
                        "relational_group"
                    ],
            ),

        "input_order":
            majority_accuracy(
                rows,
                lambda r:
                    r[
                        "input_order"
                    ],
            ),

        "surface_variant":
            majority_accuracy(
                rows,
                lambda r:
                    r[
                        "surface_variant"
                    ],
            ),

        "group_variant":
            majority_accuracy(
                rows,
                lambda r:
                    r[
                        "group_variant"
                    ],
            ),

        "world_index":
            majority_accuracy(
                rows,
                lambda r:
                    r[
                        "world_index"
                    ],
            ),

        "target_index":
            majority_accuracy(
                rows,
                lambda r:
                    r[
                        "target_index"
                    ],
            ),

        "target_name":
            majority_accuracy(
                rows,
                lambda r:
                    r[
                        "target_name"
                    ],
            ),

        "target_evidence_ref":
            majority_accuracy(
                rows,
                lambda r:
                    r[
                        "target_evidence_ref"
                    ],
            ),

        "world_target_pair":
            majority_accuracy(
                rows,
                lambda r:
                    (
                        r[
                            "world_index"
                        ],
                        r[
                            "target_index"
                        ],
                    ),
            ),

        "world_target_surface":
            majority_accuracy(
                rows,
                lambda r:
                    (
                        r[
                            "world_index"
                        ],
                        r[
                            "target_index"
                        ],
                        r[
                            "surface_variant"
                        ],
                    ),
            ),

        "world_ref_surface":
            majority_accuracy(
                rows,
                lambda r:
                    (
                        r[
                            "world_index"
                        ],
                        r[
                            "target_evidence_ref"
                        ],
                        r[
                            "surface_variant"
                        ],
                    ),
            ),

        "exact_question":
            majority_accuracy(
                rows,
                lambda r:
                    r[
                        "question"
                    ],
            ),

        "evidence_hash":
            majority_accuracy(
                rows,
                lambda r:
                    r[
                        "evidence_hash"
                    ],
            ),
    }


    lexical = {
        "question_word":
            grouped_text_accuracy(
                rows,
                lambda r:
                    r["question"],
                "word",
                (1, 2),
                n_splits,
            ),

        "question_char":
            grouped_text_accuracy(
                rows,
                lambda r:
                    r["question"],
                "char_wb",
                (3, 5),
                n_splits,
            ),

        "evidence_word":
            grouped_text_accuracy(
                rows,
                lambda r:
                    r["evidence"],
                "word",
                (1, 2),
                n_splits,
            ),

        "evidence_char":
            grouped_text_accuracy(
                rows,
                lambda r:
                    r["evidence"],
                "char_wb",
                (3, 5),
                n_splits,
            ),

        "raw_input_word":
            grouped_text_accuracy(
                rows,
                lambda r:
                    r["input"],
                "word",
                (1, 2),
                n_splits,
            ),
    }


    warnings = []

    hard_failures = []

    for category, results in (
        (
            "exact",
            exact,
        ),
        (
            "lexical",
            lexical,
        ),
    ):
        for probe, score in results.items():
            item = {
                "category":
                    category,
                "probe":
                    probe,
                "score":
                    score,
            }

            if score > HARD_FAIL:
                hard_failures.append(
                    item
                )

            elif score > WARNING:
                warnings.append(
                    item
                )


    return {
        "name":
            name,

        "rows":
            len(rows),

        "exact":
            exact,

        "lexical":
            lexical,

        "warnings":
            warnings,

        "hard_failures":
            hard_failures,
    }


def main():
    train = load_jsonl(
        TRAIN
    )

    dev = load_jsonl(
        DEV
    )


    train_result = audit_split(
        "TRAIN",
        train,
        5,
    )

    dev_result = audit_split(
        "DEVELOPMENT",
        dev,
        4,
    )


    hard_failures = (
        train_result[
            "hard_failures"
        ]
        + dev_result[
            "hard_failures"
        ]
    )

    warnings = (
        train_result[
            "warnings"
        ]
        + dev_result[
            "warnings"
        ]
    )


    status = (
        "FAIL"
        if hard_failures
        else
        "PASS"
    )


    report = {
        "status":
            status,

        "chance":
            CHANCE,

        "warning_threshold":
            WARNING,

        "hard_failure_threshold":
            HARD_FAIL,

        "train":
            train_result,

        "development":
            dev_result,

        "warning_count":
            len(
                warnings
            ),

        "hard_failure_count":
            len(
                hard_failures
            ),

        "sentinel_model_loads":
            0,

        "sentinel_inference_calls":
            0,

        "challenge_v4_files_read":
            0,
    }


    OUT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    lines = [
        "SENTINEL V6 FULL CURRICULUM — SHORTCUT AUDIT",
        "=" * 78,
        "",
        f"STATUS: {status}",
        f"chance: {CHANCE:.4f}",
        f"warning threshold: >{WARNING:.2f}",
        f"hard failure threshold: >{HARD_FAIL:.2f}",
    ]


    for result in (
        train_result,
        dev_result,
    ):
        lines.extend([
            "",
            result[
                "name"
            ],
            "-" * 78,
            "EXACT FEATURES",
        ])

        for probe, score in sorted(
            result[
                "exact"
            ].items()
        ):
            lines.append(
                f"{probe:30s} {score:.4f}"
            )

        lines.append(
            ""
        )

        lines.append(
            "GROUPED TF-IDF PROBES"
        )

        for probe, score in sorted(
            result[
                "lexical"
            ].items()
        ):
            lines.append(
                f"{probe:30s} {score:.4f}"
            )


    lines.extend([
        "",
        f"Warnings: {len(warnings)}",
        f"Hard failures: {len(hard_failures)}",
        "",
        "Sentinel model loads: 0",
        "Sentinel inference: 0",
        "Challenge V4 files read: 0",
    ])


    OUT_TXT.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


    print()
    print("=" * 78)
    print("SENTINEL V6 FULL CURRICULUM SHORTCUT AUDIT")
    print("=" * 78)

    for result in (
        train_result,
        dev_result,
    ):
        print()
        print(
            result[
                "name"
            ]
        )

        print(
            "-" * 78
        )

        print(
            "Exact-feature majority accuracies:"
        )

        for probe, score in sorted(
            result[
                "exact"
            ].items()
        ):
            print(
                f"  {probe:28s} "
                f"{score:.4f}"
            )

        print()
        print(
            "Grouped TF-IDF accuracies:"
        )

        for probe, score in sorted(
            result[
                "lexical"
            ].items()
        ):
            print(
                f"  {probe:28s} "
                f"{score:.4f}"
            )


    print()
    print(
        "Warnings:",
        len(
            warnings
        ),
    )

    print(
        "Hard failures:",
        len(
            hard_failures
        ),
    )

    print()
    print(
        "Audit JSON SHA256:",
        sha(
            OUT_JSON
        ),
    )

    print(
        "Audit TXT SHA256:",
        sha(
            OUT_TXT
        ),
    )

    print()
    print(
        "Sentinel model loads: 0"
    )

    print(
        "Sentinel inference: 0"
    )

    print(
        "Challenge V4 files read: 0"
    )

    if hard_failures:
        raise SystemExit(
            "ERROR: V6 full curriculum "
            "shortcut audit HARD FAILURE"
        )

    print()
    print(
        "V6 FULL CURRICULUM SHORTCUT AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
