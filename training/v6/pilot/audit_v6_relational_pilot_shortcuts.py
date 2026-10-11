"""
Sentinel V6 128-row relational pilot shortcut audit.

Uses only cheap statistical classifiers.

NO Sentinel model.
NO LLM inference.
NO benchmark access.
"""

from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import sys

import numpy as np

from scipy.sparse import hstack
from scipy.stats import binomtest

from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score


BASE = Path("/workspace")

DATASET = (
    BASE
    / "training/v6/pilot/"
      "sentinel_v6_relational_pilot.jsonl"
)

RESULTS = (
    BASE
    / "experiments/v6/"
      "v6_relational_pilot_shortcut_results.json"
)


EXPECTED_SHA = (
    "77fd17759cd11b05793b57c76767f04b"
    "dbf5a1638f3316c312d4c72bbe085fea"
)


LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)


HARD_MAX = 0.35
WARNING_MAX = 0.30
CHANCE = 0.25


def sha256(path):

    import hashlib

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def load_rows():

    rows = []

    with DATASET.open() as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def evidence_count(text):

    return len(
        set(
            re.findall(
                r"\[(E\d+)\]",
                text,
            )
        )
    )


def classifier():

    return LogisticRegression(
        solver="liblinear",
        max_iter=5000,
        random_state=42,
    )


def word_vectorizer():

    return TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )


def char_vectorizer():

    return TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )


def grouped_single_text_cv(
    rows,
    *,
    field,
    vectorizer_factory,
):

    texts = np.asarray(
        [
            str(
                row[
                    field
                ]
            )
            for row in rows
        ],
        dtype=object,
    )

    labels = np.asarray(
        [
            row[
                "target"
            ]
            for row in rows
        ],
        dtype=object,
    )

    groups = np.asarray(
        [
            row[
                "relational_group"
            ]
            for row in rows
        ],
        dtype=object,
    )


    logo = LeaveOneGroupOut()

    predictions = np.empty(
        len(
            rows
        ),
        dtype=object,
    )

    fold_results = []


    for fold_index, (
        train_index,
        test_index,
    ) in enumerate(
        logo.split(
            texts,
            labels,
            groups,
        )
    ):

        vectorizer = vectorizer_factory()

        train_x = vectorizer.fit_transform(
            texts[
                train_index
            ]
        )

        test_x = vectorizer.transform(
            texts[
                test_index
            ]
        )


        clf = classifier()

        clf.fit(
            train_x,
            labels[
                train_index
            ],
        )


        pred = clf.predict(
            test_x
        )


        predictions[
            test_index
        ] = pred


        fold_results.append({

            "fold":
                fold_index,

            "held_out_group":
                str(
                    groups[
                        test_index[
                            0
                        ]
                    ]
                ),

            "rows":
                int(
                    len(
                        test_index
                    )
                ),

            "accuracy":
                float(
                    accuracy_score(
                        labels[
                            test_index
                        ],
                        pred,
                    )
                ),
        })


    correct = int(
        np.sum(
            predictions
            == labels
        )
    )

    accuracy = (
        correct
        / len(
            rows
        )
    )


    p_value = float(
        binomtest(
            correct,
            len(
                rows
            ),
            CHANCE,
            alternative="greater",
        ).pvalue
    )


    return {

        "correct":
            correct,

        "rows":
            len(
                rows
            ),

        "accuracy":
            accuracy,

        "one_sided_binomial_p_vs_025":
            p_value,

        "folds":
            fold_results,
    }


def grouped_additive_cv(
    rows,
    *,
    vectorizer_factory,
):

    questions = np.asarray(
        [
            row[
                "question"
            ]
            for row in rows
        ],
        dtype=object,
    )

    evidence = np.asarray(
        [
            row[
                "evidence"
            ]
            for row in rows
        ],
        dtype=object,
    )

    labels = np.asarray(
        [
            row[
                "target"
            ]
            for row in rows
        ],
        dtype=object,
    )

    groups = np.asarray(
        [
            row[
                "relational_group"
            ]
            for row in rows
        ],
        dtype=object,
    )


    logo = LeaveOneGroupOut()

    predictions = np.empty(
        len(
            rows
        ),
        dtype=object,
    )

    fold_results = []


    for fold_index, (
        train_index,
        test_index,
    ) in enumerate(
        logo.split(
            questions,
            labels,
            groups,
        )
    ):

        question_vectorizer = (
            vectorizer_factory()
        )

        evidence_vectorizer = (
            vectorizer_factory()
        )


        q_train = (
            question_vectorizer.fit_transform(
                questions[
                    train_index
                ]
            )
        )

        q_test = (
            question_vectorizer.transform(
                questions[
                    test_index
                ]
            )
        )


        e_train = (
            evidence_vectorizer.fit_transform(
                evidence[
                    train_index
                ]
            )
        )

        e_test = (
            evidence_vectorizer.transform(
                evidence[
                    test_index
                ]
            )
        )


        train_x = hstack(
            [
                q_train,
                e_train,
            ],
            format="csr",
        )

        test_x = hstack(
            [
                q_test,
                e_test,
            ],
            format="csr",
        )


        clf = classifier()

        clf.fit(
            train_x,
            labels[
                train_index
            ],
        )


        pred = clf.predict(
            test_x
        )


        predictions[
            test_index
        ] = pred


        fold_results.append({

            "fold":
                fold_index,

            "held_out_group":
                str(
                    groups[
                        test_index[
                            0
                        ]
                    ]
                ),

            "rows":
                int(
                    len(
                        test_index
                    )
                ),

            "accuracy":
                float(
                    accuracy_score(
                        labels[
                            test_index
                        ],
                        pred,
                    )
                ),
        })


    correct = int(
        np.sum(
            predictions
            == labels
        )
    )

    accuracy = (
        correct
        / len(
            rows
        )
    )


    p_value = float(
        binomtest(
            correct,
            len(
                rows
            ),
            CHANCE,
            alternative="greater",
        ).pvalue
    )


    return {

        "correct":
            correct,

        "rows":
            len(
                rows
            ),

        "accuracy":
            accuracy,

        "one_sided_binomial_p_vs_025":
            p_value,

        "folds":
            fold_results,
    }


def marginal_oracle(
    rows,
    values,
):

    buckets = defaultdict(
        Counter
    )


    for row, value in zip(
        rows,
        values,
    ):

        buckets[
            str(
                value
            )
        ][
            row[
                "target"
            ]
        ] += 1


    correct = sum(
        max(
            counter.values()
        )
        for counter
        in buckets.values()
    )


    return {

        "categories":
            len(
                buckets
            ),

        "correct":
            correct,

        "rows":
            len(
                rows
            ),

        "accuracy":
            correct
            / len(
                rows
            ),
    }


def hidden_latin_oracle(
    rows,
):

    correct = 0


    for row in rows:

        predicted = LABELS[
            (
                int(
                    row[
                        "world_index"
                    ]
                )
                +
                int(
                    row[
                        "target_index"
                    ]
                )
            )
            % 4
        ]


        if predicted == row[
            "target"
        ]:

            correct += 1


    return {

        "correct":
            correct,

        "rows":
            len(
                rows
            ),

        "accuracy":
            correct
            / len(
                rows
            ),
    }


def main():

    if RESULTS.exists():

        raise RuntimeError(
            "Results file already exists."
        )


    if sha256(
        DATASET
    ) != EXPECTED_SHA:

        raise RuntimeError(
            "Frozen pilot SHA mismatch."
        )


    rows = load_rows()


    if len(
        rows
    ) != 128:

        raise RuntimeError(
            "Expected 128 frozen pilot rows."
        )


    label_counts = Counter(
        row[
            "target"
        ]
        for row in rows
    )


    if label_counts != Counter({
        "YES": 32,
        "NO": 32,
        "CONDITIONAL": 32,
        "INSUFFICIENT": 32,
    }):

        raise RuntimeError(
            "Pilot label distribution changed."
        )


    if len({
        row[
            "relational_group"
        ]
        for row in rows
    }) != 8:

        raise RuntimeError(
            "Expected eight relational groups."
        )


    # ================================================================
    # GROUPED LEXICAL BASELINES
    # ================================================================

    lexical = {}


    lexical[
        "question_word"
    ] = grouped_single_text_cv(
        rows,
        field="question",
        vectorizer_factory=word_vectorizer,
    )


    lexical[
        "question_char"
    ] = grouped_single_text_cv(
        rows,
        field="question",
        vectorizer_factory=char_vectorizer,
    )


    lexical[
        "evidence_word"
    ] = grouped_single_text_cv(
        rows,
        field="evidence",
        vectorizer_factory=word_vectorizer,
    )


    lexical[
        "evidence_char"
    ] = grouped_single_text_cv(
        rows,
        field="evidence",
        vectorizer_factory=char_vectorizer,
    )


    lexical[
        "additive_question_evidence_word"
    ] = grouped_additive_cv(
        rows,
        vectorizer_factory=word_vectorizer,
    )


    lexical[
        "additive_question_evidence_char"
    ] = grouped_additive_cv(
        rows,
        vectorizer_factory=char_vectorizer,
    )


    lexical[
        "target_entity_word"
    ] = grouped_single_text_cv(
        rows,
        field="target_name",
        vectorizer_factory=word_vectorizer,
    )


    lexical[
        "target_entity_char"
    ] = grouped_single_text_cv(
        rows,
        field="target_name",
        vectorizer_factory=char_vectorizer,
    )


    lexical[
        "raw_input_word"
    ] = grouped_single_text_cv(
        rows,
        field="input",
        vectorizer_factory=word_vectorizer,
    )


    lexical[
        "raw_input_char"
    ] = grouped_single_text_cv(
        rows,
        field="input",
        vectorizer_factory=char_vectorizer,
    )


    # ================================================================
    # OBSERVABLE MARGINAL ORACLES
    # ================================================================

    marginal = {}


    marginal[
        "exact_question"
    ] = marginal_oracle(
        rows,
        [
            row[
                "question"
            ]
            for row in rows
        ],
    )


    marginal[
        "evidence_hash"
    ] = marginal_oracle(
        rows,
        [
            row[
                "evidence_hash"
            ]
            for row in rows
        ],
    )


    marginal[
        "target_name"
    ] = marginal_oracle(
        rows,
        [
            row[
                "target_name"
            ]
            for row in rows
        ],
    )


    marginal[
        "family"
    ] = marginal_oracle(
        rows,
        [
            row[
                "family"
            ]
            for row in rows
        ],
    )


    marginal[
        "operator"
    ] = marginal_oracle(
        rows,
        [
            row[
                "operator"
            ]
            for row in rows
        ],
    )


    marginal[
        "relational_group"
    ] = marginal_oracle(
        rows,
        [
            row[
                "relational_group"
            ]
            for row in rows
        ],
    )


    marginal[
        "input_order"
    ] = marginal_oracle(
        rows,
        [
            row[
                "input_order"
            ]
            for row in rows
        ],
    )


    marginal[
        "evidence_count"
    ] = marginal_oracle(
        rows,
        [
            evidence_count(
                row[
                    "evidence"
                ]
            )
            for row in rows
        ],
    )


    marginal[
        "surface_variant"
    ] = marginal_oracle(
        rows,
        [
            row[
                "surface_variant"
            ]
            for row in rows
        ],
    )


    # ================================================================
    # HIDDEN-CONSTRUCTION SANITY CHECKS
    # ================================================================

    hidden = {}


    hidden[
        "world_index_only"
    ] = marginal_oracle(
        rows,
        [
            row[
                "world_index"
            ]
            for row in rows
        ],
    )


    hidden[
        "target_index_only"
    ] = marginal_oracle(
        rows,
        [
            row[
                "target_index"
            ]
            for row in rows
        ],
    )


    hidden[
        "world_plus_target_latin_oracle"
    ] = hidden_latin_oracle(
        rows
    )


    # ================================================================
    # PRECOMMITTED GATES
    # ================================================================

    hard_failures = []

    warnings = []


    for name, result in lexical.items():

        accuracy = result[
            "accuracy"
        ]


        if accuracy > HARD_MAX:

            hard_failures.append(
                (
                    f"{name}: grouped lexical accuracy "
                    f"{accuracy:.4f} > {HARD_MAX:.2f}"
                )
            )


        elif accuracy > WARNING_MAX:

            warnings.append(
                (
                    f"{name}: grouped lexical accuracy "
                    f"{accuracy:.4f} > warning "
                    f"{WARNING_MAX:.2f}"
                )
            )


    for name, result in marginal.items():

        accuracy = result[
            "accuracy"
        ]


        if abs(
            accuracy
            - CHANCE
        ) > 1e-12:

            hard_failures.append(
                (
                    f"{name}: marginal oracle "
                    f"{accuracy:.4f} != 0.2500"
                )
            )


    for name in (
        "world_index_only",
        "target_index_only",
    ):

        accuracy = hidden[
            name
        ][
            "accuracy"
        ]


        if abs(
            accuracy
            - CHANCE
        ) > 1e-12:

            hard_failures.append(
                (
                    f"{name}: hidden marginal "
                    f"{accuracy:.4f} != 0.2500"
                )
            )


    oracle_accuracy = hidden[
        "world_plus_target_latin_oracle"
    ][
        "accuracy"
    ]


    if abs(
        oracle_accuracy
        - 1.0
    ) > 1e-12:

        hard_failures.append(
            (
                "world_plus_target_latin_oracle: "
                f"{oracle_accuracy:.4f} != 1.0000"
            )
        )


    status = (
        "PASS"
        if not hard_failures
        else "FAIL"
    )


    report = {

        "dataset_sha256":
            EXPECTED_SHA,

        "rows":
            len(
                rows
            ),

        "labels":
            dict(
                sorted(
                    label_counts.items()
                )
            ),

        "relational_groups":
            8,

        "sklearn_version":
            sklearn_version,

        "chance":
            CHANCE,

        "warning_threshold":
            WARNING_MAX,

        "hard_failure_threshold":
            HARD_MAX,

        "grouped_lexical_baselines":
            lexical,

        "observable_marginal_oracles":
            marginal,

        "hidden_sanity_checks":
            hidden,

        "warnings":
            warnings,

        "hard_failures":
            hard_failures,

        "status":
            status,
    }


    RESULTS.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


    # ================================================================
    # HUMAN-READABLE REPORT
    # ================================================================

    print("=" * 88)
    print("SENTINEL V6 RELATIONAL PILOT SHORTCUT AUDIT")
    print("=" * 88)

    print(
        "rows:",
        len(
            rows
        )
    )

    print(
        "relational groups:",
        8
    )

    print(
        "labels:",
        dict(
            sorted(
                label_counts.items()
            )
        )
    )

    print(
        "scikit-learn:",
        sklearn_version
    )


    print()
    print("=" * 88)
    print("GROUPED LEXICAL BASELINES")
    print("Leave-one-relational-group-out")
    print("=" * 88)


    for name, result in lexical.items():

        print(
            f"{name:38s} "
            f"{result['correct']:3d}/128 "
            f"= {result['accuracy']:.4f} "
            f"p={result['one_sided_binomial_p_vs_025']:.6g}"
        )


    print()
    print("=" * 88)
    print("OBSERVABLE EXACT-FEATURE MARGINAL ORACLES")
    print("=" * 88)


    for name, result in marginal.items():

        print(
            f"{name:38s} "
            f"{result['correct']:3d}/128 "
            f"= {result['accuracy']:.4f} "
            f"categories={result['categories']}"
        )


    print()
    print("=" * 88)
    print("HIDDEN CONSTRUCTION SANITY CHECKS")
    print("=" * 88)


    for name, result in hidden.items():

        print(
            f"{name:38s} "
            f"{result['correct']:3d}/128 "
            f"= {result['accuracy']:.4f}"
        )


    print()
    print("=" * 88)
    print("GATES")
    print("=" * 88)

    print(
        "chance:",
        f"{CHANCE:.2f}"
    )

    print(
        "warning threshold:",
        f">{WARNING_MAX:.2f}"
    )

    print(
        "hard failure threshold:",
        f">{HARD_MAX:.2f}"
    )

    print(
        "warnings:",
        len(
            warnings
        )
    )


    for warning in warnings:

        print(
            "WARNING:",
            warning
        )


    print(
        "hard failures:",
        len(
            hard_failures
        )
    )


    for failure in hard_failures:

        print(
            "FAILURE:",
            failure
        )


    print()
    print(
        "SENTINEL V6 RELATIONAL PILOT SHORTCUT AUDIT:",
        status
    )


    if hard_failures:

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()
