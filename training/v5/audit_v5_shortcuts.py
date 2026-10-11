"""
Sentinel V5 pre-training shortcut audit.

Tests whether cheap lexical or metadata-based classifiers can predict
the gold label without learning the intended question/evidence relation.

This audit uses ONLY:

    sentinel_v5_train.jsonl
    sentinel_v5_development.jsonl

It does NOT read Challenge V1.
It does NOT read Challenge V2.
It does NOT train Sentinel.
"""

from collections import Counter
from pathlib import Path

import hashlib
import json
import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline


BASE = Path("/workspace")

TRAIN = (
    BASE
    / "training/v5/datasets/"
      "sentinel_v5_train.jsonl"
)

DEV = (
    BASE
    / "training/v5/datasets/"
      "sentinel_v5_development.jsonl"
)


EXPECTED_TRAIN_SHA = (
    "945eef0841648911aaee5b88a5f7efd6"
    "c84bcad7399697f7f786f4774645d972"
)

EXPECTED_DEV_SHA = (
    "4e9fe2effa908d8994fdfb3be486661f"
    "9ef178860725fbd887d60b5181900124"
)


Q_THRESHOLD = 0.30
E_THRESHOLD = 0.30
ADDITIVE_THRESHOLD = 0.35

EXTRA_LEXICAL_WARNING = 0.35


FAILURES = []
WARNINGS = []


def fail(
    message,
):
    FAILURES.append(
        message
    )


def warn(
    message,
):
    WARNINGS.append(
        message
    )


def sha256(
    path,
):
    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def load_jsonl(
    path,
):

    rows = []

    with path.open() as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(
                    json.loads(
                        line
                    )
                )

    return pd.DataFrame(
        rows
    )


def word_vectorizer():

    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=75000,
    )


def char_vectorizer():

    return TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=50000,
    )


def classifier():

    return LogisticRegression(
        max_iter=5000,
        solver="liblinear",
        random_state=20260816,
    )


def text_pipeline(
    analyzer,
):

    vectorizer = (
        word_vectorizer()
        if analyzer == "word"
        else char_vectorizer()
    )

    return Pipeline([
        (
            "tfidf",
            vectorizer,
        ),

        (
            "classifier",
            classifier(),
        ),
    ])


def additive_pipeline(
    analyzer,
):

    vectorizer_factory = (
        word_vectorizer
        if analyzer == "word"
        else char_vectorizer
    )

    return Pipeline([
        (
            "features",
            ColumnTransformer(
                [
                    (
                        "question",
                        vectorizer_factory(),
                        "question",
                    ),

                    (
                        "evidence",
                        vectorizer_factory(),
                        "evidence",
                    ),
                ],
                sparse_threshold=1.0,
            ),
        ),

        (
            "classifier",
            classifier(),
        ),
    ])


def fit_score(
    mode,
    train,
    test,
    analyzer="word",
):

    if mode == "question":

        pipeline = text_pipeline(
            analyzer
        )

        pipeline.fit(
            train[
                "question"
            ],
            train[
                "target"
            ],
        )

        score = pipeline.score(
            test[
                "question"
            ],
            test[
                "target"
            ],
        )

        return float(
            score
        )


    if mode == "evidence":

        pipeline = text_pipeline(
            analyzer
        )

        pipeline.fit(
            train[
                "evidence"
            ],
            train[
                "target"
            ],
        )

        score = pipeline.score(
            test[
                "evidence"
            ],
            test[
                "target"
            ],
        )

        return float(
            score
        )


    if mode == "input":

        pipeline = text_pipeline(
            analyzer
        )

        pipeline.fit(
            train[
                "input"
            ],
            train[
                "target"
            ],
        )

        score = pipeline.score(
            test[
                "input"
            ],
            test[
                "target"
            ],
        )

        return float(
            score
        )


    if mode == "additive":

        pipeline = additive_pipeline(
            analyzer
        )

        pipeline.fit(
            train[
                [
                    "question",
                    "evidence",
                ]
            ],
            train[
                "target"
            ],
        )

        score = pipeline.score(
            test[
                [
                    "question",
                    "evidence",
                ]
            ],
            test[
                "target"
            ],
        )

        return float(
            score
        )


    raise ValueError(
        mode
    )


def leave_one_family_out(
    frame,
    mode,
):

    logo = LeaveOneGroupOut()

    groups = frame[
        "family"
    ].to_numpy()

    scores = []


    for train_index, test_index in logo.split(
        frame,
        frame[
            "target"
        ],
        groups,
    ):

        fold_train = frame.iloc[
            train_index
        ]

        fold_test = frame.iloc[
            test_index
        ]

        score = fit_score(
            mode=mode,
            train=fold_train,
            test=fold_test,
            analyzer="word",
        )

        held_out = sorted(
            set(
                fold_test[
                    "family"
                ]
            )
        )

        assert len(
            held_out
        ) == 1

        scores.append(
            (
                held_out[
                    0
                ],
                score,
            )
        )


    return scores


def lookup_majority_accuracy(
    frame,
    fields,
):
    """
    Best possible deterministic lookup classifier using ONLY the
    specified exact metadata fields.
    """

    correct = 0

    grouped = frame.groupby(
        fields,
        dropna=False,
    )

    for _, group in grouped:

        counts = Counter(
            group[
                "target"
            ]
        )

        correct += max(
            counts.values()
        )

    return (
        correct
        / len(
            frame
        )
    )


def print_score(
    name,
    score,
):

    print(
        f"{name:48s} "
        f"{score:.4f}"
    )


def main():

    print(
        "=" * 88
    )

    print(
        "SENTINEL V5 PRE-TRAINING SHORTCUT AUDIT"
    )

    print(
        "=" * 88
    )


    # ================================================================
    # DATASET INTEGRITY
    # ================================================================

    train_sha = sha256(
        TRAIN
    )

    dev_sha = sha256(
        DEV
    )

    print()
    print(
        "TRAIN SHA:",
        train_sha,
    )

    print(
        "DEV SHA:  ",
        dev_sha,
    )


    if (
        train_sha
        != EXPECTED_TRAIN_SHA
    ):

        fail(
            "TRAIN dataset hash mismatch."
        )


    if (
        dev_sha
        != EXPECTED_DEV_SHA
    ):

        fail(
            "DEVELOPMENT dataset hash mismatch."
        )


    train = load_jsonl(
        TRAIN
    )

    dev = load_jsonl(
        DEV
    )


    print()
    print(
        "TRAIN rows:",
        len(
            train
        ),
    )

    print(
        "DEV rows:",
        len(
            dev
        ),
    )


    if len(
        train
    ) != 6144:

        fail(
            "TRAIN row count is not 6144."
        )


    if len(
        dev
    ) != 1024:

        fail(
            "DEVELOPMENT row count is not 1024."
        )


    # ================================================================
    # EXACT LOOKUP / METADATA BASELINES
    # ================================================================

    print()
    print(
        "=" * 88
    )

    print(
        "EXACT-IDENTITY / VISIBLE-METADATA BASELINES — DEVELOPMENT"
    )

    print(
        "=" * 88
    )


    lookup_tests = [

        (
            "exact question",
            [
                "question",
            ],
        ),

        (
            "exact evidence world",
            [
                "evidence_hash",
            ],
        ),

        (
            "asset identity",
            [
                "asset",
            ],
        ),

        (
            "input order",
            [
                "input_order",
            ],
        ),

        (
            "surface variant",
            [
                "surface_variant",
            ],
        ),

        (
            "operator",
            [
                "operator",
            ],
        ),

        (
            "family",
            [
                "family",
            ],
        ),

        (
            "input order + surface",
            [
                "input_order",
                "surface_variant",
            ],
        ),

        (
            "operator + order + surface",
            [
                "operator",
                "input_order",
                "surface_variant",
            ],
        ),
    ]


    for name, fields in lookup_tests:

        score = lookup_majority_accuracy(
            dev,
            fields,
        )

        print_score(
            name,
            score,
        )


        # These constructions are intended to be exactly balanced.
        if score > 0.250001:

            fail(
                f"Metadata/identity shortcut above chance: "
                f"{name}={score:.4f}"
            )


    # ================================================================
    # RELATIONAL ORACLE SANITY CHECK
    #
    # world + target_index is deliberately sufficient because those
    # are the hidden construction coordinates of the Latin square.
    #
    # They are NOT part of the model input.
    # ================================================================

    relational_oracle = lookup_majority_accuracy(
        dev,
        [
            "world",
            "target_index",
        ],
    )

    print()
    print_score(
        "hidden relational oracle: world + target_index",
        relational_oracle,
    )


    if relational_oracle != 1.0:

        fail(
            "Latin-square relational oracle is not exactly 1.0."
        )


    # ================================================================
    # CONTRACT LEXICAL PROBES: TRAIN -> HELD-OUT DEVELOPMENT
    # ================================================================

    print()
    print(
        "=" * 88
    )

    print(
        "WORD TF-IDF — TRAIN → HELD-OUT DEVELOPMENT"
    )

    print(
        "=" * 88
    )


    q_score = fit_score(
        "question",
        train,
        dev,
        "word",
    )

    e_score = fit_score(
        "evidence",
        train,
        dev,
        "word",
    )

    additive_score = fit_score(
        "additive",
        train,
        dev,
        "word",
    )

    input_score = fit_score(
        "input",
        train,
        dev,
        "word",
    )


    print_score(
        "question-only",
        q_score,
    )

    print_score(
        "evidence-only",
        e_score,
    )

    print_score(
        "additive question + evidence",
        additive_score,
    )

    print_score(
        "raw rendered input",
        input_score,
    )


    if q_score > Q_THRESHOLD:

        fail(
            f"Question-only TF-IDF exceeds "
            f"{Q_THRESHOLD:.2f}: {q_score:.4f}"
        )


    if e_score > E_THRESHOLD:

        fail(
            f"Evidence-only TF-IDF exceeds "
            f"{E_THRESHOLD:.2f}: {e_score:.4f}"
        )


    if additive_score > ADDITIVE_THRESHOLD:

        fail(
            f"Additive TF-IDF exceeds "
            f"{ADDITIVE_THRESHOLD:.2f}: "
            f"{additive_score:.4f}"
        )


    if input_score > ADDITIVE_THRESHOLD:

        fail(
            f"Raw-input TF-IDF exceeds "
            f"{ADDITIVE_THRESHOLD:.2f}: "
            f"{input_score:.4f}"
        )


    # ================================================================
    # DEVELOPMENT-INTERNAL LEAVE-ONE-FAMILY-OUT PROBES
    # ================================================================

    print()
    print(
        "=" * 88
    )

    print(
        "WORD TF-IDF — DEVELOPMENT LEAVE-ONE-FAMILY-OUT"
    )

    print(
        "=" * 88
    )


    grouped_results = {}


    for mode in (
        "question",
        "evidence",
        "additive",
    ):

        results = leave_one_family_out(
            dev,
            mode,
        )

        grouped_results[
            mode
        ] = results

        mean = float(
            np.mean(
                [
                    score
                    for _, score
                    in results
                ]
            )
        )

        print()
        print(
            mode.upper()
        )

        for family, score in results:

            print(
                f"  {family:38s} "
                f"{score:.4f}"
            )

        print(
            f"  MEAN{'':34s} "
            f"{mean:.4f}"
        )


        threshold = (
            ADDITIVE_THRESHOLD
            if mode == "additive"
            else (
                Q_THRESHOLD
                if mode == "question"
                else E_THRESHOLD
            )
        )


        if mean > threshold:

            fail(
                f"Development leave-one-family-out "
                f"{mode} mean exceeds threshold: "
                f"{mean:.4f} > {threshold:.2f}"
            )


    # ================================================================
    # STRONGER CHARACTER N-GRAM PROBES
    # ================================================================

    print()
    print(
        "=" * 88
    )

    print(
        "CHARACTER N-GRAM TF-IDF — TRAIN → DEVELOPMENT"
    )

    print(
        "=" * 88
    )


    char_q = fit_score(
        "question",
        train,
        dev,
        "char",
    )

    char_e = fit_score(
        "evidence",
        train,
        dev,
        "char",
    )

    char_additive = fit_score(
        "additive",
        train,
        dev,
        "char",
    )

    char_input = fit_score(
        "input",
        train,
        dev,
        "char",
    )


    print_score(
        "char question-only",
        char_q,
    )

    print_score(
        "char evidence-only",
        char_e,
    )

    print_score(
        "char additive question + evidence",
        char_additive,
    )

    print_score(
        "char raw rendered input",
        char_input,
    )


    # These are additional conservative probes rather than the
    # thresholds written in the V5 contract.

    for name, score in (

        (
            "char question-only",
            char_q,
        ),

        (
            "char evidence-only",
            char_e,
        ),

        (
            "char additive",
            char_additive,
        ),

        (
            "char raw input",
            char_input,
        ),
    ):

        if score > EXTRA_LEXICAL_WARNING:

            warn(
                f"{name} is above "
                f"{EXTRA_LEXICAL_WARNING:.2f}: "
                f"{score:.4f}"
            )


    # ================================================================
    # LABEL DISTRIBUTIONS
    # ================================================================

    print()
    print(
        "=" * 88
    )

    print(
        "LABEL DISTRIBUTIONS"
    )

    print(
        "=" * 88
    )


    print(
        "TRAIN:",
        dict(
            sorted(
                Counter(
                    train[
                        "target"
                    ]
                ).items()
            )
        ),
    )

    print(
        "DEV:",
        dict(
            sorted(
                Counter(
                    dev[
                        "target"
                    ]
                ).items()
            )
        ),
    )


    # ================================================================
    # VERDICT
    # ================================================================

    print()
    print(
        "=" * 88
    )

    print(
        "SHORTCUT AUDIT VERDICT"
    )

    print(
        "=" * 88
    )


    if WARNINGS:

        print()
        print(
            "WARNINGS:"
        )

        for item in WARNINGS:

            print(
                "  -",
                item,
            )


    if FAILURES:

        print()
        print(
            "FAILURES:",
            len(
                FAILURES
            ),
        )

        for item in FAILURES:

            print(
                "  -",
                item,
            )

        print()
        print(
            "V5 SHORTCUT AUDIT: FAIL"
        )

        print(
            "DO NOT TRAIN."
        )

        sys.exit(
            1
        )


    print()
    print(
        "FAILURES: 0"
    )

    print(
        "V5 SHORTCUT AUDIT: PASS"
    )

    print(
        "Dataset remains unfrozen."
    )

    print(
        "No Sentinel model training performed."
    )


if __name__ == "__main__":
    main()
