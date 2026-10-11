from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import sys

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


BASE = Path("/workspace")

PILOT = (
    BASE
    / "training/v5/pilot/"
      "sentinel_v5_relational_pilot.jsonl"
)

CHALLENGE_V2 = (
    BASE
    / "experiments/challenge_v2/"
      "sentinel_challenge_v2.json"
)

EXPECTED_CHALLENGE_SHA = (
    "30eb37ff39382f0d205adb7c17383b240"
    "25601ea3db1dd61f22b715af6c1e482"
)

LABELS = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
}

EXPECTED_OPERATORS = {
    "operative_rule_selection",
    "prohibition_short_circuit",
    "permission_condition_ledger",
    "exception_activation_chain",
    "authority_state_resolution",
    "category_rule_composition",
    "scope_destination_resolution",
    "evidence_relevance_boundary",
}

ERRORS = []


def fail(message):
    ERRORS.append(
        message
    )


def norm(text):

    return re.sub(
        r"\s+",
        " ",
        str(text).strip().lower(),
    )


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

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


def evidence_ids(text):

    return set(
        re.findall(
            r"\[(E\d+)\]",
            text,
        )
    )


def tfidf_probe(
    frame,
    fields,
    name,
):

    transformers = []

    for field in fields:

        transformers.append(
            (
                field,
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
                field,
            )
        )


    pipeline = Pipeline([
        (
            "features",
            ColumnTransformer(
                transformers
            ),
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                random_state=20260816,
            ),
        ),
    ])


    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=20260816,
    )


    scores = cross_val_score(
        pipeline,
        frame[
            [
                "question",
                "evidence",
            ]
        ],
        frame[
            "label"
        ],
        cv=cv,
        scoring="accuracy",
    )


    mean = float(
        scores.mean()
    )


    print(
        f"{name:34s} "
        f"mean={mean:.4f} "
        "folds=["
        + ", ".join(
            f"{score:.4f}"
            for score in scores
        )
        + "]"
    )


    return mean


def main():

    print("=" * 88)
    print(
        "SENTINEL V5 RELATIONAL PILOT — AUDIT"
    )
    print("=" * 88)


    # ========================================================
    # EXTERNAL BENCHMARK SAFETY
    # ========================================================

    challenge_sha = sha256(
        CHALLENGE_V2
    )

    print(
        "Challenge V2 SHA:",
        challenge_sha,
    )


    if (
        challenge_sha
        != EXPECTED_CHALLENGE_SHA
    ):

        fail(
            "Challenge V2 hash mismatch."
        )


    # ========================================================
    # LOAD PILOT
    # ========================================================

    rows = []

    with PILOT.open() as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(
                    json.loads(
                        line
                    )
                )


    print()
    print(
        "rows:",
        len(
            rows
        ),
    )


    if len(
        rows
    ) != 256:

        fail(
            f"Expected 256 rows, got {len(rows)}."
        )


    labels = Counter(
        row[
            "target"
        ]
        for row in rows
    )

    print(
        "labels:",
        dict(
            sorted(
                labels.items()
            )
        ),
    )


    if labels != Counter({
        label: 64
        for label in LABELS
    }):

        fail(
            "Pilot labels are not exactly balanced."
        )


    operators = Counter(
        row[
            "operator"
        ]
        for row in rows
    )


    print(
        "operators:",
        dict(
            sorted(
                operators.items()
            )
        ),
    )


    if set(
        operators
    ) != EXPECTED_OPERATORS:

        fail(
            "Operator identity mismatch."
        )


    if any(
        count != 32
        for count
        in operators.values()
    ):

        fail(
            "Every operator must contain exactly 32 rows."
        )


    orders = Counter(
        row[
            "input_order"
        ]
        for row in rows
    )


    print(
        "input order:",
        dict(
            sorted(
                orders.items()
            )
        ),
    )


    if orders != Counter({
        "question_first": 128,
        "evidence_first": 128,
    }):

        fail(
            "Input order must be exactly 128/128."
        )


    # ========================================================
    # DUPLICATES / AMBIGUITY
    # ========================================================

    ids = [
        row[
            "scenario_id"
        ]
        for row in rows
    ]


    duplicate_ids = (
        len(ids)
        - len(
            set(
                ids
            )
        )
    )


    pair_counts = Counter()

    pair_labels = defaultdict(
        set
    )


    for row in rows:

        pair = (
            norm(
                row[
                    "question"
                ]
            )
            + "\n<PAIR>\n"
            + norm(
                row[
                    "evidence"
                ]
            )
        )

        pair_counts[
            pair
        ] += 1

        pair_labels[
            pair
        ].add(
            row[
                "target"
            ]
        )


    duplicate_pairs = sum(
        value - 1
        for value
        in pair_counts.values()
        if value > 1
    )


    ambiguous_pairs = sum(
        1
        for values
        in pair_labels.values()
        if len(
            values
        ) > 1
    )


    print()
    print(
        "duplicate IDs:",
        duplicate_ids,
    )

    print(
        "duplicate question/evidence pairs:",
        duplicate_pairs,
    )

    print(
        "ambiguous identical pairs:",
        ambiguous_pairs,
    )


    if duplicate_ids:
        fail(
            "Duplicate scenario IDs."
        )

    if duplicate_pairs:
        fail(
            "Duplicate question/evidence pairs."
        )

    if ambiguous_pairs:
        fail(
            "Ambiguous identical question/evidence pairs."
        )


    # ========================================================
    # RELATIONAL COUNTERFACTUALS
    # ========================================================

    question_map = defaultdict(
        set
    )

    evidence_map = defaultdict(
        set
    )


    for row in rows:

        question_key = (
            row[
                "operator"
            ],
            norm(
                row[
                    "question"
                ]
            ),
        )

        question_map[
            question_key
        ].add(
            row[
                "target"
            ]
        )


        evidence_key = (
            row[
                "operator"
            ],
            row[
                "evidence_hash"
            ],
        )

        evidence_map[
            evidence_key
        ].add(
            row[
                "target"
            ]
        )


    question_failures = sum(
        1
        for values
        in question_map.values()
        if values != LABELS
    )


    evidence_failures = sum(
        1
        for values
        in evidence_map.values()
        if values != LABELS
    )


    print()
    print(
        "unique question identities:",
        len(
            question_map
        ),
    )

    print(
        "question counterfactual failures:",
        question_failures,
    )

    print(
        "unique evidence worlds:",
        len(
            evidence_map
        ),
    )

    print(
        "evidence counterfactual failures:",
        evidence_failures,
    )


    # 8 operators × 4 targets × 2 surfaces.
    if len(
        question_map
    ) != 64:

        fail(
            f"Expected 64 question identities, "
            f"got {len(question_map)}."
        )


    # 8 operators × 4 evidence worlds.
    if len(
        evidence_map
    ) != 32:

        fail(
            f"Expected 32 evidence identities, "
            f"got {len(evidence_map)}."
        )


    if question_failures:

        fail(
            f"{question_failures} question identities "
            "do not map to all four labels."
        )


    if evidence_failures:

        fail(
            f"{evidence_failures} evidence worlds "
            "do not map to all four labels."
        )


    # ========================================================
    # OUTPUT / CITATION CONTRACT
    # ========================================================

    output_errors = 0
    citation_errors = 0
    condition_errors = 0

    citation_cardinality = Counter()

    operator_multi = Counter()


    for row in rows:

        try:

            output = json.loads(
                row[
                    "output"
                ]
            )

        except Exception:

            output_errors += 1

            continue


        required = {
            "decision",
            "citations",
            "conditions",
            "rationale",
        }


        if set(
            output
        ) != required:

            output_errors += 1

            continue


        if (
            output[
                "decision"
            ]
            != row[
                "target"
            ]
        ):

            output_errors += 1


        citations = output[
            "citations"
        ]


        if not isinstance(
            citations,
            list,
        ) or not citations:

            citation_errors += 1

            continue


        known = evidence_ids(
            row[
                "evidence"
            ]
        )


        if any(
            citation not in known
            for citation in citations
        ):

            citation_errors += 1


        if len(
            citations
        ) != len(
            set(
                citations
            )
        ):

            citation_errors += 1


        citation_cardinality[
            len(
                citations
            )
        ] += 1


        if len(
            citations
        ) >= 2:

            operator_multi[
                row[
                    "operator"
                ]
            ] += 1


        conditions = output[
            "conditions"
        ]


        if (
            row[
                "target"
            ]
            == "CONDITIONAL"
        ):

            if (
                not isinstance(
                    conditions,
                    list,
                )
                or not conditions
            ):

                condition_errors += 1

        else:

            if conditions != []:

                condition_errors += 1


    print()
    print(
        "output-contract errors:",
        output_errors,
    )

    print(
        "citation errors:",
        citation_errors,
    )

    print(
        "condition errors:",
        condition_errors,
    )

    print(
        "citation cardinality:",
        dict(
            sorted(
                citation_cardinality.items()
            )
        ),
    )

    print(
        "multi-citation rows/operator:",
        dict(
            sorted(
                operator_multi.items()
            )
        ),
    )


    if output_errors:
        fail(
            f"{output_errors} output-contract errors."
        )

    if citation_errors:
        fail(
            f"{citation_errors} citation errors."
        )

    if condition_errors:
        fail(
            f"{condition_errors} condition errors."
        )


    multi_total = sum(
        count
        for cardinality, count
        in citation_cardinality.items()
        if cardinality >= 2
    )


    if multi_total < 64:

        fail(
            "Pilot has too few multi-citation rows."
        )


    # --------------------------------------------------------
    # Multi-citation coverage
    # --------------------------------------------------------
    #
    # Multi-citation examples should appear broadly across the
    # pilot, but they are NOT required for operators whose selected
    # pilot family is intentionally atomic.
    #
    # A direct controlling prohibition may legitimately require
    # only one governing citation.
    #
    # Likewise, an evidence-relevance boundary can be established
    # by one governing/non-governing channel rule.
    #
    # Requiring artificial extra citations in those cases would
    # conflict with the V5 minimal-citation objective.

    naturally_atomic_pilot_operators = {
        "prohibition_short_circuit",
        "evidence_relevance_boundary",
    }

    required_multi_operators = (
        EXPECTED_OPERATORS
        - naturally_atomic_pilot_operators
    )

    observed_multi_operators = set(
        operator_multi
    )

    missing_required_multi = (
        required_multi_operators
        - observed_multi_operators
    )

    print(
        "naturally atomic pilot operators:",
        sorted(
            naturally_atomic_pilot_operators
        ),
    )

    print(
        "required multi-citation operators:",
        sorted(
            required_multi_operators
        ),
    )

    print(
        "observed multi-citation operators:",
        sorted(
            observed_multi_operators
        ),
    )

    if missing_required_multi:

        fail(
            "Required operators missing multi-citation examples: "
            + str(
                sorted(
                    missing_required_multi
                )
            )
        )

    # The pilot must still demonstrate both atomic and compositional
    # citation behavior.

    if citation_cardinality.get(
        1,
        0,
    ) == 0:

        fail(
            "Pilot contains no single-citation examples."
        )

    if sum(
        count
        for cardinality, count
        in citation_cardinality.items()
        if cardinality >= 2
    ) == 0:

        fail(
            "Pilot contains no multi-citation examples."
        )


    # ========================================================
    # METADATA BALANCE
    # ========================================================

    order_label = defaultdict(
        Counter
    )


    for row in rows:

        order_label[
            row[
                "input_order"
            ]
        ][
            row[
                "target"
            ]
        ] += 1


    print()
    print(
        "label by input order:"
    )


    for order in sorted(
        order_label
    ):

        print(
            " ",
            order,
            dict(
                sorted(
                    order_label[
                        order
                    ].items()
                )
            ),
        )


    for order in [
        "question_first",
        "evidence_first",
    ]:

        if order_label[
            order
        ] != Counter({
            label: 32
            for label in LABELS
        }):

            fail(
                f"{order} is correlated with label."
            )


    # ========================================================
    # TF-IDF SHORTCUT PROBES
    # ========================================================

    print()
    print("=" * 88)
    print(
        "TF-IDF SHORTCUT PROBES"
    )
    print("=" * 88)


    frame = pd.DataFrame({
        "question": [
            row[
                "question"
            ]
            for row in rows
        ],

        "evidence": [
            row[
                "evidence"
            ]
            for row in rows
        ],

        "label": [
            row[
                "target"
            ]
            for row in rows
        ],
    })


    q_score = tfidf_probe(
        frame,
        [
            "question",
        ],
        "question-only",
    )


    e_score = tfidf_probe(
        frame,
        [
            "evidence",
        ],
        "evidence-only",
    )


    joint_score = tfidf_probe(
        frame,
        [
            "question",
            "evidence",
        ],
        "additive question+evidence",
    )


    if q_score > 0.30:

        fail(
            f"Question-only shortcut too high: "
            f"{q_score:.4f}"
        )


    if e_score > 0.30:

        fail(
            f"Evidence-only shortcut too high: "
            f"{e_score:.4f}"
        )


    if joint_score > 0.35:

        fail(
            f"Additive shortcut too high: "
            f"{joint_score:.4f}"
        )


    # ========================================================
    # SOURCE SAFETY
    # ========================================================

    source = (
        BASE
        / "training/v5/pilot/"
          "build_v5_relational_pilot.py"
    ).read_text()


    dangerous = [
        token
        for token in [
            "sentinel_challenge_v2.json",
            "CHALLENGE_V2",
            "FROZEN_SHA256",
        ]
        if token in source
    ]


    print()
    print(
        "builder Challenge V2 references:",
        dangerous,
    )


    if dangerous:

        fail(
            "Pilot builder references Challenge V2."
        )


    # ========================================================
    # VERDICT
    # ========================================================

    print()
    print("=" * 88)
    print(
        "V5 PILOT AUDIT VERDICT"
    )
    print("=" * 88)


    if ERRORS:

        print(
            "FAILURES:",
            len(
                ERRORS
            ),
        )


        for error in ERRORS:

            print(
                "FAIL:",
                error,
            )


        print()
        print(
            "V5 RELATIONAL PILOT: FAIL"
        )

        sys.exit(1)


    print(
        "FAILURES: 0"
    )

    print()
    print(
        "V5 RELATIONAL PILOT: PASS"
    )

    print(
        "Generalized relational engine preserved "
        "dual-counterfactual balance across all 8 operators."
    )

    print(
        "No model training has occurred."
    )


if __name__ == "__main__":
    main()
