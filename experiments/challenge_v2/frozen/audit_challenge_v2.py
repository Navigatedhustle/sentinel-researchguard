from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import re
import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


BASE = Path("/workspace")

CHALLENGE = (
    BASE
    / "experiments/challenge_v2/"
      "sentinel_challenge_v2.json"
)

PRIOR_DATASETS = [
    (
        "v43_train",
        BASE
        / "training/datasets/"
          "sentinel_train_v43.jsonl",
    ),
    (
        "v44_augmented",
        BASE
        / "training/datasets/"
          "sentinel_train_v44_augmented.jsonl",
    ),
    (
        "v44_gate_train",
        BASE
        / "training/datasets/"
          "sentinel_train_v44_gate_pilot.jsonl",
    ),
]

CHALLENGE_V1 = (
    BASE
    / "experiments/challenge_v1/"
      "sentinel_challenge_v1.json"
)

LABELS = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
}

EXPECTED_COMPOSITIONS = {
    "specific_rule_precedence",
    "superseding_rule_precedence",
    "exception_activation",
    "nested_exception_requirements",
    "delegated_authority_validity",
    "temporal_authorization",
    "definition_then_authorization",
    "purpose_scope_authorization",
    "jurisdiction_then_requirement",
    "prohibition_vs_unresolved_fact",
    "permission_with_material_condition",
    "evidence_relevance_sufficiency",
}

ERRORS = []
WARNINGS = []


def fail(message):
    ERRORS.append(message)


def warn(message):
    WARNINGS.append(message)


def norm(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).strip().lower(),
    )


def strip_eids(text):
    return re.sub(
        r"\[e\d+\]",
        "[e]",
        norm(text),
        flags=re.I,
    )


def canonical_pair(question, evidence):
    return (
        norm(question)
        + "\n<PAIR>\n"
        + strip_eids(evidence)
    )


def split_input(row):
    """
    Recover question/evidence across all known Sentinel dataset formats.

    Supported:

    Challenge V2:
        row["question"]
        row["evidence_text"]

    V4.4 gate pilot:
        row["question"]
        row["evidence"]

    V4.3 relational:
        SUPPLIED EVIDENCE

        ...

        QUESTION

        ...

    Challenge V1 / older explicit format:
        QUESTION:
        ...

        EVIDENCE:
        ...

    and the reverse ordering.
    """

    # --------------------------------------------------------
    # Explicit structured fields.
    # --------------------------------------------------------

    question = row.get(
        "question"
    )

    if isinstance(
        question,
        str,
    ) and question.strip():

        evidence = row.get(
            "evidence_text"
        )

        if not (
            isinstance(
                evidence,
                str,
            )
            and evidence.strip()
        ):
            evidence = row.get(
                "evidence"
            )

        if (
            isinstance(
                evidence,
                str,
            )
            and evidence.strip()
        ):
            return (
                question.strip(),
                evidence.strip(),
            )

    # --------------------------------------------------------
    # Serialized input.
    # --------------------------------------------------------

    text = row.get(
        "input",
        ""
    )

    if not isinstance(
        text,
        str,
    ):
        return "", ""

    text = text.strip()

    # --------------------------------------------------------
    # V4.3 relational format:
    #
    # SUPPLIED EVIDENCE
    #
    # ...
    #
    # QUESTION
    #
    # ...
    # --------------------------------------------------------

    v43_header = (
        "SUPPLIED EVIDENCE\n"
    )

    if text.startswith(
        v43_header
    ):

        body = text[
            len(v43_header):
        ].lstrip("\n")

        marker = (
            "\n\nQUESTION\n"
        )

        if marker in body:

            evidence, question = (
                body.rsplit(
                    marker,
                    1,
                )
            )

            return (
                question.strip(),
                evidence.strip(),
            )

        # Defensive fallback in case blank-line
        # normalization differs.
        marker = "\nQUESTION\n"

        if marker in body:

            evidence, question = (
                body.rsplit(
                    marker,
                    1,
                )
            )

            return (
                question.strip(),
                evidence.strip(),
            )

    # --------------------------------------------------------
    # Challenge V1 / explicit QUESTION:/EVIDENCE: format.
    # --------------------------------------------------------

    q_marker = (
        "QUESTION:\n"
    )

    e_marker = (
        "EVIDENCE:\n"
    )

    if text.startswith(
        q_marker
    ):

        body = text[
            len(q_marker):
        ]

        marker = (
            "\n\nEVIDENCE:\n"
        )

        if marker not in body:
            return "", ""

        question, evidence = (
            body.split(
                marker,
                1,
            )
        )

        return (
            question.strip(),
            evidence.strip(),
        )

    if text.startswith(
        e_marker
    ):

        body = text[
            len(e_marker):
        ]

        marker = (
            "\n\nQUESTION:\n"
        )

        if marker not in body:
            return "", ""

        evidence, question = (
            body.rsplit(
                marker,
                1,
            )
        )

        return (
            question.strip(),
            evidence.strip(),
        )

    return "", ""

def load_jsonl(path):
    rows = []

    if not path.exists():
        fail(
            f"Missing prior dataset: {path}"
        )
        return rows

    with path.open() as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def load_json(path):
    if not path.exists():
        fail(
            f"Missing JSON file: {path}"
        )
        return []

    obj = json.loads(
        path.read_text()
    )

    if not isinstance(obj, list):
        fail(
            f"Expected list in {path}"
        )
        return []

    return obj


def evidence_ids(text):
    return set(
        re.findall(
            r"\[(E\d+)\]",
            text,
        )
    )


def citation_text_map(text):
    """
    Challenge V2 passages are serialized as:

    [E1] text

    [E2] text
    """

    pattern = re.compile(
        r"\[(E\d+)\]\s*(.*?)"
        r"(?=\n\n\[E\d+\]|\Z)",
        flags=re.S,
    )

    return {
        eid: body.strip()
        for eid, body
        in pattern.findall(text)
    }


def print_counter(title, values):
    print(title)

    for key, value in sorted(
        Counter(values).items(),
        key=lambda kv: str(kv[0]),
    ):
        print(
            f"  {str(key):40s} {value}"
        )


# ============================================================
# TF-IDF PROBES
# ============================================================


def tfidf_probe(
    frame,
    columns,
    name,
):
    """
    Fit vectorizers only inside each CV training fold.
    """

    if columns == ["question"]:

        features = ColumnTransformer([
            (
                "question",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
                "question",
            ),
        ])

    elif columns == ["evidence_text"]:

        features = ColumnTransformer([
            (
                "evidence",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
                "evidence_text",
            ),
        ])

    elif columns == [
        "question",
        "evidence_text",
    ]:

        features = ColumnTransformer([
            (
                "question",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
                "question",
            ),
            (
                "evidence",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
                "evidence_text",
            ),
        ])

    else:
        raise ValueError(columns)

    model = Pipeline([
        (
            "features",
            features,
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
        model,
        frame[
            [
                "question",
                "evidence_text",
            ]
        ],
        frame["label"],
        cv=cv,
        scoring="accuracy",
    )

    print(
        f"{name:34s} "
        f"mean={scores.mean():.4f} "
        f"folds="
        + "["
        + ", ".join(
            f"{x:.4f}"
            for x in scores
        )
        + "]"
    )

    return float(
        scores.mean()
    )


# ============================================================
# SIMILARITY
# ============================================================


def cross_similarity(
    left,
    right,
):
    if not left or not right:
        return 0.0

    vect = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )

    matrix = vect.fit_transform(
        list(left)
        + list(right)
    )

    a = matrix[
        : len(left)
    ]

    b = matrix[
        len(left):
    ]

    sim = (
        a
        @ b.T
    )

    return float(
        sim.max()
    )


def internal_cross_composition_similarity(
    rows,
):
    texts = [
        canonical_pair(
            row["question"],
            row["evidence_text"],
        )
        for row in rows
    ]

    vect = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )

    matrix = vect.fit_transform(
        texts
    )

    sim = (
        matrix
        @ matrix.T
    ).toarray()

    maximum = 0.0
    above_90 = 0

    for i in range(
        len(rows)
    ):

        for j in range(
            i + 1,
            len(rows),
        ):

            if (
                rows[i]["composition"]
                == rows[j]["composition"]
            ):
                # High similarity is expected within a
                # counterfactual relational family.
                continue

            value = float(
                sim[i, j]
            )

            maximum = max(
                maximum,
                value,
            )

            if value >= 0.90:
                above_90 += 1

    return (
        maximum,
        above_90,
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 88)
    print(
        "SENTINEL CHALLENGE V2 — ADVERSARIAL AUDIT"
    )
    print("=" * 88)

    rows = load_json(
        CHALLENGE
    )

    if not rows:
        fail(
            "Challenge V2 did not load."
        )

        finish()
        return


    # ========================================================
    # BASIC STRUCTURE
    # ========================================================

    print()
    print("=" * 88)
    print("1. BASIC STRUCTURE")
    print("=" * 88)

    print(
        "cases:",
        len(rows),
    )

    if len(rows) != 240:
        fail(
            f"Expected 240 rows, got {len(rows)}"
        )

    labels = Counter(
        row.get(
            "gold",
            {},
        ).get(
            "decision"
        )
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

    expected_labels = Counter({
        label: 60
        for label in LABELS
    })

    if labels != expected_labels:
        fail(
            f"Label imbalance: {labels}"
        )

    compositions = Counter(
        row.get(
            "composition"
        )
        for row in rows
    )

    print(
        "compositions:",
        dict(
            sorted(
                compositions.items()
            )
        ),
    )

    if (
        set(compositions)
        != EXPECTED_COMPOSITIONS
    ):
        fail(
            "Composition identity mismatch."
        )

    if any(
        count != 20
        for count in compositions.values()
    ):
        fail(
            "Every composition must contain 20 rows."
        )

    orders = Counter(
        row.get(
            "input_order"
        )
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
        "question_first": 120,
        "evidence_first": 120,
    }):
        fail(
            "Input order is not exactly 120/120."
        )

    evidence_counts = Counter(
        row.get(
            "evidence_count"
        )
        for row in rows
    )

    print(
        "evidence counts:",
        dict(
            sorted(
                evidence_counts.items()
            )
        ),
    )

    if evidence_counts != Counter({
        1: 24,
        2: 72,
        3: 72,
        4: 48,
        5: 24,
    }):
        fail(
            "Evidence-count schedule mismatch."
        )

    question_types = Counter(
        row.get(
            "question_type"
        )
        for row in rows
    )

    print(
        "question types:",
        dict(
            sorted(
                question_types.items()
            )
        ),
    )

    if question_types != Counter({
        "relational_counterfactual": 192,
        "stress": 48,
    }):
        fail(
            "Question-type schedule mismatch."
        )


    # ========================================================
    # CASE IDS / PAIRS
    # ========================================================

    print()
    print("=" * 88)
    print("2. IDENTITY / DUPLICATE AUDIT")
    print("=" * 88)

    ids = [
        row.get(
            "case_id"
        )
        for row in rows
    ]

    duplicate_ids = (
        len(ids)
        - len(set(ids))
    )

    print(
        "duplicate case IDs:",
        duplicate_ids,
    )

    if duplicate_ids:
        fail(
            f"{duplicate_ids} duplicate case IDs."
        )

    pair_labels = defaultdict(
        set
    )

    pair_counts = Counter()

    for row in rows:

        pair = canonical_pair(
            row.get(
                "question",
                ""
            ),
            row.get(
                "evidence_text",
                ""
            ),
        )

        pair_counts[
            pair
        ] += 1

        pair_labels[
            pair
        ].add(
            row["gold"]["decision"]
        )

    duplicate_pairs = sum(
        count - 1
        for count
        in pair_counts.values()
        if count > 1
    )

    ambiguous_pairs = sum(
        1
        for labels_
        in pair_labels.values()
        if len(labels_) > 1
    )

    print(
        "duplicate normalized pairs:",
        duplicate_pairs,
    )

    print(
        "ambiguous normalized pairs:",
        ambiguous_pairs,
    )

    if duplicate_pairs:
        fail(
            f"{duplicate_pairs} duplicate normalized pairs."
        )

    if ambiguous_pairs:
        fail(
            f"{ambiguous_pairs} ambiguous normalized pairs."
        )


    # ========================================================
    # INPUT SERIALIZATION
    # ========================================================

    print()
    print("=" * 88)
    print("3. INPUT SERIALIZATION")
    print("=" * 88)

    serialization_errors = 0

    for row in rows:

        question, evidence = split_input(
            row
        )

        if (
            norm(question)
            != norm(
                row.get(
                    "question",
                    ""
                )
            )
        ):
            serialization_errors += 1

        if (
            norm(evidence)
            != norm(
                row.get(
                    "evidence_text",
                    ""
                )
            )
        ):
            serialization_errors += 1

        text = row.get(
            "input",
            ""
        )

        order = row.get(
            "input_order"
        )

        if (
            order == "question_first"
            and not text.startswith(
                "QUESTION:\n"
            )
        ):
            serialization_errors += 1

        if (
            order == "evidence_first"
            and not text.startswith(
                "EVIDENCE:\n"
            )
        ):
            serialization_errors += 1

    print(
        "serialization errors:",
        serialization_errors,
    )

    if serialization_errors:
        fail(
            f"{serialization_errors} input serialization errors."
        )


    # ========================================================
    # GOLD / CITATION SCHEMA
    # ========================================================

    print()
    print("=" * 88)
    print("4. GOLD / CITATION AUDIT")
    print("=" * 88)

    schema_errors = 0
    citation_errors = 0
    citation_cardinality = Counter()
    sufficiency_errors = 0
    condition_errors = 0

    for row in rows:

        gold = row.get(
            "gold"
        )

        if not isinstance(
            gold,
            dict,
        ):
            schema_errors += 1
            continue

        required = {
            "decision",
            "citations",
            "conditions",
            "answer",
            "evidence_sufficient",
        }

        if not required.issubset(
            gold
        ):
            schema_errors += 1
            continue

        label = gold[
            "decision"
        ]

        if label not in LABELS:
            schema_errors += 1

        if not isinstance(
            gold["citations"],
            list,
        ):
            schema_errors += 1
            continue

        if not isinstance(
            gold["conditions"],
            list,
        ):
            schema_errors += 1

        if not isinstance(
            gold["answer"],
            str,
        ) or not gold[
            "answer"
        ].strip():
            schema_errors += 1

        if not isinstance(
            gold[
                "evidence_sufficient"
            ],
            bool,
        ):
            schema_errors += 1

        expected_sufficient = (
            label != "INSUFFICIENT"
        )

        if (
            gold[
                "evidence_sufficient"
            ]
            != expected_sufficient
        ):
            sufficiency_errors += 1

        if (
            label == "CONDITIONAL"
            and len(
                gold["conditions"]
            ) == 0
        ):
            condition_errors += 1

        if (
            label in {
                "NO",
                "INSUFFICIENT",
            }
            and len(
                gold["conditions"]
            ) != 0
        ):
            condition_errors += 1

        known_ids = evidence_ids(
            row.get(
                "evidence_text",
                ""
            )
        )

        citation_map = citation_text_map(
            row.get(
                "evidence_text",
                ""
            )
        )

        citations = gold[
            "citations"
        ]

        citation_cardinality[
            len(citations)
        ] += 1

        if (
            not citations
            or len(citations)
            != len(
                set(citations)
            )
        ):
            citation_errors += 1

        for citation in citations:

            if citation not in known_ids:
                citation_errors += 1

            if (
                citation not in citation_map
                or not citation_map[
                    citation
                ].strip()
            ):
                citation_errors += 1

    print(
        "schema errors:",
        schema_errors,
    )

    print(
        "citation errors:",
        citation_errors,
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
        "sufficiency errors:",
        sufficiency_errors,
    )

    print(
        "condition errors:",
        condition_errors,
    )

    if schema_errors:
        fail(
            f"{schema_errors} gold schema errors."
        )

    if citation_errors:
        fail(
            f"{citation_errors} citation errors."
        )

    if sufficiency_errors:
        fail(
            f"{sufficiency_errors} sufficiency errors."
        )

    if condition_errors:
        fail(
            f"{condition_errors} condition-state errors."
        )


    # ========================================================
    # RELATIONAL CAUSALITY
    # ========================================================

    print()
    print("=" * 88)
    print("5. DUAL-COUNTERFACTUAL CAUSALITY")
    print("=" * 88)

    relational = [
        row
        for row in rows
        if (
            row.get(
                "question_type"
            )
            == "relational_counterfactual"
        )
    ]

    groups = defaultdict(
        list
    )

    for row in relational:

        groups[
            row.get(
                "relational_group"
            )
        ].append(
            row
        )

    group_errors = 0
    question_counterfactual_errors = 0
    evidence_counterfactual_errors = 0

    for group_name, group_rows in groups.items():

        if len(
            group_rows
        ) != 16:
            group_errors += 1

        questions = defaultdict(
            set
        )

        packs = defaultdict(
            set
        )

        for row in group_rows:

            questions[
                norm(
                    row["question"]
                )
            ].add(
                row[
                    "gold"
                ][
                    "decision"
                ]
            )

            packs[
                row[
                    "policy_pack"
                ]
            ].add(
                row[
                    "gold"
                ][
                    "decision"
                ]
            )

        if len(
            questions
        ) != 4:
            group_errors += 1

        if len(
            packs
        ) != 4:
            group_errors += 1

        for label_set in questions.values():

            if label_set != LABELS:
                question_counterfactual_errors += 1

        for label_set in packs.values():

            if label_set != LABELS:
                evidence_counterfactual_errors += 1

    print(
        "relational groups:",
        len(groups),
    )

    print(
        "group structure errors:",
        group_errors,
    )

    print(
        "question counterfactual errors:",
        question_counterfactual_errors,
    )

    print(
        "evidence counterfactual errors:",
        evidence_counterfactual_errors,
    )

    if len(
        groups
    ) != 12:
        fail(
            f"Expected 12 relational groups, got {len(groups)}."
        )

    if group_errors:
        fail(
            f"{group_errors} relational-group structure errors."
        )

    if question_counterfactual_errors:
        fail(
            f"{question_counterfactual_errors} question counterfactual failures."
        )

    if evidence_counterfactual_errors:
        fail(
            f"{evidence_counterfactual_errors} evidence counterfactual failures."
        )


    # ========================================================
    # SHORTCUT PROBES
    # ========================================================

    print()
    print("=" * 88)
    print("6. TF-IDF SHORTCUT PROBES")
    print("=" * 88)

    frame = pd.DataFrame({
        "question": [
            row["question"]
            for row in rows
        ],
        "evidence_text": [
            row["evidence_text"]
            for row in rows
        ],
        "label": [
            row[
                "gold"
            ][
                "decision"
            ]
            for row in rows
        ],
    })

    rel_frame = pd.DataFrame({
        "question": [
            row["question"]
            for row in relational
        ],
        "evidence_text": [
            row["evidence_text"]
            for row in relational
        ],
        "label": [
            row[
                "gold"
            ][
                "decision"
            ]
            for row in relational
        ],
    })

    print(
        "FULL 240-CASE BENCHMARK"
    )

    q_full = tfidf_probe(
        frame,
        ["question"],
        "question-only",
    )

    e_full = tfidf_probe(
        frame,
        ["evidence_text"],
        "evidence-only",
    )

    joint_full = tfidf_probe(
        frame,
        [
            "question",
            "evidence_text",
        ],
        "additive question+evidence",
    )

    print()
    print(
        "RELATIONAL 192-CASE SUBSET"
    )

    q_rel = tfidf_probe(
        rel_frame,
        ["question"],
        "question-only relational",
    )

    e_rel = tfidf_probe(
        rel_frame,
        ["evidence_text"],
        "evidence-only relational",
    )

    joint_rel = tfidf_probe(
        rel_frame,
        [
            "question",
            "evidence_text",
        ],
        "additive relational",
    )

    if q_full > 0.30:
        fail(
            f"Question-only TF-IDF too high: {q_full:.4f} > 0.30"
        )

    if e_full > 0.30:
        fail(
            f"Evidence-only TF-IDF too high: {e_full:.4f} > 0.30"
        )

    if joint_full > 0.35:
        fail(
            f"Additive TF-IDF too high: {joint_full:.4f} > 0.35"
        )

    # Relational core should be almost exactly chance.
    if q_rel > 0.30:
        fail(
            f"Relational question-only shortcut: {q_rel:.4f}"
        )

    if e_rel > 0.30:
        fail(
            f"Relational evidence-only shortcut: {e_rel:.4f}"
        )

    if joint_rel > 0.35:
        fail(
            f"Relational additive shortcut: {joint_rel:.4f}"
        )


    # ========================================================
    # INTERNAL NEAR-DUPLICATES
    # ========================================================

    print()
    print("=" * 88)
    print("7. INTERNAL NEAR-DUPLICATE AUDIT")
    print("=" * 88)

    (
        internal_max,
        internal_above_90,
    ) = internal_cross_composition_similarity(
        rows
    )

    print(
        "max cross-composition TF-IDF cosine:",
        f"{internal_max:.4f}",
    )

    print(
        "cross-composition pairs >= 0.90:",
        internal_above_90,
    )

    if internal_above_90:
        fail(
            f"{internal_above_90} cross-composition near-duplicate pairs >= 0.90"
        )


    # ========================================================
    # PRIOR TRAINING CONTAMINATION
    # ========================================================

    print()
    print("=" * 88)
    print("8. PRIOR-TRAINING CONTAMINATION")
    print("=" * 88)

    challenge_pairs = [
        canonical_pair(
            row["question"],
            row["evidence_text"],
        )
        for row in rows
    ]

    challenge_pair_set = set(
        challenge_pairs
    )

    for name, path in PRIOR_DATASETS:

        prior = load_jsonl(
            path
        )

        prior_pairs = []

        for row in prior:

            question, evidence = split_input(
                row
            )

            if question and evidence:

                prior_pairs.append(
                    canonical_pair(
                        question,
                        evidence,
                    )
                )

        if len(prior_pairs) != len(prior):

            fail(
                f"{name}: recovered only "
                f"{len(prior_pairs)}/{len(prior)} "
                "question/evidence pairs."
            )

        exact = len(
            challenge_pair_set.intersection(
                set(
                    prior_pairs
                )
            )
        )

        max_sim = cross_similarity(
            challenge_pairs,
            prior_pairs,
        )

        print(
            f"{name:20s} "
            f"rows={len(prior):4d} "
            f"usable_pairs={len(prior_pairs):4d} "
            f"exact={exact:3d} "
            f"max_cosine={max_sim:.4f}"
        )

        if exact:
            fail(
                f"{name}: {exact} exact Challenge V2 overlaps."
            )

        # High threshold is only a contamination alarm.
        # Moderate semantic analogy is expected.
        if max_sim >= 0.90:
            fail(
                f"{name}: suspicious near-copy cosine {max_sim:.4f}"
            )


    # ========================================================
    # CHALLENGE V1 CONTAMINATION
    # ========================================================

    print()
    print("=" * 88)
    print("9. CHALLENGE V1 CONTAMINATION")
    print("=" * 88)

    old = load_json(
        CHALLENGE_V1
    )

    old_pairs = []

    for row in old:

        question, evidence = split_input(
            row
        )

        if question and evidence:

            old_pairs.append(
                canonical_pair(
                    question,
                    evidence,
                )
            )

    exact_v1 = len(
        challenge_pair_set.intersection(
            set(
                old_pairs
            )
        )
    )

    v1_max = cross_similarity(
        challenge_pairs,
        old_pairs,
    )

    print(
        "Challenge V1 rows:",
        len(old),
    )

    print(
        "usable V1 pairs:",
        len(old_pairs),
    )

    print(
        "exact normalized full-case overlap:",
        exact_v1,
    )

    print(
        "max V2 <-> V1 TF-IDF cosine:",
        f"{v1_max:.4f}",
    )

    if exact_v1:
        fail(
            f"{exact_v1} exact Challenge V1 overlaps."
        )

    if v1_max >= 0.90:
        fail(
            f"Suspicious Challenge V1 near-copy cosine {v1_max:.4f}"
        )


    # ========================================================
    # LABEL / ORDER / COUNT CORRELATION
    # ========================================================

    print()
    print("=" * 88)
    print("10. SIMPLE METADATA SHORTCUTS")
    print("=" * 88)

    for field in [
        "input_order",
        "evidence_count",
        "question_type",
    ]:

        table = defaultdict(
            Counter
        )

        for row in rows:

            table[
                row[field]
            ][
                row[
                    "gold"
                ][
                    "decision"
                ]
            ] += 1

        print()
        print(
            field + ":"
        )

        for value in sorted(
            table,
            key=str,
        ):

            print(
                f"  {str(value):28s} "
                f"{dict(sorted(table[value].items()))}"
            )


    finish()


def finish():

    print()
    print("=" * 88)
    print("AUDIT VERDICT")
    print("=" * 88)

    if WARNINGS:

        print(
            f"WARNINGS: {len(WARNINGS)}"
        )

        for message in WARNINGS:
            print(
                "WARN:",
                message,
            )

    else:

        print(
            "WARNINGS: 0"
        )

    if ERRORS:

        print(
            f"FAILURES: {len(ERRORS)}"
        )

        for message in ERRORS:

            print(
                "FAIL:",
                message,
            )

        print()
        print(
            "CHALLENGE V2 STATUS: FAIL — DO NOT FREEZE"
        )

        sys.exit(1)

    print(
        "FAILURES: 0"
    )

    print()
    print(
        "CHALLENGE V2 STRUCTURAL/SHORTCUT STATUS: PASS"
    )

    print(
        "Semantic audit is still required before freeze."
    )


if __name__ == "__main__":
    main()
