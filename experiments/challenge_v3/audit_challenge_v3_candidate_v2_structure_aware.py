from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
)


# =====================================================================
# CONFIG
# =====================================================================

BASE = Path("/workspace")

CANDIDATE = (
    BASE
    / "experiments/challenge_v3/candidates/"
      "sentinel_challenge_v3_candidate_v2.json"
)

HIDDEN = (
    BASE
    / "experiments/challenge_v3/candidates/"
      "sentinel_challenge_v3_candidate_v2_hidden_semantics.jsonl"
)

EXPECTED_CANDIDATE_SHA = (
    "1bfb6c5a31a7ca0a2c069582f4b454b571ebf0c490d10e54880e8d5d086d90b0"
)

EXPECTED_HIDDEN_SHA = (
    "c3a24b4ea4e02691481b1563258ea4da0cebd0027e823cb3048e925116dc4b22"
)

LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)

LABEL_SET = set(LABELS)

SEED = 20260817

CHANCE = 0.25
HARD_THRESHOLD = 0.35

SVD_COMPONENTS = 24


# =====================================================================
# HELPERS
# =====================================================================

def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def die(message):
    raise RuntimeError(message)


def stratified_splits(y):
    y = np.asarray(y)

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    return list(
        splitter.split(
            np.zeros(len(y)),
            y,
        )
    )


def grouped_splits(y, groups):
    y = np.asarray(y)

    groups = np.asarray(
        groups,
        dtype=object,
    )

    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=SEED,
    )

    return list(
        splitter.split(
            np.zeros(len(y)),
            y,
            groups,
        )
    )


def fit_projection(train_texts, test_texts):
    """
    Character TF-IDF -> compact dense projection.

    Character n-grams are deliberately used because they preserve
    entity names, identifiers, wording fragments, and morphology
    without requiring any semantic model.
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=25000,
    )

    train_sparse = vectorizer.fit_transform(
        train_texts
    )

    test_sparse = vectorizer.transform(
        test_texts
    )

    max_components = min(
        SVD_COMPONENTS,
        train_sparse.shape[0] - 1,
        train_sparse.shape[1] - 1,
    )

    if max_components < 2:
        die(
            "Insufficient dimensionality for SVD projection."
        )

    svd = TruncatedSVD(
        n_components=max_components,
        random_state=SEED,
    )

    train_dense = svd.fit_transform(
        train_sparse
    )

    test_dense = svd.transform(
        test_sparse
    )

    return (
        train_dense,
        test_dense,
    )


def interaction_features(q, e):
    """
    Explicitly expose question x evidence interactions.

    Linear TF-IDF cannot naturally represent XOR / Latin-square
    interactions. The outer product gives a downstream classifier
    direct access to bilinear relationships.
    """

    cross = np.einsum(
        "bi,bj->bij",
        q,
        e,
    ).reshape(
        len(q),
        -1,
    )

    return np.hstack([
        q,
        e,
        cross,
    ])


def nonlinear_cv(
    questions,
    evidence,
    labels,
    splits,
):
    labels = np.asarray(
        labels
    )

    logistic_predictions = np.empty(
        len(labels),
        dtype=labels.dtype,
    )

    tree_predictions = np.empty(
        len(labels),
        dtype=labels.dtype,
    )

    for fold_index, (
        train_idx,
        test_idx,
    ) in enumerate(
        splits
    ):

        train_q = [
            questions[i]
            for i in train_idx
        ]

        test_q = [
            questions[i]
            for i in test_idx
        ]

        train_e = [
            evidence[i]
            for i in train_idx
        ]

        test_e = [
            evidence[i]
            for i in test_idx
        ]


        q_train, q_test = fit_projection(
            train_q,
            test_q,
        )

        e_train, e_test = fit_projection(
            train_e,
            test_e,
        )


        X_train = interaction_features(
            q_train,
            e_train,
        )

        X_test = interaction_features(
            q_test,
            e_test,
        )


        logistic = LogisticRegression(
            max_iter=6000,
            C=1.0,
            random_state=SEED + fold_index,
        )

        logistic.fit(
            X_train,
            labels[
                train_idx
            ],
        )

        logistic_predictions[
            test_idx
        ] = logistic.predict(
            X_test
        )


        trees = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=SEED + fold_index,
            n_jobs=-1,
        )

        trees.fit(
            X_train,
            labels[
                train_idx
            ],
        )

        tree_predictions[
            test_idx
        ] = trees.predict(
            X_test
        )


    logistic_acc = float(
        accuracy_score(
            labels,
            logistic_predictions,
        )
    )

    tree_acc = float(
        accuracy_score(
            labels,
            tree_predictions,
        )
    )

    return {
        "bilinear_logistic": logistic_acc,
        "extra_trees": tree_acc,
        "best": max(
            logistic_acc,
            tree_acc,
        ),
    }


# =====================================================================
# LOAD
# =====================================================================

print("=" * 92)
print("CHALLENGE V3 CANDIDATE V2")
print("FINAL STRUCTURE-AWARE SHORTCUT AUDIT")
print("=" * 92)

if sha256_file(
    CANDIDATE
) != EXPECTED_CANDIDATE_SHA:
    die(
        "Candidate identity mismatch."
    )

if sha256_file(
    HIDDEN
) != EXPECTED_HIDDEN_SHA:
    die(
        "Hidden identity mismatch."
    )


candidate = json.loads(
    CANDIDATE.read_text()
)

hidden_rows = []

with HIDDEN.open() as f:

    for line in f:

        line = line.strip()

        if line:

            hidden_rows.append(
                json.loads(
                    line
                )
            )


hidden_by_id = {
    row["case_id"]: row
    for row in hidden_rows
}


rows = []

for case in candidate:

    hidden = hidden_by_id[
        case["case_id"]
    ]

    rows.append({
        "case": case,
        "hidden": hidden,
        "label": case["gold"]["decision"],
    })


relational = [
    row
    for row in rows
    if row["case"]["question_type"]
    == "relational_counterfactual"
]


if len(relational) != 192:
    die(
        "Expected 192 relational cases."
    )


labels = [
    row["label"]
    for row in relational
]

questions = [
    row["case"]["question"]
    for row in relational
]

evidence = [
    row["case"]["evidence_text"]
    for row in relational
]

question_hashes = [
    row["hidden"]["question_hash"]
    for row in relational
]

evidence_hashes = [
    row["hidden"]["evidence_hash"]
    for row in relational
]

compositions = [
    row["case"]["composition"]
    for row in relational
]


print("candidate identity: VERIFIED")
print("hidden identity:    VERIFIED")
print("relational rows:    192")
print("individual cases printed: 0")


# =====================================================================
# CASE-LOCAL NONLINEAR ATTACKS
# =====================================================================

attack_specs = [
    (
        "random stratified",
        stratified_splits(
            labels
        ),
    ),
    (
        "exact-question groups held out",
        grouped_splits(
            labels,
            question_hashes,
        ),
    ),
    (
        "exact-evidence groups held out",
        grouped_splits(
            labels,
            evidence_hashes,
        ),
    ),
    (
        "composition held out",
        grouped_splits(
            labels,
            compositions,
        ),
    ),
]


results = {}


print()
print("================ CASE-LOCAL NONLINEAR ATTACKS ================")

for name, splits in attack_specs:

    result = nonlinear_cv(
        questions,
        evidence,
        labels,
        splits,
    )

    results[
        name
    ] = result

    print(name)
    print(
        "  bilinear logistic:",
        f"{result['bilinear_logistic']:.4f}",
    )
    print(
        "  ExtraTrees:",
        f"{result['extra_trees']:.4f}",
    )
    print(
        "  best:",
        f"{result['best']:.4f}",
    )


best_attack_name = max(
    results,
    key=lambda name: results[name]["best"],
)

best_attack = results[
    best_attack_name
][
    "best"
]


print()
print(
    "best nonlinear visible attack:",
    best_attack_name,
    f"acc={best_attack:.4f}",
)


# =====================================================================
# TRANSDUCTIVE LATIN-COMPLETION DIAGNOSTIC
#
# IMPORTANT:
# This is intentionally NOT a valid benchmark attack.
#
# It gives the diagnostic access to gold labels from OTHER Challenge
# V3 cases in the training folds.
#
# A production/canonical evaluation MUST NOT do this.
#
# The purpose is to measure how dangerous benchmark-wide adaptation
# would be if evaluation isolation were violated.
# =====================================================================

random_splits = stratified_splits(
    labels
)

transductive_total = 0
transductive_recovered = 0
transductive_correct = 0
transductive_conflicts = 0


for train_idx, test_idx in random_splits:

    known = {}

    all_cells = {}


    for i in range(
        len(relational)
    ):

        cell = (
            question_hashes[i],
            evidence_hashes[i],
        )

        all_cells[
            cell
        ] = i


    for i in train_idx:

        known[
            (
                question_hashes[i],
                evidence_hashes[i],
            )
        ] = labels[i]


    changed = True

    while changed:

        changed = False


        # ------------------------------------------------------------
        # Complete question rows.
        # ------------------------------------------------------------

        rows_by_question = defaultdict(
            list
        )

        for cell in all_cells:

            rows_by_question[
                cell[0]
            ].append(
                cell
            )


        for qhash, cells in rows_by_question.items():

            known_cells = [
                cell
                for cell in cells
                if cell in known
            ]

            unknown_cells = [
                cell
                for cell in cells
                if cell not in known
            ]

            if (
                len(known_cells) == 3
                and len(unknown_cells) == 1
            ):

                observed = {
                    known[cell]
                    for cell in known_cells
                }

                missing = (
                    LABEL_SET
                    - observed
                )

                if len(missing) == 1:

                    cell = unknown_cells[
                        0
                    ]

                    inferred = next(
                        iter(
                            missing
                        )
                    )

                    known[
                        cell
                    ] = inferred

                    changed = True


        # ------------------------------------------------------------
        # Complete evidence columns.
        # ------------------------------------------------------------

        columns_by_evidence = defaultdict(
            list
        )

        for cell in all_cells:

            columns_by_evidence[
                cell[1]
            ].append(
                cell
            )


        for ehash, cells in columns_by_evidence.items():

            known_cells = [
                cell
                for cell in cells
                if cell in known
            ]

            unknown_cells = [
                cell
                for cell in cells
                if cell not in known
            ]

            if (
                len(known_cells) == 3
                and len(unknown_cells) == 1
            ):

                observed = {
                    known[cell]
                    for cell in known_cells
                }

                missing = (
                    LABEL_SET
                    - observed
                )

                if len(missing) == 1:

                    cell = unknown_cells[
                        0
                    ]

                    inferred = next(
                        iter(
                            missing
                        )
                    )

                    if (
                        cell in known
                        and known[cell]
                        != inferred
                    ):

                        transductive_conflicts += 1

                    else:

                        known[
                            cell
                        ] = inferred

                        changed = True


    for i in test_idx:

        transductive_total += 1

        cell = (
            question_hashes[i],
            evidence_hashes[i],
        )

        if cell in known:

            transductive_recovered += 1

            if (
                known[
                    cell
                ]
                == labels[i]
            ):

                transductive_correct += 1


transductive_coverage = (
    transductive_recovered
    / transductive_total
)

if transductive_recovered:

    transductive_accuracy = (
        transductive_correct
        / transductive_recovered
    )

else:

    transductive_accuracy = 0.0


print()
print("================ TRANSDUCTIVE LEAKAGE DIAGNOSTIC ================")
print("INVALID evaluation assumption: other V3 gold labels are exposed.")
print()
print(
    "held-out cells:",
    transductive_total,
)
print(
    "Latin-completion recovered:",
    transductive_recovered,
)
print(
    "coverage:",
    f"{transductive_coverage:.4f}",
)
print(
    "accuracy when recovered:",
    f"{transductive_accuracy:.4f}",
)
print(
    "completion conflicts:",
    transductive_conflicts,
)
print()
print(
    "NOTE: this is NOT a hard benchmark gate."
)
print(
    "      It defines an evaluation-protocol isolation requirement."
)


# =====================================================================
# FINAL HARD-GATE DECISION
# =====================================================================

issues = []

for name, result in results.items():

    if (
        result[
            "best"
        ]
        > HARD_THRESHOLD
    ):

        issues.append(
            f"nonlinear case-local shortcut exceeds threshold: {name}"
        )


print()
print("================ FINAL HARD-GATE SUMMARY ================")
print(
    "four-class chance:",
    f"{CHANCE:.4f}",
)
print(
    "hard visible threshold:",
    f"{HARD_THRESHOLD:.4f}",
)
print(
    "best case-local nonlinear attack:",
    f"{best_attack:.4f}",
)
print(
    "hard-gate issues:",
    len(
        issues
    ),
)


if issues:

    print()
    print("FINAL STRUCTURE-AWARE AUDIT: NOT CLEAN")

    for issue in issues:

        print(
            f"  - {issue}"
        )

    print()
    print("Candidate remains NON-CANONICAL.")
    print("Candidate remains UNCONSUMED.")

    raise SystemExit(
        2
    )


print()
print("=" * 92)
print("FINAL STRUCTURE-AWARE AUDIT: PASS")
print("=" * 92)
print()
print("No case-local nonlinear visible attack exceeded 0.35.")
print()
print("Transductive completion is diagnostic only.")
print("Canonical evaluation must isolate cases and never expose")
print("gold labels from other Challenge V3 examples.")
print()
print("Candidate remains NON-CANONICAL pending explicit promotion.")
print("Candidate remains UNCONSUMED.")
print("No Sentinel model loaded.")
print("No LLM inference performed.")
print("Individual cases printed: 0")
