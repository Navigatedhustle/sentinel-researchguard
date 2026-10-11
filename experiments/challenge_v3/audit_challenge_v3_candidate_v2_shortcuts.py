from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations, permutations, product
from pathlib import Path
import hashlib
import json

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.pipeline import (
    FeatureUnion,
    Pipeline,
    make_pipeline,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# =====================================================================
# CONFIGURATION
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

CHANCE_4 = 0.25

# Four-class visible shortcut gates.
#
# 0.35 is deliberately conservative:
# materially above the 0.25 chance baseline but still low enough that
# we investigate moderate shortcut signal rather than rationalize it.
VISIBLE_CLEAN_THRESHOLD = 0.35

# Stress-only subset has only 48 examples and is diagnostic rather
# than an automatic rejection gate.
STRESS_DIAGNOSTIC_THRESHOLD = 0.40


# =====================================================================
# UTILITIES
# =====================================================================

def die(message):
    raise RuntimeError(message)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def majority(values):
    counts = Counter(values)

    return sorted(
        counts,
        key=lambda value: (
            -counts[value],
            str(value),
        ),
    )[0]


def make_text_probe(mode):
    word = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_features=30000,
    )

    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=2,
        max_features=40000,
    )

    if mode == "word":
        features = word

    elif mode == "char":
        features = char

    elif mode == "hybrid":
        features = FeatureUnion([
            ("word", word),
            ("char", char),
        ])

    else:
        raise ValueError(
            f"Unknown text-probe mode: {mode}"
        )

    return Pipeline([
        (
            "features",
            features,
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=4000,
                random_state=SEED,
            ),
        ),
    ])


def stratified_splits(
    y,
    n_splits=5,
):
    y = np.asarray(y)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    return list(
        splitter.split(
            np.zeros(
                len(y)
            ),
            y,
        )
    )


def grouped_splits(
    y,
    groups,
    n_splits=4,
):
    y = np.asarray(y)

    groups = np.asarray(
        groups,
        dtype=object,
    )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    return list(
        splitter.split(
            np.zeros(
                len(y)
            ),
            y,
            groups,
        )
    )


def text_cv(
    texts,
    y,
    splits,
    mode,
):
    texts = list(
        texts
    )

    y = np.asarray(
        y
    )

    predictions = np.empty(
        len(y),
        dtype=y.dtype,
    )

    for train_idx, test_idx in splits:

        model = make_text_probe(
            mode
        )

        model.fit(
            [
                texts[i]
                for i in train_idx
            ],
            y[
                train_idx
            ],
        )

        predictions[
            test_idx
        ] = model.predict(
            [
                texts[i]
                for i in test_idx
            ]
        )

    return float(
        accuracy_score(
            y,
            predictions,
        )
    )


def run_text_attack(
    *,
    name,
    texts,
    labels,
    splits,
):
    scores = {}

    for mode in (
        "word",
        "char",
        "hybrid",
    ):
        scores[
            mode
        ] = text_cv(
            texts,
            labels,
            splits,
            mode,
        )

    best_mode = max(
        scores,
        key=scores.get,
    )

    best = scores[
        best_mode
    ]

    print(name)
    print(
        f"  word:   {scores['word']:.4f}"
    )
    print(
        f"  char:   {scores['char']:.4f}"
    )
    print(
        f"  hybrid: {scores['hybrid']:.4f}"
    )
    print(
        f"  max:    {best:.4f} ({best_mode})"
    )

    return {
        "name": name,
        "scores": scores,
        "best_mode": best_mode,
        "best": best,
    }


# =====================================================================
# LOAD + IDENTITY
# =====================================================================

print("=" * 92)
print("SENTINEL CHALLENGE V3 CANDIDATE V2")
print("CORRECTED AGGREGATE SHORTCUT AUDIT")
print("=" * 92)

candidate_sha = sha256_file(
    CANDIDATE
)

hidden_sha = sha256_file(
    HIDDEN
)

if candidate_sha != EXPECTED_CANDIDATE_SHA:
    die(
        "Candidate SHA256 mismatch."
    )

if hidden_sha != EXPECTED_HIDDEN_SHA:
    die(
        "Hidden semantics SHA256 mismatch."
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


if len(candidate) != 240:
    die(
        f"Expected 240 candidate rows; found {len(candidate)}."
    )

if len(hidden_rows) != 240:
    die(
        f"Expected 240 hidden rows; found {len(hidden_rows)}."
    )


hidden_by_id = {
    row["case_id"]: row
    for row in hidden_rows
}

if set(
    row["case_id"]
    for row in candidate
) != set(
    hidden_by_id
):
    die(
        "Candidate/hidden case IDs differ."
    )


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


labels_all = [
    row["label"]
    for row in rows
]

if Counter(
    labels_all
) != Counter({
    label: 60
    for label in LABELS
}):
    die(
        "Overall labels are not balanced."
    )


relational = [
    row
    for row in rows
    if row["case"]["question_type"]
    == "relational_counterfactual"
]

stress = [
    row
    for row in rows
    if row["case"]["question_type"]
    == "stress"
]


if len(relational) != 192:
    die(
        "Relational row count mismatch."
    )

if len(stress) != 48:
    die(
        "Stress row count mismatch."
    )


print("candidate identity: VERIFIED")
print("hidden identity:    VERIFIED")
print("cases:              240")
print("relational:         192")
print("stress:             48")
print("individual cases printed: 0")


# =====================================================================
# RECONFIRM COUNTERFACTUAL GROUPS
# =====================================================================

question_groups = defaultdict(
    list
)

evidence_groups = defaultdict(
    list
)

for row in relational:

    question_groups[
        row["hidden"]["question_hash"]
    ].append(
        row["label"]
    )

    evidence_groups[
        row["hidden"]["evidence_hash"]
    ].append(
        row["label"]
    )


if len(question_groups) != 48:
    die(
        "Expected 48 exact-question groups."
    )

if len(evidence_groups) != 48:
    die(
        "Expected 48 exact-evidence groups."
    )


for groups in (
    question_groups,
    evidence_groups,
):

    for values in groups.values():

        if (
            len(values) != 4
            or set(values) != LABEL_SET
        ):
            die(
                "Malformed four-label counterfactual rotation."
            )


print()
print("================ COUNTERFACTUAL BASELINE ================")
print("same-question rotations: 48/48")
print("same-evidence rotations: 48/48")


# =====================================================================
# HIDDEN GENERATOR STATE — DIAGNOSTIC ONLY
#
# These values are NOT directly part of the model input.
#
# Candidate V1 was incorrectly rejected by treating hidden state as
# equivalent to a model-visible shortcut.
#
# Candidate V2 nevertheless intentionally forces target/world PAIRS
# to chance globally.
#
# composition + target + world will remain deterministic by design,
# because those variables define the generated semantic world.
# =====================================================================

pair_labels = defaultdict(
    list
)

triple_labels = defaultdict(
    list
)

for row in relational:

    hidden = row[
        "hidden"
    ]

    pair = (
        hidden["target_index"],
        hidden["world_index"],
    )

    triple = (
        hidden["composition"],
        hidden["target_index"],
        hidden["world_index"],
    )

    pair_labels[
        pair
    ].append(
        row["label"]
    )

    triple_labels[
        triple
    ].append(
        row["label"]
    )


pair_correct = sum(
    max(
        Counter(values).values()
    )
    for values in pair_labels.values()
)

pair_total = sum(
    len(values)
    for values in pair_labels.values()
)

pair_accuracy = (
    pair_correct
    / pair_total
)


triple_correct = sum(
    max(
        Counter(values).values()
    )
    for values in triple_labels.values()
)

triple_total = sum(
    len(values)
    for values in triple_labels.values()
)

triple_accuracy = (
    triple_correct
    / triple_total
)


if abs(
    pair_accuracy
    - 0.25
) > 1e-12:
    die(
        "Candidate V2 global target/world pair balance regressed."
    )


print()
print("================ HIDDEN-STATE DIAGNOSTIC ================")
print(
    "target/world pair-majority:",
    f"{pair_accuracy:.4f}",
)
print(
    "composition+target+world:",
    f"{triple_accuracy:.4f}",
)
print(
    "NOTE: triple determinism is expected generator state,"
)
print(
    "      not itself a model-visible shortcut."
)


# =====================================================================
# STRUCTURAL / LENGTH / FORMAT ATTACK
#
# These features can be mechanically derived from visible input.
# =====================================================================

def shape_vector(row):

    case = row[
        "case"
    ]

    q = case[
        "question"
    ]

    e = case[
        "evidence_text"
    ]

    model_input = case[
        "input"
    ]

    return [
        len(q),
        len(q.split()),
        q.count("?"),
        q.count(","),
        q.count(";"),

        len(e),
        len(e.split()),
        e.count("\n"),
        e.count("."),
        e.count(","),
        e.count(";"),

        len(model_input),
        len(model_input.split()),

        row["hidden"]["evidence_count"],

        int(
            case["input_order"]
            == "question_first"
        ),
    ]


X_shape = np.asarray(
    [
        shape_vector(
            row
        )
        for row in rows
    ],
    dtype=float,
)

y_all = np.asarray(
    labels_all
)

shape_splits = stratified_splits(
    labels_all,
    n_splits=5,
)


def numeric_cv(
    estimator_factory,
):

    predictions = np.empty(
        len(y_all),
        dtype=y_all.dtype,
    )

    for train_idx, test_idx in shape_splits:

        estimator = estimator_factory()

        estimator.fit(
            X_shape[
                train_idx
            ],
            y_all[
                train_idx
            ],
        )

        predictions[
            test_idx
        ] = estimator.predict(
            X_shape[
                test_idx
            ]
        )

    return float(
        accuracy_score(
            y_all,
            predictions,
        )
    )


shape_logistic = numeric_cv(
    lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            random_state=SEED,
        ),
    )
)

shape_tree = numeric_cv(
    lambda: DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=8,
        random_state=SEED,
    )
)

shape_best = max(
    shape_logistic,
    shape_tree,
)


print()
print("================ VISIBLE SHAPE / FORMAT ATTACK ================")
print(
    "scaled logistic:",
    f"{shape_logistic:.4f}",
)
print(
    "shallow tree:",
    f"{shape_tree:.4f}",
)
print(
    "best:",
    f"{shape_best:.4f}",
)


# =====================================================================
# RELATIONAL ONE-SIDED TEXT ATTACKS
# =====================================================================

rel_labels = [
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

full_inputs = [
    row["case"]["input"]
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


q_group_splits = grouped_splits(
    rel_labels,
    question_hashes,
    n_splits=4,
)

e_group_splits = grouped_splits(
    rel_labels,
    evidence_hashes,
    n_splits=4,
)

composition_splits = grouped_splits(
    rel_labels,
    compositions,
    n_splits=4,
)

random_rel_splits = stratified_splits(
    rel_labels,
    n_splits=5,
)


print()
print("================ ONE-SIDED RELATIONAL TEXT ATTACKS ================")

text_hard_results = []


text_hard_results.append(
    run_text_attack(
        name="QUESTION ONLY — exact-question held out",
        texts=questions,
        labels=rel_labels,
        splits=q_group_splits,
    )
)

text_hard_results.append(
    run_text_attack(
        name="EVIDENCE ONLY — exact-evidence held out",
        texts=evidence,
        labels=rel_labels,
        splits=e_group_splits,
    )
)

text_hard_results.append(
    run_text_attack(
        name="QUESTION ONLY — composition held out",
        texts=questions,
        labels=rel_labels,
        splits=composition_splits,
    )
)

text_hard_results.append(
    run_text_attack(
        name="EVIDENCE ONLY — composition held out",
        texts=evidence,
        labels=rel_labels,
        splits=composition_splits,
    )
)


# =====================================================================
# FULL-INPUT INTERACTION ATTACKS
#
# These are the major addition missing from the original V1 audit.
#
# A benchmark can have question-only == chance and evidence-only ==
# chance while the combination is shortcut-solvable.
# =====================================================================

print()
print("================ FULL-INPUT INTERACTION ATTACKS ================")


text_hard_results.append(
    run_text_attack(
        name="FULL INPUT — random stratified CV",
        texts=full_inputs,
        labels=rel_labels,
        splits=random_rel_splits,
    )
)

text_hard_results.append(
    run_text_attack(
        name="FULL INPUT — exact-question groups held out",
        texts=full_inputs,
        labels=rel_labels,
        splits=q_group_splits,
    )
)

text_hard_results.append(
    run_text_attack(
        name="FULL INPUT — exact-evidence groups held out",
        texts=full_inputs,
        labels=rel_labels,
        splits=e_group_splits,
    )
)

text_hard_results.append(
    run_text_attack(
        name="FULL INPUT — composition held out",
        texts=full_inputs,
        labels=rel_labels,
        splits=composition_splits,
    )
)


best_visible_text_result = max(
    text_hard_results,
    key=lambda result: result["best"],
)

best_visible_text = (
    best_visible_text_result[
        "best"
    ]
)


print()
print(
    "best visible lexical shortcut:",
    best_visible_text_result["name"],
    f"acc={best_visible_text:.4f}",
    f"mode={best_visible_text_result['best_mode']}",
)


# =====================================================================
# LATENT INDEX RECOVERABILITY
#
# target_index/world_index are hidden generator variables.
#
# The benchmark should not encode their arbitrary absolute index
# consistently across unseen compositions.
#
# Therefore composition-held-out recoverability materially above chance
# is a visible structural warning and is treated as a hard gate.
# =====================================================================

targets = [
    row["hidden"]["target_index"]
    for row in relational
]

worlds = [
    row["hidden"]["world_index"]
    for row in relational
]


target_comp_holdout = text_cv(
    questions,
    targets,
    composition_splits,
    "hybrid",
)

world_comp_holdout = text_cv(
    evidence,
    worlds,
    composition_splits,
    "hybrid",
)


print()
print("================ LATENT INDEX RECOVERABILITY ================")
print(
    "question -> target_index",
    "(composition held out):",
    f"{target_comp_holdout:.4f}",
)
print(
    "evidence -> world_index",
    "(composition held out):",
    f"{world_comp_holdout:.4f}",
)
print(
    "four-class chance:",
    f"{CHANCE_4:.4f}",
)


# =====================================================================
# VISIBLE LATENT-INTERACTION EXPLOIT
#
# This deliberately attacks the nonlinear interaction that ordinary
# TF-IDF logistic regression may miss.
#
# Outer folds: random stratified relational CV.
#
# For each fold:
#
# 1. infer composition from full model-visible input
# 2. infer target_index from the question
# 3. infer world_index from the evidence
# 4. infer the best affine Latin-square rule for each composition from
#    TRAINING rows only
# 5. use the predicted visible coordinates to predict held-out gold
#
# The schedule module is NOT called here.
#
# This approximates a cheap structural attacker discovering the latent
# coordinate mechanism from benchmark-visible patterns.
# =====================================================================

def infer_rule(
    train_rows,
):
    """
    Infer a composition-local affine Latin rule from training examples.

    Search:
        label_order[(a*t + b*w + c) % 4]

    The search is intentionally broader than Candidate V2's actual
    schedule and allows arbitrary label permutations.
    """

    best = None

    for label_order in permutations(
        LABELS
    ):

        for a, b, c in product(
            range(4),
            range(4),
            range(4),
        ):

            correct = 0

            for target, world, gold in train_rows:

                predicted = label_order[
                    (
                        a * target
                        + b * world
                        + c
                    )
                    % 4
                ]

                correct += int(
                    predicted
                    == gold
                )

            candidate_key = (
                correct,
                tuple(
                    label_order
                ),
                a,
                b,
                c,
            )

            if (
                best is None
                or candidate_key
                > best[
                    0
                ]
            ):
                best = (
                    candidate_key,
                    (
                        tuple(
                            label_order
                        ),
                        a,
                        b,
                        c,
                    ),
                )

    return best[
        1
    ]


rel_y = np.asarray(
    rel_labels
)

rel_targets = np.asarray(
    targets
)

rel_worlds = np.asarray(
    worlds
)

rel_compositions = np.asarray(
    compositions
)


exploit_predictions = np.empty(
    len(relational),
    dtype=rel_y.dtype,
)

true_comp_predictions = np.empty(
    len(relational),
    dtype=rel_y.dtype,
)

target_oof = np.empty(
    len(relational),
    dtype=rel_targets.dtype,
)

world_oof = np.empty(
    len(relational),
    dtype=rel_worlds.dtype,
)

composition_oof = np.empty(
    len(relational),
    dtype=rel_compositions.dtype,
)


for train_idx, test_idx in random_rel_splits:

    target_model = make_text_probe(
        "hybrid"
    )

    world_model = make_text_probe(
        "hybrid"
    )

    composition_model = make_text_probe(
        "hybrid"
    )


    target_model.fit(
        [
            questions[i]
            for i in train_idx
        ],
        rel_targets[
            train_idx
        ],
    )

    world_model.fit(
        [
            evidence[i]
            for i in train_idx
        ],
        rel_worlds[
            train_idx
        ],
    )

    composition_model.fit(
        [
            full_inputs[i]
            for i in train_idx
        ],
        rel_compositions[
            train_idx
        ],
    )


    predicted_targets = target_model.predict(
        [
            questions[i]
            for i in test_idx
        ]
    )

    predicted_worlds = world_model.predict(
        [
            evidence[i]
            for i in test_idx
        ]
    )

    predicted_compositions = composition_model.predict(
        [
            full_inputs[i]
            for i in test_idx
        ]
    )


    target_oof[
        test_idx
    ] = predicted_targets

    world_oof[
        test_idx
    ] = predicted_worlds

    composition_oof[
        test_idx
    ] = predicted_compositions


    # Infer one relational rule per composition from TRAINING DATA ONLY.
    rules = {}

    train_by_composition = defaultdict(
        list
    )

    for i in train_idx:

        train_by_composition[
            rel_compositions[
                i
            ]
        ].append(
            (
                int(
                    rel_targets[
                        i
                    ]
                ),
                int(
                    rel_worlds[
                        i
                    ]
                ),
                rel_y[
                    i
                ],
            )
        )


    for composition, observations in train_by_composition.items():

        rules[
            composition
        ] = infer_rule(
            observations
        )


    default_label = majority(
        rel_y[
            train_idx
        ]
    )


    for local_i, global_i in enumerate(
        test_idx
    ):

        predicted_composition = predicted_compositions[
            local_i
        ]

        true_composition = rel_compositions[
            global_i
        ]

        predicted_target = int(
            predicted_targets[
                local_i
            ]
        )

        predicted_world = int(
            predicted_worlds[
                local_i
            ]
        )


        # ------------------------------------------------------------
        # Fully visible exploit:
        # composition, target and world are all text-predicted.
        # ------------------------------------------------------------

        if predicted_composition in rules:

            label_order, a, b, c = rules[
                predicted_composition
            ]

            exploit_predictions[
                global_i
            ] = label_order[
                (
                    a * predicted_target
                    + b * predicted_world
                    + c
                )
                % 4
            ]

        else:

            exploit_predictions[
                global_i
            ] = default_label


        # ------------------------------------------------------------
        # Upper-bound diagnostic:
        # give the attack TRUE composition but still require visible
        # recovery of target and world.
        #
        # This is not itself a hard failure because composition is
        # generator metadata, but it tells us whether coordinate
        # recovery would become dangerous if composition identification
        # were perfect.
        # ------------------------------------------------------------

        if true_composition in rules:

            label_order, a, b, c = rules[
                true_composition
            ]

            true_comp_predictions[
                global_i
            ] = label_order[
                (
                    a * predicted_target
                    + b * predicted_world
                    + c
                )
                % 4
            ]

        else:

            true_comp_predictions[
                global_i
            ] = default_label


target_random_acc = float(
    accuracy_score(
        rel_targets,
        target_oof,
    )
)

world_random_acc = float(
    accuracy_score(
        rel_worlds,
        world_oof,
    )
)

composition_random_acc = float(
    accuracy_score(
        rel_compositions,
        composition_oof,
    )
)

visible_latent_exploit = float(
    accuracy_score(
        rel_y,
        exploit_predictions,
    )
)

true_comp_upper_bound = float(
    accuracy_score(
        rel_y,
        true_comp_predictions,
    )
)


print()
print("================ VISIBLE LATENT-INTERACTION EXPLOIT ================")
print(
    "question -> target_index",
    "(random CV):",
    f"{target_random_acc:.4f}",
)
print(
    "evidence -> world_index",
    "(random CV):",
    f"{world_random_acc:.4f}",
)
print(
    "full input -> composition",
    "(random CV):",
    f"{composition_random_acc:.4f}",
)
print()
print(
    "predicted composition/target/world -> gold:",
    f"{visible_latent_exploit:.4f}",
)
print(
    "TRUE composition + predicted target/world -> gold:",
    f"{true_comp_upper_bound:.4f}",
)
print()
print(
    "NOTE: first exploit is a HARD visible shortcut gate."
)
print(
    "      TRUE-composition result is diagnostic upper bound only."
)


# =====================================================================
# OVERALL FULL-INPUT PROBE
#
# Includes both relational and stress cases.
# =====================================================================

all_inputs = [
    row["case"]["input"]
    for row in rows
]

all_random_splits = stratified_splits(
    labels_all,
    n_splits=5,
)

overall_full_result = run_text_attack(
    name="OVERALL 240-CASE FULL INPUT — random stratified CV",
    texts=all_inputs,
    labels=labels_all,
    splits=all_random_splits,
)


# =====================================================================
# STRESS-ONLY DIAGNOSTICS
# =====================================================================

stress_labels = [
    row["label"]
    for row in stress
]

stress_questions = [
    row["case"]["question"]
    for row in stress
]

stress_evidence = [
    row["case"]["evidence_text"]
    for row in stress
]

stress_inputs = [
    row["case"]["input"]
    for row in stress
]


stress_splits = stratified_splits(
    stress_labels,
    n_splits=4,
)


stress_q = text_cv(
    stress_questions,
    stress_labels,
    stress_splits,
    "hybrid",
)

stress_e = text_cv(
    stress_evidence,
    stress_labels,
    stress_splits,
    "hybrid",
)

stress_full = text_cv(
    stress_inputs,
    stress_labels,
    stress_splits,
    "hybrid",
)


print()
print("================ STRESS-ONLY DIAGNOSTIC ================")
print("48 rows — diagnostic, not automatic promotion gate.")
print(
    "question-only hybrid:",
    f"{stress_q:.4f}",
)
print(
    "evidence-only hybrid:",
    f"{stress_e:.4f}",
)
print(
    "full-input hybrid:",
    f"{stress_full:.4f}",
)


# =====================================================================
# PROMOTION-GATE DECISION
# =====================================================================

issues = []


if shape_best > VISIBLE_CLEAN_THRESHOLD:

    issues.append(
        "visible input length/format predicts labels above clean threshold"
    )


for result in text_hard_results:

    if (
        result[
            "best"
        ]
        > VISIBLE_CLEAN_THRESHOLD
    ):

        issues.append(
            "visible lexical shortcut exceeds clean threshold: "
            + result[
                "name"
            ]
        )


if (
    overall_full_result[
        "best"
    ]
    > VISIBLE_CLEAN_THRESHOLD
):

    issues.append(
        "overall 240-case full-input shortcut exceeds clean threshold"
    )


if (
    target_comp_holdout
    > VISIBLE_CLEAN_THRESHOLD
):

    issues.append(
        "arbitrary target_index is recoverable across unseen compositions"
    )


if (
    world_comp_holdout
    > VISIBLE_CLEAN_THRESHOLD
):

    issues.append(
        "arbitrary world_index is recoverable across unseen compositions"
    )


if (
    visible_latent_exploit
    > VISIBLE_CLEAN_THRESHOLD
):

    issues.append(
        "visible latent-coordinate interaction exploit exceeds clean threshold"
    )


print()
print("================ PROMOTION GATE SUMMARY ================")
print(
    "four-class chance:",
    f"{CHANCE_4:.4f}",
)
print(
    "visible clean threshold:",
    f"{VISIBLE_CLEAN_THRESHOLD:.4f}",
)
print()
print(
    "hidden target/world pair:",
    f"{pair_accuracy:.4f}",
)
print(
    "best visible shape:",
    f"{shape_best:.4f}",
)
print(
    "best relational visible text:",
    f"{best_visible_text:.4f}",
)
print(
    "overall full-input:",
    f"{overall_full_result['best']:.4f}",
)
print(
    "target index composition-holdout:",
    f"{target_comp_holdout:.4f}",
)
print(
    "world index composition-holdout:",
    f"{world_comp_holdout:.4f}",
)
print(
    "visible latent interaction exploit:",
    f"{visible_latent_exploit:.4f}",
)
print(
    "true-composition upper bound:",
    f"{true_comp_upper_bound:.4f}",
)
print()
print(
    "stress question/evidence/full:",
    f"{stress_q:.4f} / {stress_e:.4f} / {stress_full:.4f}",
)
print()
print(
    "hard-gate issues:",
    len(
        issues
    ),
)


if issues:

    print()
    print("CHALLENGE V3 CANDIDATE V2 SHORTCUT AUDIT: NOT CLEAN")

    for issue in issues:

        print(
            f"  - {issue}"
        )

    print()
    print("Candidate remains NON-CANONICAL.")
    print("Candidate remains UNCONSUMED.")
    print("No model evaluation performed.")
    print("Individual cases printed: 0")

    raise SystemExit(
        2
    )


print()
print("=" * 92)
print("CHALLENGE V3 CANDIDATE V2 SHORTCUT AUDIT: PASS")
print("=" * 92)
print()
print("No hard model-visible shortcut exceeded the promotion threshold.")
print()
print("Candidate remains NON-CANONICAL pending explicit promotion.")
print("Candidate remains UNCONSUMED.")
print("No Sentinel model loaded.")
print("No LLM inference performed.")
print("Individual cases printed: 0")
