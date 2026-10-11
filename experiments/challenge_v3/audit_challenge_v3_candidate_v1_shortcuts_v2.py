from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# =====================================================================
# FROZEN IDENTITIES
# =====================================================================

BASE = Path("/workspace")

CANDIDATE = (
    BASE
    / "experiments/challenge_v3/candidates/"
      "sentinel_challenge_v3_candidate_v1.json"
)

HIDDEN = (
    BASE
    / "experiments/challenge_v3/candidates/"
      "sentinel_challenge_v3_candidate_v1_hidden_semantics.jsonl"
)

EXPECTED_CANDIDATE_SHA = (
    "ddd2f221f435d0a2b832276137a08f0504c99283556795402c881f373c335032"
)

EXPECTED_HIDDEN_SHA = (
    "2a28696463ed5eb238b556ae85ae5bd1a4ec105311f4a77c2f2d4312bc5d34ee"
)


LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)

LABEL_SET = set(LABELS)

CHANCE = 0.25

# A benchmark is not called shortcut-clean merely because a cheap probe
# fails to become excellent. 35% is intentionally strict against a
# four-class 25% baseline.
CLEAN_ACCURACY_THRESHOLD = 0.35

# Strong leakage is rejected regardless of permutation significance.
HARD_LEAK_THRESHOLD = 0.40

# Metadata signals between 35-40% are considered material only if they
# survive a family-wise permutation check.
MATERIAL_P = 0.05

PERMUTATIONS = 500
SEED = 20260817


# =====================================================================
# UTILITIES
# =====================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def die(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def check_equal(name, actual, expected):
    if actual != expected:
        die(f"{name} mismatch: actual={actual!r} expected={expected!r}")


def distribution(values):
    return dict(sorted(Counter(values).items(), key=lambda kv: str(kv[0])))


def majority_label(labels):
    counts = Counter(labels)

    # Deterministic tie-breaking only.
    return sorted(
        counts.items(),
        key=lambda kv: (-kv[1], LABELS.index(kv[0]))
    )[0][0]


# =====================================================================
# LOAD + INTEGRITY
# =====================================================================

print("=" * 92)
print("SENTINEL CHALLENGE V3 CANDIDATE V1 — AGGREGATE SHORTCUT AUDIT V2")
print("=" * 92)

candidate_sha = sha256_file(CANDIDATE)
hidden_sha = sha256_file(HIDDEN)

check_equal(
    "candidate SHA256",
    candidate_sha,
    EXPECTED_CANDIDATE_SHA,
)

check_equal(
    "hidden semantics SHA256",
    hidden_sha,
    EXPECTED_HIDDEN_SHA,
)

print("candidate identity: VERIFIED")
print("hidden identity:    VERIFIED")

candidate = json.loads(CANDIDATE.read_text())

if not isinstance(candidate, list):
    die("candidate must be a top-level list")

hidden_rows = []

with HIDDEN.open() as f:
    for line in f:
        line = line.strip()
        if line:
            hidden_rows.append(json.loads(line))

check_equal("candidate rows", len(candidate), 240)
check_equal("hidden rows", len(hidden_rows), 240)

candidate_ids = [x["case_id"] for x in candidate]
hidden_ids = [x["case_id"] for x in hidden_rows]

if len(set(candidate_ids)) != len(candidate_ids):
    die("duplicate candidate case_id values")

if len(set(hidden_ids)) != len(hidden_ids):
    die("duplicate hidden case_id values")

if set(candidate_ids) != set(hidden_ids):
    die("candidate/hidden case_id sets differ")

hidden_by_id = {
    row["case_id"]: row
    for row in hidden_rows
}

rows = []

for case in candidate:
    h = hidden_by_id[case["case_id"]]

    if case["composition"] != h["composition"]:
        die("candidate/hidden composition mismatch")

    if case["question_type"] != h["question_type"]:
        die("candidate/hidden question_type mismatch")

    gold = case.get("gold")

    if not isinstance(gold, dict):
        die("gold must be a dictionary")

    label = gold.get("decision")

    if label not in LABEL_SET:
        die("invalid decision label")

    rows.append({
        "case": case,
        "hidden": h,
        "label": label,
    })

print("case binding:       VERIFIED")
print("individual content printed: 0")


# =====================================================================
# BASIC DISTRIBUTIONS
# =====================================================================

labels = [r["label"] for r in rows]

check_equal(
    "overall labels",
    Counter(labels),
    Counter({
        "YES": 60,
        "NO": 60,
        "CONDITIONAL": 60,
        "INSUFFICIENT": 60,
    }),
)

relational = [
    r for r in rows
    if r["case"]["question_type"] == "relational_counterfactual"
]

stress = [
    r for r in rows
    if r["case"]["question_type"] == "stress"
]

check_equal("relational rows", len(relational), 192)
check_equal("stress rows", len(stress), 48)

check_equal(
    "relational label balance",
    Counter(r["label"] for r in relational),
    Counter({
        "YES": 48,
        "NO": 48,
        "CONDITIONAL": 48,
        "INSUFFICIENT": 48,
    }),
)

check_equal(
    "stress label balance",
    Counter(r["label"] for r in stress),
    Counter({
        "YES": 12,
        "NO": 12,
        "CONDITIONAL": 12,
        "INSUFFICIENT": 12,
    }),
)

print()
print("================ DISTRIBUTIONS ================")
print("cases:", len(rows))
print("relational:", len(relational))
print("stress:", len(stress))
print("labels:", distribution(labels))
print(
    "input order:",
    distribution(r["case"]["input_order"] for r in rows),
)
print(
    "evidence counts:",
    distribution(r["hidden"]["evidence_count"] for r in rows),
)


# =====================================================================
# COUNTERFACTUAL ROTATION GATES
# =====================================================================

def exact_rotation_gate(subset, hash_field, expected_groups, name):
    groups = defaultdict(list)

    for row in subset:
        groups[row["hidden"][hash_field]].append(row["label"])

    if len(groups) != expected_groups:
        die(
            f"{name}: expected {expected_groups} groups, "
            f"found {len(groups)}"
        )

    sizes = Counter()
    perfect = 0
    bad = 0

    for group_labels in groups.values():
        sizes[len(group_labels)] += 1

        counts = Counter(group_labels)

        if (
            len(group_labels) == 4
            and set(group_labels) == LABEL_SET
            and all(counts[label] == 1 for label in LABELS)
        ):
            perfect += 1
        else:
            bad += 1

    print(f"{name}:")
    print(f"  groups: {len(groups)}")
    print(f"  size distribution: {dict(sorted(sizes.items()))}")
    print(f"  exact four-label rotations: {perfect}/{len(groups)}")
    print(f"  malformed rotations: {bad}")

    if bad:
        die(f"{name}: counterfactual rotation failure")

    return groups


print()
print("================ EXACT COUNTERFACTUAL ROTATIONS ================")

question_groups = exact_rotation_gate(
    relational,
    "question_hash",
    48,
    "same-question / different-evidence",
)

evidence_groups = exact_rotation_gate(
    relational,
    "evidence_hash",
    48,
    "same-evidence / different-question",
)


# =====================================================================
# CATEGORICAL METADATA LEAKAGE
# =====================================================================

def metadata_value(row, name):
    c = row["case"]
    h = row["hidden"]

    # Candidate-visible structural fields are required.
    if name == "input_order":
        return c["input_order"]

    if name == "question_type":
        return c["question_type"]

    if name == "composition":
        return c["composition"]

    # Hidden structural fields may be schema-conditional.
    #
    # Missingness is itself preserved as an explicit categorical value
    # rather than dropping rows or imputing another semantic value.
    # This is conservative for leakage detection: if field presence or
    # absence predicts the label, the metadata probe can discover it.
    if name in {
        "evidence_count",
        "relational_system",
        "target_index",
        "world_index",
    }:
        return h.get(name, "__MISSING__")

    raise KeyError(name)


META_FIELDS = (
    "input_order",
    "question_type",
    "composition",
    "evidence_count",
    "relational_system",
    "target_index",
    "world_index",
)


def stratified_splits(y, n_splits=5):
    y = np.asarray(y, dtype=object)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    return list(
        splitter.split(
            np.zeros(len(y)),
            y,
        )
    )


def lookup_cv_accuracy(values, y, splits):
    values = np.asarray(values, dtype=object)
    y = np.asarray(y, dtype=object)

    predictions = np.empty(len(y), dtype=object)

    for train_idx, test_idx in splits:
        mapping = defaultdict(list)

        for i in train_idx:
            mapping[values[i]].append(y[i])

        global_default = majority_label(y[train_idx])

        for i in test_idx:
            if values[i] in mapping:
                predictions[i] = majority_label(mapping[values[i]])
            else:
                predictions[i] = global_default

    return float(accuracy_score(y, predictions))


meta_splits = stratified_splits(labels, n_splits=5)

single_features = {}

for field in META_FIELDS:
    single_features[field] = [
        metadata_value(row, field)
        for row in rows
    ]

single_scores = {
    field: lookup_cv_accuracy(values, labels, meta_splits)
    for field, values in single_features.items()
}

pair_features = {}

for a, b in combinations(META_FIELDS, 2):
    name = f"{a} + {b}"

    pair_features[name] = [
        (single_features[a][i], single_features[b][i])
        for i in range(len(rows))
    ]

pair_scores = {
    name: lookup_cv_accuracy(values, labels, meta_splits)
    for name, values in pair_features.items()
}


def familywise_permutation_p(feature_sets, observed_max, y, splits):
    rng = np.random.default_rng(SEED)
    y = np.asarray(y, dtype=object)

    exceed = 0

    feature_values = list(feature_sets.values())

    for _ in range(PERMUTATIONS):
        permuted = rng.permutation(y)

        perm_max = 0.0

        for values in feature_values:
            score = lookup_cv_accuracy(
                values,
                permuted,
                splits,
            )

            if score > perm_max:
                perm_max = score

        if perm_max >= observed_max - 1e-12:
            exceed += 1

    return (exceed + 1) / (PERMUTATIONS + 1)


single_best_name = max(single_scores, key=single_scores.get)
single_best_score = single_scores[single_best_name]

pair_best_name = max(pair_scores, key=pair_scores.get)
pair_best_score = pair_scores[pair_best_name]

single_perm_p = familywise_permutation_p(
    single_features,
    single_best_score,
    labels,
    meta_splits,
)

pair_perm_p = familywise_permutation_p(
    pair_features,
    pair_best_score,
    labels,
    meta_splits,
)

print()
print("================ STRUCTURAL METADATA PROBES ================")

for field in META_FIELDS:
    cardinality = len(set(single_features[field]))

    print(
        f"{field:20s} "
        f"values={cardinality:3d} "
        f"cv_acc={single_scores[field]:.4f}"
    )

print()
print(
    "best single metadata probe:",
    single_best_name,
    f"acc={single_best_score:.4f}",
    f"familywise_perm_p={single_perm_p:.4f}",
)

print(
    "best metadata-pair probe:",
    pair_best_name,
    f"acc={pair_best_score:.4f}",
    f"familywise_perm_p={pair_perm_p:.4f}",
)

top_pairs = sorted(
    pair_scores.items(),
    key=lambda kv: (-kv[1], kv[0]),
)[:5]

print("top metadata pairs:")

for name, score in top_pairs:
    print(f"  {name:45s} acc={score:.4f}")


# =====================================================================
# LENGTH / SHAPE PROBE
# =====================================================================

def shape_vector(row):
    q = row["case"]["question"]
    e = row["case"]["evidence_text"]

    return [
        len(q),
        len(q.split()),
        q.count("?"),
        q.count(","),
        len(e),
        len(e.split()),
        e.count("\n"),
        e.count("."),
        row["hidden"]["evidence_count"],
    ]


X_shape = np.asarray(
    [shape_vector(r) for r in rows],
    dtype=float,
)

y_arr = np.asarray(labels, dtype=object)

shape_splits = stratified_splits(labels, n_splits=5)


def numeric_cv_accuracy(estimator_factory):
    predictions = np.empty(len(rows), dtype=object)

    for train_idx, test_idx in shape_splits:
        estimator = estimator_factory()

        estimator.fit(
            X_shape[train_idx],
            y_arr[train_idx],
        )

        predictions[test_idx] = estimator.predict(
            X_shape[test_idx]
        )

    return float(
        accuracy_score(
            y_arr,
            predictions,
        )
    )


shape_logistic = numeric_cv_accuracy(
    lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            random_state=SEED,
        ),
    )
)

shape_tree = numeric_cv_accuracy(
    lambda: DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=8,
        random_state=SEED,
    )
)

shape_best = max(shape_logistic, shape_tree)

print()
print("================ LENGTH / SHAPE PROBES ================")
print(f"scaled logistic accuracy: {shape_logistic:.4f}")
print(f"shallow tree accuracy:    {shape_tree:.4f}")
print(f"best shape accuracy:       {shape_best:.4f}")


# =====================================================================
# TEXTUAL SHORTCUT PROBES
# =====================================================================

def build_text_probe(mode):
    word = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_features=25000,
    )

    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=2,
        max_features=35000,
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
        raise ValueError(mode)

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                random_state=SEED,
            ),
        ),
    ])


def grouped_splits(y, groups, n_splits=4):
    y = np.asarray(y, dtype=object)
    groups = np.asarray(groups, dtype=object)

    unique_groups = len(set(groups))

    if unique_groups < n_splits:
        die(
            f"not enough groups for {n_splits}-fold grouped CV: "
            f"{unique_groups}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    splits = list(
        splitter.split(
            np.zeros(len(y)),
            y,
            groups,
        )
    )

    for train_idx, test_idx in splits:
        train_labels = set(y[train_idx])
        test_labels = set(y[test_idx])

        if train_labels != LABEL_SET:
            die("grouped CV training fold missing labels")

        if test_labels != LABEL_SET:
            die("grouped CV test fold missing labels")

    return splits


def text_probe_accuracy(texts, y, splits, mode):
    texts = np.asarray(texts, dtype=object)
    y = np.asarray(y, dtype=object)

    predictions = np.empty(len(y), dtype=object)

    for train_idx, test_idx in splits:
        estimator = build_text_probe(mode)

        estimator.fit(
            texts[train_idx].tolist(),
            y[train_idx],
        )

        predictions[test_idx] = estimator.predict(
            texts[test_idx].tolist()
        )

    return float(
        accuracy_score(
            y,
            predictions,
        )
    )


def run_text_probe(name, subset, text_field, group_field):
    texts = [
        row["case"][text_field]
        for row in subset
    ]

    y = [
        row["label"]
        for row in subset
    ]

    if group_field in row_hidden_fields:
        groups = [
            row["hidden"][group_field]
            for row in subset
        ]
    else:
        groups = [
            row["case"][group_field]
            for row in subset
        ]

    splits = grouped_splits(
        y,
        groups,
        n_splits=4,
    )

    scores = {}

    for mode in ("word", "char", "hybrid"):
        scores[mode] = text_probe_accuracy(
            texts,
            y,
            splits,
            mode,
        )

    best_mode = max(scores, key=scores.get)

    print(name)
    print(f"  grouped by: {group_field}")
    print(f"  groups: {len(set(groups))}")
    print(f"  word:   {scores['word']:.4f}")
    print(f"  char:   {scores['char']:.4f}")
    print(f"  hybrid: {scores['hybrid']:.4f}")
    print(
        f"  max:    {scores[best_mode]:.4f} "
        f"({best_mode})"
    )

    return {
        "name": name,
        "scores": scores,
        "best_mode": best_mode,
        "best": scores[best_mode],
    }


row_hidden_fields = {
    "question_hash",
    "evidence_hash",
    "evidence_count",
    "relational_system",
    "target_index",
    "world_index",
}

print()
print("================ RELATIONAL TEXT SHORTCUT PROBES ================")

text_results = []

text_results.append(
    run_text_probe(
        "QUESTION ONLY — exact-question held out",
        relational,
        "question",
        "question_hash",
    )
)

text_results.append(
    run_text_probe(
        "EVIDENCE ONLY — exact-evidence held out",
        relational,
        "evidence_text",
        "evidence_hash",
    )
)

text_results.append(
    run_text_probe(
        "QUESTION ONLY — composition held out",
        relational,
        "question",
        "composition",
    )
)

text_results.append(
    run_text_probe(
        "EVIDENCE ONLY — composition held out",
        relational,
        "evidence_text",
        "composition",
    )
)

best_text_result = max(
    text_results,
    key=lambda x: x["best"],
)

best_relational_text = best_text_result["best"]

print()
print(
    "best relational text shortcut:",
    best_text_result["name"],
    f"acc={best_relational_text:.4f}",
    f"mode={best_text_result['best_mode']}",
)


# =====================================================================
# STRESS-ONLY DIAGNOSTIC
#
# Reported but not used as a hard gate. With only 48 rows, variance is
# materially larger than in the relational counterfactual core.
# =====================================================================

stress_y = [
    row["label"]
    for row in stress
]

stress_texts_q = [
    row["case"]["question"]
    for row in stress
]

stress_texts_e = [
    row["case"]["evidence_text"]
    for row in stress
]

stress_splitter = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=SEED,
)

stress_splits = list(
    stress_splitter.split(
        np.zeros(len(stress_y)),
        np.asarray(stress_y, dtype=object),
    )
)

stress_q_acc = text_probe_accuracy(
    stress_texts_q,
    stress_y,
    stress_splits,
    "hybrid",
)

stress_e_acc = text_probe_accuracy(
    stress_texts_e,
    stress_y,
    stress_splits,
    "hybrid",
)

print()
print("================ STRESS-ONLY DIAGNOSTIC ================")
print("NOTE: diagnostic only; not a hard promotion gate.")
print(f"question-only hybrid accuracy: {stress_q_acc:.4f}")
print(f"evidence-only hybrid accuracy: {stress_e_acc:.4f}")


# =====================================================================
# DECISION
# =====================================================================

issues = []

# Counterfactual rotations were already hard-gated above.

if (
    single_best_score > HARD_LEAK_THRESHOLD
    or (
        single_best_score > CLEAN_ACCURACY_THRESHOLD
        and single_perm_p < MATERIAL_P
    )
):
    issues.append(
        "single structural metadata predicts labels materially above chance"
    )

if (
    pair_best_score > HARD_LEAK_THRESHOLD
    or (
        pair_best_score > CLEAN_ACCURACY_THRESHOLD
        and pair_perm_p < MATERIAL_P
    )
):
    issues.append(
        "paired structural metadata predicts labels materially above chance"
    )

if shape_best > CLEAN_ACCURACY_THRESHOLD:
    issues.append(
        "question/evidence length or formatting predicts labels above clean threshold"
    )

for result in text_results:
    if result["best"] > CLEAN_ACCURACY_THRESHOLD:
        issues.append(
            f"text shortcut exceeds clean threshold: {result['name']}"
        )


print()
print("================ PROMOTION GATE SUMMARY ================")
print(f"chance baseline:              {CHANCE:.4f}")
print(f"clean accuracy threshold:     {CLEAN_ACCURACY_THRESHOLD:.4f}")
print(f"hard metadata leak threshold: {HARD_LEAK_THRESHOLD:.4f}")
print(f"metadata significance level:  {MATERIAL_P:.4f}")
print(f"permutations:                 {PERMUTATIONS}")

print()
print(
    f"best single metadata:   {single_best_score:.4f} "
    f"(p={single_perm_p:.4f})"
)
print(
    f"best metadata pair:     {pair_best_score:.4f} "
    f"(p={pair_perm_p:.4f})"
)
print(f"best length/shape:      {shape_best:.4f}")
print(f"best relational text:   {best_relational_text:.4f}")
print(f"issues:                 {len(issues)}")

if issues:
    print()
    print("SHORTCUT AUDIT RESULT: NOT CLEAN")

    for issue in issues:
        print(f"  - {issue}")

    print()
    print("Candidate remains non-canonical.")
    print("No benchmark model evaluation performed.")
    print("Individual cases printed: 0")

    raise SystemExit(2)

print()
print("CHALLENGE V3 CANDIDATE V1 SHORTCUT AUDIT: PASS")
print()
print("Candidate remains non-canonical pending explicit promotion.")
print("No Sentinel model loaded.")
print("No LLM inference performed.")
print("Individual cases printed: 0")
