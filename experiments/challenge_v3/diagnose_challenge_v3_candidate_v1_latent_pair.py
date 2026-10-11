from __future__ import annotations

from collections import Counter, defaultdict
from itertools import permutations, product
from pathlib import Path
import hashlib
import json

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline


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

SEED = 20260817


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def die(message):
    raise RuntimeError(message)


def majority(values):
    counts = Counter(values)

    return sorted(
        counts,
        key=lambda x: (-counts[x], str(x))
    )[0]


def make_probe():
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

    return Pipeline([
        (
            "features",
            FeatureUnion([
                ("word", word),
                ("char", char),
            ]),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=4000,
                random_state=SEED,
            ),
        ),
    ])


def grouped_splits(y, groups, n_splits=4):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    return list(
        splitter.split(
            np.zeros(len(y)),
            np.asarray(y, dtype=object),
            np.asarray(groups, dtype=object),
        )
    )


def text_cv(texts, y, groups):
    texts = list(texts)
    y = np.asarray(y, dtype=object)

    predictions = np.empty(len(y), dtype=object)

    for train_idx, test_idx in grouped_splits(
        y,
        groups,
        n_splits=4,
    ):
        model = make_probe()

        model.fit(
            [texts[i] for i in train_idx],
            y[train_idx],
        )

        predictions[test_idx] = model.predict(
            [texts[i] for i in test_idx]
        )

    return float(
        accuracy_score(
            y,
            predictions,
        )
    )


# ===================================================================
# LOAD
# ===================================================================

candidate = json.loads(CANDIDATE.read_text())

hidden_rows = []

with HIDDEN.open() as f:
    for line in f:
        line = line.strip()

        if line:
            hidden_rows.append(json.loads(line))

hidden_by_id = {
    row["case_id"]: row
    for row in hidden_rows
}

rows = []

for case in candidate:
    hidden = hidden_by_id[case["case_id"]]

    if case["question_type"] != "relational_counterfactual":
        continue

    rows.append({
        "case": case,
        "hidden": hidden,
        "label": case["gold"]["decision"],
    })

if len(rows) != 192:
    die(f"Expected 192 relational rows, got {len(rows)}")


print("=" * 92)
print("CHALLENGE V3 CANDIDATE V1 — LATENT PAIR OBSERVABILITY")
print("=" * 92)

print("candidate identity:",
      sha256_file(CANDIDATE) == EXPECTED_CANDIDATE_SHA)

print("hidden identity:",
      sha256_file(HIDDEN) == EXPECTED_HIDDEN_SHA)

print("relational rows:", len(rows))
print("individual case content printed: 0")


# ===================================================================
# BASIC LATENT STRUCTURE
# ===================================================================

targets = [
    r["hidden"]["target_index"]
    for r in rows
]

worlds = [
    r["hidden"]["world_index"]
    for r in rows
]

labels = [
    r["label"]
    for r in rows
]

pairs = list(zip(targets, worlds))

print()
print("================ LATENT AXES ================")
print("target_index distribution:",
      dict(sorted(Counter(targets).items())))

print("world_index distribution:",
      dict(sorted(Counter(worlds).items())))

print("unique target/world cells:", len(set(pairs)))


# ===================================================================
# PAIR PURITY
#
# How deterministic is label given latent pair?
# Do NOT print the pair -> label mapping.
# ===================================================================

cell_labels = defaultdict(list)

for pair, label in zip(pairs, labels):
    cell_labels[pair].append(label)

correct = 0
pure_cells = 0

cell_sizes = Counter()
cell_label_cardinality = Counter()

for values in cell_labels.values():
    counts = Counter(values)

    cell_sizes[len(values)] += 1
    cell_label_cardinality[len(counts)] += 1

    correct += max(counts.values())

    if len(counts) == 1:
        pure_cells += 1

pair_purity_accuracy = correct / len(rows)

print()
print("================ LATENT PAIR PURITY ================")
print("pair cells:", len(cell_labels))
print("cell size distribution:",
      dict(sorted(cell_sizes.items())))

print("label cardinality per cell:",
      dict(sorted(cell_label_cardinality.items())))

print(
    "pure pair cells:",
    f"{pure_cells}/{len(cell_labels)}"
)

print(
    "pair -> majority-label accuracy:",
    f"{pair_purity_accuracy:.4f}"
)


# ===================================================================
# SIMPLE MOD-4 RULE
#
# Detect whether the design is effectively a global Latin-square
# arithmetic rule over target_index/world_index.
#
# We do not print the winning rule or label permutation.
# ===================================================================

unique_targets = sorted(set(targets))
unique_worlds = sorted(set(worlds))

if (
    unique_targets == [0, 1, 2, 3]
    and unique_worlds == [0, 1, 2, 3]
):
    best_affine = 0.0

    for label_order in permutations(LABELS):
        label_to_int = {
            label: i
            for i, label in enumerate(label_order)
        }

        encoded = [
            label_to_int[label]
            for label in labels
        ]

        for a, b, c in product(
            range(4),
            range(4),
            range(4),
        ):
            predicted = [
                (a * target + b * world + c) % 4
                for target, world in pairs
            ]

            acc = np.mean(
                np.asarray(predicted)
                == np.asarray(encoded)
            )

            best_affine = max(
                best_affine,
                float(acc),
            )

    print()
    print("================ SIMPLE LATIN-SQUARE TEST ================")
    print(
        "best affine mod-4 latent rule accuracy:",
        f"{best_affine:.4f}"
    )

else:
    best_affine = None

    print()
    print("================ SIMPLE LATIN-SQUARE TEST ================")
    print("SKIP: latent indices are not exactly 0..3.")


# ===================================================================
# CROSS-COMPOSITION LATENT-PAIR LOOKUP
#
# If pair -> label is a global generator rule, it should generalize
# even when entire compositions are held out.
# ===================================================================

groups = [
    r["case"]["composition"]
    for r in rows
]

splits = grouped_splits(
    labels,
    groups,
    n_splits=4,
)

predictions = np.empty(
    len(rows),
    dtype=object,
)

for train_idx, test_idx in splits:
    lookup = defaultdict(list)

    for i in train_idx:
        lookup[pairs[i]].append(labels[i])

    default = majority(
        [labels[i] for i in train_idx]
    )

    for i in test_idx:
        if pairs[i] in lookup:
            predictions[i] = majority(
                lookup[pairs[i]]
            )
        else:
            predictions[i] = default

cross_composition_pair_acc = float(
    accuracy_score(
        np.asarray(labels, dtype=object),
        predictions,
    )
)

print()
print("================ HIDDEN-PAIR GENERALIZATION ================")
print(
    "composition-held-out pair lookup accuracy:",
    f"{cross_composition_pair_acc:.4f}"
)


# ===================================================================
# OBSERVABILITY OF EACH LATENT AXIS
#
# Can surface text recover the latent indices across unseen
# compositions?
# ===================================================================

questions = [
    r["case"]["question"]
    for r in rows
]

evidence = [
    r["case"]["evidence_text"]
    for r in rows
]

target_from_question = text_cv(
    questions,
    targets,
    groups,
)

world_from_evidence = text_cv(
    evidence,
    worlds,
    groups,
)

target_from_evidence = text_cv(
    evidence,
    targets,
    groups,
)

world_from_question = text_cv(
    questions,
    worlds,
    groups,
)

print()
print("================ LATENT AXIS OBSERVABILITY ================")

print(
    "question -> target_index:",
    f"{target_from_question:.4f}"
)

print(
    "evidence -> world_index:",
    f"{world_from_evidence:.4f}"
)

print(
    "evidence -> target_index:",
    f"{target_from_evidence:.4f}"
)

print(
    "question -> world_index:",
    f"{world_from_question:.4f}"
)

print("four-class chance: 0.2500")


# ===================================================================
# FULL MODEL-VISIBLE INPUT PROBE
#
# This was missing from Shortcut Auditor V3.
# ===================================================================

full_text = [
    (
        r["case"]["question"]
        + "\n\n"
        + r["case"]["evidence_text"]
    )
    for r in rows
]

full_input_gold = text_cv(
    full_text,
    labels,
    groups,
)

print()
print("================ FULL INPUT LEXICAL PROBE ================")

print(
    "question + evidence -> gold",
    "(composition held out):",
    f"{full_input_gold:.4f}"
)

print("chance: 0.2500")


# ===================================================================
# PREDICTED-PAIR EXPLOIT
#
# Outer composition-held-out folds.
#
# 1. Learn question -> target_index from visible text.
# 2. Learn evidence -> world_index from visible text.
# 3. Learn latent pair -> label mapping from training rows.
# 4. Combine predicted indices on held-out compositions.
#
# This tests whether the dangerous latent interaction is recoverable
# from model-visible surface text.
# ===================================================================

exploit_predictions = np.empty(
    len(rows),
    dtype=object,
)

for train_idx, test_idx in splits:

    q_model = make_probe()
    e_model = make_probe()

    q_model.fit(
        [questions[i] for i in train_idx],
        np.asarray(
            [targets[i] for i in train_idx],
            dtype=object,
        ),
    )

    e_model.fit(
        [evidence[i] for i in train_idx],
        np.asarray(
            [worlds[i] for i in train_idx],
            dtype=object,
        ),
    )

    predicted_targets = q_model.predict(
        [questions[i] for i in test_idx]
    )

    predicted_worlds = e_model.predict(
        [evidence[i] for i in test_idx]
    )

    pair_lookup = defaultdict(list)

    for i in train_idx:
        pair_lookup[pairs[i]].append(labels[i])

    default = majority(
        [labels[i] for i in train_idx]
    )

    for local_i, global_i in enumerate(test_idx):
        predicted_pair = (
            predicted_targets[local_i],
            predicted_worlds[local_i],
        )

        if predicted_pair in pair_lookup:
            exploit_predictions[global_i] = majority(
                pair_lookup[predicted_pair]
            )
        else:
            exploit_predictions[global_i] = default


predicted_pair_exploit = float(
    accuracy_score(
        np.asarray(labels, dtype=object),
        exploit_predictions,
    )
)

print()
print("================ VISIBLE PAIR-EXPLOIT PROBE ================")

print(
    "predicted target/world pair -> gold:",
    f"{predicted_pair_exploit:.4f}"
)

print("chance: 0.2500")


# ===================================================================
# INTERPRETATION BANDS
#
# Do not automatically promote/reject.
# ===================================================================

print()
print("================ DIAGNOSTIC SUMMARY ================")

print(
    "latent pair purity:",
    f"{pair_purity_accuracy:.4f}"
)

if best_affine is not None:
    print(
        "simple latent affine rule:",
        f"{best_affine:.4f}"
    )

print(
    "hidden pair cross-composition:",
    f"{cross_composition_pair_acc:.4f}"
)

print(
    "question->target observability:",
    f"{target_from_question:.4f}"
)

print(
    "evidence->world observability:",
    f"{world_from_evidence:.4f}"
)

print(
    "full visible input -> gold:",
    f"{full_input_gold:.4f}"
)

print(
    "visible predicted-pair exploit:",
    f"{predicted_pair_exploit:.4f}"
)

print()
print("No promotion decision made by this diagnostic.")
print("No benchmark case content printed.")
print("No Sentinel model loaded.")
print("No LLM inference performed.")
print()
print("LATENT PAIR OBSERVABILITY DIAGNOSTIC: COMPLETE")
