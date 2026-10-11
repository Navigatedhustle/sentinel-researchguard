from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import json


from experiments.challenge_v3.challenge_v3_blueprint import (
    COMPOSITIONS,
    GOLD_KEYS,
    INPUT_ORDERS,
    LABELS,
    PUBLIC_CASE_KEYS,
)

from experiments.challenge_v3.challenge_v3_candidate_v2_schedule import (
    relational_label_v2,
)

from experiments.challenge_v3.challenge_v3_oracle import (
    decide,
    state_from_dict,
)


# =====================================================================
# FROZEN IDENTITIES
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


LABEL_SET = set(LABELS)

EXPECTED_LABEL_COUNTS = Counter({
    "YES": 60,
    "NO": 60,
    "CONDITIONAL": 60,
    "INSUFFICIENT": 60,
})


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


def text_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def normalize(value):
    """
    Normalize dataclass / tuple / list / dict structures so oracle
    objects can be compared safely with serialized JSON structures.
    """

    if is_dataclass(value):
        return normalize(
            asdict(value)
        )

    if isinstance(value, dict):
        return {
            str(key): normalize(item)
            for key, item in sorted(
                value.items(),
                key=lambda kv: str(kv[0]),
            )
        }

    if isinstance(value, (tuple, list)):
        return [
            normalize(item)
            for item in value
        ]

    return value


# =====================================================================
# LOAD SEALED BYTES
# =====================================================================

print("=" * 92)
print("SENTINEL CHALLENGE V3 CANDIDATE V2")
print("SEALED SEMANTIC / STRUCTURAL AUDIT")
print("=" * 92)

candidate_sha = sha256_file(
    CANDIDATE
)

hidden_sha = sha256_file(
    HIDDEN
)

if candidate_sha != EXPECTED_CANDIDATE_SHA:
    die("Candidate V2 SHA256 mismatch.")

if hidden_sha != EXPECTED_HIDDEN_SHA:
    die("Hidden-semantics V2 SHA256 mismatch.")

candidate = json.loads(
    CANDIDATE.read_text()
)

hidden_rows = []

with HIDDEN.open() as f:
    for line in f:
        line = line.strip()

        if line:
            hidden_rows.append(
                json.loads(line)
            )


if not isinstance(candidate, list):
    die("Candidate must be a top-level list.")

if len(candidate) != 240:
    die(
        f"Expected 240 public cases; found {len(candidate)}."
    )

if len(hidden_rows) != 240:
    die(
        f"Expected 240 hidden rows; found {len(hidden_rows)}."
    )

print("candidate identity: VERIFIED")
print("hidden identity:    VERIFIED")
print("public rows:        240")
print("hidden rows:        240")


# =====================================================================
# SCHEMA
# =====================================================================

public_case_keys = set(
    PUBLIC_CASE_KEYS
)

gold_keys = set(
    GOLD_KEYS
)

for index, case in enumerate(
    candidate
):

    if not isinstance(case, dict):
        die(
            f"Public row {index} is not a dictionary."
        )

    if set(case.keys()) != public_case_keys:
        die(
            "Public-case schema mismatch."
        )

    gold = case.get(
        "gold"
    )

    if not isinstance(gold, dict):
        die(
            "Gold object is not a dictionary."
        )

    if set(gold.keys()) != gold_keys:
        die(
            "Gold-object schema mismatch."
        )

    if gold["decision"] not in LABEL_SET:
        die(
            "Invalid gold decision."
        )

    if not isinstance(
        gold["rationale"],
        str,
    ):
        die(
            "Gold rationale must be a string."
        )

    if not gold["rationale"].strip():
        die(
            "Gold rationale must not be empty."
        )

    if not isinstance(
        gold["citations"],
        list,
    ):
        die(
            "Gold citations must be a list."
        )

    if not isinstance(
        gold["conditions"],
        list,
    ):
        die(
            "Gold conditions must be a list."
        )

print("public schema:       VERIFIED")


# =====================================================================
# CASE-ID BINDING
# =====================================================================

public_ids = [
    row["case_id"]
    for row in candidate
]

hidden_ids = [
    row["case_id"]
    for row in hidden_rows
]

if len(set(public_ids)) != 240:
    die(
        "Duplicate public case_id."
    )

if len(set(hidden_ids)) != 240:
    die(
        "Duplicate hidden case_id."
    )

if set(public_ids) != set(hidden_ids):
    die(
        "Public/hidden case_id sets differ."
    )

public_by_id = {
    row["case_id"]: row
    for row in candidate
}

hidden_by_id = {
    row["case_id"]: row
    for row in hidden_rows
}

print("case-id binding:     VERIFIED")


# =====================================================================
# PUBLIC/HIDDEN METADATA BINDING + TEXT HASHES
# =====================================================================

rows = []

for case_id in public_ids:

    case = public_by_id[
        case_id
    ]

    hidden = hidden_by_id[
        case_id
    ]

    for field in (
        "composition",
        "question_type",
    ):
        if case[field] != hidden[field]:
            die(
                f"Public/hidden {field} mismatch."
            )

    if (
        text_hash(
            case["question"]
        )
        != hidden["question_hash"]
    ):
        die(
            "Question hash mismatch."
        )

    if (
        text_hash(
            case["evidence_text"]
        )
        != hidden["evidence_hash"]
    ):
        die(
            "Evidence hash mismatch."
        )

    if (
        case["input_order"]
        not in set(INPUT_ORDERS)
    ):
        die(
            "Invalid input_order."
        )

    rows.append({
        "case": case,
        "hidden": hidden,
        "label": case["gold"]["decision"],
    })

print("metadata binding:    VERIFIED")
print("question hashes:     240/240")
print("evidence hashes:     240/240")


# =====================================================================
# MODEL-INPUT STRUCTURE
#
# We do not regenerate input with builder code.
# Instead we validate that the exact sealed input contains its exact
# question/evidence payloads in the declared ordering.
# =====================================================================

input_order_counts = Counter()

for row in rows:

    case = row["case"]

    model_input = case[
        "input"
    ]

    question = case[
        "question"
    ]

    evidence = case[
        "evidence_text"
    ]

    if not isinstance(
        model_input,
        str,
    ):
        die(
            "Model input must be a string."
        )

    qpos = model_input.find(
        question
    )

    epos = model_input.find(
        evidence
    )

    if qpos < 0:
        die(
            "Question missing from sealed model input."
        )

    if epos < 0:
        die(
            "Evidence missing from sealed model input."
        )

    order = case[
        "input_order"
    ]

    input_order_counts[
        order
    ] += 1

    if (
        order == "question_first"
        and not qpos < epos
    ):
        die(
            "question_first input-order mismatch."
        )

    if (
        order == "evidence_first"
        and not epos < qpos
    ):
        die(
            "evidence_first input-order mismatch."
        )


if input_order_counts != Counter({
    "question_first": 120,
    "evidence_first": 120,
}):
    die(
        f"Input-order imbalance: {input_order_counts}"
    )

print("input binding:       VERIFIED")
print(
    "input order:",
    dict(
        sorted(
            input_order_counts.items()
        )
    ),
)


# =====================================================================
# BASIC DISTRIBUTIONS
# =====================================================================

labels = Counter(
    row["label"]
    for row in rows
)

if labels != EXPECTED_LABEL_COUNTS:
    die(
        f"Label imbalance: {labels}"
    )


question_types = Counter(
    row["case"]["question_type"]
    for row in rows
)

if question_types != Counter({
    "relational_counterfactual": 192,
    "stress": 48,
}):
    die(
        f"Question-type imbalance: {question_types}"
    )


composition_counts = Counter(
    row["case"]["composition"]
    for row in rows
)

if set(composition_counts) != set(
    COMPOSITIONS
):
    die(
        "Composition set mismatch."
    )

if any(
    count != 20
    for count in composition_counts.values()
):
    die(
        f"Composition count imbalance: {composition_counts}"
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


relational_labels = Counter(
    row["label"]
    for row in relational
)

stress_labels = Counter(
    row["label"]
    for row in stress
)


if relational_labels != Counter({
    label: 48
    for label in LABELS
}):
    die(
        f"Relational label imbalance: {relational_labels}"
    )


if stress_labels != Counter({
    label: 12
    for label in LABELS
}):
    die(
        f"Stress label imbalance: {stress_labels}"
    )


print()
print("================ DISTRIBUTIONS ================")
print(
    "labels:",
    dict(
        sorted(
            labels.items()
        )
    ),
)
print(
    "question types:",
    dict(
        sorted(
            question_types.items()
        )
    ),
)
print(
    "compositions:",
    len(
        composition_counts
    ),
)
print(
    "rows/composition:",
    sorted(
        set(
            composition_counts.values()
        )
    ),
)


# =====================================================================
# ORACLE REPLAY
#
# This is the main semantic gate:
#
# serialized hidden semantic state
#        ->
# benchmark-local oracle
#        ->
# sealed public gold decision
#
# The builder is NOT used to reconstruct the state.
# =====================================================================

oracle_decision_matches = 0
oracle_condition_matches = 0

for row in rows:

    case = row[
        "case"
    ]

    hidden = row[
        "hidden"
    ]

    semantic_dict = hidden.get(
        "semantic_state"
    )

    if not isinstance(
        semantic_dict,
        dict,
    ):
        die(
            "Hidden semantic_state missing or invalid."
        )

    state = state_from_dict(
        semantic_dict
    )

    result = decide(
        state
    )

    if (
        result.decision
        != case["gold"]["decision"]
    ):
        die(
            "Oracle replay disagrees with sealed gold decision."
        )

    oracle_decision_matches += 1

    # The builder's gold object is expected to serialize the oracle's
    # unresolved conditions. Compare if the oracle result exposes them.
    if hasattr(
        result,
        "conditions",
    ):

        expected_conditions = normalize(
            result.conditions
        )

        actual_conditions = normalize(
            case["gold"]["conditions"]
        )

        if (
            expected_conditions
            != actual_conditions
        ):
            die(
                "Oracle conditions disagree with sealed gold conditions."
            )

        oracle_condition_matches += 1


if oracle_decision_matches != 240:
    die(
        "Oracle decision replay incomplete."
    )


print()
print("================ ORACLE REPLAY ================")
print(
    "decision matches:",
    f"{oracle_decision_matches}/240",
)

if oracle_condition_matches:
    print(
        "condition matches:",
        f"{oracle_condition_matches}/240",
    )
else:
    print(
        "condition matches:",
        "N/A — OracleResult exposes no conditions field",
    )


# =====================================================================
# V2 SCHEDULE REPLAY
#
# Independent of semantic-state reconstruction.
# =====================================================================

composition_index = {
    composition: index
    for index, composition in enumerate(
        COMPOSITIONS
    )
}

schedule_matches = 0

for row in relational:

    case = row[
        "case"
    ]

    hidden = row[
        "hidden"
    ]

    for required in (
        "world_index",
        "target_index",
        "relational_system",
        "target_id",
        "target_entity",
        "target_evidence_ref",
    ):
        if required not in hidden:
            die(
                f"Relational hidden field missing: {required}"
            )

    expected = relational_label_v2(
        composition_index[
            case["composition"]
        ],
        hidden[
            "world_index"
        ],
        hidden[
            "target_index"
        ],
    )

    if (
        expected
        != case["gold"]["decision"]
    ):
        die(
            "V2 schedule replay disagrees with sealed gold."
        )

    schedule_matches += 1


if schedule_matches != 192:
    die(
        "V2 schedule replay incomplete."
    )


for row in stress:

    hidden = row[
        "hidden"
    ]

    # Relational-only coordinate metadata must not leak into stress
    # rows.
    for forbidden in (
        "relational_system",
        "target_id",
        "target_entity",
        "target_index",
        "world_index",
    ):
        if forbidden in hidden:
            die(
                f"Stress row contains relational-only field: {forbidden}"
            )


print()
print("================ V2 SCHEDULE REPLAY ================")
print(
    "relational matches:",
    f"{schedule_matches}/192",
)
print(
    "stress schema separation:",
    "48/48",
)


# =====================================================================
# EXACT COUNTERFACTUAL ROTATIONS
# =====================================================================

question_groups = defaultdict(
    list
)

evidence_groups = defaultdict(
    list
)

for row in relational:

    hidden = row[
        "hidden"
    ]

    question_groups[
        hidden["question_hash"]
    ].append(
        row["label"]
    )

    evidence_groups[
        hidden["evidence_hash"]
    ].append(
        row["label"]
    )


if len(question_groups) != 48:
    die(
        f"Expected 48 question groups; found {len(question_groups)}."
    )

if len(evidence_groups) != 48:
    die(
        f"Expected 48 evidence groups; found {len(evidence_groups)}."
    )


def verify_four_label_groups(
    groups,
    name,
):

    perfect = 0

    for values in groups.values():

        if len(values) != 4:
            die(
                f"{name}: group size is not 4."
            )

        if set(values) != LABEL_SET:
            die(
                f"{name}: group does not contain all four labels."
            )

        counts = Counter(
            values
        )

        if any(
            counts[label] != 1
            for label in LABELS
        ):
            die(
                f"{name}: labels do not occur exactly once."
            )

        perfect += 1

    return perfect


perfect_questions = verify_four_label_groups(
    question_groups,
    "question rotation",
)

perfect_evidence = verify_four_label_groups(
    evidence_groups,
    "evidence rotation",
)


print()
print("================ COUNTERFACTUAL ROTATIONS ================")
print(
    "same-question rotations:",
    f"{perfect_questions}/48",
)
print(
    "same-evidence rotations:",
    f"{perfect_evidence}/48",
)


# =====================================================================
# GENERATED-BYTE LATIN-SQUARE / GLOBAL COORDINATE GATE
# =====================================================================

composition_cells = defaultdict(
    dict
)

coordinate_labels = defaultdict(
    list
)

for row in relational:

    hidden = row[
        "hidden"
    ]

    coordinate = (
        hidden[
            "target_index"
        ],
        hidden[
            "world_index"
        ],
    )

    composition_cells[
        row["case"]["composition"]
    ][
        coordinate
    ] = row[
        "label"
    ]

    coordinate_labels[
        coordinate
    ].append(
        row[
            "label"
        ]
    )


if len(composition_cells) != 12:
    die(
        "Expected 12 relational composition grids."
    )

if len(coordinate_labels) != 16:
    die(
        "Expected 16 absolute coordinate cells."
    )


latin_compositions = 0

for composition, cells in composition_cells.items():

    if len(cells) != 16:
        die(
            "Relational composition grid is incomplete."
        )

    for target_index in range(4):

        values = {
            cells[
                (
                    target_index,
                    world_index,
                )
            ]
            for world_index in range(4)
        }

        if values != LABEL_SET:
            die(
                "Target row is not a four-label Latin row."
            )

    for world_index in range(4):

        values = {
            cells[
                (
                    target_index,
                    world_index,
                )
            ]
            for target_index in range(4)
        }

        if values != LABEL_SET:
            die(
                "World column is not a four-label Latin column."
            )

    latin_compositions += 1


expected_coordinate_counts = Counter({
    label: 3
    for label in LABELS
})


for values in coordinate_labels.values():

    if len(values) != 12:
        die(
            "Absolute coordinate does not contain 12 rows."
        )

    if Counter(values) != expected_coordinate_counts:
        die(
            "Absolute coordinate is not perfectly label-balanced."
        )


pair_majority_correct = sum(
    max(
        Counter(values).values()
    )
    for values in coordinate_labels.values()
)

pair_majority_total = sum(
    len(values)
    for values in coordinate_labels.values()
)

pair_majority_accuracy = (
    pair_majority_correct
    / pair_majority_total
)

if abs(
    pair_majority_accuracy
    - 0.25
) > 1e-12:
    die(
        "Absolute coordinate pair predicts above chance."
    )


print()
print("================ LATENT STRUCTURE ================")
print(
    "Latin compositions:",
    f"{latin_compositions}/12",
)
print(
    "absolute coordinate cells:",
    len(
        coordinate_labels
    ),
)
print(
    "per-coordinate label counts:",
    dict(
        sorted(
            expected_coordinate_counts.items()
        )
    ),
)
print(
    "pair-majority accuracy:",
    f"{pair_majority_accuracy:.4f}",
)


# =====================================================================
# FULL INPUT UNIQUENESS / CONFLICT GATE
# =====================================================================

input_to_labels = defaultdict(
    set
)

for row in rows:

    input_to_labels[
        row["case"]["input"]
    ].add(
        row["label"]
    )


unique_inputs = len(
    input_to_labels
)

conflicting_identical_inputs = sum(
    len(values) > 1
    for values in input_to_labels.values()
)


if unique_inputs != 240:
    die(
        f"Expected 240 unique model inputs; found {unique_inputs}."
    )

if conflicting_identical_inputs != 0:
    die(
        "Identical model inputs have conflicting labels."
    )


print()
print("================ INPUT UNIQUENESS ================")
print(
    "unique full model inputs:",
    unique_inputs,
)
print(
    "conflicting identical inputs:",
    conflicting_identical_inputs,
)


# =====================================================================
# EVIDENCE-COUNT REPORT
# =====================================================================

evidence_counts = Counter(
    row["hidden"]["evidence_count"]
    for row in rows
)

print()
print("================ EVIDENCE STRUCTURE ================")
print(
    "evidence counts:",
    dict(
        sorted(
            evidence_counts.items()
        )
    ),
)


# =====================================================================
# FINAL
# =====================================================================

print()
print("=" * 92)
print("CHALLENGE V3 CANDIDATE V2 SEMANTIC / STRUCTURAL AUDIT: PASS")
print("=" * 92)
print()
print("candidate SHA256:")
print(f"  {candidate_sha}")
print()
print("hidden semantics SHA256:")
print(f"  {hidden_sha}")
print()
print("individual cases printed: 0")
print("Sentinel models loaded: 0")
print("LLM inference performed: 0")
print()
print("Candidate V2 remains NON-CANONICAL.")
print("Candidate V2 remains UNCONSUMED.")
