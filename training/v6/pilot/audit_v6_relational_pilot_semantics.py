"""
Sentinel V6 relational pilot semantic/causal audit.

No model loading.
No training.
"""

from collections import Counter, defaultdict
from pathlib import Path

import hashlib
import json
import re
import sys


from training.v6.v6_relational_state_engine import (
    decide,
)

from training.v6.pilot.v6_relational_pilot_spec import (
    LABELS,
    PILOT_FAMILIES,
    TARGET_IDS,
    WORLD_IDS,
    build_state,
    intended_label,
)


BASE = Path(
    "/workspace"
)

DATASET = (
    BASE
    / "training/v6/pilot/"
      "sentinel_v6_relational_pilot.jsonl"
)


errors = []


def check(
    condition,
    message,
):

    if not condition:

        errors.append(
            message
        )


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


rows = load_rows()


# ====================================================================
# BASIC STRUCTURE
# ====================================================================

check(
    len(
        rows
    ) == 128,
    "expected exactly 128 rows",
)


row_ids = [
    row[
        "row_id"
    ]
    for row in rows
]


check(
    len(
        row_ids
    )
    == len(
        set(
            row_ids
        )
    ),
    "duplicate row_id",
)


families = Counter(
    row[
        "family"
    ]
    for row in rows
)


check(
    set(
        families
    )
    == set(
        PILOT_FAMILIES
    ),
    "pilot family set mismatch",
)


for family in PILOT_FAMILIES:

    check(
        families[
            family
        ] == 32,
        f"{family}: expected 32 rows",
    )


# ====================================================================
# LABEL + INPUT ORDER BALANCE
# ====================================================================

labels = Counter(
    row[
        "target"
    ]
    for row in rows
)


check(
    labels
    == Counter({
        "YES": 32,
        "NO": 32,
        "CONDITIONAL": 32,
        "INSUFFICIENT": 32,
    }),
    "global label balance mismatch",
)


orders = Counter(
    row[
        "input_order"
    ]
    for row in rows
)


check(
    orders
    == Counter({
        "evidence_first": 64,
        "question_first": 64,
    }),
    "global input order imbalance",
)


for label in LABELS:

    per_label_order = Counter(
        row[
            "input_order"
        ]
        for row in rows
        if row[
            "target"
        ] == label
    )


    check(
        per_label_order
        == Counter({
            "evidence_first": 16,
            "question_first": 16,
        }),
        f"{label}: input-order leakage",
    )


# ====================================================================
# ENGINE REGENERATION
# ====================================================================

for row in rows:

    state = build_state(
        row[
            "family"
        ],
        row[
            "target"
        ],
        row[
            "group_variant"
        ],
    )


    result = decide(
        state
    )


    check(
        result.decision.value
        == row[
            "target"
        ],
        (
            f"{row['row_id']}: "
            "engine decision mismatch"
        ),
    )


    expected_latin = intended_label(
        row[
            "world_index"
        ],
        row[
            "target_index"
        ],
    )


    check(
        expected_latin
        == row[
            "target"
        ],
        (
            f"{row['row_id']}: "
            "Latin square mismatch"
        ),
    )


    try:

        output = json.loads(
            row[
                "output"
            ]
        )

    except Exception as exc:

        errors.append(
            f"{row['row_id']}: invalid output JSON: {exc}"
        )

        continue


    check(
        set(
            output
        )
        == {
            "decision",
            "citations",
            "conditions",
            "rationale",
        },
        (
            f"{row['row_id']}: "
            "output key mismatch"
        ),
    )


    check(
        output.get(
            "decision"
        )
        == row[
            "target"
        ],
        (
            f"{row['row_id']}: "
            "output decision mismatch"
        ),
    )


    check(
        output.get(
            "conditions"
        )
        == list(
            result.conditions
        ),
        (
            f"{row['row_id']}: "
            "condition ledger mismatch"
        ),
    )


    expected_citations = [
        row[
            "target_evidence_ref"
        ]
    ]


    check(
        output.get(
            "citations"
        )
        == expected_citations,
        (
            f"{row['row_id']}: "
            "citation binding mismatch"
        ),
    )


# ====================================================================
# CITATION REFERENCE VALIDITY
# ====================================================================

for row in rows:

    refs = set(
        re.findall(
            r"\[(E\d+)\]",
            row[
                "evidence"
            ],
        )
    )


    check(
        row[
            "target_evidence_ref"
        ]
        in refs,
        (
            f"{row['row_id']}: "
            "target evidence ref absent"
        ),
    )


    output = json.loads(
        row[
            "output"
        ]
    )


    check(
        all(
            citation
            in refs
            for citation
            in output[
                "citations"
            ]
        ),
        (
            f"{row['row_id']}: "
            "invalid citation reference"
        ),
    )


# ====================================================================
# EVIDENCE HASH VALIDITY
# ====================================================================

for row in rows:

    actual = hashlib.sha256(
        row[
            "evidence"
        ].encode(
            "utf-8"
        )
    ).hexdigest()


    check(
        actual
        == row[
            "evidence_hash"
        ],
        (
            f"{row['row_id']}: "
            "evidence hash mismatch"
        ),
    )


# ====================================================================
# SAME EVIDENCE WORLD -> ALL FOUR LABELS
# ====================================================================

world_groups = defaultdict(
    list
)


for row in rows:

    key = (
        row[
            "relational_group"
        ],
        row[
            "world"
        ],
    )

    world_groups[
        key
    ].append(
        row
    )


check(
    len(
        world_groups
    ) == 32,
    "expected 32 evidence worlds",
)


for key, group_rows in world_groups.items():

    check(
        len(
            group_rows
        ) == 4,
        f"{key}: expected four target questions",
    )


    hashes = {
        row[
            "evidence_hash"
        ]
        for row
        in group_rows
    }


    check(
        len(
            hashes
        ) == 1,
        f"{key}: evidence differs across target questions",
    )


    group_labels = {
        row[
            "target"
        ]
        for row
        in group_rows
    }


    check(
        group_labels
        == set(
            LABELS
        ),
        f"{key}: evidence world does not span all labels",
    )


# ====================================================================
# SAME EXACT QUESTION -> ALL FOUR LABELS
# ====================================================================

question_groups = defaultdict(
    list
)


for row in rows:

    key = (
        row[
            "relational_group"
        ],
        row[
            "target_id"
        ],
    )

    question_groups[
        key
    ].append(
        row
    )


check(
    len(
        question_groups
    ) == 32,
    "expected 32 target-question groups",
)


for key, group_rows in question_groups.items():

    check(
        len(
            group_rows
        ) == 4,
        f"{key}: expected four worlds",
    )


    questions = {
        row[
            "question"
        ]
        for row
        in group_rows
    }


    check(
        len(
            questions
        ) == 1,
        f"{key}: question changed across worlds",
    )


    group_labels = {
        row[
            "target"
        ]
        for row
        in group_rows
    }


    check(
        group_labels
        == set(
            LABELS
        ),
        f"{key}: exact question does not span all labels",
    )


# ====================================================================
# FAMILY + RELATIONAL GROUP BALANCE
# ====================================================================

relational_groups = Counter(
    row[
        "relational_group"
    ]
    for row in rows
)


check(
    len(
        relational_groups
    ) == 8,
    "expected exactly eight relational groups",
)


for group, count in relational_groups.items():

    check(
        count == 16,
        f"{group}: expected 16 rows",
    )


    group_labels = Counter(
        row[
            "target"
        ]
        for row in rows
        if row[
            "relational_group"
        ] == group
    )


    check(
        group_labels
        == Counter({
            "YES": 4,
            "NO": 4,
            "CONDITIONAL": 4,
            "INSUFFICIENT": 4,
        }),
        f"{group}: label imbalance",
    )


# ====================================================================
# FULL INPUT UNIQUENESS / AMBIGUITY
# ====================================================================

input_to_targets = defaultdict(
    set
)


for row in rows:

    input_to_targets[
        row[
            "input"
        ]
    ].add(
        row[
            "target"
        ]
    )


ambiguous_inputs = {
    text:
        targets

    for text, targets
    in input_to_targets.items()

    if len(
        targets
    ) > 1
}


check(
    not ambiguous_inputs,
    "identical model input has conflicting targets",
)


check(
    len(
        input_to_targets
    ) == 128,
    "full model inputs are not unique",
)


# ====================================================================
# MODEL-VISIBLE HIDDEN METADATA LEAK
# ====================================================================

for row in rows:

    visible = (
        row[
            "instruction"
        ]
        + "\n"
        + row[
            "input"
        ]
    ).lower()


    forbidden = (
        row[
            "family"
        ].lower(),
        row[
            "relational_group"
        ].lower(),
        row[
            "world"
        ].lower(),
        row[
            "row_id"
        ].lower(),
        "target_id",
        "target_index",
        "world_index",
        "group_variant",
    )


    for token in forbidden:

        check(
            token not in visible,
            (
                f"{row['row_id']}: "
                f"hidden metadata leaked: {token}"
            ),
        )


# ====================================================================
# DECISION / LEDGER INVARIANTS
# ====================================================================

for row in rows:

    output = json.loads(
        row[
            "output"
        ]
    )


    conditions = output[
        "conditions"
    ]

    decision = output[
        "decision"
    ]


    if decision == "INSUFFICIENT":

        check(
            conditions == [],
            f"{row['row_id']}: INSUFFICIENT ledger nonempty",
        )


    elif decision == "YES":

        check(
            not any(
                item.startswith(
                    "FAILED | "
                )
                or item.startswith(
                    "UNRESOLVED | "
                )
                for item
                in conditions
            ),
            f"{row['row_id']}: YES ledger inconsistent",
        )


    elif decision == "CONDITIONAL":

        check(
            any(
                item.startswith(
                    "UNRESOLVED | "
                )
                for item
                in conditions
            ),
            f"{row['row_id']}: CONDITIONAL lacks unresolved state",
        )

        check(
            not any(
                item.startswith(
                    "FAILED | "
                )
                for item
                in conditions
            ),
            f"{row['row_id']}: CONDITIONAL contains failed state",
        )


    elif decision == "NO":

        if conditions:

            check(
                any(
                    item.startswith(
                        "FAILED | "
                    )
                    for item
                    in conditions
                ),
                f"{row['row_id']}: ledger-based NO lacks FAILED state",
            )


# ====================================================================
# REPORT
# ====================================================================

print("=" * 88)
print("SENTINEL V6 RELATIONAL PILOT SEMANTIC / CAUSAL AUDIT")
print("=" * 88)

print(
    "rows:",
    len(
        rows
    )
)

print(
    "families:",
    len(
        families
    )
)

print(
    "relational groups:",
    len(
        relational_groups
    )
)

print(
    "evidence worlds:",
    len(
        world_groups
    )
)

print(
    "exact question groups:",
    len(
        question_groups
    )
)

print(
    "labels:",
    dict(
        sorted(
            labels.items()
        )
    )
)

print(
    "input order:",
    dict(
        sorted(
            orders.items()
        )
    )
)

print(
    "unique full inputs:",
    len(
        input_to_targets
    )
)

print(
    "ambiguous identical inputs:",
    len(
        ambiguous_inputs
    )
)

print(
    "errors:",
    len(
        errors
    )
)


if errors:

    print()

    for error in errors[
        :100
    ]:

        print(
            "ERROR:",
            error
        )


    print()
    print(
        "SENTINEL V6 RELATIONAL PILOT SEMANTIC AUDIT: FAIL"
    )

    sys.exit(
        1
    )


print()
print(
    "SENTINEL V6 RELATIONAL PILOT SEMANTIC AUDIT: PASS"
)
