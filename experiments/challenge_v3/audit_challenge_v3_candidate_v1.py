"""
Challenge V3 Candidate V1 aggregate semantic/structural audit.

Never prints individual benchmark content.
No model inference.
"""

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import sys


from experiments.challenge_v3.challenge_v3_blueprint import (
    COMPOSITIONS,
    GOLD_KEYS,
    LABELS,
    PUBLIC_CASE_KEYS,
    relational_label,
)

from experiments.challenge_v3.challenge_v3_oracle import (
    decide,
    state_from_dict,
)


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


errors = []


def check(
    condition,
    message,
):

    if not condition:

        errors.append(
            message
        )


def text_hash(
    text,
):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


cases = json.loads(
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
    row[
        "case_id"
    ]:
        row

    for row
    in hidden_rows
}


# ====================================================================
# ROOT / SCHEMA
# ====================================================================

check(
    isinstance(
        cases,
        list,
    ),
    "canonical root must be list",
)

check(
    len(
        cases
    ) == 240,
    "expected 240 cases",
)

check(
    len(
        hidden_rows
    ) == 240,
    "expected 240 hidden semantic rows",
)


case_ids = [
    case[
        "case_id"
    ]
    for case
    in cases
]


check(
    len(
        case_ids
    )
    == len(
        set(
            case_ids
        )
    ),
    "duplicate case IDs",
)


check(
    set(
        case_ids
    )
    == set(
        hidden_by_id
    ),
    "public/hidden case identity mismatch",
)


for case in cases:

    check(
        tuple(
            case.keys()
        )
        == tuple(
            sorted(
                PUBLIC_CASE_KEYS
            )
        ),
        "public case schema mismatch",
    )


    check(
        tuple(
            case[
                "gold"
            ].keys()
        )
        == tuple(
            sorted(
                GOLD_KEYS
            )
        ),
        "gold schema mismatch",
    )


# ====================================================================
# AGGREGATE COUNTS
# ====================================================================

labels = Counter(
    case[
        "gold"
    ][
        "decision"
    ]
    for case
    in cases
)


check(
    labels
    == Counter({
        "YES": 60,
        "NO": 60,
        "CONDITIONAL": 60,
        "INSUFFICIENT": 60,
    }),
    "global label imbalance",
)


compositions = Counter(
    case[
        "composition"
    ]
    for case
    in cases
)


check(
    set(
        compositions
    )
    == set(
        COMPOSITIONS
    ),
    "composition set mismatch",
)


for composition in COMPOSITIONS:

    check(
        compositions[
            composition
        ] == 20,
        f"{composition}: expected 20 cases",
    )


    composition_labels = Counter(
        case[
            "gold"
        ][
            "decision"
        ]
        for case
        in cases
        if case[
            "composition"
        ] == composition
    )


    check(
        composition_labels
        == Counter({
            "YES": 5,
            "NO": 5,
            "CONDITIONAL": 5,
            "INSUFFICIENT": 5,
        }),
        f"{composition}: label imbalance",
    )


question_types = Counter(
    case[
        "question_type"
    ]
    for case
    in cases
)


check(
    question_types
    == Counter({
        "relational_counterfactual": 192,
        "stress": 48,
    }),
    "question-type imbalance",
)


input_orders = Counter(
    case[
        "input_order"
    ]
    for case
    in cases
)


check(
    input_orders
    == Counter({
        "evidence_first": 120,
        "question_first": 120,
    }),
    "input-order imbalance",
)


for label in LABELS:

    counts = Counter(
        case[
            "input_order"
        ]
        for case
        in cases
        if case[
            "gold"
        ][
            "decision"
        ] == label
    )


    check(
        counts
        == Counter({
            "evidence_first": 30,
            "question_first": 30,
        }),
        f"{label}: input-order leakage",
    )


# ====================================================================
# ORACLE REGENERATION + LEDGER
# ====================================================================

for case in cases:

    hidden = hidden_by_id[
        case[
            "case_id"
        ]
    ]


    state = state_from_dict(
        hidden[
            "semantic_state"
        ]
    )


    result = decide(
        state
    )


    gold = case[
        "gold"
    ]


    check(
        result.decision
        == gold[
            "decision"
        ],
        "oracle decision mismatch",
    )


    check(
        list(
            result.conditions
        )
        == gold[
            "conditions"
        ],
        "oracle condition-ledger mismatch",
    )


    decision = gold[
        "decision"
    ]

    conditions = gold[
        "conditions"
    ]


    if decision == "INSUFFICIENT":

        check(
            conditions == [],
            "INSUFFICIENT conditions must be empty",
        )


    elif decision == "YES":

        check(
            all(
                condition.startswith(
                    "SATISFIED | "
                )
                for condition
                in conditions
            ),
            "YES contains non-satisfied ledger state",
        )


    elif decision == "CONDITIONAL":

        check(
            any(
                condition.startswith(
                    "UNRESOLVED | "
                )
                for condition
                in conditions
            ),
            "CONDITIONAL lacks unresolved state",
        )


        check(
            not any(
                condition.startswith(
                    "FAILED | "
                )
                for condition
                in conditions
            ),
            "CONDITIONAL contains failed state",
        )


    elif decision == "NO":

        if conditions:

            check(
                any(
                    condition.startswith(
                        "FAILED | "
                    )
                    for condition
                    in conditions
                ),
                "ledger-based NO lacks failed state",
            )


# ====================================================================
# CITATION VALIDITY
# ====================================================================

for case in cases:

    refs = set(
        re.findall(
            r"\[(E\d+)\]",
            case[
                "evidence_text"
            ],
        )
    )


    gold_citations = case[
        "gold"
    ][
        "citations"
    ]


    check(
        len(
            gold_citations
        ) >= 1,
        "gold citation list unexpectedly empty",
    )


    check(
        len(
            gold_citations
        )
        == len(
            set(
                gold_citations
            )
        ),
        "duplicate gold citation",
    )


    check(
        all(
            citation in refs
            for citation
            in gold_citations
        ),
        "invalid citation reference",
    )


    hidden = hidden_by_id[
        case[
            "case_id"
        ]
    ]


    check(
        gold_citations
        == [
            hidden[
                "target_evidence_ref"
            ]
        ],
        "target citation binding mismatch",
    )


# ====================================================================
# INPUT CONSTRUCTION
# ====================================================================

for case in cases:

    evidence = case[
        "evidence_text"
    ]

    question = case[
        "question"
    ]


    if case[
        "input_order"
    ] == "evidence_first":

        expected_input = (
            "Evidence:\n"
            + evidence
            + "\n\nQuestion:\n"
            + question
        )


    else:

        expected_input = (
            "Question:\n"
            + question
            + "\n\nEvidence:\n"
            + evidence
        )


    check(
        case[
            "input"
        ] == expected_input,
        "input assembly mismatch",
    )


# ====================================================================
# EVIDENCE COUNT
# ====================================================================

evidence_counts = Counter()


evidence_count_label = defaultdict(
    Counter
)


for case in cases:

    count = len(
        set(
            re.findall(
                r"\[(E\d+)\]",
                case[
                    "evidence_text"
                ],
            )
        )
    )


    evidence_counts[
        count
    ] += 1


    evidence_count_label[
        count
    ][
        case[
            "gold"
        ][
            "decision"
        ]
    ] += 1


    hidden = hidden_by_id[
        case[
            "case_id"
        ]
    ]


    check(
        count
        == hidden[
            "evidence_count"
        ],
        "hidden/public evidence-count mismatch",
    )


check(
    set(
        evidence_counts
    )
    == {
        1,
        2,
        3,
        4,
        5,
    },
    "evidence counts must cover 1..5",
)


for count, label_counts_here in evidence_count_label.items():

    check(
        set(
            label_counts_here
        )
        == set(
            LABELS
        ),
        f"evidence count {count}: missing labels",
    )


# ====================================================================
# RELATIONAL INVARIANTS
# ====================================================================

relational = [
    case
    for case
    in cases
    if case[
        "question_type"
    ] == "relational_counterfactual"
]


rel_hidden = {
    case[
        "case_id"
    ]:
        hidden_by_id[
            case[
                "case_id"
            ]
        ]

    for case
    in relational
}


question_groups = defaultdict(
    list
)


evidence_worlds = defaultdict(
    list
)


for case in relational:

    hidden = rel_hidden[
        case[
            "case_id"
        ]
    ]


    question_groups[
        (
            case[
                "composition"
            ],
            hidden[
                "target_id"
            ],
        )
    ].append(
        case
    )


    evidence_worlds[
        (
            case[
                "composition"
            ],
            hidden[
                "world_index"
            ],
        )
    ].append(
        case
    )


check(
    len(
        question_groups
    ) == 48,
    "expected 48 relational exact-question groups",
)


check(
    len(
        evidence_worlds
    ) == 48,
    "expected 48 relational evidence worlds",
)


for key, group in question_groups.items():

    check(
        len(
            group
        ) == 4,
        f"{key}: expected four world cases",
    )


    questions = {
        case[
            "question"
        ]
        for case
        in group
    }


    check(
        len(
            questions
        ) == 1,
        f"{key}: question changed across worlds",
    )


    group_labels = {
        case[
            "gold"
        ][
            "decision"
        ]
        for case
        in group
    }


    check(
        group_labels
        == set(
            LABELS
        ),
        f"{key}: exact question does not span all labels",
    )


for key, group in evidence_worlds.items():

    check(
        len(
            group
        ) == 4,
        f"{key}: expected four target cases",
    )


    evidence = {
        case[
            "evidence_text"
        ]
        for case
        in group
    }


    check(
        len(
            evidence
        ) == 1,
        f"{key}: evidence changed across targets",
    )


    group_labels = {
        case[
            "gold"
        ][
            "decision"
        ]
        for case
        in group
    }


    check(
        group_labels
        == set(
            LABELS
        ),
        f"{key}: evidence world does not span all labels",
    )


# ====================================================================
# RELATIONAL LATIN ORACLE
# ====================================================================

latin_correct = 0


for case in relational:

    hidden = rel_hidden[
        case[
            "case_id"
        ]
    ]


    predicted = relational_label(

        hidden[
            "world_index"
        ],

        hidden[
            "target_index"
        ],
    )


    if (
        predicted
        == case[
            "gold"
        ][
            "decision"
        ]
    ):

        latin_correct += 1


check(
    latin_correct == 192,
    "hidden relational oracle not 100%",
)


# ====================================================================
# DUPLICATES / AMBIGUITY
# ====================================================================

inputs = defaultdict(
    set
)


for case in cases:

    inputs[
        (
            case[
                "instruction"
            ],
            case[
                "input"
            ],
        )
    ].add(
        case[
            "gold"
        ][
            "decision"
        ]
    )


conflicting_inputs = sum(
    1
    for targets
    in inputs.values()
    if len(
        targets
    ) > 1
)


check(
    conflicting_inputs == 0,
    "identical full model input has conflicting gold",
)


check(
    len(
        inputs
    ) == 240,
    "full model inputs are not unique",
)


# ====================================================================
# HASH CONSISTENCY
# ====================================================================

for case in cases:

    hidden = hidden_by_id[
        case[
            "case_id"
        ]
    ]


    check(
        text_hash(
            case[
                "question"
            ]
        )
        == hidden[
            "question_hash"
        ],
        "question hash mismatch",
    )


    check(
        text_hash(
            case[
                "evidence_text"
            ]
        )
        == hidden[
            "evidence_hash"
        ],
        "evidence hash mismatch",
    )


# ====================================================================
# REPORT — AGGREGATES ONLY
# ====================================================================

print("=" * 92)
print("SENTINEL CHALLENGE V3 CANDIDATE V1 SEMANTIC / STRUCTURAL AUDIT")
print("=" * 92)

print(
    "cases:",
    len(
        cases
    )
)

print(
    "compositions:",
    len(
        compositions
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
    "question types:",
    dict(
        sorted(
            question_types.items()
        )
    )
)

print(
    "input order:",
    dict(
        sorted(
            input_orders.items()
        )
    )
)

print(
    "evidence counts:",
    dict(
        sorted(
            evidence_counts.items()
        )
    )
)

print(
    "relational exact-question groups:",
    len(
        question_groups
    )
)

print(
    "relational evidence worlds:",
    len(
        evidence_worlds
    )
)

print(
    "hidden relational oracle:",
    f"{latin_correct}/192"
)

print(
    "unique full model inputs:",
    len(
        inputs
    )
)

print(
    "conflicting identical inputs:",
    conflicting_inputs
)

print(
    "individual cases printed:",
    0
)

print(
    "errors:",
    len(
        errors
    )
)


if errors:

    for error in errors[
        :100
    ]:

        print(
            "ERROR:",
            error
        )


    print()
    print(
        "SENTINEL CHALLENGE V3 CANDIDATE V1 SEMANTIC AUDIT: FAIL"
    )

    sys.exit(
        1
    )


print()
print(
    "SENTINEL CHALLENGE V3 CANDIDATE V1 SEMANTIC AUDIT: PASS"
)
