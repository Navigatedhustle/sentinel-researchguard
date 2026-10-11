from collections import Counter
from pathlib import Path
import ast
import sys


BASE = Path("/workspace")

BLUEPRINT = (
    BASE
    / "experiments/challenge_v3/"
      "challenge_v3_blueprint.py"
)


spec_text = BLUEPRINT.read_text()


errors = []


def check(condition, message):

    if not condition:

        errors.append(
            message
        )


namespace = {}

exec(
    compile(
        spec_text,
        str(
            BLUEPRINT
        ),
        "exec",
    ),
    namespace,
)


LABELS = namespace[
    "LABELS"
]

COMPOSITIONS = namespace[
    "COMPOSITIONS"
]

CAPABILITIES = namespace[
    "COMPOSITION_PRIMARY_CAPABILITY"
]

RELATIONAL_WORLDS = namespace[
    "RELATIONAL_WORLDS"
]

RELATIONAL_TARGETS = namespace[
    "RELATIONAL_TARGETS"
]

EXPECTED = namespace[
    "EXPECTED"
]

PUBLIC_CASE_KEYS = namespace[
    "PUBLIC_CASE_KEYS"
]

GOLD_KEYS = namespace[
    "GOLD_KEYS"
]

relational_label = namespace[
    "relational_label"
]


# ====================================================================
# BASIC STRUCTURE
# ====================================================================

check(
    len(
        LABELS
    ) == 4,
    "expected four labels",
)

check(
    len(
        set(
            LABELS
        )
    ) == 4,
    "labels not unique",
)

check(
    len(
        COMPOSITIONS
    )
    == EXPECTED[
        "compositions"
    ],
    "composition count mismatch",
)

check(
    len(
        set(
            COMPOSITIONS
        )
    )
    == len(
        COMPOSITIONS
    ),
    "duplicate composition",
)

check(
    set(
        COMPOSITIONS
    )
    == set(
        CAPABILITIES
    ),
    "capability mapping coverage mismatch",
)

check(
    len(
        RELATIONAL_WORLDS
    ) == 4,
    "expected four relational worlds",
)

check(
    len(
        RELATIONAL_TARGETS
    ) == 4,
    "expected four relational targets",
)


# ====================================================================
# EXPECTED CARDINALITY
# ====================================================================

check(
    EXPECTED[
        "compositions"
    ]
    *
    EXPECTED[
        "cases_per_composition"
    ]
    ==
    EXPECTED[
        "total_cases"
    ],
    "total case geometry mismatch",
)

check(
    EXPECTED[
        "relational_per_composition"
    ]
    +
    EXPECTED[
        "stress_per_composition"
    ]
    ==
    EXPECTED[
        "cases_per_composition"
    ],
    "per-composition geometry mismatch",
)

check(
    EXPECTED[
        "relational_cases"
    ]
    +
    EXPECTED[
        "stress_cases"
    ]
    ==
    EXPECTED[
        "total_cases"
    ],
    "question-type total mismatch",
)

check(
    EXPECTED[
        "cases_per_label"
    ]
    * 4
    ==
    EXPECTED[
        "total_cases"
    ],
    "label total mismatch",
)

check(
    EXPECTED[
        "evidence_first"
    ]
    +
    EXPECTED[
        "question_first"
    ]
    ==
    EXPECTED[
        "total_cases"
    ],
    "input-order total mismatch",
)


# ====================================================================
# RELATIONAL LATIN-SQUARE INVARIANTS
# ====================================================================

relational_rows = []


for world_index, world in enumerate(
    RELATIONAL_WORLDS
):

    for target_index, target in enumerate(
        RELATIONAL_TARGETS
    ):

        relational_rows.append({

            "world":
                world,

            "target":
                target,

            "label":
                relational_label(
                    world_index,
                    target_index,
                ),
        })


check(
    len(
        relational_rows
    ) == 16,
    "relational system must contain 16 positions",
)


labels = Counter(
    row[
        "label"
    ]
    for row in relational_rows
)


check(
    labels
    == Counter({
        "YES": 4,
        "NO": 4,
        "CONDITIONAL": 4,
        "INSUFFICIENT": 4,
    }),
    "relational labels not balanced",
)


for target in RELATIONAL_TARGETS:

    target_labels = {
        row[
            "label"
        ]
        for row
        in relational_rows
        if row[
            "target"
        ] == target
    }


    check(
        target_labels
        == set(
            LABELS
        ),
        f"target {target} does not span all labels",
    )


for world in RELATIONAL_WORLDS:

    world_labels = {
        row[
            "label"
        ]
        for row
        in relational_rows
        if row[
            "world"
        ] == world
    }


    check(
        world_labels
        == set(
            LABELS
        ),
        f"world {world} does not span all labels",
    )


# ====================================================================
# SCHEMA
# ====================================================================

check(
    len(
        PUBLIC_CASE_KEYS
    ) == 9,
    "public schema key count mismatch",
)

check(
    set(
        GOLD_KEYS
    )
    == {
        "decision",
        "citations",
        "conditions",
        "rationale",
    },
    "gold schema mismatch",
)


# ====================================================================
# INDEPENDENCE STATIC CHECK
# ====================================================================

tree = ast.parse(
    spec_text
)


for node in ast.walk(
    tree
):

    if isinstance(
        node,
        ast.Import,
    ):

        for alias in node.names:

            if alias.name.startswith(
                "training.v6"
            ):

                errors.append(
                    "Challenge V3 blueprint imports training.v6"
                )


    if isinstance(
        node,
        ast.ImportFrom,
    ):

        if (
            node.module
            and node.module.startswith(
                "training.v6"
            )
        ):

            errors.append(
                "Challenge V3 blueprint imports training.v6"
            )


# ====================================================================
# FORBIDDEN PRIOR BENCHMARK CASE IDS
# ====================================================================

lower = spec_text.lower()


for forbidden in (
    "challenge_v1__",
    "challenge_v2__",
):

    check(
        forbidden not in lower,
        f"forbidden prior benchmark case identifier: {forbidden}",
    )


# ====================================================================
# REPORT
# ====================================================================

print("=" * 92)
print("SENTINEL CHALLENGE V3 BLUEPRINT VALIDATION")
print("=" * 92)

print(
    "compositions:",
    len(
        COMPOSITIONS
    )
)

print(
    "cases per composition:",
    EXPECTED[
        "cases_per_composition"
    ]
)

print(
    "relational cases:",
    EXPECTED[
        "relational_cases"
    ]
)

print(
    "stress cases:",
    EXPECTED[
        "stress_cases"
    ]
)

print(
    "total cases:",
    EXPECTED[
        "total_cases"
    ]
)

print(
    "labels:",
    {
        label:
            EXPECTED[
                "cases_per_label"
            ]
        for label
        in LABELS
    }
)

print(
    "input order target:",
    {
        "evidence_first":
            EXPECTED[
                "evidence_first"
            ],

        "question_first":
            EXPECTED[
                "question_first"
            ],
    }
)

print(
    "public schema keys:",
    len(
        PUBLIC_CASE_KEYS
    )
)

print(
    "gold schema:",
    GOLD_KEYS
)

print(
    "training.v6 imports:",
    "0 required"
)

print()
print(
    "errors:",
    len(
        errors
    )
)


if errors:

    for error in errors:

        print(
            "ERROR:",
            error
        )


    print()
    print(
        "SENTINEL CHALLENGE V3 BLUEPRINT: FAIL"
    )

    sys.exit(
        1
    )


print()
print(
    "SENTINEL CHALLENGE V3 BLUEPRINT: PASS"
)
