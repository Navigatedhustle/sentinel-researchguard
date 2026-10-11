from collections import Counter, defaultdict
import sys


from training.v6.v6_operator_ontology import (
    FAMILY_TO_OPERATOR as ONTOLOGY_FAMILY_TO_OPERATOR,
    FAMILY_TO_SPLIT,
)

from training.v6.v6_relational_state_engine import (
    decide,
)

from training.v6.pilot.v6_relational_pilot_spec import (
    EXPECTED_FAMILY_COUNT,
    EXPECTED_GROUPS_PER_FAMILY,
    EXPECTED_ROWS_PER_FAMILY,
    EXPECTED_ROWS_PER_GROUP,
    EXPECTED_TARGETS_PER_WORLD,
    EXPECTED_TOTAL_ROWS,
    EXPECTED_WORLDS_PER_GROUP,
    FAMILY_TO_OPERATOR,
    GROUP_VARIANTS,
    LABELS,
    PILOT_FAMILIES,
    TARGET_IDS,
    WORLD_IDS,
    build_state,
    input_order,
    intended_label,
)


errors = []

rows = []


def check(
    condition,
    message,
):

    if not condition:

        errors.append(
            message
        )


# ====================================================================
# STRUCTURE
# ====================================================================

check(
    len(
        PILOT_FAMILIES
    )
    == EXPECTED_FAMILY_COUNT,
    "pilot family count mismatch",
)

check(
    len(
        GROUP_VARIANTS
    )
    == EXPECTED_GROUPS_PER_FAMILY,
    "group variant count mismatch",
)

check(
    len(
        WORLD_IDS
    )
    == EXPECTED_WORLDS_PER_GROUP,
    "world count mismatch",
)

check(
    len(
        TARGET_IDS
    )
    == EXPECTED_TARGETS_PER_WORLD,
    "target count mismatch",
)


# ====================================================================
# ONTOLOGY MEMBERSHIP
# ====================================================================

for family in PILOT_FAMILIES:

    check(
        family in ONTOLOGY_FAMILY_TO_OPERATOR,
        f"{family}: family absent from frozen ontology",
    )


    if family in ONTOLOGY_FAMILY_TO_OPERATOR:

        check(
            FAMILY_TO_SPLIT[
                family
            ]
            == "train",
            f"{family}: pilot must use TRAIN family",
        )


        check(
            ONTOLOGY_FAMILY_TO_OPERATOR[
                family
            ]
            == FAMILY_TO_OPERATOR[
                family
            ],
            f"{family}: operator mismatch",
        )


# ====================================================================
# REGENERATE EVERY INTENDED PILOT SEMANTIC ROW
# ====================================================================

for family_index, family in enumerate(
    PILOT_FAMILIES
):

    for variant in GROUP_VARIANTS:

        relational_group = (
            f"{family}__pilot_g{variant}"
        )


        for world_index, world_id in enumerate(
            WORLD_IDS
        ):

            for target_index, target_id in enumerate(
                TARGET_IDS
            ):

                intended = intended_label(
                    world_index,
                    target_index,
                )


                state = build_state(
                    family,
                    intended,
                    variant,
                )


                result = decide(
                    state
                )


                actual = (
                    result.decision.value
                )


                if actual != intended:

                    errors.append(
                        f"{family}/g{variant}/"
                        f"{world_id}/{target_id}: "
                        f"intended {intended}, engine {actual}"
                    )


                order = input_order(
                    family_index=family_index,
                    group_variant=variant,
                    world_index=world_index,
                    target_index=target_index,
                )


                rows.append({
                    "family":
                        family,

                    "relational_group":
                        relational_group,

                    "world":
                        world_id,

                    "target":
                        target_id,

                    "label":
                        actual,

                    "input_order":
                        order,

                    "conditions":
                        result.conditions,
                })


# ====================================================================
# CARDINALITY
# ====================================================================

check(
    len(
        rows
    )
    == EXPECTED_TOTAL_ROWS,
    "pilot total row count mismatch",
)


family_counts = Counter(
    row[
        "family"
    ]
    for row in rows
)


for family in PILOT_FAMILIES:

    check(
        family_counts[
            family
        ]
        == EXPECTED_ROWS_PER_FAMILY,
        f"{family}: row count mismatch",
    )


group_counts = Counter(
    row[
        "relational_group"
    ]
    for row in rows
)


for group, count in group_counts.items():

    check(
        count
        == EXPECTED_ROWS_PER_GROUP,
        f"{group}: expected 16 rows",
    )


# ====================================================================
# LABEL BALANCE
# ====================================================================

label_counts = Counter(
    row[
        "label"
    ]
    for row in rows
)


check(
    label_counts
    == Counter({
        "YES": 32,
        "NO": 32,
        "CONDITIONAL": 32,
        "INSUFFICIENT": 32,
    }),
    "global label distribution mismatch",
)


for family in PILOT_FAMILIES:

    counts = Counter(
        row[
            "label"
        ]
        for row in rows
        if row[
            "family"
        ] == family
    )


    check(
        counts
        == Counter({
            "YES": 8,
            "NO": 8,
            "CONDITIONAL": 8,
            "INSUFFICIENT": 8,
        }),
        f"{family}: label distribution mismatch",
    )


for group in group_counts:

    counts = Counter(
        row[
            "label"
        ]
        for row in rows
        if row[
            "relational_group"
        ] == group
    )


    check(
        counts
        == Counter({
            "YES": 4,
            "NO": 4,
            "CONDITIONAL": 4,
            "INSUFFICIENT": 4,
        }),
        f"{group}: relational label distribution mismatch",
    )


# ====================================================================
# SAME TARGET ACROSS WORLDS MUST SEE ALL LABELS
# ====================================================================

for family in PILOT_FAMILIES:

    for variant in GROUP_VARIANTS:

        group = (
            f"{family}__pilot_g{variant}"
        )


        for target in TARGET_IDS:

            labels = {
                row[
                    "label"
                ]
                for row in rows
                if row[
                    "relational_group"
                ] == group
                and row[
                    "target"
                ] == target
            }


            check(
                labels
                == set(
                    LABELS
                ),
                (
                    f"{group}/{target}: "
                    "same-target worlds do not span all labels"
                ),
            )


# ====================================================================
# SAME WORLD ACROSS TARGETS MUST SEE ALL LABELS
# ====================================================================

for family in PILOT_FAMILIES:

    for variant in GROUP_VARIANTS:

        group = (
            f"{family}__pilot_g{variant}"
        )


        for world in WORLD_IDS:

            labels = {
                row[
                    "label"
                ]
                for row in rows
                if row[
                    "relational_group"
                ] == group
                and row[
                    "world"
                ] == world
            }


            check(
                labels
                == set(
                    LABELS
                ),
                (
                    f"{group}/{world}: "
                    "same-world targets do not span all labels"
                ),
            )


# ====================================================================
# INPUT ORDER BALANCE
# ====================================================================

order_counts = Counter(
    row[
        "input_order"
    ]
    for row in rows
)


check(
    order_counts
    == Counter({
        "evidence_first": 64,
        "question_first": 64,
    }),
    "global input-order balance mismatch",
)


for label in LABELS:

    counts = Counter(
        row[
            "input_order"
        ]
        for row in rows
        if row[
            "label"
        ] == label
    )


    check(
        counts
        == Counter({
            "evidence_first": 16,
            "question_first": 16,
        }),
        f"{label}: input-order leak",
    )


# ====================================================================
# LEDGER INVARIANTS
# ====================================================================

for row in rows:

    conditions = row[
        "conditions"
    ]

    label = row[
        "label"
    ]


    if label == "INSUFFICIENT":

        check(
            conditions == (),
            "INSUFFICIENT row has nonempty ledger",
        )


    elif label == "YES":

        check(
            not any(
                item.startswith(
                    "FAILED | "
                )
                or item.startswith(
                    "UNRESOLVED | "
                )
                for item in conditions
            ),
            "YES row contains non-satisfied condition",
        )


    elif label == "CONDITIONAL":

        check(
            any(
                item.startswith(
                    "UNRESOLVED | "
                )
                for item in conditions
            ),
            "CONDITIONAL row lacks unresolved condition",
        )


        check(
            not any(
                item.startswith(
                    "FAILED | "
                )
                for item in conditions
            ),
            "CONDITIONAL row contains failed condition",
        )


# ====================================================================
# REPORT
# ====================================================================

print("=" * 88)
print("SENTINEL V6 RELATIONAL PILOT SPEC VALIDATION")
print("=" * 88)

print(
    "families:",
    len(
        PILOT_FAMILIES
    )
)

print(
    "relational groups:",
    len(
        group_counts
    )
)

print(
    "semantic rows:",
    len(
        rows
    )
)

print(
    "label distribution:",
    dict(
        sorted(
            label_counts.items()
        )
    )
)

print(
    "input order:",
    dict(
        sorted(
            order_counts.items()
        )
    )
)

print()


for family in PILOT_FAMILIES:

    counts = Counter(
        row[
            "label"
        ]
        for row in rows
        if row[
            "family"
        ] == family
    )


    print(
        family
    )

    print(
        "  operator:",
        FAMILY_TO_OPERATOR[
            family
        ]
    )

    print(
        "  rows:",
        family_counts[
            family
        ]
    )

    print(
        "  labels:",
        dict(
            sorted(
                counts.items()
            )
        )
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
        "SENTINEL V6 RELATIONAL PILOT SPEC: FAIL"
    )

    sys.exit(
        1
    )


print()
print(
    "SENTINEL V6 RELATIONAL PILOT SPEC: PASS"
)
