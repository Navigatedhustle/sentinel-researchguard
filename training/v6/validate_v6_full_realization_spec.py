from collections import Counter, defaultdict
import sys


from training.v6.v6_operator_ontology import (
    DEVELOPMENT_FAMILIES,
    OPERATOR_ORDER,
    TRAIN_FAMILIES,
)

from training.v6.v6_full_realization_spec import (
    DEVELOPMENT_SURFACES,
    EXPECTED,
    LABELS,
    SEMANTIC_GROUPS,
    TARGET_IDS,
    TRAIN_SURFACES,
    WORLD_IDS,
    iter_virtual_rows,
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


# ====================================================================
# FROZEN STRUCTURE
# ====================================================================

check(
    len(
        TRAIN_FAMILIES
    ) == 32,
    "expected 32 TRAIN families",
)

check(
    len(
        DEVELOPMENT_FAMILIES
    ) == 8,
    "expected 8 DEVELOPMENT families",
)

check(
    not (
        set(
            TRAIN_FAMILIES
        )
        &
        set(
            DEVELOPMENT_FAMILIES
        )
    ),
    "TRAIN/DEVELOPMENT family overlap",
)

check(
    len(
        SEMANTIC_GROUPS
    ) == 4,
    "expected four semantic groups",
)

check(
    len(
        TRAIN_SURFACES
    ) == 3,
    "expected three TRAIN surfaces",
)

check(
    len(
        DEVELOPMENT_SURFACES
    ) == 2,
    "expected two DEVELOPMENT surfaces",
)

check(
    not (
        set(
            TRAIN_SURFACES
        )
        &
        set(
            DEVELOPMENT_SURFACES
        )
    ),
    "TRAIN/DEVELOPMENT surface overlap",
)

check(
    len(
        WORLD_IDS
    ) == 4,
    "expected four worlds",
)

check(
    len(
        TARGET_IDS
    ) == 4,
    "expected four targets",
)


# ====================================================================
# VIRTUAL DATASET VALIDATION
# ====================================================================

all_split_rows = {}


for split in (
    "train",
    "development",
):

    rows = list(
        iter_virtual_rows(
            split
        )
    )

    all_split_rows[
        split
    ] = rows


    expected = EXPECTED[
        split
    ]


    # ---------------------------------------------------------------
    # Cardinality
    # ---------------------------------------------------------------

    check(
        len(
            rows
        )
        == expected[
            "rows"
        ],
        f"{split}: row count mismatch",
    )


    families = {
        row[
            "family"
        ]
        for row in rows
    }


    check(
        len(
            families
        )
        == expected[
            "families"
        ],
        f"{split}: family count mismatch",
    )


    relational_groups = defaultdict(
        list
    )

    evidence_worlds = defaultdict(
        list
    )

    question_groups = defaultdict(
        list
    )


    for row in rows:

        relational_groups[
            row[
                "relational_group"
            ]
        ].append(
            row
        )


        evidence_worlds[
            (
                row[
                    "relational_group"
                ],
                row[
                    "world"
                ],
            )
        ].append(
            row
        )


        question_groups[
            (
                row[
                    "relational_group"
                ],
                row[
                    "target_id"
                ],
            )
        ].append(
            row
        )


    check(
        len(
            relational_groups
        )
        == expected[
            "relational_groups"
        ],
        f"{split}: relational-group count mismatch",
    )


    check(
        len(
            evidence_worlds
        )
        == expected[
            "evidence_worlds"
        ],
        f"{split}: evidence-world count mismatch",
    )


    check(
        len(
            question_groups
        )
        == expected[
            "question_groups"
        ],
        f"{split}: question-group count mismatch",
    )


    # ---------------------------------------------------------------
    # Global labels
    # ---------------------------------------------------------------

    labels = Counter(
        row[
            "label"
        ]
        for row in rows
    )


    expected_labels = Counter({
        label:
            expected[
                "rows_per_label"
            ]
        for label
        in LABELS
    })


    check(
        labels
        == expected_labels,
        f"{split}: global label imbalance {labels}",
    )


    # ---------------------------------------------------------------
    # Per-family labels
    # ---------------------------------------------------------------

    for family in families:

        family_rows = [
            row
            for row in rows
            if row[
                "family"
            ] == family
        ]


        family_labels = Counter(
            row[
                "label"
            ]
            for row
            in family_rows
        )


        per_label = (
            len(
                family_rows
            )
            // 4
        )


        check(
            family_labels
            == Counter({
                label:
                    per_label
                for label
                in LABELS
            }),
            f"{split}/{family}: label imbalance",
        )


    # ---------------------------------------------------------------
    # Per-relational-group structure
    # ---------------------------------------------------------------

    for group, group_rows in relational_groups.items():

        check(
            len(
                group_rows
            ) == 16,
            f"{group}: expected 16 rows",
        )


        group_labels = Counter(
            row[
                "label"
            ]
            for row
            in group_rows
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


        orders = Counter(
            row[
                "input_order"
            ]
            for row
            in group_rows
        )


        check(
            orders
            == Counter({
                "evidence_first": 8,
                "question_first": 8,
            }),
            f"{group}: input-order imbalance",
        )


        for label in LABELS:

            label_orders = Counter(
                row[
                    "input_order"
                ]
                for row
                in group_rows
                if row[
                    "label"
                ] == label
            )


            check(
                label_orders
                == Counter({
                    "evidence_first": 2,
                    "question_first": 2,
                }),
                (
                    f"{group}/{label}: "
                    "input-order label leak"
                ),
            )


    # ---------------------------------------------------------------
    # Same evidence world -> all four labels
    # ---------------------------------------------------------------

    for key, group_rows in evidence_worlds.items():

        check(
            len(
                group_rows
            ) == 4,
            f"{key}: expected four target rows",
        )


        labels_here = {
            row[
                "label"
            ]
            for row
            in group_rows
        }


        check(
            labels_here
            == set(
                LABELS
            ),
            f"{key}: evidence world does not span all labels",
        )


    # ---------------------------------------------------------------
    # Same target question -> all four labels
    # ---------------------------------------------------------------

    for key, group_rows in question_groups.items():

        check(
            len(
                group_rows
            ) == 4,
            f"{key}: expected four world rows",
        )


        labels_here = {
            row[
                "label"
            ]
            for row
            in group_rows
        }


        check(
            labels_here
            == set(
                LABELS
            ),
            f"{key}: target group does not span all labels",
        )


    # ---------------------------------------------------------------
    # Observable metadata marginal balance
    # ---------------------------------------------------------------

    for field in (
        "family",
        "operator",
        "semantic_group",
        "surface",
        "input_order",
    ):

        buckets = defaultdict(
            Counter
        )


        for row in rows:

            buckets[
                row[
                    field
                ]
            ][
                row[
                    "label"
                ]
            ] += 1


        for value, counts in buckets.items():

            values = list(
                counts.values()
            )


            check(
                len(
                    counts
                ) == 4
                and len(
                    set(
                        values
                    )
                ) == 1,
                (
                    f"{split}/{field}={value}: "
                    "metadata marginal label imbalance"
                ),
            )


    # ---------------------------------------------------------------
    # Hidden construction sanity
    # ---------------------------------------------------------------

    for field in (
        "world_index",
        "target_index",
    ):

        buckets = defaultdict(
            Counter
        )


        for row in rows:

            buckets[
                row[
                    field
                ]
            ][
                row[
                    "label"
                ]
            ] += 1


        for value, counts in buckets.items():

            check(
                len(
                    counts
                ) == 4
                and len(
                    set(
                        counts.values()
                    )
                ) == 1,
                f"{split}/{field}={value}: hidden marginal leak",
            )


    latin_correct = sum(

        LABELS[
            (
                row[
                    "world_index"
                ]
                +
                row[
                    "target_index"
                ]
            )
            % 4
        ]
        == row[
            "label"
        ]

        for row
        in rows
    )


    check(
        latin_correct
        == len(
            rows
        ),
        f"{split}: hidden Latin oracle not 100%",
    )


# ====================================================================
# REPORT
# ====================================================================

print("=" * 92)
print("SENTINEL V6 FULL REALIZATION SPEC VALIDATION")
print("=" * 92)

for split in (
    "train",
    "development",
):

    rows = all_split_rows[
        split
    ]

    expected = EXPECTED[
        split
    ]


    print()
    print(
        split.upper()
    )

    print(
        "  rows:",
        len(
            rows
        )
    )

    print(
        "  families:",
        len({
            row[
                "family"
            ]
            for row in rows
        })
    )

    print(
        "  relational groups:",
        len({
            row[
                "relational_group"
            ]
            for row in rows
        })
    )

    print(
        "  labels:",
        dict(
            sorted(
                Counter(
                    row[
                        "label"
                    ]
                    for row in rows
                ).items()
            )
        )
    )

    print(
        "  expected rows:",
        expected[
            "rows"
        ]
    )


print()
print(
    "operators:",
    len(
        OPERATOR_ORDER
    )
)

print(
    "semantic groups per family:",
    len(
        SEMANTIC_GROUPS
    )
)

print(
    "train surfaces:",
    len(
        TRAIN_SURFACES
    )
)

print(
    "development surfaces:",
    len(
        DEVELOPMENT_SURFACES
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

    for error in errors[
        :100
    ]:

        print(
            "ERROR:",
            error
        )


    print()
    print(
        "SENTINEL V6 FULL REALIZATION SPEC: FAIL"
    )

    sys.exit(
        1
    )


print()
print(
    "SENTINEL V6 FULL REALIZATION SPEC: PASS"
)
