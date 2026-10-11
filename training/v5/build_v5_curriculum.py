"""
Sentinel V5 full relational curriculum builder.

Architecture
------------

Frozen conceptual ontology:
    training/v5/v5_family_specs.py

Executable family semantics:
    training/v5/v5_family_realizers.py

Generic relational operator engine:
    training/v5/v5_relational_engine.py

This builder creates:

TRAIN
    32 families
    x 4 relational worlds
    x 4 targets
    x 12 controlled surfaces
    = 6144 rows

DEVELOPMENT
    8 held-out families
    x 4 relational worlds
    x 4 targets
    x 8 controlled surfaces
    = 1024 rows

IMPORTANT
---------

- Does NOT read Challenge V1.
- Does NOT read Challenge V2.
- Does NOT evaluate Challenge V2.
- Does NOT train a model.
"""

from collections import Counter, defaultdict
from pathlib import Path

import hashlib
import json
import random
import re

from training.v5.v5_family_realizers import (
    FAMILY_INDEX,
    family_config,
    validate_realizers,
)

from training.v5.v5_relational_engine import (
    LABELS,
    validate_state,
)


BASE = Path(
    "/workspace"
)

OUT_DIR = (
    BASE
    / "training/v5/datasets"
)

TRAIN_OUT = (
    OUT_DIR
    / "sentinel_v5_train.jsonl"
)

DEV_OUT = (
    OUT_DIR
    / "sentinel_v5_development.jsonl"
)


SEED = 20260816


EXPECTED_TRAIN_ROWS = 6144
EXPECTED_DEV_ROWS = 1024


# =====================================================================
# INPUT-ORDER ORTHOGONALIZATION
# =====================================================================

# 0 = question_first
# 1 = evidence_first
#
# This matrix has:
#
# - row sums = 2
# - column sums = 2
# - every cyclic Latin-square label diagonal sum = 2
#
# Therefore input order remains balanced not merely globally, but also
# relative to target identity, evidence world, and gold label.

ORDER_MATRIX = (
    (0, 0, 1, 1),
    (1, 0, 0, 1),
    (1, 1, 0, 0),
    (0, 1, 1, 0),
)


# =====================================================================
# MODEL INSTRUCTION
# =====================================================================

INSTRUCTION = """You are Sentinel, a policy-governance decision model.

Use ONLY the supplied evidence.

Classify the exact question as one of:

YES
NO
CONDITIONAL
INSUFFICIENT

INSUFFICIENT:
Choose INSUFFICIENT when the supplied evidence does not directly govern
the exact issue asked about.

NO:
Choose NO when the operative evidence directly prohibits the exact action
or directly establishes that the required proposition is false.

CONDITIONAL:
Choose CONDITIONAL when the action may be permitted, but one or more
material requirements remain unresolved.

YES:
Choose YES only when the operative evidence permits the exact action and
all material requirements are established as satisfied.

Determine the operative rule before evaluating downstream conditions.

Silence is not prohibition.
Semantic similarity is not evidence.

Return valid JSON only.
""".strip()


# =====================================================================
# QUESTION SURFACES
# =====================================================================

TRAIN_QUESTION_TEMPLATES = [

    (
        "Under the supplied evidence, "
        "may someone {action}?"
    ),

    (
        "Based only on these provisions, "
        "is someone permitted to {action}?"
    ),

    (
        "Considering only the supplied rules, "
        "may someone {action}?"
    ),

    (
        "What decision follows from the evidence "
        "about whether someone may {action}?"
    ),

    (
        "Do the provided provisions permit someone "
        "to {action}?"
    ),

    (
        "Using only this evidence, "
        "is it permitted to {action}?"
    ),

    (
        "Under these supplied provisions, "
        "would someone be allowed to {action}?"
    ),

    (
        "Does the governing evidence authorize "
        "someone to {action}?"
    ),

    (
        "Based strictly on the evidence below, "
        "may someone {action}?"
    ),

    (
        "What classification applies, based only on "
        "the supplied evidence, to whether someone may {action}?"
    ),

    (
        "Do these provisions establish that someone "
        "may {action}?"
    ),

    (
        "From the supplied evidence alone, "
        "is someone allowed to {action}?"
    ),
]


DEVELOPMENT_QUESTION_TEMPLATES = [

    (
        "According to only the provided material, "
        "would it be permissible to {action}?"
    ),

    (
        "What does this evidence establish about "
        "whether someone can {action}?"
    ),

    (
        "Restricting the analysis to these provisions, "
        "is someone authorized to {action}?"
    ),

    (
        "Does the evidence supplied here allow "
        "someone to {action}?"
    ),

    (
        "How should Sentinel classify whether "
        "someone may {action}, using only this evidence?"
    ),

    (
        "Do the operative provisions support permission "
        "for someone to {action}?"
    ),

    (
        "On this evidence alone, "
        "would the requested action—{action}—be permitted?"
    ),

    (
        "Considering no information beyond these provisions, "
        "is there authority for someone to {action}?"
    ),
]


assert len(
    TRAIN_QUESTION_TEMPLATES
) == 12

assert len(
    DEVELOPMENT_QUESTION_TEMPLATES
) == 8

assert (
    set(
        TRAIN_QUESTION_TEMPLATES
    )
    .isdisjoint(
        DEVELOPMENT_QUESTION_TEMPLATES
    )
)


# =====================================================================
# ENTITY SURFACES
# =====================================================================

ENTITY_ROOTS = [

    "Aster",
    "Beryl",
    "Cobalt",
    "Drift",
    "Elm",
    "Flint",
    "Grove",
    "Haven",
    "Iris",
    "Jade",
    "Kite",
    "Lumen",
    "Mesa",
    "North",
    "Opal",
    "Pine",
]


def assets_for(
    family_index,
    asset_kind,
):

    assets = []

    for target_index in range(
        4
    ):

        root = ENTITY_ROOTS[
            (
                family_index * 5
                + target_index * 3
            )
            % len(
                ENTITY_ROOTS
            )
        ]

        assets.append(
            f"{root} {asset_kind}"
        )

    assert len(
        set(
            assets
        )
    ) == 4

    return assets


# =====================================================================
# GENERAL DISTRACTORS
# =====================================================================

GENERAL_DISTRACTORS = [

    (
        "Institution-owned devices must remain listed "
        "in the central equipment inventory."
    ),

    (
        "Annual travel reimbursement requests require "
        "itemized receipts."
    ),

    (
        "Published institutional web pages must satisfy "
        "accessibility requirements."
    ),

    (
        "Administrative accounts require multifactor "
        "authentication."
    ),

    (
        "Routine procurement records must retain the "
        "responsible cost center."
    ),

    (
        "Facilities teams must maintain annual inspection "
        "logs for controlled work areas."
    ),

    (
        "Internal training records are retained according "
        "to the standard records schedule."
    ),

    (
        "Department managers must reconcile equipment "
        "inventories at the end of each quarter."
    ),
]


# =====================================================================
# HELPERS
# =====================================================================

def render_evidence(
    chunks,
):

    return "\n\n".join(
        f"[{eid}] {text}"
        for eid, text
        in chunks
    )


def render_input(
    question,
    evidence,
    order,
):

    if (
        order
        == "evidence_first"
    ):

        return (
            "SUPPLIED EVIDENCE\n\n"
            f"{evidence}"
            "\n\nQUESTION\n\n"
            f"{question}"
        )

    if (
        order
        == "question_first"
    ):

        return (
            "QUESTION\n\n"
            f"{question}"
            "\n\nSUPPLIED EVIDENCE\n\n"
            f"{evidence}"
        )

    raise ValueError(
        f"Unknown input order: {order}"
    )


def hash_text(
    text,
):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def norm(
    text,
):

    return re.sub(
        r"\s+",
        " ",
        str(
            text
        ).strip().lower(),
    )


def evidence_ids(
    text,
):

    return set(
        re.findall(
            r"\[(E\d+)\]",
            text,
        )
    )


# =====================================================================
# SPLIT BUILDER
# =====================================================================

def build_split(
    split,
    question_templates,
):

    family_items = sorted(
        (
            family,
            spec,
        )
        for family, spec
        in FAMILY_INDEX.items()
        if spec[
            "split"
        ] == split
    )

    rows = []

    for family_index, (
        family,
        spec,
    ) in enumerate(
        family_items
    ):

        operator = spec[
            "operator"
        ]

        cfg = family_config(
            family
        )

        assets = assets_for(
            family_index=family_index,
            asset_kind=cfg[
                "asset_kind"
            ],
        )

        relational_group = (
            f"{operator}__{family}"
        )


        # -------------------------------------------------------------
        # Each surface defines both:
        #
        #   question realization
        #   evidence realization/order
        #
        # Evidence remains identical across the four questions inside
        # the same family/world/surface relation.
        # -------------------------------------------------------------

        for surface_index, question_template in enumerate(
            question_templates
        ):

            semantic_variant = (
                surface_index
                % 3
            )


            # ---------------------------------------------------------
            # Questions are world-independent.
            #
            # Therefore the exact same question identity receives
            # each of the four labels across the four worlds.
            # ---------------------------------------------------------

            questions = {}

            for target_index, asset in enumerate(
                assets
            ):

                yes_state = validate_state(
                    operator=operator,
                    family=family,
                    asset=asset,
                    label="YES",
                    variant=semantic_variant,
                )

                questions[
                    target_index
                ] = question_template.format(
                    action=yes_state[
                        "action"
                    ]
                )


            # ---------------------------------------------------------
            # Four Latin-square relational worlds.
            # ---------------------------------------------------------

            for world_index in range(
                4
            ):

                raw_chunks = []

                target_gold = {}


                for target_index, asset in enumerate(
                    assets
                ):

                    label = LABELS[
                        (
                            target_index
                            + world_index
                        )
                        % 4
                    ]

                    state = validate_state(
                        operator=operator,
                        family=family,
                        asset=asset,
                        label=label,
                        variant=semantic_variant,
                    )

                    fact_keys = []


                    # -------------------------------------------------
                    # Governing evidence.
                    # -------------------------------------------------

                    for fact_index, fact in enumerate(
                        state[
                            "governing_facts"
                        ]
                    ):

                        key = (
                            "governing",
                            target_index,
                            fact_index,
                        )

                        fact_keys.append(
                            key
                        )

                        raw_chunks.append(
                            (
                                key,
                                fact,
                            )
                        )


                    # -------------------------------------------------
                    # Target-local non-governing evidence.
                    # -------------------------------------------------

                    for distractor_index, distractor in enumerate(
                        state[
                            "distractors"
                        ]
                    ):

                        raw_chunks.append(
                            (
                                (
                                    "local_distractor",
                                    target_index,
                                    distractor_index,
                                ),
                                distractor,
                            )
                        )


                    target_gold[
                        target_index
                    ] = {

                        "decision":
                            label,

                        "conditions":
                            state[
                                "conditions"
                            ],

                        "rationale":
                            state[
                                "rationale"
                            ],

                        "fact_keys":
                            fact_keys,
                    }


                # -----------------------------------------------------
                # Two world-level unrelated administrative distractors.
                # -----------------------------------------------------

                d1_index = (
                    family_index
                    + world_index
                    + surface_index
                ) % len(
                    GENERAL_DISTRACTORS
                )

                d2_index = (
                    family_index * 3
                    + world_index
                    + surface_index * 2
                    + 1
                ) % len(
                    GENERAL_DISTRACTORS
                )

                if (
                    d2_index
                    == d1_index
                ):

                    d2_index = (
                        d2_index
                        + 1
                    ) % len(
                        GENERAL_DISTRACTORS
                    )


                raw_chunks.append(
                    (
                        (
                            "world_distractor",
                            0,
                        ),
                        GENERAL_DISTRACTORS[
                            d1_index
                        ],
                    )
                )

                raw_chunks.append(
                    (
                        (
                            "world_distractor",
                            1,
                        ),
                        GENERAL_DISTRACTORS[
                            d2_index
                        ],
                    )
                )


                # -----------------------------------------------------
                # Deterministic evidence ordering.
                # -----------------------------------------------------

                split_offset = (
                    0
                    if split == "train"
                    else 1_000_000
                )

                rng = random.Random(
                    SEED
                    + split_offset
                    + family_index * 10_000
                    + world_index * 100
                    + surface_index
                )

                rng.shuffle(
                    raw_chunks
                )


                key_to_eid = {}

                evidence_chunks = []


                for evidence_index, (
                    key,
                    text,
                ) in enumerate(
                    raw_chunks,
                    start=1,
                ):

                    eid = (
                        f"E{evidence_index}"
                    )

                    key_to_eid[
                        key
                    ] = eid

                    evidence_chunks.append(
                        (
                            eid,
                            text,
                        )
                    )


                evidence_text = render_evidence(
                    evidence_chunks
                )

                evidence_hash = hash_text(
                    evidence_text
                )


                # -----------------------------------------------------
                # Same evidence world paired with all four questions.
                # -----------------------------------------------------

                for target_index, asset in enumerate(
                    assets
                ):

                    gold = target_gold[
                        target_index
                    ]

                    question = questions[
                        target_index
                    ]

                    citations = [
                        key_to_eid[
                            key
                        ]
                        for key
                        in gold[
                            "fact_keys"
                        ]
                    ]


                    # -------------------------------------------------
                    # Input-order assignment.
                    #
                    # IMPORTANT:
                    #
                    # Do NOT derive order from (world_index + target_index)
                    # parity because that quantity also determines the
                    # Latin-square label.
                    #
                    # Instead use an orthogonal balanced matrix.
                    # -------------------------------------------------

                    matrix_world = (
                        world_index
                        + family_index
                        + surface_index
                    ) % 4

                    matrix_target = (
                        target_index
                        + 2 * family_index
                        + surface_index
                    ) % 4

                    order_bit = ORDER_MATRIX[
                        matrix_world
                    ][
                        matrix_target
                    ]

                    if order_bit == 0:

                        input_order = (
                            "question_first"
                        )

                    else:

                        input_order = (
                            "evidence_first"
                        )


                    output = {

                        "decision":
                            gold[
                                "decision"
                            ],

                        "citations":
                            citations,

                        "conditions":
                            gold[
                                "conditions"
                            ],

                        "rationale":
                            gold[
                                "rationale"
                            ],
                    }


                    scenario_id = (
                        f"v5__{split}__"
                        f"{operator}__"
                        f"{family}__"
                        f"world{world_index}__"
                        f"target{target_index}__"
                        f"surface{surface_index}"
                    )


                    row = {

                        "asset":
                            asset,

                        "evidence":
                            evidence_text,

                        "evidence_hash":
                            evidence_hash,

                        "family":
                            family,

                        "input":
                            render_input(
                                question,
                                evidence_text,
                                input_order,
                            ),

                        "input_order":
                            input_order,

                        "instruction":
                            INSTRUCTION,

                        "operator":
                            operator,

                        "output":
                            json.dumps(
                                output,
                                sort_keys=True,
                            ),

                        "question":
                            question,

                        "relational_group":
                            relational_group,

                        "scenario_id":
                            scenario_id,

                        "split":
                            split,

                        "surface_variant":
                            surface_index,

                        "target":
                            gold[
                                "decision"
                            ],

                        "target_index":
                            target_index,

                        "world":
                            world_index,
                    }


                    rows.append(
                        row
                    )


    return rows


# =====================================================================
# DATASET AUDIT
# =====================================================================

EXPECTED_ROW_KEYS = {

    "asset",
    "evidence",
    "evidence_hash",
    "family",
    "input",
    "input_order",
    "instruction",
    "operator",
    "output",
    "question",
    "relational_group",
    "scenario_id",
    "split",
    "surface_variant",
    "target",
    "target_index",
    "world",
}


EXPECTED_OUTPUT_KEYS = {

    "decision",
    "citations",
    "conditions",
    "rationale",
}


def audit_split(
    rows,
    split,
    expected_rows,
    expected_family_rows,
):

    print()
    print(
        "=" * 88
    )

    print(
        f"AUDIT: {split.upper()}"
    )

    print(
        "=" * 88
    )


    assert len(
        rows
    ) == expected_rows


    # -------------------------------------------------------------
    # Schema.
    # -------------------------------------------------------------

    for row in rows:

        assert set(
            row
        ) == EXPECTED_ROW_KEYS

        assert row[
            "split"
        ] == split

        assert row[
            "target"
        ] in LABELS

        assert row[
            "input_order"
        ] in {
            "question_first",
            "evidence_first",
        }

        assert row[
            "world"
        ] in {
            0,
            1,
            2,
            3,
        }

        assert row[
            "evidence_hash"
        ] == hash_text(
            row[
                "evidence"
            ]
        )

        output = json.loads(
            row[
                "output"
            ]
        )

        assert set(
            output
        ) == EXPECTED_OUTPUT_KEYS

        assert output[
            "decision"
        ] == row[
            "target"
        ]

        available_ids = evidence_ids(
            row[
                "evidence"
            ]
        )

        assert set(
            output[
                "citations"
            ]
        ).issubset(
            available_ids
        )

        if (
            row[
                "target"
            ]
            == "INSUFFICIENT"
        ):

            assert output[
                "citations"
            ] == []

            assert output[
                "conditions"
            ] == []

        elif (
            row[
                "target"
            ]
            == "CONDITIONAL"
        ):

            assert output[
                "citations"
            ]

            assert output[
                "conditions"
            ]

        else:

            assert output[
                "citations"
            ]

            assert output[
                "conditions"
            ] == []


    # -------------------------------------------------------------
    # Exact balance.
    # -------------------------------------------------------------

    labels = Counter(
        row[
            "target"
        ]
        for row in rows
    )

    expected_label_count = (
        expected_rows
        // 4
    )

    assert labels == Counter({
        label:
            expected_label_count
        for label
        in LABELS
    })


    orders = Counter(
        row[
            "input_order"
        ]
        for row in rows
    )

    assert orders == Counter({

        "question_first":
            expected_rows
            // 2,

        "evidence_first":
            expected_rows
            // 2,
    })


    # -------------------------------------------------------------
    # CONDITIONAL INPUT-ORDER INDEPENDENCE
    #
    # Global 50/50 balance is insufficient.
    #
    # We require 2/2 order balance inside:
    #
    #   family + surface + target
    #   family + surface + evidence world
    #   family + surface + label
    #
    # This prevents order from acting as a hidden label-parity cue.
    # -------------------------------------------------------------

    expected_local_order = Counter({
        "question_first": 2,
        "evidence_first": 2,
    })


    order_by_target = defaultdict(
        Counter
    )

    order_by_world = defaultdict(
        Counter
    )

    order_by_label = defaultdict(
        Counter
    )


    for row in rows:

        order_by_target[
            (
                row[
                    "family"
                ],
                row[
                    "surface_variant"
                ],
                row[
                    "target_index"
                ],
            )
        ][
            row[
                "input_order"
            ]
        ] += 1


        order_by_world[
            (
                row[
                    "family"
                ],
                row[
                    "surface_variant"
                ],
                row[
                    "world"
                ],
            )
        ][
            row[
                "input_order"
            ]
        ] += 1


        order_by_label[
            (
                row[
                    "family"
                ],
                row[
                    "surface_variant"
                ],
                row[
                    "target"
                ],
            )
        ][
            row[
                "input_order"
            ]
        ] += 1


    assert all(
        counter
        == expected_local_order
        for counter
        in order_by_target.values()
    )


    assert all(
        counter
        == expected_local_order
        for counter
        in order_by_world.values()
    )


    assert all(
        counter
        == expected_local_order
        for counter
        in order_by_label.values()
    )


    print(
        "order balance by target groups:",
        len(
            order_by_target
        ),
        "PASS",
    )

    print(
        "order balance by evidence-world groups:",
        len(
            order_by_world
        ),
        "PASS",
    )

    print(
        "order balance by label groups:",
        len(
            order_by_label
        ),
        "PASS",
    )


    # -------------------------------------------------------------
    # Family counts.
    # -------------------------------------------------------------

    families = Counter(
        row[
            "family"
        ]
        for row in rows
    )

    assert all(
        count
        == expected_family_rows
        for count
        in families.values()
    )


    expected_family_count = (
        32
        if split == "train"
        else 8
    )

    assert len(
        families
    ) == expected_family_count


    # -------------------------------------------------------------
    # Operator counts.
    # -------------------------------------------------------------

    operators = Counter(
        row[
            "operator"
        ]
        for row in rows
    )

    assert len(
        operators
    ) == 8

    expected_operator_rows = (
        expected_rows
        // 8
    )

    assert all(
        count
        == expected_operator_rows
        for count
        in operators.values()
    )


    # -------------------------------------------------------------
    # Scenario identity.
    # -------------------------------------------------------------

    scenario_ids = [
        row[
            "scenario_id"
        ]
        for row
        in rows
    ]

    assert len(
        scenario_ids
    ) == len(
        set(
            scenario_ids
        )
    )


    # -------------------------------------------------------------
    # Duplicate / ambiguity audit.
    # -------------------------------------------------------------

    pair_counts = Counter()

    pair_labels = defaultdict(
        set
    )


    for row in rows:

        pair = (
            norm(
                row[
                    "question"
                ]
            )
            + "\n<PAIR>\n"
            + norm(
                row[
                    "evidence"
                ]
            )
        )

        pair_counts[
            pair
        ] += 1

        pair_labels[
            pair
        ].add(
            row[
                "target"
            ]
        )


    duplicate_pairs = sum(
        count - 1
        for count
        in pair_counts.values()
        if count > 1
    )

    ambiguous_pairs = sum(
        1
        for labels_for_pair
        in pair_labels.values()
        if len(
            labels_for_pair
        ) > 1
    )


    assert duplicate_pairs == 0
    assert ambiguous_pairs == 0


    # -------------------------------------------------------------
    # QUESTION COUNTERFACTUAL.
    #
    # Exact question identity must receive all four labels across
    # different evidence worlds.
    # -------------------------------------------------------------

    question_labels = defaultdict(
        set
    )

    question_counts = Counter()


    for row in rows:

        question_key = norm(
            row[
                "question"
            ]
        )

        question_labels[
            question_key
        ].add(
            row[
                "target"
            ]
        )

        question_counts[
            question_key
        ] += 1


    bad_questions = [
        question
        for question, labels_for_question
        in question_labels.items()
        if labels_for_question
        != set(
            LABELS
        )
    ]


    assert not bad_questions


    # Every intended question identity should normally occur
    # once in each of four worlds.

    assert all(
        count >= 4
        and count % 4 == 0
        for count
        in question_counts.values()
    )


    # -------------------------------------------------------------
    # EVIDENCE COUNTERFACTUAL.
    #
    # Exact evidence identity must be paired with all four labels.
    # -------------------------------------------------------------

    evidence_labels = defaultdict(
        set
    )

    evidence_counts = Counter()


    for row in rows:

        evidence_key = row[
            "evidence_hash"
        ]

        evidence_labels[
            evidence_key
        ].add(
            row[
                "target"
            ]
        )

        evidence_counts[
            evidence_key
        ] += 1


    bad_evidence = [
        evidence_key
        for evidence_key, labels_for_evidence
        in evidence_labels.items()
        if labels_for_evidence
        != set(
            LABELS
        )
    ]


    assert not bad_evidence


    # Every evidence world is intentionally paired with
    # exactly four target questions.

    assert all(
        count == 4
        for count
        in evidence_counts.values()
    )


    # -------------------------------------------------------------
    # Citation cardinality.
    # -------------------------------------------------------------

    citation_counts = Counter()

    for row in rows:

        output = json.loads(
            row[
                "output"
            ]
        )

        citation_counts[
            len(
                output[
                    "citations"
                ]
            )
        ] += 1


    # We deliberately require 0 / 1 / 2 / 3 citation examples.

    assert {
        0,
        1,
        2,
        3,
    }.issubset(
        citation_counts
    )


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
        "operators:",
        dict(
            sorted(
                operators.items()
            )
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
        "unique questions:",
        len(
            question_labels
        )
    )

    print(
        "unique evidence worlds:",
        len(
            evidence_labels
        )
    )

    print(
        "duplicate question/evidence pairs:",
        duplicate_pairs
    )

    print(
        "ambiguous identical pairs:",
        ambiguous_pairs
    )

    print(
        "citation cardinality:",
        dict(
            sorted(
                citation_counts.items()
            )
        )
    )


    return {

        "families":
            set(
                families
            ),

        "questions":
            set(
                question_labels
            ),

        "evidence_hashes":
            set(
                evidence_labels
            ),
    }


# =====================================================================
# WRITE
# =====================================================================

def write_jsonl(
    path,
    rows,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w"
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
                + "\n"
            )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print(
        "=" * 88
    )

    print(
        "SENTINEL V5 FULL RELATIONAL CURRICULUM"
    )

    print(
        "=" * 88
    )


    # -------------------------------------------------------------
    # Ontology ↔ executable semantics contract.
    # -------------------------------------------------------------

    realizer_summary = validate_realizers()

    print(
        "realizer contract:",
        realizer_summary,
    )


    # -------------------------------------------------------------
    # Build in memory first.
    # Nothing is written until all structural generation succeeds.
    # -------------------------------------------------------------

    train_rows = build_split(
        split="train",
        question_templates=TRAIN_QUESTION_TEMPLATES,
    )

    dev_rows = build_split(
        split="development",
        question_templates=DEVELOPMENT_QUESTION_TEMPLATES,
    )


    # -------------------------------------------------------------
    # Full build-time audits.
    # -------------------------------------------------------------

    train_audit = audit_split(
        rows=train_rows,
        split="train",
        expected_rows=EXPECTED_TRAIN_ROWS,
        expected_family_rows=192,
    )

    dev_audit = audit_split(
        rows=dev_rows,
        split="development",
        expected_rows=EXPECTED_DEV_ROWS,
        expected_family_rows=128,
    )


    # -------------------------------------------------------------
    # Split isolation.
    # -------------------------------------------------------------

    print()
    print(
        "=" * 88
    )

    print(
        "TRAIN / DEVELOPMENT ISOLATION"
    )

    print(
        "=" * 88
    )


    family_overlap = (
        train_audit[
            "families"
        ]
        & dev_audit[
            "families"
        ]
    )

    assert not family_overlap


    # Development uses held-out families and disjoint question surfaces.
    # Exact question/evidence overlap should therefore also be zero.

    question_overlap = (
        train_audit[
            "questions"
        ]
        & dev_audit[
            "questions"
        ]
    )

    evidence_overlap = (
        train_audit[
            "evidence_hashes"
        ]
        & dev_audit[
            "evidence_hashes"
        ]
    )


    assert not question_overlap
    assert not evidence_overlap


    print(
        "family overlap:",
        len(
            family_overlap
        )
    )

    print(
        "exact question overlap:",
        len(
            question_overlap
        )
    )

    print(
        "exact evidence-world overlap:",
        len(
            evidence_overlap
        )
    )


    # -------------------------------------------------------------
    # Write only after all checks pass.
    # -------------------------------------------------------------

    write_jsonl(
        TRAIN_OUT,
        train_rows,
    )

    write_jsonl(
        DEV_OUT,
        dev_rows,
    )


    print()
    print(
        "=" * 88
    )

    print(
        "WRITE COMPLETE"
    )

    print(
        "=" * 88
    )

    print(
        "TRAIN:",
        TRAIN_OUT,
    )

    print(
        "DEVELOPMENT:",
        DEV_OUT,
    )

    print()

    print(
        "TRAIN SHA256:",
        hash_text(
            TRAIN_OUT.read_text()
        ),
    )

    print(
        "DEVELOPMENT SHA256:",
        hash_text(
            DEV_OUT.read_text()
        ),
    )

    print()

    print(
        "V5 FULL CURRICULUM BUILD: PASS"
    )

    print(
        "NO MODEL TRAINING PERFORMED"
    )

    print(
        "DATASET NOT YET FROZEN"
    )


if __name__ == "__main__":
    main()
