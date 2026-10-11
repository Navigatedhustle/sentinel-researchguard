"""
Sentinel V5 relational pilot.

PURPOSE:

Test whether one generalized relational engine can express all
eight V5 reasoning operators before full curriculum generation.

This builder:

- uses ONE TRAIN family from each operator
- produces 8 relational groups
- uses 4 target entities per group
- uses 4 Latin-square evidence worlds
- uses 2 question surface variants
- produces 256 rows total
- balances all four labels
- supports multi-citation gold sets
- does NOT read Challenge V2
- does NOT evaluate any model
"""

from collections import Counter
from pathlib import Path
import hashlib
import json
import random

from training.v5.v5_family_specs import OPERATORS


BASE = Path("/workspace")

OUT = (
    BASE
    / "training/v5/pilot/"
      "sentinel_v5_relational_pilot.jsonl"
)

SEED = 20260816

LABELS = [
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
]


SELECTED_FAMILIES = {

    "operative_rule_selection":
        "project_charter_priority",

    "prohibition_short_circuit":
        "blocked_interface_route",

    "permission_condition_ledger":
        "review_notice_ledger",

    "exception_activation_chain":
        "maintenance_window_exception",

    "authority_state_resolution":
        "temporary_reviewer_authority",

    "category_rule_composition":
        "managed_compute_category",

    "scope_destination_resolution":
        "recipient_scope_resolution",

    "evidence_relevance_boundary":
        "adjacent_channel_nonrule",
}


ASSET_SETS = {

    "operative_rule_selection": [
        "Aster packet",
        "Beryl packet",
        "Cobalt packet",
        "Drift packet",
    ],

    "prohibition_short_circuit": [
        "Elm records",
        "Flint records",
        "Grove records",
        "Haven records",
    ],

    "permission_condition_ledger": [
        "Iris archive",
        "Jade archive",
        "Kite archive",
        "Lumen archive",
    ],

    "exception_activation_chain": [
        "Mesa dataset",
        "North dataset",
        "Opal dataset",
        "Pine dataset",
    ],

    "authority_state_resolution": [
        "Quartz files",
        "Reed files",
        "Slate files",
        "Thorn files",
    ],

    "category_rule_composition": [
        "Umber corpus",
        "Vale corpus",
        "Wren corpus",
        "Xylem corpus",
    ],

    "scope_destination_resolution": [
        "Yarrow records",
        "Zenith records",
        "Amber records",
        "Brook records",
    ],

    "evidence_relevance_boundary": [
        "Crest files",
        "Dawn files",
        "Ember files",
        "Fjord files",
    ],
}


DISTRACTORS = [
    (
        "Institution-owned devices must remain "
        "listed in the equipment inventory."
    ),

    (
        "Annual travel reimbursement requests "
        "must include itemized receipts."
    ),

    (
        "Published institutional web pages must "
        "meet accessibility requirements."
    ),

    (
        "Administrative accounts require "
        "multifactor authentication."
    ),
]


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


def family_exists(
    operator,
    family,
):

    names = {
        item["family"]
        for item
        in OPERATORS[
            operator
        ][
            "train"
        ]
    }

    return (
        family
        in names
    )


def state_for(
    operator,
    asset,
    label,
):
    """
    Return:

        action
        governing fact strings
        conditions
        rationale

    Every governing fact explicitly names the target asset so
    citation minimality can later be audited mechanically.
    """

    # ========================================================
    # 1. OPERATIVE RULE SELECTION
    # ========================================================

    if (
        operator
        == "operative_rule_selection"
    ):

        action = (
            f"stage the {asset} "
            "in the project exchange zone"
        )

        precedence = (
            f"For the {asset}, when the active project charter "
            "conflicts with the organization-wide staging rule, "
            "the active project charter is the operative rule."
        )

        if label == "YES":

            facts = [
                precedence,

                (
                    f"The organization-wide rule would block staging "
                    f"of the {asset}, but the active project charter "
                    "permits staging in the project exchange zone "
                    "after technical review. Technical review is complete."
                ),
            ]

            conditions = []

            rationale = (
                "The project charter is operative and its "
                "material review requirement is satisfied."
            )

        elif label == "NO":

            facts = [
                precedence,

                (
                    f"The organization-wide rule would permit staging "
                    f"of the {asset}, but the active project charter "
                    "expressly prohibits staging it in the "
                    "project exchange zone."
                ),
            ]

            conditions = []

            rationale = (
                "The higher-priority project charter controls "
                "and prohibits the requested staging."
            )

        elif label == "CONDITIONAL":

            facts = [
                precedence,

                (
                    f"The active project charter permits staging "
                    f"of the {asset} in the project exchange zone "
                    "only after technical review. Technical-review "
                    "status is not established."
                ),
            ]

            conditions = [
                "technical review unresolved",
            ]

            rationale = (
                "The operative charter may permit staging, "
                "but technical review remains unresolved."
            )

        else:

            facts = [
                (
                    f"The project charter for the {asset} assigns "
                    "reporting ownership and monthly inventory duties. "
                    "No supplied rule governs whether it may be staged "
                    "in the project exchange zone."
                ),
            ]

            conditions = []

            rationale = (
                "The supplied charter is administratively relevant "
                "but does not govern the requested staging."
            )

        return (
            action,
            facts,
            conditions,
            rationale,
        )


    # ========================================================
    # 2. PROHIBITION SHORT-CIRCUIT
    # ========================================================

    if (
        operator
        == "prohibition_short_circuit"
    ):

        action = (
            f"transmit the {asset} "
            "through Interface R"
        )

        if label == "YES":

            facts = [
                (
                    f"The controlling interface rule permits the "
                    f"{asset} to be transmitted through Interface R "
                    "when transport encryption is active. "
                    "Transport encryption is active."
                ),
            ]

            conditions = []

            rationale = (
                "The controlling rule permits the route and "
                "its encryption requirement is satisfied."
            )

        elif label == "NO":

            facts = [
                (
                    f"The controlling interface rule expressly "
                    f"prohibits transmission of the {asset} "
                    "through Interface R. A separate annual review "
                    "status has not yet been established."
                ),
            ]

            conditions = []

            rationale = (
                "The direct Interface R prohibition resolves "
                "the request despite unrelated unresolved review status."
            )

        elif label == "CONDITIONAL":

            facts = [
                (
                    f"The controlling interface rule permits "
                    f"transmission of the {asset} through Interface R "
                    "only with release authorization. "
                    "Release-authorization status is not established."
                ),
            ]

            conditions = [
                "release authorization unresolved",
            ]

            rationale = (
                "The route may be permitted, but release "
                "authorization remains unresolved."
            )

        else:

            facts = [
                (
                    f"The supplied rule governs transmission of the "
                    f"{asset} through Interface S. It provides no rule "
                    "for transmission through Interface R."
                ),
            ]

            conditions = []

            rationale = (
                "The supplied evidence governs a different interface."
            )

        return (
            action,
            facts,
            conditions,
            rationale,
        )


    # ========================================================
    # 3. PERMISSION CONDITION LEDGER
    # ========================================================

    if (
        operator
        == "permission_condition_ledger"
    ):

        action = (
            f"release the {asset} "
            "to the registered research workspace"
        )

        permission = (
            f"Release of the {asset} to the registered research "
            "workspace is permitted only after technical review "
            "and advance notice."
        )

        if label == "YES":

            facts = [
                permission,

                (
                    f"For the {asset}, technical review is complete "
                    "and advance notice has been recorded."
                ),
            ]

            conditions = []

            rationale = (
                "Direct permission applies and both material "
                "requirements are satisfied."
            )

        elif label == "NO":

            facts = [
                (
                    f"The governing release rule expressly prohibits "
                    f"release of the {asset} to the registered "
                    "research workspace."
                ),
            ]

            conditions = []

            rationale = (
                "The governing rule directly prohibits the release."
            )

        elif label == "CONDITIONAL":

            facts = [
                permission,

                (
                    f"For the {asset}, technical review is complete, "
                    "but advance-notice status is not established."
                ),
            ]

            conditions = [
                "advance notice unresolved",
            ]

            rationale = (
                "Permission exists, but one material requirement "
                "remains unresolved."
            )

        else:

            facts = [
                (
                    f"The {asset} is listed in the internal archive "
                    "catalog and subject to quarterly inventory review. "
                    "No supplied provision governs release to the "
                    "registered research workspace."
                ),
            ]

            conditions = []

            rationale = (
                "The evidence does not supply a governing release rule."
            )

        return (
            action,
            facts,
            conditions,
            rationale,
        )


    # ========================================================
    # 4. EXCEPTION ACTIVATION CHAIN
    # ========================================================

    if (
        operator
        == "exception_activation_chain"
    ):

        action = (
            f"move the {asset} outside "
            "the normal processing enclave"
        )

        base = (
            f"The standing rule prohibits moving the {asset} "
            "outside the normal processing enclave."
        )

        if label == "YES":

            facts = [
                base,

                (
                    f"For the {asset}, a declared maintenance window "
                    "activates an exception to the standing prohibition "
                    "when operations clearance is recorded. "
                    "The maintenance window is active and operations "
                    "clearance is recorded."
                ),
            ]

            conditions = []

            rationale = (
                "The exception is active and its clearance "
                "requirement is satisfied."
            )

        elif label == "NO":

            facts = [
                base,

                (
                    f"For the {asset}, the exception is available only "
                    "during a declared maintenance window. No maintenance "
                    "window is active, although operations clearance "
                    "has already been recorded."
                ),
            ]

            conditions = []

            rationale = (
                "The exception is inactive, so the standing "
                "prohibition controls."
            )

        elif label == "CONDITIONAL":

            facts = [
                base,

                (
                    f"For the {asset}, a declared maintenance window "
                    "activates an exception if operations clearance "
                    "is recorded. The maintenance window is active, "
                    "but operations-clearance status is not established."
                ),
            ]

            conditions = [
                "operations clearance unresolved",
            ]

            rationale = (
                "The exception is active, but its material clearance "
                "requirement remains unresolved."
            )

        else:

            facts = [
                (
                    f"The {asset} is subject to maintenance logging "
                    "and equipment-status reporting. No supplied rule "
                    "governs whether it may leave the normal "
                    "processing enclave."
                ),
            ]

            conditions = []

            rationale = (
                "Maintenance administration is described, "
                "but no governing movement rule is supplied."
            )

        return (
            action,
            facts,
            conditions,
            rationale,
        )


    # ========================================================
    # 5. AUTHORITY STATE RESOLUTION
    # ========================================================

    if (
        operator
        == "authority_state_resolution"
    ):

        action = (
            "have the temporary reviewer approve "
            f"publication of the {asset}"
        )

        authority = (
            f"Publication of the {asset} may be approved by the "
            "principal reviewer or a temporary reviewer whose "
            "assignment is currently effective."
        )

        if label == "YES":

            facts = [
                authority,

                (
                    f"The temporary reviewer assigned to the {asset} "
                    "is recorded as currently effective."
                ),
            ]

            conditions = []

            rationale = (
                "The temporary reviewer currently holds the "
                "authority required to approve publication."
            )

        elif label == "NO":

            facts = [
                authority,

                (
                    f"The temporary reviewer's assignment for the "
                    f"{asset} was revoked before this publication request."
                ),
            ]

            conditions = []

            rationale = (
                "Revocation ended the authority required "
                "for the requested approval."
            )

        elif label == "CONDITIONAL":

            facts = [
                authority,

                (
                    f"The temporary reviewer was previously assigned "
                    f"to the {asset}, but the evidence does not establish "
                    "whether that assignment is currently effective."
                ),
            ]

            conditions = [
                "current reviewer assignment unresolved",
            ]

            rationale = (
                "Approval may be valid, but current authority "
                "status remains unresolved."
            )

        else:

            facts = [
                (
                    f"The temporary reviewer performs routine metadata "
                    f"checks for the {asset}. The supplied evidence does "
                    "not state whether a temporary reviewer may approve "
                    "publication."
                ),
            ]

            conditions = []

            rationale = (
                "The actor's duties are described, but approval "
                "authority is not established."
            )

        return (
            action,
            facts,
            conditions,
            rationale,
        )


    # ========================================================
    # 6. CATEGORY + RULE COMPOSITION
    # ========================================================

    if (
        operator
        == "category_rule_composition"
    ):

        action = (
            f"run the {asset} "
            "on the shared analysis cluster"
        )

        if label == "YES":

            facts = [
                (
                    f"The {asset} satisfies the definition of a "
                    "managed-compute workload."
                ),

                (
                    f"Managed-compute workloads such as the {asset} "
                    "may run on the shared analysis cluster when the "
                    "cluster is certified. The cluster is certified."
                ),
            ]

            conditions = []

            rationale = (
                "Category membership is established and the "
                "category's authorization rule permits the action."
            )

        elif label == "NO":

            facts = [
                (
                    f"The {asset} satisfies the definition of a "
                    "restricted-compute workload."
                ),

                (
                    f"Restricted-compute workloads such as the {asset} "
                    "are expressly prohibited from running on the "
                    "shared analysis cluster."
                ),
            ]

            conditions = []

            rationale = (
                "The asset falls within a category whose governing "
                "rule prohibits the requested processing."
            )

        elif label == "CONDITIONAL":

            facts = [
                (
                    f"The {asset} satisfies the definition of a "
                    "managed-compute workload."
                ),

                (
                    f"Managed-compute workloads such as the {asset} "
                    "may run on the shared analysis cluster only while "
                    "cluster certification is current. Current "
                    "certification status is not established."
                ),
            ]

            conditions = [
                "cluster certification unresolved",
            ]

            rationale = (
                "Category membership is established, but the "
                "authorization condition remains unresolved."
            )

        else:

            facts = [
                (
                    f"The {asset} satisfies the definition of a "
                    "managed-compute workload. No supplied provision "
                    "states whether managed-compute workloads may run "
                    "on the shared analysis cluster."
                ),
            ]

            conditions = []

            rationale = (
                "The evidence establishes category membership "
                "but supplies no governing authorization rule."
            )

        return (
            action,
            facts,
            conditions,
            rationale,
        )


    # ========================================================
    # 7. SCOPE / DESTINATION RESOLUTION
    # ========================================================

    if (
        operator
        == "scope_destination_resolution"
    ):

        action = (
            f"send the {asset} "
            "to recipient group Delta"
        )

        if label == "YES":

            facts = [
                (
                    f"For the {asset}, recipient group Delta is "
                    "classified as an approved recipient class."
                ),

                (
                    f"The {asset} may be sent to approved recipient "
                    "classes when recipient controls are active. "
                    "The required recipient controls are active."
                ),
            ]

            conditions = []

            rationale = (
                "The recipient is inside the governed scope and "
                "the required controls are satisfied."
            )

        elif label == "NO":

            facts = [
                (
                    f"For the {asset}, recipient group Delta is "
                    "explicitly classified as an excluded recipient class."
                ),

                (
                    f"The {asset} may not be sent to excluded "
                    "recipient classes."
                ),
            ]

            conditions = []

            rationale = (
                "The requested recipient falls explicitly outside "
                "the permitted recipient scope."
            )

        elif label == "CONDITIONAL":

            facts = [
                (
                    f"For the {asset}, recipient group Delta is "
                    "classified as an approved recipient class."
                ),

                (
                    f"The {asset} may be sent to approved recipients "
                    "only while a recipient agreement is active. "
                    "Agreement status is not established."
                ),
            ]

            conditions = [
                "recipient agreement unresolved",
            ]

            rationale = (
                "Recipient scope is approved, but a material "
                "agreement requirement remains unresolved."
            )

        else:

            facts = [
                (
                    f"The policy for the {asset} distinguishes approved "
                    "and excluded recipient classes, but the supplied "
                    "evidence does not classify recipient group Delta."
                ),
            ]

            conditions = []

            rationale = (
                "The governing scope framework exists, but the "
                "requested recipient classification is missing."
            )

        return (
            action,
            facts,
            conditions,
            rationale,
        )


    # ========================================================
    # 8. EVIDENCE RELEVANCE BOUNDARY
    # ========================================================

    if (
        operator
        == "evidence_relevance_boundary"
    ):

        action = (
            f"submit the {asset} "
            "through Channel K"
        )

        if label == "YES":

            facts = [
                (
                    f"The Channel K rule permits submission of the "
                    f"{asset} when transmission validation is complete. "
                    "Transmission validation is complete."
                ),
            ]

            conditions = []

            rationale = (
                "The evidence directly governs Channel K and "
                "its material validation requirement is satisfied."
            )

        elif label == "NO":

            facts = [
                (
                    f"The Channel K rule expressly prohibits submission "
                    f"of the {asset} through Channel K."
                ),
            ]

            conditions = []

            rationale = (
                "The supplied evidence directly prohibits "
                "the exact Channel K action."
            )

        elif label == "CONDITIONAL":

            facts = [
                (
                    f"The Channel K rule permits submission of the "
                    f"{asset} only after transmission validation. "
                    "Validation status is not established."
                ),
            ]

            conditions = [
                "transmission validation unresolved",
            ]

            rationale = (
                "The evidence directly governs the action, "
                "but validation remains unresolved."
            )

        else:

            facts = [
                (
                    f"The supplied rule permits submission of the "
                    f"{asset} through Channel J under specified controls. "
                    "It contains no provision governing Channel K."
                ),
            ]

            conditions = []

            rationale = (
                "The evidence is topically related but governs "
                "a different channel."
            )

        return (
            action,
            facts,
            conditions,
            rationale,
        )


    raise ValueError(
        operator
    )


def question_variants(
    action,
):

    return [

        (
            "Under the supplied evidence, "
            f"may someone {action}?"
        ),

        (
            "Based only on these provisions, "
            f"is someone permitted to {action}?"
        ),
    ]


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

    return (
        "QUESTION\n\n"
        f"{question}"
        "\n\nSUPPLIED EVIDENCE\n\n"
        f"{evidence}"
    )


def hash_text(
    text,
):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def main():

    for operator, family in (
        SELECTED_FAMILIES.items()
    ):

        assert family_exists(
            operator,
            family,
        ), (
            operator,
            family,
        )


    rows = []


    for operator_index, (
        operator,
        family,
    ) in enumerate(
        SELECTED_FAMILIES.items()
    ):

        assets = ASSET_SETS[
            operator
        ]

        group = (
            f"{operator}__{family}"
        )


        # ----------------------------------------------------
        # Build canonical questions.
        # ----------------------------------------------------

        actions = {}

        surfaces = {}


        for target_index, asset in enumerate(
            assets
        ):

            action = state_for(
                operator,
                asset,
                "YES",
            )[0]

            actions[
                target_index
            ] = action

            surfaces[
                target_index
            ] = question_variants(
                action
            )


        # ----------------------------------------------------
        # Four Latin-square evidence worlds.
        # ----------------------------------------------------

        for world_index in range(4):

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

                (
                    action,
                    facts,
                    conditions,
                    rationale,
                ) = state_for(
                    operator,
                    asset,
                    label,
                )


                fact_keys = []

                for fact_index, fact in enumerate(
                    facts
                ):

                    key = (
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


                target_gold[
                    target_index
                ] = {
                    "decision":
                        label,

                    "conditions":
                        conditions,

                    "rationale":
                        rationale,

                    "fact_keys":
                        fact_keys,
                }


            # One unrelated administrative distractor/world.

            raw_chunks.append(
                (
                    (
                        "distractor",
                        world_index,
                    ),
                    DISTRACTORS[
                        (
                            operator_index
                            + world_index
                        )
                        % len(
                            DISTRACTORS
                        )
                    ],
                )
            )


            rng = random.Random(
                SEED
                + operator_index * 100
                + world_index
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


            evidence_text = (
                render_evidence(
                    evidence_chunks
                )
            )

            evidence_hash = hash_text(
                evidence_text
            )


            # ------------------------------------------------
            # Pair same world with every question target.
            # Two surface variants per target.
            # ------------------------------------------------

            for target_index, asset in enumerate(
                assets
            ):

                gold = target_gold[
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


                for surface_index, question in enumerate(
                    surfaces[
                        target_index
                    ]
                ):

                    # Every logical question/evidence pair gets
                    # one question-first and one evidence-first
                    # surface across its two variants.

                    if (
                        (
                            operator_index
                            + world_index
                            + target_index
                            + surface_index
                        )
                        % 2
                        == 0
                    ):

                        order = (
                            "question_first"
                        )

                    else:

                        order = (
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


                    rows.append({

                        "scenario_id":
                            (
                                "v5pilot__"
                                f"{operator}__"
                                f"world{world_index}__"
                                f"target{target_index}__"
                                f"surface{surface_index}"
                            ),

                        "split":
                            "pilot",

                        "operator":
                            operator,

                        "family":
                            family,

                        "world":
                            world_index,

                        "target_index":
                            target_index,

                        "surface_variant":
                            surface_index,

                        "asset":
                            asset,

                        "relational_group":
                            group,

                        "question":
                            question,

                        "evidence":
                            evidence_text,

                        "evidence_hash":
                            evidence_hash,

                        "input_order":
                            order,

                        "instruction":
                            INSTRUCTION,

                        "input":
                            render_input(
                                question,
                                evidence_text,
                                order,
                            ),

                        "target":
                            gold[
                                "decision"
                            ],

                        "output":
                            json.dumps(
                                output,
                                sort_keys=True,
                            ),
                    })


    # ========================================================
    # BUILD-TIME ASSERTIONS
    # ========================================================

    assert len(
        rows
    ) == 256


    assert Counter(
        row[
            "target"
        ]
        for row in rows
    ) == Counter({
        label: 64
        for label in LABELS
    })


    assert Counter(
        row[
            "operator"
        ]
        for row in rows
    ) == Counter({
        operator: 32
        for operator
        in SELECTED_FAMILIES
    })


    assert Counter(
        row[
            "input_order"
        ]
        for row in rows
    ) == Counter({
        "question_first": 128,
        "evidence_first": 128,
    })


    # Each operator:
    # 4 worlds × 4 targets × 2 surfaces = 32.

    for operator in SELECTED_FAMILIES:

        subset = [
            row
            for row in rows
            if row[
                "operator"
            ] == operator
        ]

        assert len(
            subset
        ) == 32


    # ========================================================
    # WRITE
    # ========================================================

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with OUT.open(
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


    print("=" * 88)
    print(
        "SENTINEL V5 RELATIONAL PILOT BUILD"
    )
    print("=" * 88)

    print(
        "rows:",
        len(
            rows
        ),
    )

    print(
        "operators:",
        dict(
            sorted(
                Counter(
                    row[
                        "operator"
                    ]
                    for row in rows
                ).items()
            )
        ),
    )

    print(
        "labels:",
        dict(
            sorted(
                Counter(
                    row[
                        "target"
                    ]
                    for row in rows
                ).items()
            )
        ),
    )

    print(
        "input order:",
        dict(
            sorted(
                Counter(
                    row[
                        "input_order"
                    ]
                    for row in rows
                ).items()
            )
        ),
    )

    print(
        "citation cardinality:",
        dict(
            sorted(
                citation_counts.items()
            )
        ),
    )

    print(
        "WROTE:",
        OUT,
    )


if __name__ == "__main__":
    main()
