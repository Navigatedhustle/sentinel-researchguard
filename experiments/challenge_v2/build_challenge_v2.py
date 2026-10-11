from collections import Counter
from pathlib import Path
import json
import random
import re


BASE = Path("/workspace")
OUT = BASE / "experiments/challenge_v2"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260816
RNG = random.Random(SEED)

LABELS = [
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
]


INSTRUCTION = """You are Sentinel, a policy-governance decision model.

Use ONLY the supplied evidence.

Classify the exact question as one of:

YES
NO
CONDITIONAL
INSUFFICIENT

Apply these rules in order:

INSUFFICIENT:
Choose INSUFFICIENT when the supplied evidence does not directly govern
the exact issue asked about. Related evidence is not enough. Do not invent
missing approvers, technologies, durations, thresholds, locations, or rules.

NO:
Choose NO when the evidence directly prohibits the exact action or directly
establishes that the proposition in the question is false.

CONDITIONAL:
Choose CONDITIONAL when the action may be permitted, but one or more
material requirements remain unresolved.

YES:
Choose YES only when the evidence directly permits the exact action and
every material condition is already established as satisfied by the question.

Silence is not prohibition.
Semantic similarity is not evidence.

Return valid JSON only.
""".strip()


# Each composition supplies:
#
# action template
# label -> (
#     governing evidence,
#     gold answer,
#     material conditions
# )
#
# These are independently authored V2 structures.
# They are NOT generated from Challenge V1 failures.


SPECS = {

    "specific_rule_precedence": {

        "action":
            "share the {a} with an external collaborator",

        "YES": (
            "A general collaboration rule permits external sharing. "
            "For the {a}, a specific rule controls and permits external "
            "sharing when disclosure review is approved. "
            "The disclosure review is recorded as approved.",

            "The specific rule controls and its review condition is satisfied.",

            [
                "specific rule applies",
                "disclosure review approved",
            ],
        ),

        "NO": (
            "A general collaboration rule permits external sharing. "
            "For the {a}, the specific controlled-data rule controls "
            "and expressly prohibits external sharing.",

            "The specific rule overrides the general permission and "
            "directly prohibits the action.",

            [],
        ),

        "CONDITIONAL": (
            "A general collaboration rule permits external sharing. "
            "For the {a}, a specific rule controls and permits external "
            "sharing only while a data-use agreement is active. "
            "The supplied evidence does not establish whether that "
            "agreement is active.",

            "The specific rule may permit sharing, but active agreement "
            "status remains unresolved.",

            [
                "active data-use agreement unresolved",
            ],
        ),

        "INSUFFICIENT": (
            "The {a} is listed in the records inventory and is subject "
            "to retention logging. The supplied provisions do not state "
            "a rule governing external collaborator sharing of the {a}.",

            "The evidence is related to the asset but does not govern "
            "the requested sharing.",

            [],
        ),
    },


    "superseding_rule_precedence": {

        "action":
            "release the {a} through the sponsor portal",

        "YES": (
            "A 2024 release rule restricted sponsor-portal release of "
            "the {a}. A 2026 rule expressly supersedes the 2024 rule "
            "and permits release after sponsor review. "
            "Sponsor review is complete.",

            "The superseding 2026 rule controls and its review "
            "requirement is satisfied.",

            [
                "2026 rule operative",
                "sponsor review complete",
            ],
        ),

        "NO": (
            "A 2024 rule permitted sponsor-portal release of the {a}. "
            "A 2026 rule expressly supersedes it and requires sponsor "
            "signoff before release. Sponsor signoff was denied.",

            "The current superseding rule controls, and the denied "
            "signoff prevents release.",

            [],
        ),

        "CONDITIONAL": (
            "A 2024 rule addressed release of the {a}. "
            "A 2026 rule expressly supersedes it and permits "
            "sponsor-portal release only with sponsor signoff. "
            "Current signoff status is not established.",

            "The current rule may permit release, but sponsor signoff "
            "remains unresolved.",

            [
                "sponsor signoff unresolved",
            ],
        ),

        "INSUFFICIENT": (
            "A 2026 rule supersedes the prior retention schedule for "
            "the {a} and changes its archival review date. "
            "No supplied provision states whether the {a} may be "
            "released through a sponsor portal.",

            "The evidence establishes a current archival rule but not "
            "a governing release rule.",

            [],
        ),
    },


    "exception_activation": {

        "action":
            "export the {a} outside the approved environment",

        "YES": (
            "Export of the {a} outside the approved environment is "
            "prohibited. An exception applies during a declared "
            "recovery event if written security clearance is recorded. "
            "A recovery event is declared and written clearance is recorded.",

            "The exception is active and its clearance requirement is satisfied.",

            [
                "recovery event declared",
                "security clearance recorded",
            ],
        ),

        "NO": (
            "Export of the {a} outside the approved environment is "
            "prohibited. An exception becomes available only during a "
            "declared recovery event. No recovery event has been declared, "
            "although security clearance is already recorded.",

            "The exception is inactive, so the base prohibition controls "
            "despite downstream clearance.",

            [],
        ),

        "CONDITIONAL": (
            "Export of the {a} outside the approved environment is "
            "prohibited. A declared recovery event activates an exception "
            "allowing export with written security clearance. "
            "A recovery event is declared, but written-clearance status "
            "is not established.",

            "The exception is active, but its required clearance "
            "remains unresolved.",

            [
                "written security clearance unresolved",
            ],
        ),

        "INSUFFICIENT": (
            "The {a} is covered by incident notification and inventory "
            "rules. The supplied evidence does not state whether export "
            "outside an approved environment is permitted or prohibited.",

            "The evidence concerns the asset but does not govern "
            "the requested export.",

            [],
        ),
    },


    "nested_exception_requirements": {

        "action":
            "transfer the {a} to an external research partner",

        "YES": (
            "External transfer of the {a} is prohibited. "
            "A collaboration exception applies when an institutional "
            "waiver is active and sponsor consent is recorded. "
            "The waiver is active and sponsor consent is recorded.",

            "The nested exception applies and both required conditions "
            "are satisfied.",

            [
                "institutional waiver active",
                "sponsor consent recorded",
            ],
        ),

        "NO": (
            "External transfer of the {a} is prohibited. "
            "The only exception requires both an active institutional "
            "waiver and sponsor consent. The institutional waiver was "
            "denied, while sponsor consent is recorded.",

            "A required exception element failed, leaving the transfer "
            "prohibition operative.",

            [],
        ),

        "CONDITIONAL": (
            "External transfer of the {a} is prohibited. "
            "A collaboration exception requires an active institutional "
            "waiver and sponsor consent. The waiver is active, but "
            "sponsor-consent status is not established.",

            "The exception may permit transfer, but sponsor consent "
            "remains unresolved.",

            [
                "sponsor consent unresolved",
            ],
        ),

        "INSUFFICIENT": (
            "The {a} is subject to collaboration reporting and annual "
            "inventory reconciliation. The supplied evidence does not "
            "establish any rule governing external research-partner transfer.",

            "No supplied rule governs whether this external transfer "
            "may occur.",

            [],
        ),
    },


    "delegated_authority_validity": {

        "action":
            "have the acting data steward approve release of the {a}",

        "YES": (
            "Release of the {a} may be approved by the principal steward "
            "or a currently recorded delegate. The acting data steward is "
            "recorded as the current delegate, and required department "
            "approval exists.",

            "The acting steward has current delegated authority and the "
            "required approval is established.",

            [
                "delegation current",
                "department approval established",
            ],
        ),

        "NO": (
            "Release of the {a} requires approval by the principal steward "
            "or a current delegate. The acting data steward's delegation "
            "was revoked before this request.",

            "Revocation ended the authority required for the acting "
            "steward to approve release.",

            [],
        ),

        "CONDITIONAL": (
            "Release of the {a} may be approved by a currently recorded "
            "delegate. The acting data steward previously received "
            "delegation, but the evidence does not establish whether "
            "that delegation is currently recorded.",

            "Approval may be valid, but current delegation status "
            "remains unresolved.",

            [
                "current delegation status unresolved",
            ],
        ),

        "INSUFFICIENT": (
            "The policy describes routine duties of data stewards for "
            "the {a} but does not identify the acting data steward as "
            "a delegate or state who may approve this release.",

            "The evidence does not establish the authority needed to "
            "answer the approval question.",

            [],
        ),
    },


    "temporal_authorization": {

        "action":
            "access the {a} remotely",

        "YES": (
            "Remote access to the {a} is permitted only while project "
            "authorization is active and annual review is complete. "
            "Authorization runs March 1 through September 30, the request "
            "occurs June 15, and annual review is complete.",

            "The request falls within the active authorization period "
            "and the review condition is satisfied.",

            [
                "authorization active",
                "annual review complete",
            ],
        ),

        "NO": (
            "Remote access to the {a} is permitted only during the active "
            "authorization period. Authorization expired May 31, and the "
            "requested access occurs June 15.",

            "The required authorization has expired, so remote access "
            "is not permitted.",

            [],
        ),

        "CONDITIONAL": (
            "Remote access to the {a} may occur while authorization is "
            "active if annual review is complete. The request falls within "
            "the active date range, but annual-review status is not established.",

            "The authorization window is open, but annual-review completion "
            "remains unresolved.",

            [
                "annual review unresolved",
            ],
        ),

        "INSUFFICIENT": (
            "The evidence lists milestone dates and reporting deadlines "
            "for the {a}. It does not state any rule governing whether "
            "remote access is allowed.",

            "The dates are related to the project but do not establish "
            "a remote-access rule.",

            [],
        ),
    },


    "definition_then_authorization": {

        "action":
            "process the {a} on a local workstation",

        "YES": (
            "A qualified local workload is defined as institutional "
            "research data with no external service call. Processing the "
            "{a} meets that definition. Qualified local workloads may run "
            "on managed local workstations, and the workstation is "
            "recorded as managed.",

            "The workload falls within the defined category and the "
            "authorization rule permits it on the managed workstation.",

            [
                "qualified local workload",
                "managed workstation",
            ],
        ),

        "NO": (
            "A restricted local workload is defined to include the {a}. "
            "Restricted local workloads are expressly prohibited from "
            "workstation processing.",

            "The definition places the asset in the restricted category, "
            "whose governing rule prohibits workstation processing.",

            [],
        ),

        "CONDITIONAL": (
            "The {a} satisfies the definition of a qualified local workload. "
            "Qualified local workloads may be processed on a local workstation "
            "only after endpoint security approval. Endpoint-security "
            "approval status is not established.",

            "Category membership is established, but the authorization "
            "condition remains unresolved.",

            [
                "endpoint security approval unresolved",
            ],
        ),

        "INSUFFICIENT": (
            "The evidence defines the {a} as a qualified local workload "
            "and describes its technical characteristics. No supplied "
            "provision states whether qualified local workloads may be "
            "processed on a local workstation.",

            "The definition establishes category membership but supplies "
            "no governing authorization rule.",

            [],
        ),
    },


    "purpose_scope_authorization": {

        "action":
            "reuse the {a} for secondary academic research",

        "YES": (
            "The authorized purpose for the {a} includes noncommercial "
            "secondary academic research. Reuse within that purpose is "
            "permitted after ethics review, and ethics review is complete.",

            "The proposed purpose is within scope and the required "
            "review is complete.",

            [
                "purpose within scope",
                "ethics review complete",
            ],
        ),

        "NO": (
            "The authorization for the {a} permits primary analysis but "
            "explicitly excludes secondary academic reuse.",

            "The requested purpose is explicitly outside the authorized scope.",

            [],
        ),

        "CONDITIONAL": (
            "The authorized purpose for the {a} includes secondary academic "
            "research. Such reuse is permitted only after ethics review. "
            "Ethics-review status is not established.",

            "The purpose is within scope, but required ethics review "
            "remains unresolved.",

            [
                "ethics review unresolved",
            ],
        ),

        "INSUFFICIENT": (
            "The license for the {a} permits research use but does not "
            "define whether the proposed secondary academic activity falls "
            "within the licensed purpose, and no supplied definition "
            "resolves that term.",

            "The evidence does not establish whether the requested purpose "
            "falls within the governed scope.",

            [],
        ),
    },


    "jurisdiction_then_requirement": {

        "action":
            "transfer the {a} to Territory Q",

        "YES": (
            "Territory Q is classified as an approved destination for "
            "the {a}. Transfers to approved destinations are permitted "
            "when contractual clauses are active. The required clauses "
            "are active.",

            "The destination is approved and the required contractual "
            "clauses are active.",

            [
                "approved destination",
                "contractual clauses active",
            ],
        ),

        "NO": (
            "Territory Q is classified as a prohibited destination for "
            "the {a}. Transfers to prohibited destinations are not permitted.",

            "The jurisdictional classification directly places the "
            "destination under the transfer prohibition.",

            [],
        ),

        "CONDITIONAL": (
            "Territory Q is classified as an approved destination for "
            "the {a}. Transfer is permitted only while contractual clauses "
            "are active. Clause status is not established.",

            "Jurisdiction is approved, but the contractual requirement "
            "remains unresolved.",

            [
                "contractual clauses unresolved",
            ],
        ),

        "INSUFFICIENT": (
            "The transfer policy distinguishes approved and prohibited "
            "destinations for the {a}, but the supplied evidence does not "
            "classify Territory Q in either category.",

            "The framework is supplied, but Territory Q's required "
            "classification is missing.",

            [],
        ),
    },


    "prohibition_vs_unresolved_fact": {

        "action":
            "send the {a} by institutional email",

        "YES": (
            "Institutional email transmission of the {a} is expressly "
            "permitted when transport encryption is enabled. Transport "
            "encryption is enabled. A separate annual inventory review "
            "has not yet been scheduled.",

            "The email rule directly permits the action and its material "
            "encryption condition is satisfied; the inventory issue is unrelated.",

            [
                "transport encryption enabled",
            ],
        ),

        "NO": (
            "Institutional email transmission of the {a} is expressly "
            "prohibited. The evidence does not establish whether a separate "
            "disclosure review has been completed.",

            "The direct email prohibition resolves the action; unresolved "
            "disclosure-review status does not make it conditional.",

            [],
        ),

        "CONDITIONAL": (
            "Institutional email transmission of the {a} is permitted "
            "only after disclosure review approval. Approval status is "
            "not established.",

            "Email transmission may be permitted, but the material "
            "approval condition remains unresolved.",

            [
                "disclosure review approval unresolved",
            ],
        ),

        "INSUFFICIENT": (
            "The evidence governs secure file-transfer service use for "
            "the {a} and requires encryption for that service. It does "
            "not state a rule for institutional email transmission.",

            "A related transmission rule exists, but it governs a "
            "different channel.",

            [],
        ),
    },


    "permission_with_material_condition": {

        "action":
            "publish the {a} in a public repository",

        "YES": (
            "Publication of the {a} in a public repository is permitted "
            "after disclosure review and sponsor notice. Disclosure review "
            "is approved and sponsor notice was sent on time.",

            "Direct publication permission applies and every material "
            "requirement is established.",

            [
                "disclosure review approved",
                "sponsor notice satisfied",
            ],
        ),

        "NO": (
            "The public-release rule for the {a} expressly prohibits "
            "publication in public repositories.",

            "The governing public-release rule directly prohibits the "
            "requested publication.",

            [],
        ),

        "CONDITIONAL": (
            "Publication of the {a} in a public repository is permitted "
            "after disclosure review and sponsor notice. Disclosure review "
            "is approved, but sponsor-notice status is not established.",

            "Publication may be permitted, but a material notice "
            "requirement remains unresolved.",

            [
                "sponsor notice unresolved",
            ],
        ),

        "INSUFFICIENT": (
            "The evidence describes internal archival retention for "
            "the {a} and internal catalog publication. It does not state "
            "whether the asset may be published in a public repository.",

            "The supplied provisions do not govern the requested "
            "public-repository publication.",

            [],
        ),
    },


    "evidence_relevance_sufficiency": {

        "action":
            "use the {a} with an external generative-AI service",

        "YES": (
            "Use of the {a} with an external generative-AI service is "
            "permitted when provider approval and data minimization are "
            "complete. The provider is approved and data minimization "
            "is complete.",

            "The evidence directly governs external generative-AI use "
            "and all material requirements are satisfied.",

            [
                "provider approved",
                "data minimization complete",
            ],
        ),

        "NO": (
            "The external generative-AI rule expressly prohibits sending "
            "the {a} to external generative-AI services.",

            "The supplied rule directly prohibits the exact external "
            "generative-AI action.",

            [],
        ),

        "CONDITIONAL": (
            "Use of the {a} with an external generative-AI service may "
            "be permitted after provider approval. Provider-approval "
            "status is not established.",

            "The evidence directly governs the action, but provider "
            "approval remains unresolved.",

            [
                "provider approval unresolved",
            ],
        ),

        "INSUFFICIENT": (
            "The policy for the {a} governs handheld calculators and "
            "locally installed statistical software. It contains no "
            "provision addressing external generative-AI services.",

            "The evidence concerns computational tools but does not "
            "directly govern external generative-AI services.",

            [],
        ),
    },
}


ASSETS = [
    "Alder dataset",
    "Birch archive",
    "Cedar records",
    "Dune corpus",
    "Ember files",
    "Harbor dataset",
    "Juniper records",
    "Kestrel archive",
    "Maple corpus",
    "Noble files",
    "Orchid dataset",
    "Prism records",
    "Quartz archive",
    "Summit corpus",
    "Talon files",
    "Umber dataset",
    "Willow records",
    "Xenia archive",
    "Yarrow corpus",
    "Zephyr files",
]


DISTRACTORS = [
    "Institution-owned devices must remain listed in the central inventory.",
    "Institutional accounts must use multifactor authentication for remote administrative access.",
    "Published institutional web pages must follow accessibility requirements.",
    "Public training materials may be distributed internally.",
    "Annual asset inventories are due before the end of the fiscal year.",
    "Travel reimbursement requests must include itemized receipts.",
]


def slug(s):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        s.lower(),
    ).strip("_")


def render(
    composition,
    asset,
    label,
):

    spec = SPECS[
        composition
    ]

    action = (
        spec["action"]
        .format(
            a=asset
        )
    )

    statement, answer, conditions = (
        spec[
            label
        ]
    )

    return (
        action,
        statement.format(
            a=asset
        ),
        answer.format(
            a=asset
        ),
        list(
            conditions
        ),
    )


def pack(
    statements,
    count,
    salt,
):

    local = random.Random(
        SEED + salt
    )

    ordered = list(
        statements
    )

    local.shuffle(
        ordered
    )

    if count == 1:

        groups = [
            ordered
        ]

    elif count == 2:

        groups = [
            ordered[:2],
            ordered[2:],
        ]

    elif count == 3:

        groups = [
            ordered[:2],
            ordered[2:3],
            ordered[3:],
        ]

    elif count == 4:

        groups = [
            [x]
            for x in ordered
        ]

    elif count == 5:

        groups = [
            [x]
            for x in ordered
        ]

        groups.append([
            (
                -1,
                local.choice(
                    DISTRACTORS
                ),
            )
        ])

    else:

        raise ValueError(
            count
        )

    passages = []

    for group in groups:

        passages.append({
            "text":
                " ".join(
                    item[1]
                    for item in group
                ),

            "targets": [
                item[0]
                for item in group
                if item[0] >= 0
            ],
        })

    local.shuffle(
        passages
    )

    target_to_eid = {}
    final = []

    for index, passage in enumerate(
        passages,
        start=1,
    ):

        eid = f"E{index}"

        final.append(
            (
                eid,
                passage["text"],
            )
        )

        for target in passage[
            "targets"
        ]:

            target_to_eid[
                target
            ] = eid

    return (
        final,
        target_to_eid,
    )


def evidence_text(
    passages,
):

    return "\n\n".join(
        f"[{eid}] {text}"
        for eid, text
        in passages
    )


def make_input(
    question,
    evidence,
    order,
):

    if order == "question_first":

        return (
            f"QUESTION:\n{question}"
            f"\n\nEVIDENCE:\n{evidence}"
        )

    return (
        f"EVIDENCE:\n{evidence}"
        f"\n\nQUESTION:\n{question}"
    )


# ============================================================
# EVIDENCE-COUNT SCHEDULE
# ============================================================
#
# Exact final 240-case distribution:
#
# 1 passage:  24
# 2 passages: 72
# 3 passages: 72
# 4 passages: 48
# 5 passages: 24
#
# Relational evidence packs are each reused by four questions,
# so their contribution occurs in multiples of four.


rel_counts = (
    [1] * 4
    + [2] * 16
    + [3] * 16
    + [4] * 8
    + [5] * 4
)

stress_counts = (
    [1] * 8
    + [2] * 8
    + [3] * 8
    + [4] * 16
    + [5] * 8
)

RNG.shuffle(
    rel_counts
)

RNG.shuffle(
    stress_counts
)


rows = []

label_seen = Counter()

rel_cursor = 0
stress_cursor = 0
asset_cursor = 0


for composition_index, composition in enumerate(
    SPECS
):

    assets = [
        ASSETS[
            (
                asset_cursor + offset
            )
            % len(
                ASSETS
            )
        ]
        for offset
        in range(4)
    ]

    asset_cursor += 4


    # ========================================================
    # FOUR FIXED QUESTIONS
    # ========================================================
    #
    # Each question will be paired with all four evidence packs.
    #
    # Across those four evidence packs the same question receives:
    #
    # YES
    # NO
    # CONDITIONAL
    # INSUFFICIENT


    questions = []

    for asset in assets:

        action = render(
            composition,
            asset,
            "YES",
        )[0]

        questions.append(
            "Under the supplied evidence, "
            f"may someone {action}?"
        )


    # ========================================================
    # 4 x 4 LATIN-SQUARE RELATIONAL MATRIX
    # ========================================================
    #
    # 4 questions × 4 evidence packs = 16 cases.
    #
    # label(question_i, pack_j)
    #     = LABELS[(i + j) % 4]
    #
    # Therefore:
    #
    # same question + different evidence
    #     => all four labels
    #
    # same evidence + different questions
    #     => all four labels


    for pack_index in range(4):

        states = {}
        statements = []

        for question_index, asset in enumerate(
            assets
        ):

            label = LABELS[
                (
                    question_index
                    + pack_index
                )
                % 4
            ]

            (
                _,
                statement,
                answer,
                conditions,
            ) = render(
                composition,
                asset,
                label,
            )

            statements.append(
                (
                    question_index,
                    statement,
                )
            )

            states[
                question_index
            ] = (
                label,
                answer,
                conditions,
            )


        passages, citation_map = pack(
            statements,
            rel_counts[
                rel_cursor
            ],
            (
                composition_index
                * 100
                + pack_index
            ),
        )

        rel_cursor += 1

        evidence = evidence_text(
            passages
        )


        for question_index, question in enumerate(
            questions
        ):

            (
                label,
                answer,
                conditions,
            ) = states[
                question_index
            ]


            # Alternate independently within each label.
            # Across 60 examples/label this produces exactly:
            #
            # 30 question-first
            # 30 evidence-first


            if (
                label_seen[
                    label
                ]
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

            label_seen[
                label
            ] += 1


            rows.append({

                "case_id":
                    (
                        "challenge_v2__"
                        f"{composition}__"
                        "rel__"
                        f"p{pack_index}__"
                        f"q{question_index}"
                    ),

                "composition":
                    composition,

                "question_type":
                    "relational_counterfactual",

                "reasoning_tags": [
                    composition,
                    "dual_counterfactual",
                    "operative_rule_selection",
                ],

                "relational_group":
                    f"{composition}__latin4",

                "policy_pack":
                    (
                        f"{composition}__"
                        f"pack{pack_index}"
                    ),

                "input_order":
                    order,

                "evidence_count":
                    len(
                        passages
                    ),

                "instruction":
                    INSTRUCTION,

                "question":
                    question,

                "evidence_text":
                    evidence,

                "input":
                    make_input(
                        question,
                        evidence,
                        order,
                    ),

                "gold": {

                    "decision":
                        label,

                    "citations": [
                        citation_map[
                            question_index
                        ]
                    ],

                    "conditions":
                        conditions,

                    "answer":
                        answer,

                    "evidence_sufficient":
                        (
                            label
                            != "INSUFFICIENT"
                        ),
                },
            })


    # ========================================================
    # FOUR STRESS CASES
    # ========================================================
    #
    # One independent case per label.
    #
    # Latin-square cases already provide 4 of each label.
    # Stress adds one of each:
    #
    # 5 YES
    # 5 NO
    # 5 CONDITIONAL
    # 5 INSUFFICIENT
    #
    # per composition.


    for label_index, label in enumerate(
        LABELS
    ):

        count = stress_counts[
            stress_cursor
        ]

        stress_cursor += 1


        asset = (
            slug(
                composition
            )
            .replace(
                "_",
                " "
            )
            .title()
            + " stress asset "
            + str(
                label_index + 1
            )
        )


        (
            action,
            governing_statement,
            answer,
            conditions,
        ) = render(
            composition,
            asset,
            label,
        )


        local = random.Random(
            SEED
            + 10000
            + composition_index * 10
            + label_index
        )


        extras = list(
            DISTRACTORS
        )

        local.shuffle(
            extras
        )


        raw_passages = [
            (
                "GOV",
                governing_statement,
            )
        ]

        for distractor_index, text in enumerate(
            extras[
                : count - 1
            ],
            start=1,
        ):

            raw_passages.append(
                (
                    f"D{distractor_index}",
                    text,
                )
            )


        local.shuffle(
            raw_passages
        )


        final_passages = []
        governing_eid = None


        for evidence_index, (
            kind,
            text,
        ) in enumerate(
            raw_passages,
            start=1,
        ):

            eid = (
                f"E{evidence_index}"
            )

            final_passages.append(
                (
                    eid,
                    text,
                )
            )

            if kind == "GOV":

                governing_eid = (
                    eid
                )


        evidence = evidence_text(
            final_passages
        )


        question = (
            "Based only on these passages, "
            f"is it permitted to {action}?"
        )


        if (
            label_seen[
                label
            ]
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


        label_seen[
            label
        ] += 1


        rows.append({

            "case_id":
                (
                    "challenge_v2__"
                    f"{composition}__"
                    "stress__"
                    f"{label.lower()}"
                ),

            "composition":
                composition,

            "question_type":
                "stress",

            "reasoning_tags": [
                composition,
                "stress_case",
            ],

            "relational_group":
                None,

            "policy_pack":
                (
                    f"{composition}__"
                    "stress__"
                    f"{label.lower()}"
                ),

            "input_order":
                order,

            "evidence_count":
                len(
                    final_passages
                ),

            "instruction":
                INSTRUCTION,

            "question":
                question,

            "evidence_text":
                evidence,

            "input":
                make_input(
                    question,
                    evidence,
                    order,
                ),

            "gold": {

                "decision":
                    label,

                "citations": [
                    governing_eid
                ],

                "conditions":
                    conditions,

                "answer":
                    answer,

                "evidence_sufficient":
                    (
                        label
                        != "INSUFFICIENT"
                    ),
            },
        })


# ============================================================
# BUILD-TIME HARD ASSERTIONS
# ============================================================


assert len(
    rows
) == 240


assert Counter(
    row[
        "gold"
    ][
        "decision"
    ]
    for row in rows
) == Counter({
    label: 60
    for label in LABELS
})


assert all(
    count == 20
    for count
    in Counter(
        row[
            "composition"
        ]
        for row in rows
    ).values()
)


assert Counter(
    row[
        "input_order"
    ]
    for row in rows
) == Counter({
    "question_first": 120,
    "evidence_first": 120,
})


assert Counter(
    row[
        "evidence_count"
    ]
    for row in rows
) == Counter({
    1: 24,
    2: 72,
    3: 72,
    4: 48,
    5: 24,
})


# ============================================================
# WRITE CANDIDATE
# ============================================================


path = (
    OUT
    / "sentinel_challenge_v2.json"
)


path.write_text(
    json.dumps(
        rows,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)


print(
    "=" * 80
)

print(
    "CHALLENGE V2 BUILD"
)

print(
    "=" * 80
)


print(
    "cases:",
    len(
        rows
    ),
)


print(
    "labels:",
    dict(
        sorted(
            Counter(
                row[
                    "gold"
                ][
                    "decision"
                ]
                for row in rows
            ).items()
        )
    ),
)


print(
    "compositions:",
    dict(
        sorted(
            Counter(
                row[
                    "composition"
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
    "evidence counts:",
    dict(
        sorted(
            Counter(
                row[
                    "evidence_count"
                ]
                for row in rows
            ).items()
        )
    ),
)


print(
    "question types:",
    dict(
        sorted(
            Counter(
                row[
                    "question_type"
                ]
                for row in rows
            ).items()
        )
    ),
)


print(
    "WROTE:",
    path,
)
