"""
Challenge V3 benchmark-local language realization.

Independent from Sentinel V6 curriculum realization.
"""

from experiments.challenge_v3.challenge_v3_oracle import (
    ExceptionState,
    Requirement,
    SemanticState,
)


INSTRUCTION = """You are Sentinel. Decide the exact policy-governance question using only the supplied evidence.

Return exactly one decision:

YES
NO
CONDITIONAL
INSUFFICIENT

Use INSUFFICIENT when the evidence does not establish that a governing rule applies to the exact proposition.

Use NO when an applicable controlling prohibition or explicitly failed required state prevents the action.

Use CONDITIONAL when applicability is established, no controlling required state has failed, and at least one material required state is unresolved.

Use YES only when applicability is established, the operative state permits the action, and every material requirement is satisfied.

Return valid JSON with exactly:

decision
citations
conditions
rationale

For conditions, use only:

SATISFIED | requirement
FAILED | requirement
UNRESOLVED | requirement
""".strip()


SPECS = {

    "controlled_release_scope": {
        "mode":
            "permission",

        "action":
            "release through the controlled exchange",

        "policy":
            "distribution charter",

        "requirements": (
            "release review complete",
        ),

        "question":
            "May someone {action} for the {entity}?",
    },


    "delegated_review_window": {
        "mode":
            "permission",

        "action":
            "approve the scheduled review",

        "policy":
            "delegated-review framework",

        "requirements": (
            "delegation currently valid",
            "review window currently open",
        ),

        "question":
            "Under the evidence, may the scheduled review for the {entity} be approved?",
    },


    "dual_prerequisite_exception": {
        "mode":
            "exception",

        "action":
            "send to an external collaborator",

        "policy":
            "external-transfer restriction",

        "trigger":
            "collaboration exception active",

        "requirements": (
            "institutional clearance recorded",
            "partner authorization recorded",
        ),

        "question":
            "May someone send the {entity} to an external collaborator?",
    },


    "current_rule_destination_gate": {
        "mode":
            "precedence",

        "action":
            "deliver to Region Delta",

        "policy":
            "current destination directive",

        "requirements": (
            "destination authorization recorded",
        ),

        "question":
            "Does the current policy permit the {entity} to be delivered to Region Delta?",
    },


    "defined_workflow_control": {
        "mode":
            "definition",

        "action":
            "process through the assisted workflow",

        "policy":
            "workflow control standard",

        "requirements": (
            "workflow attestation complete",
        ),

        "question":
            "May the {entity} be processed through the assisted workflow?",
    },


    "purpose_boundary_with_notice": {
        "mode":
            "scope",

        "action":
            "reuse for prototype validation",

        "policy":
            "purpose-use license",

        "requirements": (
            "required notice delivered",
        ),

        "question":
            "May someone reuse the {entity} for prototype validation?",
    },


    "temporary_access_override": {
        "mode":
            "exception",

        "action":
            "access through the restricted console",

        "policy":
            "restricted-console rule",

        "trigger":
            "temporary access window active",

        "requirements": (
            "security acknowledgement recorded",
        ),

        "question":
            "May someone access the {entity} through the restricted console?",
    },


    "recipient_class_authorization": {
        "mode":
            "permission",

        "action":
            "provide to the designated recipient",

        "policy":
            "recipient authorization standard",

        "requirements": (
            "recipient qualification confirmed",
            "authorization recorded",
        ),

        "question":
            "May the {entity} be provided to the designated recipient?",
    },


    "route_specific_rule_revision": {
        "mode":
            "precedence",

        "action":
            "publish through Route Sigma",

        "policy":
            "revised route directive",

        "requirements": (
            "route approval complete",
        ),

        "question":
            "May someone publish the {entity} through Route Sigma?",
    },


    "multi_entity_mixed_state": {
        "mode":
            "permission",

        "action":
            "enter the shared analysis space",

        "policy":
            "shared-space policy",

        "requirements": (
            "access review complete",
        ),

        "question":
            "May the {entity} enter the shared analysis space?",
    },


    "authority_condition_chain": {
        "mode":
            "permission",

        "action":
            "authorize controlled processing",

        "policy":
            "authority-chain protocol",

        "requirements": (
            "acting authority valid",
            "technical review complete",
            "processing notice recorded",
        ),

        "question":
            "May controlled processing of the {entity} be authorized?",
    },


    "full_governance_state_chain": {
        "mode":
            "exception",

        "action":
            "operate through the contingency pathway",

        "policy":
            "governance pathway rule",

        "trigger":
            "contingency authority active",

        "requirements": (
            "scope confirmation complete",
            "special review recorded",
        ),

        "question":
            "May the {entity} operate through the contingency pathway?",
    },
}


RELATIONAL_ENTITIES = {

    "controlled_release_scope":
        ("Atlas folio", "Cinder register", "Dahlia packet", "Flint archive"),

    "delegated_review_window":
        ("Garnet dossier", "Hearth file", "Indigo register", "Jasper packet"),

    "dual_prerequisite_exception":
        ("Kite archive", "Lumen dossier", "Marble folio", "Nimbus register"),

    "current_rule_destination_gate":
        ("Oriel packet", "Parchment file", "Raven dossier", "Solace archive"),

    "defined_workflow_control":
        ("Tundra folio", "Umbral file", "Vector register", "Willow packet"),

    "purpose_boundary_with_notice":
        ("Xenon archive", "Yarrow dossier", "Zephyr folio", "Arbor register"),

    "temporary_access_override":
        ("Bastion packet", "Crest file", "Dovetail archive", "Elm dossier"),

    "recipient_class_authorization":
        ("Forge register", "Glacier folio", "Horizon packet", "Islet file"),

    "route_specific_rule_revision":
        ("Jade dossier", "Kiln archive", "Lagoon register", "Meadow folio"),

    "multi_entity_mixed_state":
        ("Nectar file", "Orbit packet", "Prairie dossier", "Quarry register"),

    "authority_condition_chain":
        ("Rill archive", "Stone folio", "Thicket file", "Upland packet"),

    "full_governance_state_chain":
        ("Verdant dossier", "Wharf register", "Yonder archive", "Zenith file"),
}


STRESS_ENTITY = {

    composition:
        f"{entities[0]} stress subject"

    for composition, entities
    in RELATIONAL_ENTITIES.items()
}


DISTRACTORS = (
    "Equipment assigned to shared laboratories must remain inventoried.",
    "Travel expense records require supporting receipts.",
    "Published web resources must satisfy accessibility standards.",
    "Internal training notices are retained according to the records schedule.",
    "Routine maintenance tickets must identify the responsible service group.",
    "Annual equipment reconciliation occurs at the close of the fiscal cycle.",
)


def requirement_status_sentence(
    requirement,
):

    if requirement.state == "SATISFIED":

        return (
            f"The evidence confirms that {requirement.name}."
        )


    if requirement.state == "FAILED":

        return (
            f"The evidence explicitly establishes that "
            f"{requirement.name} is not satisfied."
        )


    if requirement.state == "UNRESOLVED":

        return (
            f"The supplied evidence does not establish whether "
            f"{requirement.name}."
        )


    raise ValueError(
        requirement.state
    )


def render_target_clause(
    composition,
    entity,
    state,
):

    spec = SPECS[
        composition
    ]

    action = spec[
        "action"
    ]

    policy = spec[
        "policy"
    ]

    mode = spec[
        "mode"
    ]


    if state.applicability != "ESTABLISHED":

        if mode in (
            "scope",
            "definition",
        ):

            return (
                f"The {policy} discusses the {entity}, but the supplied "
                f"definitions do not establish whether {action} falls "
                f"within the governed class or purpose."
            )


        return (
            f"The supplied {policy} refers to the {entity}, but does not "
            f"establish whether its rule governs the requested action to "
            f"{action}."
        )


    if state.base_rule == "PROHIBITION" and state.exception is None:

        if mode == "precedence":

            return (
                f"Earlier guidance addressed the {entity}. The current "
                f"{policy} supersedes that guidance and expressly prohibits "
                f"the {entity} from being used to {action}."
            )


        return (
            f"The applicable {policy} expressly prohibits the "
            f"{entity} from being used to {action}."
        )


    if state.base_rule == "PERMISSION":

        status = " ".join(
            requirement_status_sentence(
                requirement
            )
            for requirement
            in state.requirements
        )


        requirement_names = "; ".join(
            requirement.name
            for requirement
            in state.requirements
        )


        if mode == "precedence":

            prefix = (
                f"An earlier rule addressed the {entity}. The current "
                f"{policy} supersedes the earlier rule and permits "
                f"{action} only when {requirement_names}. "
            )


        elif mode == "definition":

            prefix = (
                f"The supplied definition places the {entity} inside the "
                f"class governed by the {policy}. That standard permits "
                f"{action} only when {requirement_names}. "
            )


        elif mode == "scope":

            prefix = (
                f"The {policy} expressly places the requested use of the "
                f"{entity} within its governed purpose. It permits "
                f"{action} only when {requirement_names}. "
            )


        else:

            prefix = (
                f"The applicable {policy} permits the {entity} to "
                f"{action} only when {requirement_names}. "
            )


        return (
            prefix
            + status
        )


    if (
        state.base_rule == "PROHIBITION"
        and state.exception is not None
    ):

        trigger = state.exception.trigger

        requirements = state.exception.requirements


        status = " ".join(
            [
                requirement_status_sentence(
                    trigger
                ),
                *[
                    requirement_status_sentence(
                        requirement
                    )
                    for requirement
                    in requirements
                ],
            ]
        )


        requirement_names = "; ".join(
            [
                trigger.name,
                *[
                    requirement.name
                    for requirement
                    in requirements
                ],
            ]
        )


        return (
            f"The applicable {policy} normally prohibits the {entity} "
            f"from being used to {action}. An exception can displace that "
            f"restriction only when {requirement_names}. {status}"
        )


    raise ValueError(
        "Unsupported semantic state."
    )


def question_for(
    composition,
    entity,
):

    spec = SPECS[
        composition
    ]


    return spec[
        "question"
    ].format(
        action=spec[
            "action"
        ],
        entity=entity,
    )


def rationale_for(
    result,
):

    mapping = {

        "NO_APPLICABLE_RULE":
            "The supplied evidence does not establish an applicable rule for the exact proposition.",

        "APPLICABILITY_UNRESOLVED":
            "The supplied evidence does not establish that the governing rule applies to the exact proposition.",

        "DIRECT_PROHIBITION":
            "The currently applicable rule directly prohibits the requested action.",

        "PERMISSION_COMPLETE":
            "The applicable permission governs the request and every material requirement is satisfied.",

        "PERMISSION_REQUIREMENT_FAILED":
            "The applicable permission is blocked because a material required state explicitly failed.",

        "PERMISSION_REQUIREMENT_UNRESOLVED":
            "The applicable permission could authorize the action, but a material required state remains unresolved.",

        "EXCEPTION_COMPLETE":
            "The exception displaces the base restriction because its trigger and all material requirements are satisfied.",

        "EXCEPTION_STATE_FAILED":
            "The base restriction remains controlling because a required exception state explicitly failed.",

        "EXCEPTION_STATE_UNRESOLVED":
            "The exception could displace the restriction, but a required exception state remains unresolved.",
    }


    return mapping[
        result.reason_code
    ]


def build_input(
    evidence,
    question,
    input_order,
):

    if input_order == "evidence_first":

        return (
            "Evidence:\n"
            + evidence
            + "\n\nQuestion:\n"
            + question
        )


    if input_order == "question_first":

        return (
            "Question:\n"
            + question
            + "\n\nEvidence:\n"
            + evidence
        )


    raise ValueError(
        input_order
    )
