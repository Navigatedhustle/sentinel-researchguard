"""
Sentinel V6 relational pilot language realizers.

Language realization only.

Gold decisions and ledgers come from the frozen semantic engine.
"""

import json

from training.v6.v6_relational_state_engine import (
    Applicability,
    BaseRule,
    RequirementState,
)


INSTRUCTION = """You are Sentinel, a policy-governance decision model.

Use ONLY the supplied evidence.

Evaluate the exact proposition in causal order:

1. bind facts to the exact target
2. determine whether a governing rule applies
3. identify the operative rule
4. resolve exception state when relevant
5. resolve each material requirement as SATISFIED, FAILED, or UNRESOLVED
6. return the decision implied by the completed state

Choose exactly one:

YES
NO
CONDITIONAL
INSUFFICIENT

INSUFFICIENT:
The evidence does not establish that a governing rule applies to the exact proposition.

NO:
An applicable controlling prohibition or explicitly failed required state prevents the action.

CONDITIONAL:
Applicability is established, no controlling requirement has failed, and at least one material requirement remains unresolved.

YES:
Applicability is established and every material requirement is satisfied.

Return valid JSON only with exactly these keys:

decision
citations
conditions
rationale

conditions is a complete material state ledger using:

SATISFIED | requirement
FAILED | requirement
UNRESOLVED | requirement
""".strip()


GROUP_ENTITIES = {

    "same_passage_entity_binding": (
        (
            "Lark collection",
            "Moss ledger",
            "Pine bundle",
            "Reef package",
        ),
        (
            "Amber notebook",
            "Bronze packet",
            "Cobalt ledger",
            "Ivory bundle",
        ),
    ),


    "purpose_membership_resolution": (
        (
            "Helix dataset",
            "Lantern corpus",
            "Meridian files",
            "Quill archive",
        ),
        (
            "Aurora records",
            "Beacon corpus",
            "Cipher dataset",
            "Drift files",
        ),
    ),


    "trigger_plus_requirement_exception": (
        (
            "Finch archive",
            "Grove records",
            "Hollow dataset",
            "Ion files",
        ),
        (
            "Juncture corpus",
            "Keystone records",
            "Loom archive",
            "Mosaic files",
        ),
    ),


    "binding_exception_requirement_chain": (
        (
            "Northstar dataset",
            "Opal corpus",
            "Pylon records",
            "Ridge files",
        ),
        (
            "Sable archive",
            "Timber dataset",
            "Vale records",
            "Wren corpus",
        ),
    ),
}


TARGET_IDS = (
    "A",
    "B",
    "C",
    "D",
)


EVIDENCE_PERMUTATIONS = (
    (0, 1, 2, 3),
    (2, 0, 3, 1),
    (1, 3, 0, 2),
    (3, 2, 1, 0),
)


def entity_map(
    family,
    variant,
):

    names = GROUP_ENTITIES[
        family
    ][
        variant
    ]

    return {
        target_id:
            name

        for target_id, name
        in zip(
            TARGET_IDS,
            names,
        )
    }


def requirement_sentence(
    state,
    *,
    satisfied,
    failed,
    unresolved,
):

    if state == RequirementState.SATISFIED:
        return satisfied

    if state == RequirementState.FAILED:
        return failed

    if state == RequirementState.UNRESOLVED:
        return unresolved

    raise ValueError(
        state
    )


def question(
    family,
    target_name,
    variant,
):

    if family == "same_passage_entity_binding":

        if variant == 0:

            return (
                f"Under the supplied evidence, may someone publish "
                f"the {target_name} to the partner catalog?"
            )

        return (
            f"Based only on the supplied evidence, is publication of "
            f"the {target_name} to the partner catalog permitted?"
        )


    if family == "purpose_membership_resolution":

        if variant == 0:

            return (
                f"Under the supplied evidence, may someone use "
                f"the {target_name} for commercial product testing?"
            )

        return (
            f"Based only on these passages, is commercial product testing "
            f"of the {target_name} permitted?"
        )


    if family == "trigger_plus_requirement_exception":

        if variant == 0:

            return (
                f"Under the supplied evidence, may someone release "
                f"the {target_name} through the backup delivery channel?"
            )

        return (
            f"Based only on these passages, is backup-channel release of "
            f"the {target_name} permitted?"
        )


    if family == "binding_exception_requirement_chain":

        if variant == 0:

            return (
                f"Under the supplied evidence, may someone enable "
                f"remote maintenance for the {target_name}?"
            )

        return (
            f"Based only on these passages, is remote maintenance "
            f"permitted for the {target_name}?"
        )


    raise ValueError(
        family
    )


def same_passage_clause(
    target_name,
    state,
):

    action = (
        f"Publication of the {target_name} "
        f"to the partner catalog"
    )


    if state.applicability == Applicability.UNRESOLVED:

        return (
            f"The {target_name} is covered by inventory and catalog "
            f"administration rules, but the supplied policy does not "
            f"establish whether {action.lower()} is governed or permitted."
        )


    if state.base_rule == BaseRule.PROHIBITION:

        return (
            f"{action} is expressly prohibited."
        )


    requirement = state.requirements[
        0
    ]


    status = requirement_sentence(
        requirement.state,
        satisfied=(
            "The required review is complete."
        ),
        failed=(
            "The required review was denied."
        ),
        unresolved=(
            "Required-review status is not established."
        ),
    )


    return (
        f"{action} is permitted only when the required review is complete. "
        f"{status}"
    )


def purpose_clause(
    target_name,
    state,
):

    if state.applicability == Applicability.UNRESOLVED:

        return (
            f"The usage charter for the {target_name} permits approved "
            f"evaluation activity, but the supplied definitions do not "
            f"establish whether commercial product testing falls within "
            f"that governed purpose."
        )


    if state.base_rule == BaseRule.PROHIBITION:

        return (
            f"The usage charter expressly governs commercial product "
            f"testing of the {target_name} and prohibits that purpose."
        )


    requirement = state.requirements[
        0
    ]


    status = requirement_sentence(
        requirement.state,
        satisfied=(
            "Ethics review is approved."
        ),
        failed=(
            "Ethics review was denied."
        ),
        unresolved=(
            "Ethics-review status is not established."
        ),
    )


    return (
        f"The usage charter expressly includes commercial product testing "
        f"of the {target_name}. That use is permitted only when ethics "
        f"review is approved. {status}"
    )


def exception_clause(
    target_name,
    state,
):

    if state.applicability == Applicability.UNRESOLVED:

        return (
            f"Delivery documentation discusses the {target_name}, but the "
            f"supplied evidence does not establish whether the "
            f"backup-delivery-channel rule governs this release."
        )


    trigger = state.exception.trigger

    requirement = state.exception.requirements[
        0
    ]


    trigger_status = requirement_sentence(
        trigger.state,
        satisfied=(
            "The exception trigger is active."
        ),
        failed=(
            "The exception trigger is inactive."
        ),
        unresolved=(
            "Exception-trigger status is not established."
        ),
    )


    requirement_status = requirement_sentence(
        requirement.state,
        satisfied=(
            "Exception approval is recorded."
        ),
        failed=(
            "Exception approval was denied."
        ),
        unresolved=(
            "Exception-approval status is not established."
        ),
    )


    return (
        f"Release of the {target_name} through the backup delivery channel "
        f"is prohibited. An exception may displace that prohibition only "
        f"when the exception trigger is active and exception approval is "
        f"recorded. {trigger_status} {requirement_status}"
    )


def cross_gate_clause(
    target_name,
    state,
):

    if state.applicability == Applicability.UNRESOLVED:

        return (
            f"Service documentation references the {target_name}, but the "
            f"supplied evidence does not establish whether the remote "
            f"maintenance rule applies to this system."
        )


    trigger = state.exception.trigger

    requirement = state.exception.requirements[
        0
    ]


    trigger_status = requirement_sentence(
        trigger.state,
        satisfied=(
            "The contingency state is active."
        ),
        failed=(
            "The contingency state is inactive."
        ),
        unresolved=(
            "Contingency-state status is not established."
        ),
    )


    requirement_status = requirement_sentence(
        requirement.state,
        satisfied=(
            "Special authorization is recorded."
        ),
        failed=(
            "Special authorization was denied."
        ),
        unresolved=(
            "Special-authorization status is not established."
        ),
    )


    return (
        f"Remote maintenance for the {target_name} is prohibited by the "
        f"applicable service-control rule. A contingency exception can "
        f"override the prohibition only when the contingency state is "
        f"active and special authorization is recorded. "
        f"{trigger_status} {requirement_status}"
    )


def clause(
    family,
    target_name,
    state,
):

    if family == "same_passage_entity_binding":

        return same_passage_clause(
            target_name,
            state,
        )


    if family == "purpose_membership_resolution":

        return purpose_clause(
            target_name,
            state,
        )


    if family == "trigger_plus_requirement_exception":

        return exception_clause(
            target_name,
            state,
        )


    if family == "binding_exception_requirement_chain":

        return cross_gate_clause(
            target_name,
            state,
        )


    raise ValueError(
        family
    )


def render_world(
    *,
    family,
    variant,
    world_index,
    states,
):

    entities = entity_map(
        family,
        variant,
    )


    permutation = list(
        EVIDENCE_PERMUTATIONS[
            world_index
        ]
    )


    if variant == 1:

        permutation = permutation[
            1:
        ] + permutation[
            :1
        ]


    ordered_targets = [
        TARGET_IDS[
            index
        ]
        for index
        in permutation
    ]


    if family == "same_passage_entity_binding":

        pieces = []

        for target_id in ordered_targets:

            pieces.append(
                clause(
                    family,
                    entities[
                        target_id
                    ],
                    states[
                        target_id
                    ],
                )
            )


        evidence = (
            "[E1] "
            + " ".join(
                pieces
            )
        )


        refs = {
            target_id:
                "E1"
            for target_id
            in TARGET_IDS
        }


        return (
            evidence,
            refs,
            entities,
        )


    evidence_blocks = []

    refs = {}


    for ref_index, target_id in enumerate(
        ordered_targets,
        start=1,
    ):

        ref = (
            f"E{ref_index}"
        )

        refs[
            target_id
        ] = ref


        evidence_blocks.append(
            f"[{ref}] "
            + clause(
                family,
                entities[
                    target_id
                ],
                states[
                    target_id
                ],
            )
        )


    return (
        "\n\n".join(
            evidence_blocks
        ),
        refs,
        entities,
    )


def rationale(
    family,
    state,
    result,
):

    code = result.reason_code


    if code == "NO_GOVERNING_RULE":

        return (
            "The supplied evidence does not establish a governing rule "
            "for the exact proposition."
        )


    if code == "APPLICABILITY_UNRESOLVED":

        return (
            "The supplied evidence does not establish that the governing "
            "rule applies to the exact proposition."
        )


    if code == "DIRECT_PROHIBITION":

        return (
            "The governing rule directly prohibits the requested action."
        )


    if code == "PERMISSION_ALL_REQUIREMENTS_SATISFIED":

        return (
            "The governing permission applies and every material "
            "requirement is satisfied."
        )


    if code == "PERMISSION_REQUIREMENT_FAILED":

        return (
            "The governing rule applies, but a required permission state "
            "has explicitly failed."
        )


    if code == "PERMISSION_REQUIREMENT_UNRESOLVED":

        return (
            "The governing permission applies, but a material requirement "
            "remains unresolved."
        )


    if code == "EXCEPTION_ACTIVE_ALL_REQUIREMENTS_SATISFIED":

        return (
            "The exception displaces the base prohibition because its "
            "trigger and every material requirement are satisfied."
        )


    if code == "EXCEPTION_TRIGGER_OR_REQUIREMENT_FAILED":

        return (
            "The base prohibition remains controlling because a required "
            "exception state has explicitly failed."
        )


    if code == "EXCEPTION_TRIGGER_OR_REQUIREMENT_UNRESOLVED":

        return (
            "The exception could displace the prohibition, but a required "
            "exception state remains unresolved."
        )


    raise ValueError(
        code
    )


def output_object(
    *,
    family,
    state,
    result,
    citation_ref,
):

    return {
        "decision":
            result.decision.value,

        "citations": [
            citation_ref
        ],

        "conditions":
            list(
                result.conditions
            ),

        "rationale":
            rationale(
                family,
                state,
                result,
            ),
    }


def serialize_output(
    value,
):

    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def build_input(
    *,
    evidence,
    question_text,
    input_order,
):

    if input_order == "evidence_first":

        return (
            "Evidence:\n"
            + evidence
            + "\n\nQuestion:\n"
            + question_text
        )


    if input_order == "question_first":

        return (
            "Question:\n"
            + question_text
            + "\n\nEvidence:\n"
            + evidence
        )


    raise ValueError(
        input_order
    )
