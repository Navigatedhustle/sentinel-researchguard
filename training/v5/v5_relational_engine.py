"""
Sentinel V5 generic relational realization engine.

The engine contains one semantic implementation per abstract operator.

Family-specific nouns, rules, requirements, scopes, and mechanisms
come from v5_family_realizers.py.

This prevents a fake 40-family curriculum made from 40 copied templates.

IMPORTANT
---------

- Does NOT read Challenge V1 or Challenge V2.
- Does NOT generate curriculum JSONL.
- Does NOT train or evaluate a model.
"""

from training.v5.v5_family_realizers import (
    family_config,
)


LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)


def _variant(
    variant,
    *choices,
):

    return choices[
        variant
        % len(
            choices
        )
    ]


def _state(
    action,
    governing_facts,
    distractors,
    conditions,
    rationale,
):

    return {
        "action":
            action,

        "governing_facts":
            list(
                governing_facts
            ),

        "distractors":
            list(
                distractors
            ),

        "conditions":
            list(
                conditions
            ),

        "rationale":
            rationale,
    }


def realize_state(
    operator,
    family,
    asset,
    label,
    variant=0,
):
    """
    Realize one target state.

    Returns:

        action
        governing_facts
        distractors
        conditions
        rationale

    Governing facts become gold citation candidates.

    Distractors are included in the evidence world but must NOT
    become gold citations.
    """

    if label not in LABELS:
        raise ValueError(
            f"Unknown V5 label: {label}"
        )

    cfg = family_config(
        family
    )

    if cfg[
        "operator"
    ] != operator:

        raise ValueError(
            f"Family/operator mismatch: "
            f"{family} != {operator}"
        )

    action = cfg[
        "action"
    ].format(
        asset=asset
    )


    # =================================================================
    # 1. OPERATIVE RULE SELECTION
    # =================================================================

    if (
        operator
        == "operative_rule_selection"
    ):

        high = cfg[
            "high_rule"
        ]

        low = cfg[
            "low_rule"
        ]

        requirement = cfg[
            "requirement"
        ]

        precedence = _variant(
            variant,

            (
                f"For {asset}, when both provisions address the requested "
                f"action, the {high} takes precedence over the {low}."
            ),

            (
                f"If the {high} and {low} conflict for {asset}, "
                f"the {high} is the controlling provision."
            ),

            (
                f"The {high} governs conflicts concerning {asset}; "
                f"the {low} is subordinate for that action."
            ),
        )

        if label == "YES":

            permission = _variant(
                variant,

                (
                    f"The {high} permits someone to {action} after "
                    f"{requirement}. For {asset}, {requirement} is complete."
                ),

                (
                    f"Under the {high}, {action} is allowed once "
                    f"{requirement} has been satisfied. "
                    f"That requirement is satisfied for {asset}."
                ),

                (
                    f"The controlling {high} authorizes someone to {action} "
                    f"provided {requirement} is complete; "
                    f"the record for {asset} shows completion."
                ),
            )

            distractor = (
                f"The {low} would otherwise prohibit someone from "
                f"{action}."
            )

            return _state(
                action,
                [
                    precedence,
                    permission,
                ],
                [
                    distractor,
                ],
                [],
                (
                    f"The {high} controls and its material "
                    f"{requirement} requirement is satisfied."
                ),
            )

        if label == "NO":

            prohibition = _variant(
                variant,

                (
                    f"The {high} expressly prohibits someone from "
                    f"{action}."
                ),

                (
                    f"Under the controlling {high}, someone may not "
                    f"{action}."
                ),

                (
                    f"The operative {high} bars the requested action "
                    f"for {asset}."
                ),
            )

            distractor = (
                f"The {low} would otherwise permit someone to "
                f"{action}."
            )

            return _state(
                action,
                [
                    precedence,
                    prohibition,
                ],
                [
                    distractor,
                ],
                [],
                (
                    f"The higher-priority {high} controls and "
                    f"prohibits the requested action."
                ),
            )

        if label == "CONDITIONAL":

            permission = _variant(
                variant,

                (
                    f"The {high} permits someone to {action} only after "
                    f"{requirement}. The status of {requirement} for "
                    f"{asset} is not established."
                ),

                (
                    f"Under the {high}, {action} may occur if "
                    f"{requirement} is satisfied. The supplied material "
                    f"does not resolve that state for {asset}."
                ),

                (
                    f"The controlling {high} makes {requirement} a "
                    f"material prerequisite for {action}; its current "
                    f"status for {asset} remains unresolved."
                ),
            )

            distractor = (
                f"The subordinate {low} would prohibit someone from "
                f"{action}."
            )

            return _state(
                action,
                [
                    precedence,
                    permission,
                ],
                [
                    distractor,
                ],
                [
                    f"{requirement} unresolved",
                ],
                (
                    f"The operative rule may permit the action, "
                    f"but {requirement} remains unresolved."
                ),
            )

        # INSUFFICIENT

        related = _variant(
            variant,

            (
                f"The {high} for {asset} assigns reporting ownership "
                f"and inventory duties."
            ),

            (
                f"For {asset}, the available {high} addresses retention "
                f"schedules and administrative custodianship."
            ),

            (
                f"The supplied {high} describes audit routing and "
                f"recordkeeping responsibilities for {asset}."
            ),
        )

        return _state(
            action,
            [],
            [
                related,
            ],
            [],
            (
                "The supplied provisions are administratively related "
                "but do not directly govern the requested action."
            ),
        )


    # =================================================================
    # 2. PROHIBITION SHORT-CIRCUIT
    # =================================================================

    if (
        operator
        == "prohibition_short_circuit"
    ):

        rule = cfg[
            "rule"
        ]

        requirement = cfg[
            "requirement"
        ]

        unrelated = cfg[
            "distractor"
        ]

        neighbor_action = cfg[
            "neighbor_action"
        ].format(
            asset=asset
        )

        if label == "YES":

            fact = _variant(
                variant,

                (
                    f"The {rule} permits someone to {action} when "
                    f"{requirement} is active. For {asset}, "
                    f"{requirement} is active."
                ),

                (
                    f"Under the {rule}, {action} is allowed provided "
                    f"{requirement} is satisfied; it is satisfied "
                    f"for {asset}."
                ),

                (
                    f"The controlling {rule} authorizes {action} after "
                    f"{requirement}. The required state is confirmed "
                    f"for {asset}."
                ),
            )

            return _state(
                action,
                [
                    fact,
                ],
                [],
                [],
                (
                    "The controlling rule directly permits the action "
                    "and its material requirement is satisfied."
                ),
            )

        if label == "NO":

            fact = _variant(
                variant,

                (
                    f"The {rule} expressly prohibits someone from "
                    f"{action}."
                ),

                (
                    f"Under the controlling {rule}, someone may not "
                    f"{action}."
                ),

                (
                    f"The operative {rule} directly bars the requested "
                    f"action for {asset}."
                ),
            )

            distractor = (
                f"The status of the separate {unrelated} for {asset} "
                f"is unresolved."
            )

            return _state(
                action,
                [
                    fact,
                ],
                [
                    distractor,
                ],
                [],
                (
                    "The controlling prohibition resolves the request; "
                    "the unrelated unresolved fact does not change it."
                ),
            )

        if label == "CONDITIONAL":

            fact = _variant(
                variant,

                (
                    f"The {rule} permits someone to {action} only when "
                    f"{requirement} is satisfied. Its status for "
                    f"{asset} is unresolved."
                ),

                (
                    f"Under the {rule}, {action} may occur subject to "
                    f"{requirement}; the supplied evidence does not "
                    f"establish that state for {asset}."
                ),

                (
                    f"The controlling {rule} makes {requirement} a "
                    f"material prerequisite for {action}, and that "
                    f"prerequisite remains unresolved for {asset}."
                ),
            )

            return _state(
                action,
                [
                    fact,
                ],
                [],
                [
                    f"{requirement} unresolved",
                ],
                (
                    "The route may be permitted, but its material "
                    "requirement remains unresolved."
                ),
            )

        related = _variant(
            variant,

            (
                f"The supplied {rule} governs whether someone may "
                f"{neighbor_action}."
            ),

            (
                f"For {asset}, the available provision regulates "
                f"the neighboring action: {neighbor_action}."
            ),

            (
                f"The evidence addresses {neighbor_action} for {asset}, "
                f"rather than the requested route."
            ),
        )

        return _state(
            action,
            [],
            [
                related,
            ],
            [],
            (
                "The evidence governs a neighboring route or class "
                "rather than the exact requested action."
            ),
        )


    # =================================================================
    # 3. PERMISSION CONDITION LEDGER
    # =================================================================

    if (
        operator
        == "permission_condition_ledger"
    ):

        rule = cfg[
            "rule"
        ]

        requirements = list(
            cfg[
                "requirements"
            ]
        )

        if len(
            requirements
        ) != 2:

            raise AssertionError(
                f"{family} must define exactly two ledger requirements."
            )

        r1, r2 = requirements

        if label == "YES":

            permission = (
                f"The {rule} directly permits someone to {action} "
                f"when both {r1} and {r2} are satisfied."
            )

            fact1 = _variant(
                variant,

                f"For {asset}, {r1} is satisfied.",
                f"The record for {asset} confirms completion of {r1}.",
                f"{r1.capitalize()} has been completed for {asset}.",
            )

            fact2 = _variant(
                variant + 1,

                f"For {asset}, {r2} is satisfied.",
                f"The record for {asset} confirms completion of {r2}.",
                f"{r2.capitalize()} has been completed for {asset}.",
            )

            return _state(
                action,
                [
                    permission,
                    fact1,
                    fact2,
                ],
                [],
                [],
                (
                    "Direct permission exists and every material "
                    "ledger requirement is satisfied."
                ),
            )

        if label == "NO":

            prohibition = _variant(
                variant,

                (
                    f"The controlling {rule} expressly prohibits someone "
                    f"from {action}."
                ),

                (
                    f"Under the operative {rule}, someone may not "
                    f"{action}."
                ),

                (
                    f"The {rule} directly bars the requested action "
                    f"for {asset}."
                ),
            )

            distractors = [
                f"For {asset}, {r1} is satisfied.",
                f"For {asset}, {r2} is satisfied.",
            ]

            return _state(
                action,
                [
                    prohibition,
                ],
                distractors,
                [],
                (
                    "A controlling prohibition resolves the action even "
                    "though other requirement states are favorable."
                ),
            )

        if label == "CONDITIONAL":

            unresolved_index = (
                variant
                % 2
            )

            unresolved = requirements[
                unresolved_index
            ]

            satisfied = requirements[
                1 - unresolved_index
            ]

            permission = (
                f"The {rule} permits someone to {action} only when "
                f"both {r1} and {r2} are satisfied."
            )

            satisfied_fact = (
                f"For {asset}, {satisfied} is satisfied."
            )

            unresolved_fact = _variant(
                variant,

                (
                    f"The status of {unresolved} for {asset} "
                    f"is not established."
                ),

                (
                    f"The supplied record leaves {unresolved} "
                    f"unresolved for {asset}."
                ),

                (
                    f"No current status for {unresolved} is supplied "
                    f"for {asset}."
                ),
            )

            return _state(
                action,
                [
                    permission,
                    satisfied_fact,
                    unresolved_fact,
                ],
                [],
                [
                    f"{unresolved} unresolved",
                ],
                (
                    "Permission exists, but one material ledger "
                    "requirement remains unresolved."
                ),
            )

        related = _variant(
            variant,

            (
                f"The record for {asset} tracks {r1} and {r2} "
                f"for administrative reporting."
            ),

            (
                f"Administrative entries for {asset} reference "
                f"{r1} and {r2} as workflow metadata."
            ),

            (
                f"The supplied material records completion dates "
                f"associated with {r1} and {r2} for {asset}."
            ),
        )

        return _state(
            action,
            [],
            [
                related,
            ],
            [],
            (
                "Requirement-related administrative evidence is present, "
                "but no governing permission or prohibition is supplied."
            ),
        )


    # =================================================================
    # 4. EXCEPTION ACTIVATION CHAIN
    # =================================================================

    if (
        operator
        == "exception_activation_chain"
    ):

        base_rule = cfg[
            "base_rule"
        ]

        exception_rule = cfg[
            "exception_rule"
        ]

        trigger = cfg[
            "trigger"
        ]

        requirement = cfg[
            "requirement"
        ]

        base = (
            f"The {base_rule} prohibits someone from {action} unless "
            f"a valid exception applies."
        )

        exception = (
            f"The {exception_rule} can permit someone to {action} when "
            f"{trigger} is active and {requirement} is satisfied."
        )

        if label == "YES":

            status = _variant(
                variant,

                (
                    f"For {asset}, {trigger} is active and "
                    f"{requirement} is satisfied."
                ),

                (
                    f"The record for {asset} confirms both an active "
                    f"{trigger} and completed {requirement}."
                ),

                (
                    f"{asset} is within an active {trigger}; "
                    f"{requirement} has also been completed."
                ),
            )

            return _state(
                action,
                [
                    base,
                    exception,
                    status,
                ],
                [],
                [],
                (
                    "The exception is active and every material "
                    "exception requirement is satisfied."
                ),
            )

        if label == "NO":

            if (
                variant
                % 2
                == 0
            ):

                status = (
                    f"For {asset}, {trigger} is not active."
                )

                rationale = (
                    "The exception trigger is inactive, so the "
                    "base prohibition remains operative."
                )

            else:

                status = (
                    f"For {asset}, {trigger} is active, but the "
                    f"required {requirement} has failed."
                )

                rationale = (
                    "The exception cannot authorize the action because "
                    "a required exception element has failed."
                )

            return _state(
                action,
                [
                    base,
                    exception,
                    status,
                ],
                [],
                [],
                rationale,
            )

        if label == "CONDITIONAL":

            status = _variant(
                variant,

                (
                    f"For {asset}, {trigger} is active, but the status "
                    f"of {requirement} is not established."
                ),

                (
                    f"The {trigger} applies to {asset}; however, "
                    f"{requirement} remains unresolved."
                ),

                (
                    f"The exception trigger is active for {asset}, "
                    f"while current {requirement} status is unavailable."
                ),
            )

            return _state(
                action,
                [
                    base,
                    exception,
                    status,
                ],
                [],
                [
                    f"{requirement} unresolved",
                ],
                (
                    "The exception could permit the action, but a "
                    "material exception requirement remains unresolved."
                ),
            )

        related = _variant(
            variant,

            (
                f"The operations log for {asset} records the timing "
                f"of {trigger} for scheduling purposes."
            ),

            (
                f"For {asset}, administrative records mention "
                f"{trigger} and routine maintenance ownership."
            ),

            (
                f"The supplied status report tracks {trigger} for "
                f"{asset} but addresses operational coordination."
            ),
        )

        return _state(
            action,
            [],
            [
                related,
            ],
            [],
            (
                "The evidence discusses the event state but supplies no "
                "governing exception rule for the requested action."
            ),
        )


    # =================================================================
    # 5. AUTHORITY STATE RESOLUTION
    # =================================================================

    if (
        operator
        == "authority_state_resolution"
    ):

        role = cfg[
            "role"
        ]

        states = list(
            cfg[
                "states"
            ]
        )

        if len(
            states
        ) != 4:

            raise AssertionError(
                f"{family} must define exactly four authority states."
            )

        assignment_state = states[
            0
        ]

        positive_state = states[
            1
        ]

        negative_state = states[
            2 + (
                variant
                % 2
            )
        ]

        authority_rule = (
            f"Only a currently operative {role} may {action}."
        )

        if label == "YES":

            status = _variant(
                variant,

                (
                    f"For {asset}, the {role} is {assignment_state} "
                    f"and {positive_state}."
                ),

                (
                    f"The authority record for {asset} shows the "
                    f"{role} as {assignment_state}; its operative state "
                    f"is {positive_state}."
                ),

                (
                    f"The {role} associated with {asset} has a current "
                    f"record marked {assignment_state} and {positive_state}."
                ),
            )

            return _state(
                action,
                [
                    authority_rule,
                    status,
                ],
                [],
                [],
                (
                    "The actor possesses the current operative authority "
                    "required for the requested action."
                ),
            )

        if label == "NO":

            status = (
                f"For {asset}, the {role} authority is "
                f"{negative_state}."
            )

            return _state(
                action,
                [
                    authority_rule,
                    status,
                ],
                [],
                [],
                (
                    "The required authority is not currently operative."
                ),
            )

        if label == "CONDITIONAL":

            status = _variant(
                variant,

                (
                    f"For {asset}, the {role} is recorded as "
                    f"{assignment_state}, but current operative status "
                    f"is not established."
                ),

                (
                    f"The record assigns the {role} to {asset}; "
                    f"whether that authority is presently operative "
                    f"remains unresolved."
                ),

                (
                    f"Historical assignment of the {role} for {asset} "
                    f"is documented, but its current authority state "
                    f"is unavailable."
                ),
            )

            return _state(
                action,
                [
                    authority_rule,
                    status,
                ],
                [],
                [
                    "current authority unresolved",
                ],
                (
                    "The action may be authorized, but current operative "
                    "authority remains unresolved."
                ),
            )

        related = _variant(
            variant,

            (
                f"The personnel record for {asset} lists the {role} "
                f"for scheduling and contact purposes."
            ),

            (
                f"Administrative records associated with {asset} "
                f"identify a {role} as a coordination contact."
            ),

            (
                f"The supplied roster for {asset} records a {role} "
                f"but describes only workflow ownership."
            ),
        )

        return _state(
            action,
            [],
            [
                related,
            ],
            [],
            (
                "A role assignment is mentioned, but no supplied rule "
                "establishes that the role governs the requested action."
            ),
        )


    # =================================================================
    # 6. CATEGORY + RULE COMPOSITION
    # =================================================================

    if (
        operator
        == "category_rule_composition"
    ):

        category = cfg[
            "classification"
        ]

        rule = cfg[
            "rule"
        ]

        requirement = cfg[
            "requirement"
        ]

        classification = (
            f"The classification rule places {asset} in the "
            f"{category}."
        )

        if label == "YES":

            authorization = (
                f"For {asset}, the {rule} permits members of the "
                f"{category} to perform this action when "
                f"{requirement} is satisfied."
            )

            status = (
                f"For {asset}, {requirement} is satisfied."
            )

            return _state(
                action,
                [
                    classification,
                    authorization,
                    status,
                ],
                [],
                [],
                (
                    "Category membership is established and the separate "
                    "authorization rule permits the action with its "
                    "requirement satisfied."
                ),
            )

        if label == "NO":

            prohibition = _variant(
                variant,

                (
                    f"For {asset}, the {rule} prohibits members of "
                    f"the {category} from performing this action."
                ),

                (
                    f"For {asset}, under the {rule}, the {category} "
                    f"may not perform the requested action."
                ),

                (
                    f"For {asset}, the governing {rule} bars this "
                    f"action for the {category}."
                ),
            )

            return _state(
                action,
                [
                    classification,
                    prohibition,
                ],
                [],
                [],
                (
                    "Category membership is established and the separate "
                    "governing rule prohibits the requested action."
                ),
            )

        if label == "CONDITIONAL":

            authorization = (
                f"For {asset}, the {rule} permits members of the "
                f"{category} to perform this action only when "
                f"{requirement} is satisfied."
            )

            status = (
                f"The status of {requirement} for {asset} is unresolved."
            )

            return _state(
                action,
                [
                    classification,
                    authorization,
                    status,
                ],
                [],
                [
                    f"{requirement} unresolved",
                ],
                (
                    "Category membership is established and a governing "
                    "permission exists, but its material condition "
                    "remains unresolved."
                ),
            )

        related = _variant(
            variant,

            classification,

            (
                f"The supplied classification material identifies "
                f"{asset} as belonging to the {category}."
            ),

            (
                f"For classification purposes, {asset} is treated as "
                f"a member of the {category}."
            ),
        )

        return _state(
            action,
            [],
            [
                related,
            ],
            [],
            (
                "Classification is established, but classification alone "
                "does not grant or deny authorization for the action."
            ),
        )


    # =================================================================
    # 7. SCOPE / DESTINATION RESOLUTION
    # =================================================================

    if (
        operator
        == "scope_destination_resolution"
    ):

        rule = cfg[
            "rule"
        ]

        in_scope = cfg[
            "in_scope"
        ]

        neighbor_scope = cfg[
            "neighbor_scope"
        ]

        requirement = cfg[
            "requirement"
        ]

        if label == "YES":

            fact = _variant(
                variant,

                (
                    f"The {rule} directly permits the requested action "
                    f"within {in_scope} when {requirement} is satisfied. "
                    f"For {asset}, {requirement} is satisfied."
                ),

                (
                    f"The requested {in_scope} falls within the permitted "
                    f"scope of the {rule}; {requirement} is complete "
                    f"for {asset}."
                ),

                (
                    f"Under the {rule}, the requested action is authorized "
                    f"in {in_scope} after {requirement}. The requirement "
                    f"is complete for {asset}."
                ),
            )

            return _state(
                action,
                [
                    fact,
                ],
                [],
                [],
                (
                    "The exact requested scope is governed and permitted, "
                    "with its material requirement satisfied."
                ),
            )

        if label == "NO":

            fact = _variant(
                variant,

                (
                    f"The {rule} expressly excludes {in_scope} from "
                    f"the permitted scope for {asset}."
                ),

                (
                    f"Under the controlling {rule}, the requested "
                    f"{in_scope} is prohibited for {asset}."
                ),

                (
                    f"The operative {rule} directly bars the requested "
                    f"scope, {in_scope}, for {asset}."
                ),
            )

            return _state(
                action,
                [
                    fact,
                ],
                [],
                [],
                (
                    "The exact requested scope is directly excluded by "
                    "the controlling rule."
                ),
            )

        if label == "CONDITIONAL":

            fact = _variant(
                variant,

                (
                    f"The {rule} can permit the requested action in "
                    f"{in_scope} only after {requirement}. The status "
                    f"of {requirement} for {asset} is unresolved."
                ),

                (
                    f"The requested {in_scope} is conditionally within "
                    f"scope under the {rule}, but {requirement} has "
                    f"not been established for {asset}."
                ),

                (
                    f"Under the {rule}, {in_scope} is an authorized "
                    f"scope if {requirement} is satisfied; its current "
                    f"status for {asset} is unavailable."
                ),
            )

            return _state(
                action,
                [
                    fact,
                ],
                [],
                [
                    f"{requirement} unresolved",
                ],
                (
                    "The requested scope may be permitted, but its "
                    "material requirement remains unresolved."
                ),
            )

        related = _variant(
            variant,

            (
                f"The {rule} governs the neighboring scope "
                f"{neighbor_scope} for {asset}."
            ),

            (
                f"For {asset}, the available scope provision addresses "
                f"{neighbor_scope}."
            ),

            (
                f"The supplied rule places {neighbor_scope} within its "
                f"scope for {asset}."
            ),
        )

        return _state(
            action,
            [],
            [
                related,
            ],
            [],
            (
                "The evidence governs a neighboring scope rather than "
                "the exact requested destination, location, or purpose."
            ),
        )


    # =================================================================
    # 8. EVIDENCE RELEVANCE BOUNDARY
    # =================================================================

    if (
        operator
        == "evidence_relevance_boundary"
    ):

        rule = cfg[
            "rule"
        ]

        requirement = cfg[
            "requirement"
        ]

        neighbor_action = cfg[
            "neighbor_action"
        ].format(
            asset=asset
        )

        if label == "YES":

            fact = _variant(
                variant,

                (
                    f"The {rule} directly permits someone to {action} "
                    f"when {requirement} is satisfied. "
                    f"For {asset}, {requirement} is satisfied."
                ),

                (
                    f"The exact requested action is governed by the "
                    f"{rule}, which permits it after {requirement}; "
                    f"that requirement is complete for {asset}."
                ),

                (
                    f"For {asset}, the {rule} authorizes the precise "
                    f"requested action and the required {requirement} "
                    f"is satisfied."
                ),
            )

            return _state(
                action,
                [
                    fact,
                ],
                [],
                [],
                (
                    "The evidence directly governs the exact proposition "
                    "and permits it."
                ),
            )

        if label == "NO":

            fact = _variant(
                variant,

                (
                    f"The {rule} expressly prohibits someone from "
                    f"{action}."
                ),

                (
                    f"The exact requested action is governed by the "
                    f"{rule}, which prohibits it for {asset}."
                ),

                (
                    f"For {asset}, the controlling {rule} directly bars "
                    f"the precise requested action."
                ),
            )

            return _state(
                action,
                [
                    fact,
                ],
                [],
                [],
                (
                    "The evidence directly governs the exact proposition "
                    "and prohibits it."
                ),
            )

        if label == "CONDITIONAL":

            fact = _variant(
                variant,

                (
                    f"The {rule} permits someone to {action} only if "
                    f"{requirement} is satisfied. Its status for "
                    f"{asset} is unresolved."
                ),

                (
                    f"The exact action is governed by the {rule}, which "
                    f"requires {requirement}; that requirement remains "
                    f"unresolved for {asset}."
                ),

                (
                    f"For {asset}, the governing {rule} conditionally "
                    f"permits the precise action subject to "
                    f"{requirement}, whose status is unavailable."
                ),
            )

            return _state(
                action,
                [
                    fact,
                ],
                [],
                [
                    f"{requirement} unresolved",
                ],
                (
                    "The exact proposition is governed, but a material "
                    "permission requirement remains unresolved."
                ),
            )

        related = _variant(
            variant,

            (
                f"The supplied policy for {asset} governs whether someone "
                f"may {neighbor_action}."
            ),

            (
                f"The available evidence directly regulates the related "
                f"action '{neighbor_action}' for {asset}."
            ),

            (
                f"For {asset}, the cited provision concerns "
                f"{neighbor_action}, a neighboring proposition."
            ),
        )

        return _state(
            action,
            [],
            [
                related,
            ],
            [],
            (
                "The supplied evidence is topically related but does not "
                "directly govern the exact proposition asked."
            ),
        )


    raise ValueError(
        f"No V5 engine for operator: {operator}"
    )


def validate_state(
    operator,
    family,
    asset,
    label,
    variant,
):

    state = realize_state(
        operator=operator,
        family=family,
        asset=asset,
        label=label,
        variant=variant,
    )

    required = {
        "action",
        "governing_facts",
        "distractors",
        "conditions",
        "rationale",
    }

    assert set(
        state
    ) == required

    assert isinstance(
        state[
            "action"
        ],
        str,
    )

    assert state[
        "action"
    ].strip()

    assert isinstance(
        state[
            "governing_facts"
        ],
        list,
    )

    assert isinstance(
        state[
            "distractors"
        ],
        list,
    )

    assert isinstance(
        state[
            "conditions"
        ],
        list,
    )

    assert isinstance(
        state[
            "rationale"
        ],
        str,
    )

    assert state[
        "rationale"
    ].strip()

    # Governing decisions need governing evidence.
    if label in {
        "YES",
        "NO",
        "CONDITIONAL",
    }:

        assert state[
            "governing_facts"
        ]

    # INSUFFICIENT deliberately has no governing citation set.
    if (
        label
        == "INSUFFICIENT"
    ):

        assert not state[
            "governing_facts"
        ]

        assert state[
            "distractors"
        ]

    # Only CONDITIONAL carries unresolved material conditions.
    if (
        label
        == "CONDITIONAL"
    ):

        assert state[
            "conditions"
        ]

    else:

        assert not state[
            "conditions"
        ]

    # All local evidence should explicitly identify the target asset.
    for text in (
        state[
            "governing_facts"
        ]
        + state[
            "distractors"
        ]
    ):

        assert asset.lower() in text.lower(), (
            family,
            label,
            variant,
            text,
        )

    return state
