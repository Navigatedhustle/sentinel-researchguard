"""
Challenge V3 frozen structural blueprint.

No benchmark cases are generated here.
No V6 training modules are imported.
"""

LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)


QUESTION_TYPES = (
    "relational_counterfactual",
    "stress",
)


INPUT_ORDERS = (
    "evidence_first",
    "question_first",
)


COMPOSITIONS = (
    "controlled_release_scope",
    "delegated_review_window",
    "dual_prerequisite_exception",
    "current_rule_destination_gate",
    "defined_workflow_control",
    "purpose_boundary_with_notice",
    "temporary_access_override",
    "recipient_class_authorization",
    "route_specific_rule_revision",
    "multi_entity_mixed_state",
    "authority_condition_chain",
    "full_governance_state_chain",
)


COMPOSITION_PRIMARY_CAPABILITY = {

    "controlled_release_scope":
        "binding_applicability_requirement",

    "delegated_review_window":
        "authority_temporal_requirement",

    "dual_prerequisite_exception":
        "exception_multi_requirement",

    "current_rule_destination_gate":
        "scope_supersession_requirement",

    "defined_workflow_control":
        "definition_applicability_rule",

    "purpose_boundary_with_notice":
        "purpose_applicability_requirement",

    "temporary_access_override":
        "temporal_exception_state",

    "recipient_class_authorization":
        "recipient_binding_authorization",

    "route_specific_rule_revision":
        "route_binding_supersession",

    "multi_entity_mixed_state":
        "dense_target_binding",

    "authority_condition_chain":
        "authority_multi_requirement",

    "full_governance_state_chain":
        "cross_gate_state_composition",
}


RELATIONAL_WORLDS = (
    "world_0",
    "world_1",
    "world_2",
    "world_3",
)


RELATIONAL_TARGETS = (
    "A",
    "B",
    "C",
    "D",
)


def relational_label(
    world_index,
    target_index,
):

    return LABELS[
        (
            world_index
            + target_index
        )
        % 4
    ]


STRESS_LABELS = LABELS


EXPECTED = {

    "compositions":
        12,

    "cases_per_composition":
        20,

    "relational_per_composition":
        16,

    "stress_per_composition":
        4,

    "total_cases":
        240,

    "relational_cases":
        192,

    "stress_cases":
        48,

    "cases_per_label":
        60,

    "cases_per_label_per_composition":
        5,

    "evidence_first":
        120,

    "question_first":
        120,
}


PUBLIC_CASE_KEYS = (
    "case_id",
    "composition",
    "question_type",
    "input_order",
    "instruction",
    "question",
    "evidence_text",
    "input",
    "gold",
)


GOLD_KEYS = (
    "decision",
    "citations",
    "conditions",
    "rationale",
)
