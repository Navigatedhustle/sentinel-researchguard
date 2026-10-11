"""
Sentinel V6 frozen operator ontology.

This file defines only abstract operators and family identities.

It contains:
- no generated examples
- no Challenge V2 case content
- no Challenge V3 content
- no model code
"""

LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)


STATE_VALUES = (
    "SATISFIED",
    "FAILED",
    "UNRESOLVED",
)


OPERATORS = {

    "target_entity_binding": {

        "description":
            "Bind facts only to the exact entity and proposition asked about.",

        "depends_on": (),

        "train_families": (
            "multi_asset_opposing_states",
            "same_passage_entity_binding",
            "cross_document_target_binding",
            "multi_dimension_target_binding",
        ),

        "development_families": (
            "nested_reference_target_binding",
        ),

        "required_labels":
            LABELS,
    },


    "applicability_scope_resolution": {

        "description":
            "Determine whether supplied evidence directly governs the exact proposition.",

        "depends_on": (
            "target_entity_binding",
        ),

        "train_families": (
            "purpose_membership_resolution",
            "destination_jurisdiction_resolution",
            "category_membership_resolution",
            "route_environment_scope_resolution",
        ),

        "development_families": (
            "authority_scope_resolution",
        ),

        "required_labels":
            LABELS,
    },


    "operative_rule_precedence": {

        "description":
            "Select the controlling rule among applicable rules.",

        "depends_on": (
            "target_entity_binding",
            "applicability_scope_resolution",
        ),

        "train_families": (
            "specific_over_general_rule",
            "current_over_superseded_rule",
            "delegated_authority_priority",
            "defined_category_rule_selection",
        ),

        "development_families": (
            "purpose_specific_rule_priority",
        ),

        "required_labels":
            LABELS,
    },


    "exception_state_propagation": {

        "description":
            "Resolve exception activation and propagate it over a base prohibition.",

        "depends_on": (
            "target_entity_binding",
            "applicability_scope_resolution",
            "operative_rule_precedence",
        ),

        "train_families": (
            "single_trigger_exception",
            "trigger_plus_requirement_exception",
            "multi_requirement_exception",
            "competing_exception_entities",
        ),

        "development_families": (
            "nested_exception_state_chain",
        ),

        "required_labels":
            LABELS,
    },


    "requirement_ledger_composition": {

        "description":
            "Resolve and compose SATISFIED, FAILED, and UNRESOLVED material requirements.",

        "depends_on": (
            "applicability_scope_resolution",
            "operative_rule_precedence",
        ),

        "train_families": (
            "all_requirements_satisfied",
            "single_unresolved_requirement",
            "single_failed_requirement",
            "mixed_requirement_states",
        ),

        "development_families": (
            "failed_vs_unresolved_precedence",
        ),

        "required_labels":
            LABELS,
    },


    "final_decision_finalization": {

        "description":
            "Map the completed governing state deterministically to the final decision.",

        "depends_on": (
            "applicability_scope_resolution",
            "operative_rule_precedence",
            "requirement_ledger_composition",
        ),

        "train_families": (
            "satisfied_ledger_finalization",
            "failed_ledger_finalization",
            "unresolved_ledger_finalization",
            "nonapplicable_finalization",
        ),

        "development_families": (
            "mixed_gate_finalization",
        ),

        "required_labels":
            LABELS,
    },


    "temporal_authority_state": {

        "description":
            "Resolve time- and authority-dependent state for the exact request.",

        "depends_on": (
            "target_entity_binding",
            "applicability_scope_resolution",
        ),

        "train_families": (
            "authorization_window_state",
            "temporary_authority_state",
            "review_cycle_state",
            "delegated_term_state",
        ),

        "development_families": (
            "overlapping_temporal_authority",
        ),

        "required_labels":
            LABELS,
    },


    "cross_gate_state_composition": {

        "description":
            "Preserve causal state across multiple reasoning gates.",

        "depends_on": (
            "target_entity_binding",
            "applicability_scope_resolution",
            "operative_rule_precedence",
            "exception_state_propagation",
            "requirement_ledger_composition",
            "final_decision_finalization",
            "temporal_authority_state",
        ),

        "train_families": (
            "binding_applicability_requirement_chain",
            "binding_exception_requirement_chain",
            "applicability_precedence_ledger_chain",
            "temporal_exception_finalization_chain",
        ),

        "development_families": (
            "full_policy_state_chain",
        ),

        "required_labels":
            LABELS,
    },
}


OPERATOR_ORDER = (
    "target_entity_binding",
    "applicability_scope_resolution",
    "operative_rule_precedence",
    "exception_state_propagation",
    "requirement_ledger_composition",
    "final_decision_finalization",
    "temporal_authority_state",
    "cross_gate_state_composition",
)


TRAIN_FAMILIES = tuple(
    family
    for operator in OPERATOR_ORDER
    for family in OPERATORS[operator]["train_families"]
)


DEVELOPMENT_FAMILIES = tuple(
    family
    for operator in OPERATOR_ORDER
    for family in OPERATORS[operator]["development_families"]
)


ALL_FAMILIES = (
    TRAIN_FAMILIES
    + DEVELOPMENT_FAMILIES
)


FAMILY_TO_OPERATOR = {

    family:
        operator

    for operator in OPERATOR_ORDER

    for family in (
        OPERATORS[operator]["train_families"]
        + OPERATORS[operator]["development_families"]
    )
}


FAMILY_TO_SPLIT = {

    **{
        family:
            "train"
        for family
        in TRAIN_FAMILIES
    },

    **{
        family:
            "development"
        for family
        in DEVELOPMENT_FAMILIES
    },
}


# These are ontology-level invariants.
#
# They do not define row counts or realization mechanics.

EXPECTED_OPERATOR_COUNT = 8
EXPECTED_TRAIN_FAMILY_COUNT = 32
EXPECTED_DEVELOPMENT_FAMILY_COUNT = 8
EXPECTED_TOTAL_FAMILY_COUNT = 40
