"""
Sentinel V5 family ontology.

IMPORTANT:

- This file defines reasoning-family architecture only.
- It does NOT read Challenge V2.
- It does NOT generate training examples.
- It does NOT evaluate any model.

Design:

8 abstract operators

TRAIN:
    4 families / operator
    32 total

DEVELOPMENT:
    1 held-out family / operator
    8 total

The held-out family names and mechanisms are not used in TRAIN.
"""


OPERATORS = {

    # ============================================================
    # 1. OPERATIVE RULE SELECTION
    # ============================================================

    "operative_rule_selection": {

        "description":
            "Choose the rule that actually governs before "
            "evaluating downstream requirements.",

        "train": [

            {
                "family":
                    "project_charter_priority",

                "mechanism":
                    "A project-specific charter may control over "
                    "an organization-wide operational rule.",

                "precedence_dimension":
                    "project_specificity",

                "positive_structure":
                    "project charter governs and permits",

                "negative_structure":
                    "project charter governs and prohibits",

                "conditional_structure":
                    "project charter governs but required "
                    "approval remains unresolved",

                "insufficient_structure":
                    "candidate rules discuss project administration "
                    "without governing the requested action",
            },

            {
                "family":
                    "facility_protocol_priority",

                "mechanism":
                    "A facility-specific protocol may control over "
                    "a general institutional procedure.",

                "precedence_dimension":
                    "facility_specificity",

                "positive_structure":
                    "facility protocol permits",

                "negative_structure":
                    "facility protocol prohibits",

                "conditional_structure":
                    "facility protocol permits subject to unresolved control",

                "insufficient_structure":
                    "facility evidence exists but no governing action rule",
            },

            {
                "family":
                    "incident_directive_priority",

                "mechanism":
                    "An activated incident directive may control over "
                    "the normal operating rule.",

                "precedence_dimension":
                    "activated_emergency_directive",

                "positive_structure":
                    "incident directive permits",

                "negative_structure":
                    "incident directive blocks",

                "conditional_structure":
                    "incident directive permits subject to unresolved requirement",

                "insufficient_structure":
                    "incident status supplied but no operative action rule",
            },

            {
                "family":
                    "revision_status_priority",

                "mechanism":
                    "A currently effective revision may displace an older "
                    "operating instruction.",

                "precedence_dimension":
                    "revision_status",

                "positive_structure":
                    "current revision permits",

                "negative_structure":
                    "current revision prohibits",

                "conditional_structure":
                    "current revision permits subject to unresolved condition",

                "insufficient_structure":
                    "revision history supplied but requested action not governed",
            },
        ],

        "development": [

            {
                "family":
                    "program_addendum_priority",

                "mechanism":
                    "A program-specific addendum may control over "
                    "a standing organizational rule.",

                "precedence_dimension":
                    "program_specificity",
            },
        ],
    },


    # ============================================================
    # 2. PROHIBITION SHORT-CIRCUIT
    # ============================================================

    "prohibition_short_circuit": {

        "description":
            "A controlling prohibition resolves the action even when "
            "other satisfied or unresolved facts are present.",

        "train": [

            {
                "family":
                    "blocked_interface_route",

                "mechanism":
                    "A prohibited interface route remains NO regardless "
                    "of unrelated approval state.",

                "no_core":
                    "route directly prohibited",

                "distractor_axis":
                    "approval state",
            },

            {
                "family":
                    "restricted_recipient_class",

                "mechanism":
                    "A recipient-class prohibition controls even when "
                    "security controls are satisfied.",

                "no_core":
                    "recipient class directly prohibited",

                "distractor_axis":
                    "security controls",
            },

            {
                "family":
                    "disabled_processing_mode",

                "mechanism":
                    "A processing mode is disabled by the controlling rule "
                    "despite unrelated review status.",

                "no_core":
                    "processing mode directly prohibited",

                "distractor_axis":
                    "review status",
            },

            {
                "family":
                    "blocked_release_medium",

                "mechanism":
                    "A release medium is prohibited independently of "
                    "downstream administrative status.",

                "no_core":
                    "release medium directly prohibited",

                "distractor_axis":
                    "administrative status",
            },
        ],

        "development": [

            {
                "family":
                    "restricted_delivery_path",

                "mechanism":
                    "A delivery path is prohibited despite apparently "
                    "favorable downstream facts.",
            },
        ],
    },


    # ============================================================
    # 3. PERMISSION CONDITION LEDGER
    # ============================================================

    "permission_condition_ledger": {

        "description":
            "Separate direct permission from satisfaction of every "
            "material requirement.",

        "train": [

            {
                "family":
                    "review_notice_ledger",

                "requirements": [
                    "technical review",
                    "advance notice",
                ],

                "yes":
                    "both satisfied",

                "conditional":
                    "permission exists but one requirement unresolved",
            },

            {
                "family":
                    "registration_certification_ledger",

                "requirements": [
                    "registration",
                    "certification",
                ],

                "yes":
                    "both satisfied",

                "conditional":
                    "one material state unresolved",
            },

            {
                "family":
                    "consent_security_ledger",

                "requirements": [
                    "consent",
                    "security validation",
                ],

                "yes":
                    "both satisfied",

                "conditional":
                    "one material requirement unresolved",
            },

            {
                "family":
                    "approval_logging_ledger",

                "requirements": [
                    "approval",
                    "audit logging",
                ],

                "yes":
                    "both satisfied",

                "conditional":
                    "one material requirement unresolved",
            },
        ],

        "development": [

            {
                "family":
                    "authorization_training_ledger",

                "requirements": [
                    "authorization",
                    "required training",
                ],
            },
        ],
    },


    # ============================================================
    # 4. EXCEPTION ACTIVATION CHAIN
    # ============================================================

    "exception_activation_chain": {

        "description":
            "Resolve base rule, trigger state, exception validity, "
            "and downstream requirements in order.",

        "train": [

            {
                "family":
                    "maintenance_window_exception",

                "base":
                    "action prohibited",

                "trigger":
                    "declared maintenance window",

                "requirement":
                    "operations clearance",
            },

            {
                "family":
                    "continuity_event_exception",

                "base":
                    "action prohibited",

                "trigger":
                    "continuity event activated",

                "requirement":
                    "supervisor authorization",
            },

            {
                "family":
                    "research_pause_exception",

                "base":
                    "action restricted",

                "trigger":
                    "formal pause condition",

                "requirement":
                    "documented waiver",
            },

            {
                "family":
                    "restoration_mode_exception",

                "base":
                    "action prohibited",

                "trigger":
                    "restoration mode active",

                "requirement":
                    "security clearance",
            },
        ],

        "development": [

            {
                "family":
                    "contingency_window_exception",

                "base":
                    "action prohibited",

                "trigger":
                    "contingency window active",

                "requirement":
                    "recorded authorization",
            },
        ],
    },


    # ============================================================
    # 5. AUTHORITY STATE RESOLUTION
    # ============================================================

    "authority_state_resolution": {

        "description":
            "Determine whether an actor possesses current operative "
            "authority rather than merely historical authority.",

        "train": [

            {
                "family":
                    "temporary_reviewer_authority",

                "states": [
                    "assigned",
                    "current",
                    "expired",
                    "revoked",
                ],
            },

            {
                "family":
                    "backup_custodian_authority",

                "states": [
                    "designated",
                    "activated",
                    "inactive",
                    "withdrawn",
                ],
            },

            {
                "family":
                    "regional_approver_authority",

                "states": [
                    "appointed",
                    "in scope",
                    "out of scope",
                    "superseded",
                ],
            },

            {
                "family":
                    "acting_controller_authority",

                "states": [
                    "recorded",
                    "effective",
                    "lapsed",
                    "replaced",
                ],
            },
        ],

        "development": [

            {
                "family":
                    "interim_coordinator_authority",

                "states": [
                    "appointed",
                    "current",
                    "expired",
                    "revoked",
                ],
            },
        ],
    },


    # ============================================================
    # 6. CATEGORY + RULE COMPOSITION
    # ============================================================

    "category_rule_composition": {

        "description":
            "First establish category membership, then separately apply "
            "the governing rule for that category.",

        "train": [

            {
                "family":
                    "managed_compute_category",

                "classification":
                    "managed compute class",

                "authorization":
                    "separate processing rule",
            },

            {
                "family":
                    "controlled_document_category",

                "classification":
                    "controlled document class",

                "authorization":
                    "separate handling rule",
            },

            {
                "family":
                    "regulated_sample_category",

                "classification":
                    "regulated sample class",

                "authorization":
                    "separate usage rule",
            },

            {
                "family":
                    "protected_workspace_category",

                "classification":
                    "protected workspace class",

                "authorization":
                    "separate access rule",
            },
        ],

        "development": [

            {
                "family":
                    "verified_device_category",

                "classification":
                    "verified device class",

                "authorization":
                    "separate connection rule",
            },
        ],
    },


    # ============================================================
    # 7. SCOPE / DESTINATION RESOLUTION
    # ============================================================

    "scope_destination_resolution": {

        "description":
            "Determine whether the exact actor, purpose, channel, "
            "location, destination, or use is within governed scope.",

        "train": [

            {
                "family":
                    "recipient_scope_resolution",

                "dimension":
                    "recipient class",
            },

            {
                "family":
                    "environment_scope_resolution",

                "dimension":
                    "deployment environment",
            },

            {
                "family":
                    "workflow_channel_resolution",

                "dimension":
                    "communication/workflow channel",
            },

            {
                "family":
                    "use_purpose_resolution",

                "dimension":
                    "authorized use purpose",
            },
        ],

        "development": [

            {
                "family":
                    "location_scope_resolution",

                "dimension":
                    "operating location",
            },
        ],
    },


    # ============================================================
    # 8. EVIDENCE RELEVANCE BOUNDARY
    # ============================================================

    "evidence_relevance_boundary": {

        "description":
            "Distinguish topically related evidence from evidence that "
            "actually governs the exact proposition.",

        "train": [

            {
                "family":
                    "adjacent_channel_nonrule",

                "mismatch":
                    "rule governs neighboring channel",
            },

            {
                "family":
                    "adjacent_actor_nonrule",

                "mismatch":
                    "rule governs neighboring actor class",
            },

            {
                "family":
                    "adjacent_technology_nonrule",

                "mismatch":
                    "rule governs neighboring technology",
            },

            {
                "family":
                    "precision_absent_nonrule",

                "mismatch":
                    "general requirement exists but requested precision "
                    "is absent",
            },
        ],

        "development": [

            {
                "family":
                    "adjacent_purpose_nonrule",

                "mismatch":
                    "rule governs neighboring purpose",
            },
        ],
    },
}


# Challenge V2 composition identifiers.
#
# These names are included only as a mechanical forbidden-name list.
# No Challenge V2 file is read by this module.

FORBIDDEN_EXTERNAL_FAMILY_NAMES = {

    "specific_rule_precedence",
    "superseding_rule_precedence",
    "exception_activation",
    "nested_exception_requirements",
    "delegated_authority_validity",
    "temporal_authorization",
    "definition_then_authorization",
    "purpose_scope_authorization",
    "jurisdiction_then_requirement",
    "prohibition_vs_unresolved_fact",
    "permission_with_material_condition",
    "evidence_relevance_sufficiency",
}


EXPECTED_OPERATORS = {
    "operative_rule_selection",
    "prohibition_short_circuit",
    "permission_condition_ledger",
    "exception_activation_chain",
    "authority_state_resolution",
    "category_rule_composition",
    "scope_destination_resolution",
    "evidence_relevance_boundary",
}
