"""
Sentinel V6 full-curriculum family specification.

This file expands the frozen eight-operator V6 ontology into:

    32 TRAIN families
    8 DEVELOPMENT families

It does NOT generate data.
It does NOT train a model.
It does NOT read any Challenge benchmark.

All families use the same abstract schema so family identity itself
cannot encode decision semantics.
"""


FAMILY_SPECS = {

    # ================================================================
    # 1. TARGET ENTITY BINDING
    # ================================================================

    "target_entity_binding": {

        "train": [

            {
                "family": "paired_subject_fact_isolation",
                "mechanism":
                    "Nearby records describe two subjects with different "
                    "governance states; only facts bound to the exact target count.",
                "dimensions":
                    ["entity identity", "fact ownership", "proposition binding"],
                "stress_axes":
                    ["neighbor entity", "same vocabulary", "different state"],
            },

            {
                "family": "role_instance_binding",
                "mechanism":
                    "Multiple actors share the same role title but possess "
                    "different request-specific states.",
                "dimensions":
                    ["actor instance", "shared role", "request identity"],
                "stress_axes":
                    ["same role name", "different actor", "cross-attribution"],
            },

            {
                "family": "record_reference_binding",
                "mechanism":
                    "Several records reference related objects; only the record "
                    "whose identifier binds to the exact target proposition applies.",
                "dimensions":
                    ["record identity", "object identity", "reference binding"],
                "stress_axes":
                    ["adjacent record", "shared object type", "reference collision"],
            },

            {
                "family": "multi_object_state_isolation",
                "mechanism":
                    "One evidence pack contains several objects with opposing "
                    "permission states and requires exact object-state binding.",
                "dimensions":
                    ["object identity", "state ownership", "action binding"],
                "stress_axes":
                    ["opposing states", "same action", "multiple objects"],
            },
        ],

        "development": [

            {
                "family": "alias_identity_chain_binding",
                "mechanism":
                    "The exact target is connected through a stated alias/reference "
                    "chain while nearby entities remain distinct.",
                "dimensions":
                    ["alias resolution", "identity chain", "fact binding"],
                "stress_axes":
                    ["alias", "near-name entity", "indirect reference"],
            },
        ],
    },


    # ================================================================
    # 2. APPLICABILITY / SCOPE
    # ================================================================

    "applicability_scope_resolution": {

        "train": [

            {
                "family": "recipient_membership_applicability",
                "mechanism":
                    "A rule applies only to a defined recipient class; evidence "
                    "must establish exact recipient membership before conditions matter.",
                "dimensions":
                    ["recipient class", "membership", "applicability"],
                "stress_axes":
                    ["nearby class", "missing membership", "related recipient"],
            },

            {
                "family": "channel_membership_applicability",
                "mechanism":
                    "A rule governs a specific communication or transfer channel "
                    "and does not automatically govern adjacent channels.",
                "dimensions":
                    ["channel", "scope", "applicability"],
                "stress_axes":
                    ["adjacent channel", "shared purpose", "semantic similarity"],
            },

            {
                "family": "purpose_membership_applicability",
                "mechanism":
                    "Permission depends on the exact governed purpose rather than "
                    "a merely related purpose.",
                "dimensions":
                    ["purpose", "scope membership", "governance"],
                "stress_axes":
                    ["near purpose", "same asset", "scope ambiguity"],
            },

            {
                "family": "asset_class_applicability",
                "mechanism":
                    "A rule governs only assets in an explicitly defined class and "
                    "requires establishing class membership first.",
                "dimensions":
                    ["asset class", "classification", "applicability"],
                "stress_axes":
                    ["neighbor class", "partial definition", "topical similarity"],
            },
        ],

        "development": [

            {
                "family": "jurisdiction_membership_applicability",
                "mechanism":
                    "A rule applies only inside a stated jurisdictional boundary; "
                    "location or domain membership must be established.",
                "dimensions":
                    ["jurisdiction", "boundary", "applicability"],
                "stress_axes":
                    ["near jurisdiction", "missing location", "shared organization"],
            },
        ],
    },


    # ================================================================
    # 3. OPERATIVE RULE PRECEDENCE
    # ================================================================

    "operative_rule_precedence": {

        "train": [

            {
                "family": "specific_policy_over_general_policy",
                "mechanism":
                    "A request-specific or narrowly scoped rule controls over a "
                    "broader general rule when both are applicable.",
                "dimensions":
                    ["specificity", "rule scope", "precedence"],
                "stress_axes":
                    ["conflicting rules", "general rule salience", "specificity"],
            },

            {
                "family": "current_revision_over_archived_rule",
                "mechanism":
                    "A currently effective revision controls over an older archived "
                    "version concerning the same action.",
                "dimensions":
                    ["revision status", "current state", "precedence"],
                "stress_axes":
                    ["older permission", "older prohibition", "revision history"],
            },

            {
                "family": "route_specific_rule_precedence",
                "mechanism":
                    "A rule specific to the requested route controls over a general "
                    "rule covering the broader transfer category.",
                "dimensions":
                    ["route", "specificity", "precedence"],
                "stress_axes":
                    ["general route rule", "route similarity", "conflict"],
            },

            {
                "family": "activated_directive_precedence",
                "mechanism":
                    "An explicitly activated directive temporarily controls over the "
                    "ordinary standing rule.",
                "dimensions":
                    ["activation", "directive state", "precedence"],
                "stress_axes":
                    ["inactive directive", "standing rule", "activation evidence"],
            },
        ],

        "development": [

            {
                "family": "delegated_addendum_precedence",
                "mechanism":
                    "A valid scoped addendum issued under delegated authority may "
                    "control over the baseline procedure.",
                "dimensions":
                    ["delegation", "addendum scope", "precedence"],
                "stress_axes":
                    ["baseline rule", "delegation validity", "scope mismatch"],
            },
        ],
    },


    # ================================================================
    # 4. EXCEPTION STATE PROPAGATION
    # ================================================================

    "exception_state_propagation": {

        "train": [

            {
                "family": "declared_event_exception_state",
                "mechanism":
                    "A base prohibition can be displaced only when a declared event "
                    "activates an explicit exception.",
                "dimensions":
                    ["base prohibition", "event trigger", "exception state"],
                "stress_axes":
                    ["undeclared event", "related event", "base-rule persistence"],
            },

            {
                "family": "maintenance_mode_exception_state",
                "mechanism":
                    "A maintenance-mode exception applies only while the required "
                    "maintenance state is actually active.",
                "dimensions":
                    ["maintenance state", "exception activation", "base rule"],
                "stress_axes":
                    ["scheduled but inactive", "expired state", "near trigger"],
            },

            {
                "family": "emergency_override_exception_state",
                "mechanism":
                    "An emergency override changes the base rule only when its exact "
                    "activation criteria are established.",
                "dimensions":
                    ["override", "trigger criteria", "exception propagation"],
                "stress_axes":
                    ["warning state", "partial trigger", "unrelated emergency"],
            },

            {
                "family": "waiver_exception_state",
                "mechanism":
                    "A documented waiver activates a defined exception only for the "
                    "request and scope covered by that waiver.",
                "dimensions":
                    ["waiver", "scope", "exception activation"],
                "stress_axes":
                    ["wrong request", "expired waiver", "partial scope"],
            },
        ],

        "development": [

            {
                "family": "contingency_exception_state",
                "mechanism":
                    "A contingency exception is active only after the specified "
                    "contingency state is established for the exact request.",
                "dimensions":
                    ["contingency", "activation", "exception state"],
                "stress_axes":
                    ["planned contingency", "unconfirmed activation", "scope"],
            },
        ],
    },


    # ================================================================
    # 5. REQUIREMENT LEDGER COMPOSITION
    # ================================================================

    "requirement_ledger_composition": {

        "train": [

            {
                "family": "approval_notice_requirement_ledger",
                "mechanism":
                    "Permission requires both approval and notice, each resolved "
                    "independently as SATISFIED, FAILED, or UNRESOLVED.",
                "dimensions":
                    ["approval", "notice", "ledger composition"],
                "stress_axes":
                    ["one unresolved", "one failed", "both satisfied"],
            },

            {
                "family": "consent_security_requirement_ledger",
                "mechanism":
                    "Consent and security validation are separate material requirements "
                    "whose states must both be preserved.",
                "dimensions":
                    ["consent", "security validation", "ledger composition"],
                "stress_axes":
                    ["mixed states", "implicit satisfaction", "missing status"],
            },

            {
                "family": "registration_review_requirement_ledger",
                "mechanism":
                    "Registration and review completion are independently material and "
                    "must not be collapsed into a single approval concept.",
                "dimensions":
                    ["registration", "review", "ledger composition"],
                "stress_axes":
                    ["partial completion", "failed review", "unknown registration"],
            },

            {
                "family": "training_logging_requirement_ledger",
                "mechanism":
                    "Required training and audit logging are distinct prerequisites and "
                    "must each appear in the final ledger.",
                "dimensions":
                    ["training", "logging", "ledger composition"],
                "stress_axes":
                    ["one missing", "failed prerequisite", "both satisfied"],
            },
        ],

        "development": [

            {
                "family": "certification_attestation_requirement_ledger",
                "mechanism":
                    "Certification and attestation are separate material states whose "
                    "composition determines the final decision.",
                "dimensions":
                    ["certification", "attestation", "ledger composition"],
                "stress_axes":
                    ["unresolved attestation", "failed certification", "complete ledger"],
            },
        ],
    },


    # ================================================================
    # 6. FINAL DECISION FINALIZATION
    # ================================================================

    "final_decision_finalization": {

        "train": [

            {
                "family": "satisfied_permission_finalization",
                "mechanism":
                    "Applicability plus permission plus a fully satisfied material "
                    "ledger must finalize to YES.",
                "dimensions":
                    ["applicability", "permission", "all satisfied"],
                "stress_axes":
                    ["irrelevant uncertainty", "distractor prohibition", "complete state"],
            },

            {
                "family": "unresolved_requirement_finalization",
                "mechanism":
                    "Applicable permission with no failure but at least one unresolved "
                    "material requirement must finalize to CONDITIONAL.",
                "dimensions":
                    ["permission", "unresolved state", "finalization"],
                "stress_axes":
                    ["satisfied sibling condition", "irrelevant uncertainty", "partial ledger"],
            },

            {
                "family": "failed_requirement_finalization",
                "mechanism":
                    "Any explicitly failed controlling requirement must finalize to NO "
                    "even when other requirements are satisfied.",
                "dimensions":
                    ["failed requirement", "short circuit", "finalization"],
                "stress_axes":
                    ["many satisfied states", "single failure", "salient permission"],
            },

            {
                "family": "nonapplicable_rule_finalization",
                "mechanism":
                    "When applicability itself is not established, downstream condition "
                    "language cannot convert the result into CONDITIONAL.",
                "dimensions":
                    ["non-applicability", "epistemic boundary", "finalization"],
                "stress_axes":
                    ["related condition", "near rule", "invented prerequisite"],
            },
        ],

        "development": [

            {
                "family": "mixed_governance_state_finalization",
                "mechanism":
                    "A completed multi-part governance state must map deterministically "
                    "to one final decision without weighting irrelevant favorable facts.",
                "dimensions":
                    ["mixed evidence", "controlling state", "finalization"],
                "stress_axes":
                    ["conflicting salience", "irrelevant satisfied facts", "exact state"],
            },
        ],
    },


    # ================================================================
    # 7. TEMPORAL / AUTHORITY STATE
    # ================================================================

    "temporal_authority_state": {

        "train": [

            {
                "family": "temporary_delegate_window_state",
                "mechanism":
                    "Temporary delegated authority is operative only inside its current "
                    "authorized window.",
                "dimensions":
                    ["delegate", "time window", "authority state"],
                "stress_axes":
                    ["expired window", "future window", "historical assignment"],
            },

            {
                "family": "expiring_authorization_state",
                "mechanism":
                    "An authorization may be valid, expired, future, or unresolved and "
                    "must be evaluated at the relevant decision time.",
                "dimensions":
                    ["authorization", "time", "current validity"],
                "stress_axes":
                    ["historical approval", "future start", "missing date"],
            },

            {
                "family": "backup_authority_activation_state",
                "mechanism":
                    "Backup authority becomes operative only after its activation state "
                    "is established.",
                "dimensions":
                    ["backup authority", "activation", "current state"],
                "stress_axes":
                    ["designated only", "inactive backup", "activated backup"],
            },

            {
                "family": "superseded_authority_state",
                "mechanism":
                    "Historical authority ceases to control once a valid superseding "
                    "authority state is established.",
                "dimensions":
                    ["supersession", "authority", "current state"],
                "stress_axes":
                    ["historical authority", "replacement", "ambiguous status"],
            },
        ],

        "development": [

            {
                "family": "suspended_reviewer_authority_state",
                "mechanism":
                    "Reviewer authority must account for a currently suspended state "
                    "rather than relying on historical appointment alone.",
                "dimensions":
                    ["reviewer authority", "suspension", "current validity"],
                "stress_axes":
                    ["historical appointment", "temporary suspension", "resumption"],
            },
        ],
    },


    # ================================================================
    # 8. CROSS-GATE STATE COMPOSITION
    # ================================================================

    "cross_gate_state_composition": {

        "train": [

            {
                "family": "binding_scope_ledger_chain",
                "mechanism":
                    "Exact target binding feeds applicability, which then gates a "
                    "multi-requirement ledger and finalization.",
                "dimensions":
                    ["target binding", "applicability", "requirements", "finalization"],
                "stress_axes":
                    ["neighbor target", "scope mismatch", "partial ledger"],
            },

            {
                "family": "precedence_exception_ledger_chain",
                "mechanism":
                    "The operative-rule choice determines the base state before an "
                    "exception and downstream requirements are resolved.",
                "dimensions":
                    ["precedence", "exception", "requirements", "finalization"],
                "stress_axes":
                    ["superseded rule", "inactive exception", "mixed requirements"],
            },

            {
                "family": "authority_scope_finalization_chain",
                "mechanism":
                    "Current authority and request scope must both be resolved before "
                    "the final permission state can be finalized.",
                "dimensions":
                    ["authority", "scope", "permission", "finalization"],
                "stress_axes":
                    ["historical authority", "near scope", "favorable distractor"],
            },

            {
                "family": "entity_precedence_exception_chain",
                "mechanism":
                    "Facts must first bind to the exact entity, then the controlling "
                    "rule and any valid exception must be propagated.",
                "dimensions":
                    ["entity binding", "precedence", "exception", "finalization"],
                "stress_axes":
                    ["cross-entity rule", "older rule", "exception ambiguity"],
            },
        ],

        "development": [

            {
                "family": "temporal_exception_ledger_chain",
                "mechanism":
                    "Temporal authority, exception activation, and requirement state "
                    "must compose without losing causal ordering.",
                "dimensions":
                    ["temporal state", "exception", "requirements", "finalization"],
                "stress_axes":
                    ["expired authority", "unresolved trigger", "mixed ledger"],
            },
        ],
    },
}


EXPECTED_OPERATORS = {
    "target_entity_binding",
    "applicability_scope_resolution",
    "operative_rule_precedence",
    "exception_state_propagation",
    "requirement_ledger_composition",
    "final_decision_finalization",
    "temporal_authority_state",
    "cross_gate_state_composition",
}


REQUIRED_FAMILY_FIELDS = {
    "family",
    "mechanism",
    "dimensions",
    "stress_axes",
}


def validate_family_specs():
    errors = []

    if set(FAMILY_SPECS) != EXPECTED_OPERATORS:
        errors.append(
            "operator set mismatch"
        )

    all_names = []

    train_count = 0
    development_count = 0

    for operator in sorted(
        EXPECTED_OPERATORS
    ):
        spec = FAMILY_SPECS.get(
            operator,
            {},
        )

        train = spec.get(
            "train",
            [],
        )

        development = spec.get(
            "development",
            [],
        )

        if len(train) != 4:
            errors.append(
                f"{operator}: "
                f"train count={len(train)}"
            )

        if len(development) != 1:
            errors.append(
                f"{operator}: "
                f"development count="
                f"{len(development)}"
            )

        train_count += len(train)
        development_count += len(
            development
        )

        for split_name, families in (
            ("train", train),
            ("development", development),
        ):
            for family in families:

                if set(family) != REQUIRED_FAMILY_FIELDS:
                    errors.append(
                        f"{operator}/{split_name}/"
                        f"{family.get('family')}: "
                        "field schema mismatch"
                    )

                name = family.get(
                    "family"
                )

                if not isinstance(
                    name,
                    str,
                ) or not name:
                    errors.append(
                        f"{operator}/{split_name}: "
                        "invalid family name"
                    )

                all_names.append(
                    name
                )

                if not isinstance(
                    family.get("dimensions"),
                    list,
                ) or not family[
                    "dimensions"
                ]:
                    errors.append(
                        f"{name}: invalid dimensions"
                    )

                if not isinstance(
                    family.get("stress_axes"),
                    list,
                ) or not family[
                    "stress_axes"
                ]:
                    errors.append(
                        f"{name}: invalid stress_axes"
                    )

    if train_count != 32:
        errors.append(
            f"train family total={train_count}"
        )

    if development_count != 8:
        errors.append(
            "development family total="
            f"{development_count}"
        )

    if len(all_names) != len(
        set(all_names)
    ):
        errors.append(
            "duplicate family names"
        )

    train_names = {
        family["family"]
        for op in FAMILY_SPECS.values()
        for family in op["train"]
    }

    development_names = {
        family["family"]
        for op in FAMILY_SPECS.values()
        for family in op[
            "development"
        ]
    }

    if train_names & development_names:
        errors.append(
            "train/development family overlap"
        )

    return {
        "errors": errors,
        "train_count": train_count,
        "development_count":
            development_count,
        "total_count":
            train_count
            + development_count,
        "train_names":
            sorted(train_names),
        "development_names":
            sorted(development_names),
    }


if __name__ == "__main__":
    result = validate_family_specs()

    if result["errors"]:
        for error in result["errors"]:
            print(
                "ERROR:",
                error,
            )

        raise SystemExit(1)

    print(
        "V6 FAMILY SPEC VALIDATION: PASS"
    )
    print(
        "train families:",
        result["train_count"],
    )
    print(
        "development families:",
        result[
            "development_count"
        ],
    )
    print(
        "all families:",
        result["total_count"],
    )
