"""
Sentinel V5 executable family realizers.

PURPOSE
-------

The frozen ontology in v5_family_specs.py defines WHAT reasoning
families exist.

This module defines the additional semantic surface configuration
needed to instantiate those families without changing the ontology.

IMPORTANT
---------

- Does NOT read Challenge V1 or Challenge V2.
- Does NOT generate curriculum rows.
- Does NOT train or evaluate a model.
- Every ontology TRAIN and DEVELOPMENT family must appear exactly once.
"""

from training.v5.v5_family_specs import OPERATORS


# ---------------------------------------------------------------------
# Build canonical ontology index.
# ---------------------------------------------------------------------

FAMILY_INDEX = {}

for operator, operator_spec in OPERATORS.items():

    for split in (
        "train",
        "development",
    ):

        for spec in operator_spec[split]:

            family = spec["family"]

            if family in FAMILY_INDEX:
                raise ValueError(
                    f"Duplicate ontology family: {family}"
                )

            FAMILY_INDEX[family] = {
                "operator": operator,
                "split": split,
                **spec,
            }


# ---------------------------------------------------------------------
# Supplemental executable semantics.
#
# These are intentionally separate from the frozen conceptual ontology.
# ---------------------------------------------------------------------

SUPPLEMENTAL = {

    # =================================================================
    # 1. OPERATIVE RULE SELECTION
    # =================================================================

    "project_charter_priority": {
        "asset_kind": "packet",
        "action":
            "stage {asset} in the project exchange zone",
        "high_rule":
            "active project charter",
        "low_rule":
            "organization-wide staging rule",
        "requirement":
            "technical review",
    },

    "facility_protocol_priority": {
        "asset_kind": "specimen set",
        "action":
            "transfer {asset} into the facility handoff bay",
        "high_rule":
            "facility-specific transfer protocol",
        "low_rule":
            "institution-wide transfer procedure",
        "requirement":
            "facility clearance",
    },

    "incident_directive_priority": {
        "asset_kind": "record bundle",
        "action":
            "route {asset} through the incident processing lane",
        "high_rule":
            "activated incident directive",
        "low_rule":
            "normal operating rule",
        "requirement":
            "incident review",
    },

    "revision_status_priority": {
        "asset_kind": "release package",
        "action":
            "release {asset} through the revised distribution workflow",
        "high_rule":
            "currently effective revision",
        "low_rule":
            "legacy operating instruction",
        "requirement":
            "revision acknowledgement",
    },

    "program_addendum_priority": {
        "asset_kind": "program bundle",
        "action":
            "place {asset} into the program collaboration vault",
        "high_rule":
            "program-specific addendum",
        "low_rule":
            "standing organizational rule",
        "requirement":
            "program review",
    },


    # =================================================================
    # 2. PROHIBITION SHORT-CIRCUIT
    # =================================================================

    "blocked_interface_route": {
        "asset_kind": "record set",
        "action":
            "transmit {asset} through Interface R",
        "rule":
            "controlling interface rule",
        "requirement":
            "transport encryption",
        "distractor":
            "annual review",
        "neighbor_action":
            "transmit {asset} through Interface S",
    },

    "restricted_recipient_class": {
        "asset_kind": "research package",
        "action":
            "release {asset} to the restricted external recipient class",
        "rule":
            "controlling recipient-class rule",
        "requirement":
            "confidentiality validation",
        "distractor":
            "archive registration",
        "neighbor_action":
            "release {asset} to the approved internal recipient class",
    },

    "disabled_processing_mode": {
        "asset_kind": "analysis bundle",
        "action":
            "process {asset} using Mode Delta",
        "rule":
            "controlling processing-mode rule",
        "requirement":
            "isolation control",
        "distractor":
            "annual review",
        "neighbor_action":
            "process {asset} using Mode Gamma",
    },

    "blocked_release_medium": {
        "asset_kind": "release archive",
        "action":
            "publish {asset} through the public object store",
        "rule":
            "controlling release-medium rule",
        "requirement":
            "integrity validation",
        "distractor":
            "administrative signoff",
        "neighbor_action":
            "publish {asset} through the managed repository",
    },

    "restricted_delivery_path": {
        "asset_kind": "delivery package",
        "action":
            "deliver {asset} through Relay Path K",
        "rule":
            "controlling delivery-path rule",
        "requirement":
            "session protection",
        "distractor":
            "compliance review",
        "neighbor_action":
            "deliver {asset} through Relay Path J",
    },


    # =================================================================
    # 3. PERMISSION CONDITION LEDGER
    # =================================================================

    "review_notice_ledger": {
        "asset_kind": "submission archive",
        "action":
            "submit {asset} to the controlled analysis enclave",
        "rule":
            "controlled-submission rule",
    },

    "registration_certification_ledger": {
        "asset_kind": "deployment package",
        "action":
            "deploy {asset} in the partner laboratory",
        "rule":
            "partner-deployment rule",
    },

    "consent_security_ledger": {
        "asset_kind": "study dataset",
        "action":
            "use {asset} in the collaborative study workflow",
        "rule":
            "collaborative-use rule",
    },

    "approval_logging_ledger": {
        "asset_kind": "integration package",
        "action":
            "connect {asset} to the production workflow",
        "rule":
            "production-integration rule",
    },

    "authorization_training_ledger": {
        "asset_kind": "service package",
        "action":
            "operate {asset} in the shared service queue",
        "rule":
            "shared-service operation rule",
    },


    # =================================================================
    # 4. EXCEPTION ACTIVATION CHAIN
    # =================================================================

    "maintenance_window_exception": {
        "asset_kind": "maintenance package",
        "action":
            "move {asset} through the service network",
        "base_rule":
            "standing service restriction",
        "exception_rule":
            "maintenance-window exception",
    },

    "continuity_event_exception": {
        "asset_kind": "continuity archive",
        "action":
            "access {asset} through the continuity replica",
        "base_rule":
            "standing replica restriction",
        "exception_rule":
            "continuity-event exception",
    },

    "research_pause_exception": {
        "asset_kind": "paused-study package",
        "action":
            "transfer {asset} into the external review workspace",
        "base_rule":
            "standing paused-study restriction",
        "exception_rule":
            "research-pause exception",
    },

    "restoration_mode_exception": {
        "asset_kind": "recovery bundle",
        "action":
            "connect {asset} to the restoration segment",
        "base_rule":
            "standing restoration restriction",
        "exception_rule":
            "restoration-mode exception",
    },

    "contingency_window_exception": {
        "asset_kind": "contingency package",
        "action":
            "transfer {asset} through the contingency relay",
        "base_rule":
            "standing contingency restriction",
        "exception_rule":
            "contingency-window exception",
    },


    # =================================================================
    # 5. AUTHORITY STATE RESOLUTION
    # =================================================================

    "temporary_reviewer_authority": {
        "asset_kind": "review packet",
        "role":
            "temporary reviewer",
        "action":
            "approve release of {asset}",
    },

    "backup_custodian_authority": {
        "asset_kind": "custody archive",
        "role":
            "backup custodian",
        "action":
            "authorize transfer of {asset}",
    },

    "regional_approver_authority": {
        "asset_kind": "regional package",
        "role":
            "regional approver",
        "action":
            "approve deployment of {asset}",
    },

    "acting_controller_authority": {
        "asset_kind": "publication bundle",
        "role":
            "acting controller",
        "action":
            "authorize publication of {asset}",
    },

    "interim_coordinator_authority": {
        "asset_kind": "coordination package",
        "role":
            "interim coordinator",
        "action":
            "approve external sharing of {asset}",
    },


    # =================================================================
    # 6. CATEGORY + RULE COMPOSITION
    # =================================================================

    "managed_compute_category": {
        "asset_kind": "compute node",
        "action":
            "attach {asset} to the protected compute fabric",
        "rule":
            "managed-compute connection rule",
        "requirement":
            "endpoint attestation",
    },

    "controlled_document_category": {
        "asset_kind": "document package",
        "action":
            "share {asset} through the secure review portal",
        "rule":
            "controlled-document handling rule",
        "requirement":
            "recipient validation",
    },

    "regulated_sample_category": {
        "asset_kind": "sample package",
        "action":
            "use {asset} in the external assay workflow",
        "rule":
            "regulated-sample usage rule",
        "requirement":
            "protocol approval",
    },

    "protected_workspace_category": {
        "asset_kind": "workspace",
        "action":
            "grant remote access to {asset}",
        "rule":
            "protected-workspace access rule",
        "requirement":
            "multifactor validation",
    },

    "verified_device_category": {
        "asset_kind": "device",
        "action":
            "connect {asset} to the research backbone",
        "rule":
            "verified-device connection rule",
        "requirement":
            "network registration",
    },


    # =================================================================
    # 7. SCOPE / DESTINATION RESOLUTION
    # =================================================================

    "recipient_scope_resolution": {
        "asset_kind": "sharing package",
        "action":
            "send {asset} to the external research partner",
        "rule":
            "recipient-scope rule",
        "in_scope":
            "external research partner",
        "neighbor_scope":
            "internal service vendor",
        "requirement":
            "recipient verification",
    },

    "environment_scope_resolution": {
        "asset_kind": "deployment bundle",
        "action":
            "deploy {asset} in the on-premises analysis enclave",
        "rule":
            "deployment-environment rule",
        "in_scope":
            "on-premises analysis enclave",
        "neighbor_scope":
            "cloud test environment",
        "requirement":
            "environment validation",
    },

    "workflow_channel_resolution": {
        "asset_kind": "workflow package",
        "action":
            "transmit {asset} through the secure review queue",
        "rule":
            "workflow-channel rule",
        "in_scope":
            "secure review queue",
        "neighbor_scope":
            "general messaging channel",
        "requirement":
            "queue authorization",
    },

    "use_purpose_resolution": {
        "asset_kind": "analysis dataset",
        "action":
            "use {asset} for reproducibility analysis",
        "rule":
            "authorized-purpose rule",
        "in_scope":
            "reproducibility analysis",
        "neighbor_scope":
            "instructional demonstration",
        "requirement":
            "purpose registration",
    },

    "location_scope_resolution": {
        "asset_kind": "operations package",
        "action":
            "operate {asset} in the West Annex secure laboratory",
        "rule":
            "operating-location rule",
        "in_scope":
            "West Annex secure laboratory",
        "neighbor_scope":
            "East Annex open laboratory",
        "requirement":
            "location clearance",
    },


    # =================================================================
    # 8. EVIDENCE RELEVANCE BOUNDARY
    # =================================================================

    "adjacent_channel_nonrule": {
        "asset_kind": "communications package",
        "action":
            "send {asset} through Channel R",
        "rule":
            "Channel R communications rule",
        "requirement":
            "channel authorization",
        "neighbor_action":
            "send {asset} through Channel S",
    },

    "adjacent_actor_nonrule": {
        "asset_kind": "access package",
        "action":
            "allow the project analyst to access {asset}",
        "rule":
            "project-analyst access rule",
        "requirement":
            "access approval",
        "neighbor_action":
            "allow the project administrator to access {asset}",
    },

    "adjacent_technology_nonrule": {
        "asset_kind": "technology package",
        "action":
            "process {asset} with Engine R",
        "rule":
            "Engine R processing rule",
        "requirement":
            "engine validation",
        "neighbor_action":
            "process {asset} with Engine S",
    },

    "precision_absent_nonrule": {
        "asset_kind": "retention package",
        "action":
            "retain {asset} for exactly 45 days",
        "rule":
            "retention-duration rule",
        "requirement":
            "retention approval",
        "neighbor_action":
            "retain {asset} under the general retention program",
    },

    "adjacent_purpose_nonrule": {
        "asset_kind": "purpose package",
        "action":
            "use {asset} for external validation",
        "rule":
            "external-validation use rule",
        "requirement":
            "purpose authorization",
        "neighbor_action":
            "use {asset} for internal validation",
    },
}


def family_config(
    family,
):
    """
    Return merged ontology + executable realization configuration.
    """

    if family not in FAMILY_INDEX:
        raise KeyError(
            f"Unknown ontology family: {family}"
        )

    if family not in SUPPLEMENTAL:
        raise KeyError(
            f"No executable realizer for family: {family}"
        )

    return {
        **FAMILY_INDEX[family],
        **SUPPLEMENTAL[family],
    }


def validate_realizers():
    """
    Fail closed if executable semantics do not correspond exactly
    to the frozen ontology.
    """

    expected = set(
        FAMILY_INDEX
    )

    actual = set(
        SUPPLEMENTAL
    )

    missing = sorted(
        expected - actual
    )

    extra = sorted(
        actual - expected
    )

    if missing:
        raise AssertionError(
            "Missing V5 family realizers: "
            + ", ".join(
                missing
            )
        )

    if extra:
        raise AssertionError(
            "Unknown V5 family realizers: "
            + ", ".join(
                extra
            )
        )

    train = [
        family
        for family, spec
        in FAMILY_INDEX.items()
        if spec["split"] == "train"
    ]

    development = [
        family
        for family, spec
        in FAMILY_INDEX.items()
        if spec["split"] == "development"
    ]

    assert len(
        FAMILY_INDEX
    ) == 40

    assert len(
        train
    ) == 32

    assert len(
        development
    ) == 8

    return {
        "families":
            len(
                FAMILY_INDEX
            ),

        "train":
            len(
                train
            ),

        "development":
            len(
                development
            ),
    }


if __name__ == "__main__":

    result = validate_realizers()

    print(
        "V5 FAMILY REALIZERS: PASS"
    )

    print(
        result
    )
