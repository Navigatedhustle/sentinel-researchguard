"""
Sentinel V4.1 surface-decontaminated curriculum.

The underlying logical compositions remain those in v4_family_specs.py.

V4.1 repairs:
1. validation/test use distinct ontologies from train
2. the old exotic `unsupported` tokens are removed from INSUFFICIENT
3. INSUFFICIENT uses varied unsupported relationships between known concepts
4. obvious article errors such as "A Orion" -> "An Orion" are repaired

Challenge V1 is not imported or referenced here.
"""

from copy import deepcopy
import re

from training.v4_family_specs import (
    FAMILIES as BASE_FAMILIES,
    VARIANTS as BASE_VARIANTS,
)


BASE_BY_CODE = {
    v["code"]: v
    for v in BASE_VARIANTS
}


# ---------------------------------------------------------------------
# SPLIT-SPECIFIC SURFACE ONTOLOGIES
#
# Train keeps its original ontology.
# Validation and test use deliberately different action/domain vocabulary.
# ---------------------------------------------------------------------

VALIDATION_VARIANTS = {
    "orion": {
        "code": "apogee",
        "action": "review {asset} within the Apogee enclave",
        "c1": "enclave authorization",
        "c2": "custodian acknowledgment",
    },
    "atlas": {
        "code": "harbor",
        "action": "stage {asset} for the Harbor exchange",
        "c1": "exchange approval",
        "c2": "counterparty verification",
    },
    "nova": {
        "code": "prism",
        "action": "process {asset} through the Prism pipeline",
        "c1": "processing clearance",
        "c2": "control attestation",
    },
    "meridian": {
        "code": "summit",
        "action": "place {asset} in the Summit repository",
        "c1": "repository approval",
        "c2": "repository enrollment",
    },
    "helix": {
        "code": "ember",
        "action": "preserve {asset} in the Ember vault",
        "c1": "vault authorization",
        "c2": "retention verification",
    },
}


TEST_VARIANTS = {
    "orion": {
        "code": "juniper",
        "action": "inspect {asset} inside the Juniper domain",
        "c1": "domain approval",
        "c2": "steward verification",
    },
    "atlas": {
        "code": "cobalt",
        "action": "provide {asset} through the Cobalt exchange",
        "c1": "sharing approval",
        "c2": "receiver verification",
    },
    "nova": {
        "code": "aurora",
        "action": "operate on {asset} in the Aurora workflow",
        "c1": "workflow clearance",
        "c2": "safety verification",
    },
    "meridian": {
        "code": "delta",
        "action": "transfer {asset} to the Delta enclave",
        "c1": "transfer clearance",
        "c2": "destination verification",
    },
    "helix": {
        "code": "cedar",
        "action": "store {asset} in the Cedar repository",
        "c1": "storage approval",
        "c2": "retention confirmation",
    },
}


SPLIT_VARIANTS = {
    "validation": VALIDATION_VARIANTS,
    "test": TEST_VARIANTS,
}


# ---------------------------------------------------------------------
# INSUFFICIENT QUESTION MECHANISMS
#
# These no longer introduce a bizarre novel object such as AES-512,
# a silver token, etc.
#
# Instead, they ask whether the evidence establishes an additional
# relationship among already-known governed facts.
#
# The relationship itself is not supplied by the evidence.
# ---------------------------------------------------------------------

INSUFFICIENT_PATTERNS = {
    "condition_ledger": [
        "Must {c1} and {c2} be documented in the same record before the rule applies to {action}?",
        "Does the policy require one reviewer to establish both {c1} and {c2} for {action}?",
        "Must {c1} be recorded earlier than {c2} for the rule governing {action}?",
        "Does the rule require {c1} and {c2} to come from the same approval process for {action}?",
        "Does the policy distinguish whether {c1} and {c2} were recorded by separate systems for {action}?",
    ],

    "specific_rule_precedence": [
        "Must {c1} and {c2} appear in the same record for the specific rule governing {action}?",
        "Does the specific rule require the same reviewer to establish {c1} and {c2} for {action}?",
        "Does the specific rule require {c1} to be recorded earlier than {c2} for {action}?",
        "Must {c1} and {c2} come from one approval process under the specific rule for {action}?",
        "Does the specific rule treat separate recording systems differently when establishing {c1} and {c2} for {action}?",
    ],

    "prohibition_exception": [
        "Must {c1} and {c2} be documented together for the exception governing {action}?",
        "Does the exception require one reviewer to establish both {c1} and {c2} for {action}?",
        "Does the exception require {c1} to be recorded earlier than {c2} for {action}?",
        "Must {c1} and {c2} originate from the same approval process for the exception to govern {action}?",
        "Does the exception distinguish whether {c1} and {c2} were recorded in separate systems for {action}?",
    ],

    "definition_chain": [
        "Must {c1} and {c2} be stored in the same record for the definition governing {action}?",
        "Does the definition require one reviewer to establish both {c1} and {c2} for {action}?",
        "Does the definition require {c1} to be recorded earlier than {c2} for {action}?",
        "Must {c1} and {c2} originate from one approval process for the definition governing {action}?",
        "Does the definition change depending on whether {c1} and {c2} were recorded in separate systems for {action}?",
    ],

    "delegated_authority_status": [
        "Must the authorization and {c2} be documented in the same record for {action}?",
        "Does delegated authority require the same reviewer to establish {c1} and {c2} for {action}?",
        "Must {c1} be recorded earlier than {c2} for delegated authority over {action}?",
        "Must {c1} and {c2} originate from one approval process for delegated authority over {action}?",
        "Does delegated authority change if {c1} and {c2} were recorded in separate systems for {action}?",
    ],

    "purpose_scope_definition": [
        "Must {c1} and {c2} be documented in the same record for the purpose rule governing {action}?",
        "Does the purpose rule require one reviewer to establish both {c1} and {c2} for {action}?",
        "Does the purpose rule require {c1} to be recorded earlier than {c2} for {action}?",
        "Must {c1} and {c2} originate from one approval process for the purpose rule governing {action}?",
        "Does the purpose rule distinguish whether {c1} and {c2} were recorded in separate systems for {action}?",
    ],

    "supersession_current_rule": [
        "Must {c1} and {c2} appear in the same record under the current rule governing {action}?",
        "Does the current rule require one reviewer to establish both {c1} and {c2} for {action}?",
        "Must {c1} be recorded earlier than {c2} under the current rule governing {action}?",
        "Must {c1} and {c2} originate from one approval process under the current rule for {action}?",
        "Does the current rule distinguish whether {c1} and {c2} were recorded in separate systems for {action}?",
    ],

    "temporal_override": [
        "Must {c1} and {c2} be documented in the same record after the timing rule governing {action} is satisfied?",
        "Does the timing rule require one reviewer to establish both {c1} and {c2} for {action}?",
        "Does the timing rule require {c1} to be recorded earlier than {c2} for {action}?",
        "Must {c1} and {c2} originate from one approval process after the timing rule governing {action} is satisfied?",
        "Does the timing rule distinguish whether {c1} and {c2} were recorded in separate systems for {action}?",
    ],

    "jurisdiction_definition_authorization": [
        "Must {c1} and {c2} be documented in the same record for the jurisdiction rule governing {action}?",
        "Does the jurisdiction rule require one reviewer to establish both {c1} and {c2} for {action}?",
        "Does jurisdiction require {c1} to be recorded earlier than {c2} for {action}?",
        "Must {c1} and {c2} originate from one approval process for the jurisdiction rule governing {action}?",
        "Does jurisdiction change depending on whether {c1} and {c2} were recorded in separate systems for {action}?",
    ],

    "permission_obligation_scope": [
        "Must {c1} and {c2} be documented in the same record for the obligation rule governing {action}?",
        "Does the obligation rule require one reviewer to establish both {c1} and {c2} for {action}?",
        "Does the obligation rule require {c1} to be recorded earlier than {c2} for {action}?",
        "Must {c1} and {c2} originate from one approval process for the obligation rule governing {action}?",
        "Does the obligation rule distinguish whether {c1} and {c2} were recorded in separate systems for {action}?",
    ],
}


def replace_strings(value, replacements):
    """
    Recursively replace known ontology strings without mutating the base specs.
    """
    if isinstance(value, str):
        result = value

        # Longest first avoids partial replacement problems.
        for old, new in sorted(
            replacements.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        ):
            result = result.replace(old, new)

        # Synthetic grammar cleanup.
        result = re.sub(
            r"\bA ([AEIOUaeiou])",
            r"An \1",
            result,
        )

        return result

    if isinstance(value, list):
        return [
            replace_strings(x, replacements)
            for x in value
        ]

    if isinstance(value, dict):
        return {
            k: replace_strings(v, replacements)
            for k, v in value.items()
        }

    return value


def remap_family_ontology(family, split, original_code):
    """
    Give validation and test independent surface ontologies.
    Train intentionally keeps the original ontology.
    """
    if split not in SPLIT_VARIANTS:
        return family

    old = BASE_BY_CODE[original_code]
    new = SPLIT_VARIANTS[split][original_code]

    replacements = {
        old["action"]: new["action"],
        old["c1"]: new["c1"],
        old["c2"]: new["c2"],

        original_code.capitalize(): new["code"].capitalize(),
        original_code: new["code"],
    }

    family = replace_strings(
        family,
        replacements,
    )

    prefix = family["name"].rsplit("_", 1)[0]
    family["name"] = (
        f"{prefix}_{new['code']}"
    )

    return family


def get_surface_fields(split, original_code):
    """
    Return the active ontology after split remapping.
    """
    if split in SPLIT_VARIANTS:
        return SPLIT_VARIANTS[split][original_code]

    old = BASE_BY_CODE[original_code]

    return {
        "code": old["code"],
        "action": old["action"],
        "c1": old["c1"],
        "c2": old["c2"],
    }


def repair_insufficient_question(
    family,
    split,
    original_code,
):
    """
    Remove the old exotic unsupported-object shortcut.

    The new question asks about an additional relationship among concepts
    already present in the evidence. The policy does not govern that
    relationship, so the evidence boundary remains the target skill.
    """
    fields = get_surface_fields(
        split,
        original_code,
    )

    composition = family["composition"]

    patterns = INSUFFICIENT_PATTERNS[
        composition
    ]

    # Deterministic per-variant choice.
    variant_index = list(
        BASE_BY_CODE
    ).index(
        original_code
    )

    question = patterns[
        variant_index
    ].format(
        action=fields["action"],
        c1=fields["c1"],
        c2=fields["c2"],
        code=fields["code"],
    )

    case = family["cases"]["INSUFFICIENT"]

    case["question"] = question
    case["question_type"] = "requirement"
    case["rationale"] = (
        "The supplied evidence governs the operative requirements "
        "but does not establish the additional relationship asked about."
    )
    case["citations"] = []
    case["conditions"] = []
    case["reasoning_tags"] = [
        "evidence_boundary",
        "unsupported_relationship",
    ]


def repair_family(
    family,
    split,
):
    family = deepcopy(family)

    original_code = (
        family["name"].rsplit("_", 1)[1]
    )

    family = remap_family_ontology(
        family,
        split,
        original_code,
    )

    # Apply grammar cleanup even for train.
    family = replace_strings(
        family,
        {},
    )

    repair_insufficient_question(
        family,
        split,
        original_code,
    )

    return family


FAMILIES = {
    split: [
        repair_family(
            family,
            split,
        )
        for family in families
    ]
    for split, families in BASE_FAMILIES.items()
}
