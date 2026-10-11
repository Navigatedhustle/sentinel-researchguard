# Sentinel V4 compositional curriculum specifications.
#
# IMPORTANT:
# These are abstract reasoning skills derived from V3's weakness profile.
# They do NOT reproduce Challenge V1 cases, policy packs, entities, or wording.

VARIANTS = [
    {
        "code": "orion",
        "action": "analyze {asset} inside the Orion workspace",
        "c1": "workspace clearance",
        "c2": "data-steward confirmation",
        "unsupported": "a silver hardware token",
    },
    {
        "code": "atlas",
        "action": "release {asset} into the Atlas collaboration",
        "c1": "release clearance",
        "c2": "recipient assurance",
        "unsupported": "an exactly seven-day review",
    },
    {
        "code": "nova",
        "action": "use {asset} in the Nova research workflow",
        "c1": "workflow authorization",
        "c2": "security attestation",
        "unsupported": "a vice-president signature",
    },
    {
        "code": "meridian",
        "action": "move {asset} into the Meridian environment",
        "c1": "transfer authorization",
        "c2": "environment registration",
        "unsupported": "AES-512 encryption",
    },
    {
        "code": "helix",
        "action": "retain {asset} in the Helix archive",
        "c1": "archive approval",
        "c2": "preservation-status confirmation",
        "unsupported": "three overwrite passes",
    },
]


def E(eid, source, section, text):
    return {
        "id": eid,
        "source": source,
        "section": section,
        "text": text,
    }


def C(
    decision,
    question,
    question_type,
    citations,
    rationale,
    tags,
    conditions=None,
):
    return {
        "decision": decision,
        "question": question,
        "question_type": question_type,
        "citations": citations,
        "rationale": rationale,
        "reasoning_tags": tags,
        "conditions": conditions or [],
    }


def F(
    name,
    composition,
    evidence,
    cases,
):
    return {
        "name": name,
        "composition": composition,
        "evidence": evidence,
        "cases": {
            x["decision"]: x
            for x in cases
        },
    }


# -------------------------------------------------------------------
# 1. CONDITION LEDGER
# -------------------------------------------------------------------

def condition_ledger(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    unsupported = v["unsupported"]

    return F(
        f"ledger_{code}",
        "condition_ledger",
        [
            E(
                "R1",
                f"{code}_operations.txt",
                "AUTHORIZED ROUTE",
                f"Someone may {action} only when "
                f"{c1} is active and {c2} is confirmed.",
            ),
            E(
                "R2",
                f"{code}_operations.txt",
                "FAILED REQUIREMENT",
                "If a required control is expressly denied "
                "or inactive, the action is not permitted.",
            ),
        ],
        [
            C(
                "YES",
                f"{c1.capitalize()} is active and {c2} is confirmed. "
                f"May someone {action}?",
                "permission",
                ["R1"],
                "Both material requirements are established as satisfied.",
                ["condition_ledger", "multiple_conditions"],
            ),
            C(
                "CONDITIONAL",
                f"{c1.capitalize()} is active. "
                f"May someone {action}?",
                "permission",
                ["R1"],
                "One material requirement remains unresolved.",
                ["condition_ledger", "unresolved_condition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                f"{c1.capitalize()} is active, but {c2} was expressly denied. "
                f"May someone {action}?",
                "permission",
                ["R1", "R2"],
                "A required condition is explicitly failed.",
                ["condition_ledger", "failed_condition"],
            ),
            C(
                "INSUFFICIENT",
                f"Does the policy specifically require {unsupported} "
                f"before someone may {action}?",
                "requirement",
                [],
                "The evidence does not establish the requested specific requirement.",
                ["evidence_boundary", "unsupported_specific"],
            ),
        ],
    )


# -------------------------------------------------------------------
# 2. SPECIFIC RULE + PRECEDENCE + CONDITIONS
# -------------------------------------------------------------------

def specificity(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    unsupported = v["unsupported"]

    return F(
        f"specificity_{code}",
        "specific_rule_precedence",
        [
            E(
                "G",
                f"{code}_general.txt",
                "GENERAL RULE",
                f"Approved institutional projects may {action}.",
            ),
            E(
                "S",
                f"{code}_restricted.txt",
                "RESTRICTED MATERIAL",
                f"When restricted {{asset}} are involved, someone may "
                f"{action} only when {c1} is active and {c2} is confirmed.",
            ),
            E(
                "P",
                f"{code}_precedence.txt",
                "SPECIFICITY",
                "A rule written specifically for restricted project "
                "material controls over a conflicting general rule.",
            ),
        ],
        [
            C(
                "YES",
                f"The material is restricted {{asset}}; {c1} is active "
                f"and {c2} is confirmed. May someone {action}?",
                "permission",
                ["S", "P"],
                "The specific rule controls and all conditions are satisfied.",
                ["specificity", "policy_precedence", "multiple_conditions"],
            ),
            C(
                "CONDITIONAL",
                f"The material is restricted {{asset}} and {c1} is active. "
                f"May someone {action}?",
                "permission",
                ["S", "P"],
                "The specific rule controls, but one requirement is unresolved.",
                ["specificity", "policy_precedence", "unresolved_condition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                f"The material is restricted {{asset}}; {c1} is active, "
                f"but {c2} is expressly not confirmed. "
                f"May someone {action}?",
                "permission",
                ["S", "P"],
                "The controlling specific rule has an explicitly failed requirement.",
                ["specificity", "policy_precedence", "failed_condition"],
            ),
            C(
                "INSUFFICIENT",
                f"Does the restricted-material rule specifically require "
                f"{unsupported} before someone may {action}?",
                "requirement",
                [],
                "The requested specific requirement is not established.",
                ["specificity", "evidence_boundary"],
            ),
        ],
    )


# -------------------------------------------------------------------
# 3. DEFAULT PROHIBITION + LIMITED EXCEPTION
# -------------------------------------------------------------------

def exception_path(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    unsupported = v["unsupported"]

    return F(
        f"exception_{code}",
        "prohibition_exception",
        [
            E(
                "D",
                f"{code}_restriction.txt",
                "DEFAULT RULE",
                f"Someone must not {action}.",
            ),
            E(
                "X",
                f"{code}_restriction.txt",
                "LIMITED EXCEPTION",
                f"Despite the default rule, someone may {action} "
                f"when {c1} is active and {c2} is confirmed.",
            ),
        ],
        [
            C(
                "YES",
                f"{c1.capitalize()} is active and {c2} is confirmed. "
                f"Under the stated exception, may someone {action}?",
                "permission",
                ["D", "X"],
                "The exception applies and every exception requirement is satisfied.",
                ["exception", "multiple_conditions"],
            ),
            C(
                "CONDITIONAL",
                f"{c1.capitalize()} is active. "
                f"Under the stated exception, may someone {action}?",
                "permission",
                ["D", "X"],
                "The exception may apply, but one material requirement is unresolved.",
                ["exception", "unresolved_condition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                f"{c1.capitalize()} was expressly denied. "
                f"May someone {action}?",
                "permission",
                ["D", "X"],
                "The default prohibition remains because an exception requirement failed.",
                ["exception", "failed_exception"],
            ),
            C(
                "INSUFFICIENT",
                f"Does the exception specifically require {unsupported} "
                f"before someone may {action}?",
                "requirement",
                [],
                "The exception does not establish the requested specific detail.",
                ["exception", "evidence_boundary"],
            ),
        ],
    )


# -------------------------------------------------------------------
# 4. DEFINITION CHAIN
# -------------------------------------------------------------------

def definition_chain(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    unsupported = v["unsupported"]

    category = f"{code.capitalize()} qualified environment"

    return F(
        f"definition_{code}",
        "definition_chain",
        [
            E(
                "D",
                f"{code}_definitions.txt",
                "QUALIFIED ENVIRONMENT",
                f"A {category} is an institutional environment "
                f"for which {c1} is active and {c2} is confirmed.",
            ),
            E(
                "R",
                f"{code}_processing.txt",
                "QUALIFIED ENVIRONMENT RULE",
                f"Someone may {action} only in a {category}.",
            ),
        ],
        [
            C(
                "YES",
                f"For the environment in question, {c1} is active "
                f"and {c2} is confirmed. "
                f"Does the evidence establish that someone may {action} there?",
                "classification",
                ["D", "R"],
                "The environment satisfies the supplied definition.",
                ["definition_chain", "classification", "multiple_conditions"],
            ),
            C(
                "CONDITIONAL",
                f"For the environment in question, {c1} is active, "
                f"but the evidence does not state whether {c2} is confirmed. "
                f"May someone {action} there?",
                "classification",
                ["D", "R"],
                "One required definitional element remains unresolved.",
                ["definition_chain", "partial_definition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                f"For the environment in question, {c2} is expressly "
                f"not confirmed. Does the environment qualify for "
                f"someone to {action} there?",
                "classification",
                ["D", "R"],
                "The environment explicitly fails a required definitional element.",
                ["definition_chain", "failed_definition"],
            ),
            C(
                "INSUFFICIENT",
                f"Does {unsupported} by itself make an environment "
                f"a {category}?",
                "classification",
                [],
                "The supplied definition does not establish that property.",
                ["definition_chain", "evidence_boundary"],
            ),
        ],
    )


# -------------------------------------------------------------------
# 5. DELEGATED AUTHORITY + CURRENT STATUS
# -------------------------------------------------------------------

def delegation(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    unsupported = v["unsupported"]

    actor = f"{code.capitalize()} coordinator"

    return F(
        f"delegation_{code}",
        "delegated_authority_status",
        [
            E(
                "A",
                f"{code}_authority.txt",
                "AUTHORIZED ACTOR",
                f"A project lead may authorize someone to {action}.",
            ),
            E(
                "D",
                f"{code}_delegation.txt",
                "DELEGATION",
                f"A project lead may delegate that authority to a "
                f"{actor} only while {c1} is active.",
            ),
            E(
                "R",
                f"{code}_delegation.txt",
                "DELEGATION STATUS",
                f"A delegated authorization is valid only when "
                f"{c2} is confirmed.",
            ),
        ],
        [
            C(
                "YES",
                f"A {actor} issued the authorization; {c1} is active "
                f"and {c2} is confirmed. "
                f"Is the authorization valid for someone to {action}?",
                "actor_authority",
                ["A", "D", "R"],
                "The delegated actor has valid current authority.",
                ["delegated_authority", "multiple_conditions"],
            ),
            C(
                "CONDITIONAL",
                f"A {actor} issued the authorization and {c1} is active. "
                f"Is that enough to authorize someone to {action}?",
                "actor_authority",
                ["D", "R"],
                "Delegated authority may be valid, but one status requirement is unresolved.",
                ["delegated_authority", "unresolved_status"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                f"A {actor} issued the authorization, but {c1} had expired. "
                f"Is the authorization valid for someone to {action}?",
                "actor_authority",
                ["D"],
                "The delegation is explicitly outside its valid period.",
                ["delegated_authority", "expired_authority"],
            ),
            C(
                "INSUFFICIENT",
                f"Does holding the job title associated with "
                f"{unsupported} automatically make someone a {actor}?",
                "classification",
                [],
                "The evidence does not establish that job title as delegated authority.",
                ["delegated_authority", "evidence_boundary"],
            ),
        ],
    )


# -------------------------------------------------------------------
# 6. PURPOSE / LICENSE SCOPE + DEFINITION
# -------------------------------------------------------------------

def purpose_scope(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    unsupported = v["unsupported"]

    purpose = f"{code} scholarly purpose"

    return F(
        f"purpose_{code}",
        "purpose_scope_definition",
        [
            E(
                "L",
                f"{code}_use_terms.txt",
                "PERMITTED PURPOSE",
                f"Someone may {action} for a {purpose} "
                f"only when {c1} is active.",
            ),
            E(
                "D",
                f"{code}_purpose_definition.txt",
                "SCHOLARLY PURPOSE",
                f"A {purpose} is a noncommercial institutional activity "
                f"for which {c2} is confirmed.",
            ),
            E(
                "N",
                f"{code}_use_terms.txt",
                "OUT-OF-SCOPE USE",
                f"Use outside a {purpose} is not permitted under these terms.",
            ),
        ],
        [
            C(
                "YES",
                f"The proposed use is noncommercial institutional activity; "
                f"{c1} is active and {c2} is confirmed. "
                f"May someone {action}?",
                "permission",
                ["L", "D"],
                "The use satisfies both the purpose definition and approval condition.",
                ["purpose_limitation", "definition_chain", "multiple_conditions"],
            ),
            C(
                "CONDITIONAL",
                f"The proposed use is noncommercial institutional activity "
                f"and {c1} is active. May someone {action}?",
                "permission",
                ["L", "D"],
                "The proposed purpose may qualify, but one defining condition is unresolved.",
                ["purpose_limitation", "partial_definition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                f"The proposed use is expressly commercial. "
                f"May someone {action} under these terms?",
                "permission",
                ["D", "N"],
                "The action is outside the defined permitted purpose.",
                ["purpose_limitation", "scope_failure"],
            ),
            C(
                "INSUFFICIENT",
                f"Does receiving {unsupported} automatically make "
                f"a use commercial under these terms?",
                "classification",
                [],
                "The evidence does not define the effect of that detail.",
                ["purpose_limitation", "undefined_term"],
            ),
        ],
    )


# -------------------------------------------------------------------
# 7. SUPERSESSION + CURRENT RULE
# VALIDATION ONLY
# -------------------------------------------------------------------

def supersession(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    unsupported = v["unsupported"]

    return F(
        f"supersession_{code}",
        "supersession_current_rule",
        [
            E(
                "O",
                f"{code}_former_policy.txt",
                "FORMER RULE",
                f"The former rule allowed someone to {action} "
                f"when {c1} was active.",
            ),
            E(
                "N",
                f"{code}_current_policy.txt",
                "CURRENT RULE",
                f"The current rule allows someone to {action} "
                f"only when {c2} is confirmed.",
            ),
            E(
                "S",
                f"{code}_policy_notice.txt",
                "SUPERSESSION",
                "The current rule supersedes the former rule wherever they conflict.",
            ),
        ],
        [
            C(
                "YES",
                f"{c2.capitalize()} is confirmed. "
                f"Under the current policy, may someone {action}?",
                "permission",
                ["N", "S"],
                "The superseding current rule controls and its condition is satisfied.",
                ["supersession", "policy_precedence"],
            ),
            C(
                "CONDITIONAL",
                f"{c1.capitalize()} is active, but the evidence does not "
                f"state whether {c2} is confirmed. "
                f"Under the current policy, may someone {action}?",
                "permission",
                ["N", "S"],
                "The current rule controls, but its required condition is unresolved.",
                ["supersession", "policy_precedence", "unresolved_condition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                f"{c1.capitalize()} is active, but {c2} is expressly "
                f"not confirmed. Under the current policy, "
                f"may someone {action}?",
                "permission",
                ["N", "S"],
                "The superseding current rule has an explicitly failed condition.",
                ["supersession", "failed_condition"],
            ),
            C(
                "INSUFFICIENT",
                f"Was the current policy adopted using "
                f"{unsupported} specifically?",
                "proposition",
                [],
                "The evidence establishes supersession but not that adoption detail.",
                ["supersession", "evidence_boundary"],
            ),
        ],
    )


# -------------------------------------------------------------------
# 8. TEMPORAL RULE + OVERRIDE
# VALIDATION ONLY
# -------------------------------------------------------------------

def temporal_override(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    unsupported = v["unsupported"]

    return F(
        f"temporal_{code}",
        "temporal_override",
        [
            E(
                "T",
                f"{code}_schedule.txt",
                "TIME RULE",
                f"Someone may {action} after {c1} is satisfied.",
            ),
            E(
                "H",
                f"{code}_override.txt",
                "ACTIVE OVERRIDE",
                f"Even after that time, the action must not occur "
                f"while {c2} remains active.",
            ),
        ],
        [
            C(
                "YES",
                f"{c1.capitalize()} is satisfied and {c2} is not active. "
                f"May someone {action}?",
                "permission",
                ["T", "H"],
                "The time requirement is satisfied and the override does not apply.",
                ["temporal", "override", "multiple_conditions"],
            ),
            C(
                "CONDITIONAL",
                f"{c1.capitalize()} is satisfied. "
                f"May someone {action}?",
                "permission",
                ["T", "H"],
                "The time requirement is satisfied, but override status is unresolved.",
                ["temporal", "override", "unresolved_condition"],
                [f"{c2} must not remain active"],
            ),
            C(
                "NO",
                f"{c1.capitalize()} is satisfied, but {c2} remains active. "
                f"May someone {action}?",
                "permission",
                ["T", "H"],
                "The active override directly prevents the action.",
                ["temporal", "override", "direct_prohibition"],
            ),
            C(
                "INSUFFICIENT",
                f"Does the policy require {unsupported} specifically "
                f"after the time requirement is satisfied?",
                "requirement",
                [],
                "The evidence does not establish the requested additional requirement.",
                ["temporal", "evidence_boundary"],
            ),
        ],
    )


# -------------------------------------------------------------------
# 9. JURISDICTION DEFINITION + AUTHORIZATION
# TEST ONLY
# -------------------------------------------------------------------

def jurisdiction(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    unsupported = v["unsupported"]

    zone = f"{code.capitalize()} approved zone"

    return F(
        f"jurisdiction_{code}",
        "jurisdiction_definition_authorization",
        [
            E(
                "J",
                f"{code}_jurisdiction.txt",
                "APPROVED ZONE",
                f"A {zone} is a destination for which {c1} is confirmed.",
            ),
            E(
                "R",
                f"{code}_zone_rule.txt",
                "ZONE RULE",
                f"Someone may {action} into a {zone} "
                f"only when {c2} is active.",
            ),
            E(
                "N",
                f"{code}_zone_rule.txt",
                "OUTSIDE ZONE",
                f"The action is not permitted into a destination "
                f"established to be outside a {zone}.",
            ),
        ],
        [
            C(
                "YES",
                f"For the destination, {c1} is confirmed and {c2} is active. "
                f"May someone {action}?",
                "permission",
                ["J", "R"],
                "The destination qualifies and the required authorization is active.",
                ["jurisdiction", "definition_chain", "authorization"],
            ),
            C(
                "CONDITIONAL",
                f"For the destination, {c1} is confirmed. "
                f"May someone {action}?",
                "permission",
                ["J", "R"],
                "The destination may qualify, but authorization status is unresolved.",
                ["jurisdiction", "unresolved_authorization"],
                [f"{c2} must be active"],
            ),
            C(
                "NO",
                f"The destination is expressly established to be outside "
                f"the {zone}. May someone {action} there?",
                "permission",
                ["N"],
                "The destination is explicitly outside the permitted jurisdiction.",
                ["jurisdiction", "scope_failure"],
            ),
            C(
                "INSUFFICIENT",
                f"Does {unsupported} by itself establish that a destination "
                f"is within the {zone}?",
                "classification",
                [],
                "The supplied jurisdiction definition does not establish that criterion.",
                ["jurisdiction", "evidence_boundary"],
            ),
        ],
    )


# -------------------------------------------------------------------
# 10. PERMISSION VS OBLIGATION + SCOPE
# TEST ONLY
# -------------------------------------------------------------------

def obligation_boundary(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    unsupported = v["unsupported"]

    return F(
        f"obligation_{code}",
        "permission_obligation_scope",
        [
            E(
                "P",
                f"{code}_optional_action.txt",
                "PERMISSION",
                f"Someone may {action} when {c1} is active.",
            ),
            E(
                "M",
                f"{code}_mandatory_action.txt",
                "SEPARATE OBLIGATION",
                f"{c2.capitalize()} is mandatory only for projects "
                f"expressly designated high-risk.",
            ),
            E(
                "S",
                f"{code}_scope.txt",
                "HIGH-RISK DESIGNATION",
                "A project is high-risk only when the supplied record "
                "expressly designates it high-risk.",
            ),
        ],
        [
            C(
                "YES",
                f"{c1.capitalize()} is active. "
                f"Is someone permitted to {action}?",
                "permission",
                ["P"],
                "The permission condition is satisfied.",
                ["permission_vs_obligation"],
            ),
            C(
                "CONDITIONAL",
                f"May someone {action}?",
                "permission",
                ["P"],
                "The action may be permitted, but its stated condition is unresolved.",
                ["permission_vs_obligation", "unresolved_condition"],
                [f"{c1} must be active"],
            ),
            C(
                "NO",
                f"The project is expressly established not to be high-risk. "
                f"Does the evidence establish that {c2} is mandatory "
                f"for this project?",
                "proposition",
                ["M", "S"],
                "The mandatory rule is expressly limited to high-risk projects.",
                ["mandatory_scope", "negative_proposition"],
            ),
            C(
                "INSUFFICIENT",
                f"Does {unsupported} by itself establish that a project "
                f"is high-risk?",
                "classification",
                [],
                "The scope rule requires an express designation.",
                ["scope_boundary", "evidence_boundary"],
            ),
        ],
    )


# -------------------------------------------------------------------
# SPLITS
#
# 6 composition structures x 5 variants = 30 train families
# 2 structures x 5 variants = 10 validation families
# 2 structures x 5 variants = 10 test families
#
# Total = 50 composition families.
# -------------------------------------------------------------------

TRAIN_BUILDERS = [
    condition_ledger,
    specificity,
    exception_path,
    definition_chain,
    delegation,
    purpose_scope,
]

VALIDATION_BUILDERS = [
    supersession,
    temporal_override,
]

TEST_BUILDERS = [
    jurisdiction,
    obligation_boundary,
]


def build(builder_list):
    result = []

    for builder in builder_list:
        for variant in VARIANTS:
            result.append(
                builder(variant)
            )

    return result


TRAIN = build(TRAIN_BUILDERS)
VALIDATION = build(VALIDATION_BUILDERS)
TEST = build(TEST_BUILDERS)

FAMILIES = {
    "train": TRAIN,
    "validation": VALIDATION,
    "test": TEST,
}
