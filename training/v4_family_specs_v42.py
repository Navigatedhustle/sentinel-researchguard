"""
Sentinel V4.2 counterfactual compositional curriculum.

Core design:
- The QUESTION does not determine the label.
- Within each family, the same question appears in four different evidence
  worlds.
- YES / NO / CONDITIONAL / INSUFFICIENT therefore depend on evidence.
- Train / validation / test retain different composition structures and
  different surface ontologies.
- Challenge V1 is never imported or referenced.
"""


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
    evidence,
    question_type,
    citations,
    rationale,
    tags,
    conditions=None,
):
    return {
        "decision": decision,
        "question": question,
        "evidence": evidence,
        "question_type": question_type,
        "citations": citations,
        "rationale": rationale,
        "reasoning_tags": tags,
        "conditions": conditions or [],
    }


def F(name, composition, cases):
    return {
        "name": name,
        "composition": composition,

        # V4.2 uses case-specific evidence.
        # Kept for builder compatibility.
        "evidence": [],

        "cases": {
            case["decision"]: case
            for case in cases
        },
    }


TRAIN_VARIANTS = [
    {
        "code": "orion",
        "action": "analyze {asset} inside the Orion workspace",
        "c1": "workspace clearance",
        "c2": "data-steward confirmation",
    },
    {
        "code": "atlas",
        "action": "release {asset} into the Atlas collaboration",
        "c1": "release clearance",
        "c2": "recipient assurance",
    },
    {
        "code": "nova",
        "action": "use {asset} in the Nova research workflow",
        "c1": "workflow authorization",
        "c2": "security attestation",
    },
    {
        "code": "meridian",
        "action": "move {asset} into the Meridian environment",
        "c1": "transfer authorization",
        "c2": "environment registration",
    },
    {
        "code": "helix",
        "action": "retain {asset} in the Helix archive",
        "c1": "archive approval",
        "c2": "preservation-status confirmation",
    },
]


VALIDATION_VARIANTS = [
    {
        "code": "apogee",
        "action": "review {asset} within the Apogee enclave",
        "c1": "enclave authorization",
        "c2": "custodian acknowledgment",
    },
    {
        "code": "harbor",
        "action": "stage {asset} for the Harbor exchange",
        "c1": "exchange approval",
        "c2": "counterparty verification",
    },
    {
        "code": "prism",
        "action": "process {asset} through the Prism pipeline",
        "c1": "processing clearance",
        "c2": "control attestation",
    },
    {
        "code": "summit",
        "action": "place {asset} in the Summit repository",
        "c1": "repository approval",
        "c2": "repository enrollment",
    },
    {
        "code": "ember",
        "action": "preserve {asset} in the Ember vault",
        "c1": "vault authorization",
        "c2": "retention verification",
    },
]


TEST_VARIANTS = [
    {
        "code": "juniper",
        "action": "inspect {asset} inside the Juniper domain",
        "c1": "domain approval",
        "c2": "steward verification",
    },
    {
        "code": "cobalt",
        "action": "provide {asset} through the Cobalt exchange",
        "c1": "sharing approval",
        "c2": "receiver verification",
    },
    {
        "code": "aurora",
        "action": "operate on {asset} in the Aurora workflow",
        "c1": "workflow clearance",
        "c2": "safety verification",
    },
    {
        "code": "delta",
        "action": "transfer {asset} to the Delta enclave",
        "c1": "transfer clearance",
        "c2": "destination verification",
    },
    {
        "code": "cedar",
        "action": "store {asset} in the Cedar repository",
        "c1": "storage approval",
        "c2": "retention confirmation",
    },
]


# =====================================================================
# TRAIN COMPOSITIONS
# =====================================================================

def condition_ledger(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]

    q = (
        f"Based on the statuses in the supplied record, "
        f"may someone {action}?"
    )

    return F(
        f"ledger_{code}",
        "condition_ledger",
        [
            C(
                "YES",
                q,
                [
                    E(
                        "R",
                        f"{code}_route.txt",
                        "OPERATIVE RULE",
                        f"Someone may {action} when {c1} is active "
                        f"and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "CURRENT STATUS",
                        f"For this request, {c1} is active and "
                        f"{c2} is confirmed.",
                    ),
                ],
                "permission",
                ["R", "S"],
                "Every material condition in the operative rule is established.",
                ["condition_ledger", "multiple_conditions"],
            ),
            C(
                "CONDITIONAL",
                q,
                [
                    E(
                        "R",
                        f"{code}_route.txt",
                        "OPERATIVE RULE",
                        f"Someone may {action} when {c1} is active "
                        f"and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "CURRENT STATUS",
                        f"For this request, {c1} is active. "
                        f"The record contains no determination of {c2}.",
                    ),
                ],
                "permission",
                ["R", "S"],
                "The action may be permitted, but one material condition is unresolved.",
                ["condition_ledger", "unresolved_condition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                q,
                [
                    E(
                        "R",
                        f"{code}_route.txt",
                        "OPERATIVE RULE",
                        f"Someone may {action} only when {c1} is active "
                        f"and {c2} is confirmed.",
                    ),
                    E(
                        "F",
                        f"{code}_route.txt",
                        "FAILED CONTROL",
                        f"If {c2} is denied, {action} is not permitted.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "CURRENT STATUS",
                        f"For this request, {c1} is active and "
                        f"{c2} is denied.",
                    ),
                ],
                "permission",
                ["R", "F", "S"],
                "A material requirement is directly failed.",
                ["condition_ledger", "failed_condition"],
            ),
            C(
                "INSUFFICIENT",
                q,
                [
                    E(
                        "I",
                        f"{code}_inventory.txt",
                        "RECORDKEEPING",
                        f"Records associated with {c1} and {c2} "
                        f"must remain in the institutional inventory.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "CURRENT STATUS",
                        f"The record lists {c1} as active and "
                        f"{c2} as confirmed.",
                    ),
                ],
                "permission",
                [],
                "The supplied evidence describes related controls but does not govern whether the action is permitted.",
                ["condition_ledger", "evidence_boundary"],
            ),
        ],
    )


def specificity(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]

    q = (
        f"For the restricted material described in the supplied record, "
        f"may someone {action}?"
    )

    return F(
        f"specificity_{code}",
        "specific_rule_precedence",
        [
            C(
                "YES",
                q,
                [
                    E(
                        "G",
                        f"{code}_general.txt",
                        "GENERAL RULE",
                        f"Materials ordinarily may not be used to {action}.",
                    ),
                    E(
                        "S",
                        f"{code}_restricted.txt",
                        "SPECIFIC RULE",
                        f"Despite the general rule, restricted material may be used to "
                        f"{action} when {c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "X",
                        f"{code}_status.txt",
                        "STATUS",
                        f"The material is restricted. {c1} is active and {c2} is confirmed.",
                    ),
                ],
                "permission",
                ["G", "S", "X"],
                "The specific rule controls over the general rule and its requirements are satisfied.",
                ["specificity", "policy_precedence"],
            ),
            C(
                "CONDITIONAL",
                q,
                [
                    E(
                        "G",
                        f"{code}_general.txt",
                        "GENERAL RULE",
                        f"Materials ordinarily may not be used to {action}.",
                    ),
                    E(
                        "S",
                        f"{code}_restricted.txt",
                        "SPECIFIC RULE",
                        f"Despite the general rule, restricted material may be used to "
                        f"{action} when {c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "X",
                        f"{code}_status.txt",
                        "STATUS",
                        f"The material is restricted and {c1} is active. "
                        f"The record does not establish {c2}.",
                    ),
                ],
                "permission",
                ["G", "S", "X"],
                "The specific rule controls, but one of its requirements is unresolved.",
                ["specificity", "policy_precedence", "unresolved_condition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                q,
                [
                    E(
                        "G",
                        f"{code}_general.txt",
                        "GENERAL RULE",
                        f"Institutional material may be used to {action} when {c1} is active.",
                    ),
                    E(
                        "S",
                        f"{code}_restricted.txt",
                        "SPECIFIC RULE",
                        f"Restricted material may not be used to {action}, "
                        f"even when the general permission applies.",
                    ),
                    E(
                        "X",
                        f"{code}_status.txt",
                        "STATUS",
                        f"The material is restricted and {c1} is active.",
                    ),
                ],
                "permission",
                ["G", "S", "X"],
                "The specific prohibition controls over the general permission.",
                ["specificity", "policy_precedence", "direct_prohibition"],
            ),
            C(
                "INSUFFICIENT",
                q,
                [
                    E(
                        "S",
                        f"{code}_restricted.txt",
                        "RESTRICTED MATERIAL",
                        "Restricted material must be identified in the asset register.",
                    ),
                    E(
                        "X",
                        f"{code}_status.txt",
                        "STATUS",
                        "The material in question is classified as restricted.",
                    ),
                ],
                "permission",
                [],
                "The evidence governs classification and recordkeeping but not permission for the action.",
                ["specificity", "evidence_boundary"],
            ),
        ],
    )


def exception_path(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]

    q = (
        f"Under the exception framework in the supplied record, "
        f"may someone {action}?"
    )

    return F(
        f"exception_{code}",
        "prohibition_exception",
        [
            C(
                "YES",
                q,
                [
                    E(
                        "P",
                        f"{code}_prohibition.txt",
                        "BASE RULE",
                        f"Someone may not {action}.",
                    ),
                    E(
                        "X",
                        f"{code}_exception.txt",
                        "EXCEPTION",
                        f"An exception permits someone to {action} when "
                        f"{c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "STATUS",
                        f"{c1.capitalize()} is active and {c2} is confirmed.",
                    ),
                ],
                "permission",
                ["P", "X", "S"],
                "The exception defeats the prohibition because every exception condition is satisfied.",
                ["exception", "prohibition_exception"],
            ),
            C(
                "CONDITIONAL",
                q,
                [
                    E(
                        "P",
                        f"{code}_prohibition.txt",
                        "BASE RULE",
                        f"Someone may not {action}.",
                    ),
                    E(
                        "X",
                        f"{code}_exception.txt",
                        "EXCEPTION",
                        f"An exception permits someone to {action} when "
                        f"{c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "STATUS",
                        f"{c1.capitalize()} is active. "
                        f"The record does not establish {c2}.",
                    ),
                ],
                "permission",
                ["P", "X", "S"],
                "An exception could apply, but one exception requirement is unresolved.",
                ["exception", "unresolved_condition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                q,
                [
                    E(
                        "P",
                        f"{code}_prohibition.txt",
                        "BASE RULE",
                        f"Someone may not {action}.",
                    ),
                    E(
                        "X",
                        f"{code}_exception.txt",
                        "EXCEPTION",
                        f"The prohibition is displaced only when "
                        f"{c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "STATUS",
                        f"{c1.capitalize()} is active and {c2} is denied.",
                    ),
                ],
                "permission",
                ["P", "X", "S"],
                "The exception is unavailable, so the prohibition remains controlling.",
                ["exception", "failed_exception"],
            ),
            C(
                "INSUFFICIENT",
                q,
                [
                    E(
                        "X",
                        f"{code}_exception.txt",
                        "EXCEPTION ADMINISTRATION",
                        f"Exception requests involving {c1} and {c2} "
                        f"must be logged for audit review.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "STATUS",
                        f"A record exists for {c1} and {c2}.",
                    ),
                ],
                "permission",
                [],
                "The evidence describes exception administration but never establishes a prohibition, permission, or operative exception for the action.",
                ["exception", "evidence_boundary"],
            ),
        ],
    )


def definition_chain(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    category = f"{code.capitalize()} qualified environment"

    q = (
        f"For the environment characterized in the supplied record, "
        f"may someone {action}?"
    )

    return F(
        f"definition_{code}",
        "definition_chain",
        [
            C(
                "YES",
                q,
                [
                    E(
                        "D",
                        f"{code}_definitions.txt",
                        "QUALIFIED ENVIRONMENT",
                        f"A {category} is an environment where "
                        f"{c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "R",
                        f"{code}_usage.txt",
                        "USE RULE",
                        f"Someone may {action} in a {category}.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "STATUS",
                        f"For this environment, {c1} is active and {c2} is confirmed.",
                    ),
                ],
                "permission",
                ["D", "R", "S"],
                "The facts satisfy the definition that activates the permission.",
                ["definition_chain", "classification"],
            ),
            C(
                "CONDITIONAL",
                q,
                [
                    E(
                        "D",
                        f"{code}_definitions.txt",
                        "QUALIFIED ENVIRONMENT",
                        f"A {category} requires both {c1} and {c2}.",
                    ),
                    E(
                        "R",
                        f"{code}_usage.txt",
                        "USE RULE",
                        f"Someone may {action} in a {category}.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "STATUS",
                        f"{c1.capitalize()} is active. "
                        f"The record does not establish {c2}.",
                    ),
                ],
                "permission",
                ["D", "R", "S"],
                "Whether the environment satisfies the governing definition remains unresolved.",
                ["definition_chain", "unresolved_definition"],
                [f"{c2} must be established"],
            ),
            C(
                "NO",
                q,
                [
                    E(
                        "D",
                        f"{code}_definitions.txt",
                        "QUALIFIED ENVIRONMENT",
                        f"A {category} requires both {c1} and {c2}.",
                    ),
                    E(
                        "R",
                        f"{code}_usage.txt",
                        "USE RULE",
                        f"Someone may not {action} outside a {category}.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "STATUS",
                        f"{c1.capitalize()} is active, but {c2} is denied.",
                    ),
                ],
                "permission",
                ["D", "R", "S"],
                "The environment fails the definition and the rule prohibits the action outside that class.",
                ["definition_chain", "failed_definition"],
            ),
            C(
                "INSUFFICIENT",
                q,
                [
                    E(
                        "D",
                        f"{code}_definitions.txt",
                        "QUALIFIED ENVIRONMENT",
                        f"A {category} is an environment where "
                        f"{c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "STATUS",
                        f"This environment satisfies the definition of a {category}.",
                    ),
                ],
                "permission",
                [],
                "The evidence establishes a classification but supplies no rule governing the requested action.",
                ["definition_chain", "evidence_boundary"],
            ),
        ],
    )


def delegation(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    actor = f"{code.capitalize()} coordinator"

    q = (
        f"Is there valid authority in the supplied record "
        f"for someone to {action}?"
    )

    return F(
        f"delegation_{code}",
        "delegated_authority_status",
        [
            C(
                "YES",
                q,
                [
                    E(
                        "A",
                        f"{code}_authority.txt",
                        "DELEGATION",
                        f"A {actor} may authorize someone to {action} "
                        f"while the coordinator's delegation is active.",
                    ),
                    E(
                        "R",
                        f"{code}_authority.txt",
                        "AUTHORIZED USE",
                        f"A valid authorization permits {action} when "
                        f"{c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "CURRENT STATUS",
                        f"An {actor} issued the authorization while the delegation "
                        f"was active. {c1.capitalize()} is active and {c2} is confirmed.",
                    ),
                ],
                "actor_authority",
                ["A", "R", "S"],
                "The authorization was issued under valid delegated authority and all operative requirements are satisfied.",
                ["delegated_authority", "authority_status"],
            ),
            C(
                "CONDITIONAL",
                q,
                [
                    E(
                        "A",
                        f"{code}_authority.txt",
                        "DELEGATION",
                        f"A {actor} may authorize someone to {action} "
                        f"while the delegation is active.",
                    ),
                    E(
                        "R",
                        f"{code}_authority.txt",
                        "AUTHORIZED USE",
                        f"A valid authorization permits {action} when "
                        f"{c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "CURRENT STATUS",
                        f"An {actor} issued the authorization while the delegation "
                        f"was active. {c1.capitalize()} is active. "
                        f"The record does not establish {c2}.",
                    ),
                ],
                "actor_authority",
                ["A", "R", "S"],
                "The authority is valid, but one requirement for exercising it is unresolved.",
                ["delegated_authority", "unresolved_condition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                q,
                [
                    E(
                        "A",
                        f"{code}_authority.txt",
                        "DELEGATION",
                        f"A {actor} may authorize someone to {action} only "
                        f"while the delegation is active.",
                    ),
                    E(
                        "V",
                        f"{code}_authority.txt",
                        "REVOCATION",
                        "An authorization issued after revocation has no delegated authority.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "CURRENT STATUS",
                        f"The coordinator's delegation was revoked before "
                        f"the authorization was issued.",
                    ),
                ],
                "actor_authority",
                ["A", "V", "S"],
                "The purported authorization was issued after delegated authority had ended.",
                ["delegated_authority", "revoked_authority"],
            ),
            C(
                "INSUFFICIENT",
                q,
                [
                    E(
                        "T",
                        f"{code}_directory.txt",
                        "ROLE DIRECTORY",
                        f"The institutional directory lists a role called {actor}.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "CURRENT STATUS",
                        f"{c1.capitalize()} is active and {c2} is confirmed.",
                    ),
                ],
                "actor_authority",
                [],
                "A role title and operational statuses do not establish delegated authority to authorize the action.",
                ["delegated_authority", "evidence_boundary"],
            ),
        ],
    )


def purpose_scope(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    purpose = f"{code} scholarly purpose"

    q = (
        f"For the use described in the supplied record, "
        f"may someone {action}?"
    )

    return F(
        f"purpose_{code}",
        "purpose_scope_definition",
        [
            C(
                "YES",
                q,
                [
                    E(
                        "D",
                        f"{code}_purpose.txt",
                        "DEFINITION",
                        f"A {purpose} means noncommercial institutional research activity.",
                    ),
                    E(
                        "R",
                        f"{code}_purpose.txt",
                        "PERMISSION",
                        f"Someone may {action} for a {purpose} when "
                        f"{c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "USE STATUS",
                        f"The proposed use is noncommercial institutional research. "
                        f"{c1.capitalize()} is active and {c2} is confirmed.",
                    ),
                ],
                "permission",
                ["D", "R", "S"],
                "The proposed use falls within the defined purpose and satisfies the operative requirements.",
                ["purpose_limitation", "definition_chain"],
            ),
            C(
                "CONDITIONAL",
                q,
                [
                    E(
                        "D",
                        f"{code}_purpose.txt",
                        "DEFINITION",
                        f"A {purpose} means noncommercial institutional research activity.",
                    ),
                    E(
                        "R",
                        f"{code}_purpose.txt",
                        "PERMISSION",
                        f"Someone may {action} for a {purpose} when "
                        f"{c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "USE STATUS",
                        f"The proposed use is noncommercial institutional research. "
                        f"{c1.capitalize()} is active. "
                        f"The record does not establish {c2}.",
                    ),
                ],
                "permission",
                ["D", "R", "S"],
                "The purpose requirement is satisfied, but another material requirement remains unresolved.",
                ["purpose_limitation", "unresolved_condition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                q,
                [
                    E(
                        "D",
                        f"{code}_purpose.txt",
                        "DEFINITION",
                        f"A {purpose} excludes commercial client activity.",
                    ),
                    E(
                        "R",
                        f"{code}_purpose.txt",
                        "SCOPE",
                        f"Someone may not {action} for activity outside a {purpose}.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "USE STATUS",
                        "The proposed use is commercial client work.",
                    ),
                ],
                "permission",
                ["D", "R", "S"],
                "The proposed use falls outside the defined permitted purpose.",
                ["purpose_limitation", "scope_failure"],
            ),
            C(
                "INSUFFICIENT",
                q,
                [
                    E(
                        "D",
                        f"{code}_purpose.txt",
                        "DEFINITION",
                        f"A {purpose} means noncommercial institutional research activity.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "USE STATUS",
                        "The proposed use is noncommercial institutional research.",
                    ),
                ],
                "permission",
                [],
                "The evidence defines and classifies the purpose but does not govern whether the requested action is permitted.",
                ["purpose_limitation", "evidence_boundary"],
            ),
        ],
    )


# =====================================================================
# VALIDATION-ONLY COMPOSITIONS
# =====================================================================

def supersession(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]

    q = (
        f"Under the policy currently in force, "
        f"may someone {action}?"
    )

    return F(
        f"supersession_{code}",
        "supersession_current_rule",
        [
            C(
                "YES",
                q,
                [
                    E(
                        "O",
                        f"{code}_policy_2025.txt",
                        "OLDER RULE",
                        f"Someone may not {action}.",
                    ),
                    E(
                        "N",
                        f"{code}_policy_2026.txt",
                        "CURRENT RULE",
                        f"Someone may {action} when {c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "M",
                        f"{code}_policy_2026.txt",
                        "SUPERSESSION",
                        "This policy supersedes the prior rule on the same subject.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "STATUS",
                        f"{c1.capitalize()} is active and {c2} is confirmed.",
                    ),
                ],
                "permission",
                ["O", "N", "M", "S"],
                "The newer policy supersedes the older prohibition and its permission requirements are satisfied.",
                ["supersession", "current_rule"],
            ),
            C(
                "CONDITIONAL",
                q,
                [
                    E(
                        "O",
                        f"{code}_policy_2025.txt",
                        "OLDER RULE",
                        f"Someone may {action} whenever {c1} is active.",
                    ),
                    E(
                        "N",
                        f"{code}_policy_2026.txt",
                        "CURRENT RULE",
                        f"Someone may {action} only when {c1} is active "
                        f"and {c2} is confirmed.",
                    ),
                    E(
                        "M",
                        f"{code}_policy_2026.txt",
                        "SUPERSESSION",
                        "The current rule replaces the prior rule on this subject.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "STATUS",
                        f"{c1.capitalize()} is active. "
                        f"The record does not establish {c2}.",
                    ),
                ],
                "permission",
                ["O", "N", "M", "S"],
                "The current rule controls, but one current requirement is unresolved.",
                ["supersession", "unresolved_condition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                q,
                [
                    E(
                        "O",
                        f"{code}_policy_2025.txt",
                        "OLDER RULE",
                        f"Someone may {action} when {c1} is active.",
                    ),
                    E(
                        "N",
                        f"{code}_policy_2026.txt",
                        "CURRENT RULE",
                        f"Someone may not {action}.",
                    ),
                    E(
                        "M",
                        f"{code}_policy_2026.txt",
                        "SUPERSESSION",
                        "The current rule supersedes the prior permission.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "STATUS",
                        f"{c1.capitalize()} is active.",
                    ),
                ],
                "permission",
                ["O", "N", "M", "S"],
                "The current prohibition supersedes the older permission.",
                ["supersession", "direct_prohibition"],
            ),
            C(
                "INSUFFICIENT",
                q,
                [
                    E(
                        "O",
                        f"{code}_policy_2025.txt",
                        "OLDER RULE",
                        f"Someone may {action} when {c1} is active.",
                    ),
                    E(
                        "N",
                        f"{code}_notice_2026.txt",
                        "LATER NOTICE",
                        f"A later notice discusses documentation for {c2}, "
                        f"but does not state whether it replaces the earlier rule.",
                    ),
                ],
                "permission",
                [],
                "The evidence does not establish which rule, if any, currently controls the requested action.",
                ["supersession", "evidence_boundary"],
            ),
        ],
    )


def temporal_override(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]

    q = (
        f"At the time described in the supplied record, "
        f"may someone {action}?"
    )

    return F(
        f"temporal_{code}",
        "temporal_override",
        [
            C(
                "YES",
                q,
                [
                    E(
                        "B",
                        f"{code}_timing.txt",
                        "BASE RULE",
                        f"Before 30 elapsed days, someone may not {action}.",
                    ),
                    E(
                        "T",
                        f"{code}_timing.txt",
                        "TIME OVERRIDE",
                        f"After 30 elapsed days, someone may {action} when "
                        f"{c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "CURRENT STATUS",
                        f"Forty-five days have elapsed. "
                        f"{c1.capitalize()} is active and {c2} is confirmed.",
                    ),
                ],
                "permission",
                ["B", "T", "S"],
                "The temporal override is active and its requirements are satisfied.",
                ["temporal", "override"],
            ),
            C(
                "CONDITIONAL",
                q,
                [
                    E(
                        "B",
                        f"{code}_timing.txt",
                        "BASE RULE",
                        f"Before 30 elapsed days, someone may not {action}.",
                    ),
                    E(
                        "T",
                        f"{code}_timing.txt",
                        "TIME OVERRIDE",
                        f"After 30 elapsed days, someone may {action} when "
                        f"{c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "CURRENT STATUS",
                        f"Forty-five days have elapsed and {c1} is active. "
                        f"The record does not establish {c2}.",
                    ),
                ],
                "permission",
                ["B", "T", "S"],
                "The temporal override applies, but one requirement remains unresolved.",
                ["temporal", "override", "unresolved_condition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                q,
                [
                    E(
                        "B",
                        f"{code}_timing.txt",
                        "BASE RULE",
                        f"Before 30 elapsed days, someone may not {action}.",
                    ),
                    E(
                        "T",
                        f"{code}_timing.txt",
                        "TIME OVERRIDE",
                        f"After 30 elapsed days, someone may {action} when "
                        f"{c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "CURRENT STATUS",
                        "Only ten days have elapsed.",
                    ),
                ],
                "permission",
                ["B", "T", "S"],
                "The time-triggered override is not yet active, so the prohibition controls.",
                ["temporal", "direct_prohibition"],
            ),
            C(
                "INSUFFICIENT",
                q,
                [
                    E(
                        "T",
                        f"{code}_timing.txt",
                        "REVIEW TIMING",
                        f"Documentation associated with {c1} and {c2} "
                        f"must be reviewed after 30 elapsed days.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "CURRENT STATUS",
                        "Forty-five days have elapsed.",
                    ),
                ],
                "permission",
                [],
                "The evidence establishes a timing requirement for review but does not govern whether the requested action is permitted.",
                ["temporal", "evidence_boundary"],
            ),
        ],
    )


# =====================================================================
# TEST-ONLY COMPOSITIONS
# =====================================================================

def jurisdiction(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]
    zone = f"{code.capitalize()} approved zone"

    q = (
        f"For the destination characterized in the supplied record, "
        f"may someone {action}?"
    )

    return F(
        f"jurisdiction_{code}",
        "jurisdiction_definition_authorization",
        [
            C(
                "YES",
                q,
                [
                    E(
                        "D",
                        f"{code}_jurisdiction.txt",
                        "ZONE DEFINITION",
                        f"A {zone} is a destination where {c2} is confirmed.",
                    ),
                    E(
                        "R",
                        f"{code}_jurisdiction.txt",
                        "AUTHORIZATION",
                        f"Someone may {action} within a {zone} when {c1} is active.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "DESTINATION STATUS",
                        f"For this destination, {c1} is active and {c2} is confirmed.",
                    ),
                ],
                "permission",
                ["D", "R", "S"],
                "The destination falls within the defined jurisdiction and the authorization requirement is satisfied.",
                ["jurisdiction", "definition_chain", "authorization"],
            ),
            C(
                "CONDITIONAL",
                q,
                [
                    E(
                        "D",
                        f"{code}_jurisdiction.txt",
                        "ZONE DEFINITION",
                        f"A {zone} is a destination where {c2} is confirmed.",
                    ),
                    E(
                        "R",
                        f"{code}_jurisdiction.txt",
                        "AUTHORIZATION",
                        f"Someone may {action} within a {zone} when {c1} is active.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "DESTINATION STATUS",
                        f"{c1.capitalize()} is active. "
                        f"The record does not establish {c2}.",
                    ),
                ],
                "permission",
                ["D", "R", "S"],
                "Authorization is established, but jurisdiction remains unresolved.",
                ["jurisdiction", "unresolved_definition"],
                [f"{c2} must be confirmed"],
            ),
            C(
                "NO",
                q,
                [
                    E(
                        "D",
                        f"{code}_jurisdiction.txt",
                        "ZONE DEFINITION",
                        f"A destination is outside the {zone} when {c2} is denied.",
                    ),
                    E(
                        "R",
                        f"{code}_jurisdiction.txt",
                        "BOUNDARY RULE",
                        f"Someone may not {action} outside the {zone}.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "DESTINATION STATUS",
                        f"{c1.capitalize()} is active, but {c2} is denied.",
                    ),
                ],
                "permission",
                ["D", "R", "S"],
                "The destination is outside the governing jurisdiction and the action is prohibited there.",
                ["jurisdiction", "scope_failure"],
            ),
            C(
                "INSUFFICIENT",
                q,
                [
                    E(
                        "D",
                        f"{code}_jurisdiction.txt",
                        "ZONE DEFINITION",
                        f"A {zone} is a destination where {c2} is confirmed.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "DESTINATION STATUS",
                        f"This destination satisfies the definition of a {zone}.",
                    ),
                ],
                "permission",
                [],
                "The evidence establishes jurisdiction but supplies no rule authorizing or prohibiting the requested action.",
                ["jurisdiction", "evidence_boundary"],
            ),
        ],
    )


def obligation_boundary(v):
    code = v["code"]
    action = v["action"]
    c1 = v["c1"]
    c2 = v["c2"]

    q = (
        f"Under the supplied policy, "
        f"is someone required to {action}?"
    )

    return F(
        f"obligation_{code}",
        "permission_obligation_scope",
        [
            C(
                "YES",
                q,
                [
                    E(
                        "P",
                        f"{code}_obligation.txt",
                        "PERMISSION",
                        f"Someone is permitted to {action}.",
                    ),
                    E(
                        "D",
                        f"{code}_obligation.txt",
                        "HIGH-RISK SCOPE",
                        f"A project is high-risk when {c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "O",
                        f"{code}_obligation.txt",
                        "MANDATORY RULE",
                        f"For a high-risk project, someone is required to {action}.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "PROJECT STATUS",
                        f"{c1.capitalize()} is active and {c2} is confirmed.",
                    ),
                ],
                "requirement",
                ["P", "D", "O", "S"],
                "The project falls within the mandatory scope, so the action is required rather than merely permitted.",
                ["permission_vs_obligation", "mandatory_scope"],
            ),
            C(
                "CONDITIONAL",
                q,
                [
                    E(
                        "P",
                        f"{code}_obligation.txt",
                        "PERMISSION",
                        f"Someone is permitted to {action}.",
                    ),
                    E(
                        "D",
                        f"{code}_obligation.txt",
                        "HIGH-RISK SCOPE",
                        f"A project is high-risk when {c1} is active and {c2} is confirmed.",
                    ),
                    E(
                        "O",
                        f"{code}_obligation.txt",
                        "MANDATORY RULE",
                        f"For a high-risk project, someone is required to {action}.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "PROJECT STATUS",
                        f"{c1.capitalize()} is active. "
                        f"The record does not establish {c2}.",
                    ),
                ],
                "requirement",
                ["P", "D", "O", "S"],
                "The action is permitted, but whether the mandatory scope applies remains unresolved.",
                ["permission_vs_obligation", "unresolved_scope"],
                [f"{c2} must be confirmed to establish high-risk scope"],
            ),
            C(
                "NO",
                q,
                [
                    E(
                        "P",
                        f"{code}_obligation.txt",
                        "PERMISSION",
                        f"Someone is permitted to {action}.",
                    ),
                    E(
                        "D",
                        f"{code}_obligation.txt",
                        "STANDARD SCOPE",
                        f"A project is standard-risk when {c2} is denied.",
                    ),
                    E(
                        "O",
                        f"{code}_obligation.txt",
                        "MANDATORY RULE",
                        f"The requirement to {action} applies only to high-risk projects, "
                        f"not standard-risk projects.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "PROJECT STATUS",
                        f"{c2.capitalize()} is denied; the project is standard-risk.",
                    ),
                ],
                "requirement",
                ["P", "D", "O", "S"],
                "The action is permitted but is not mandatory for the established scope.",
                ["permission_vs_obligation", "negative_proposition"],
            ),
            C(
                "INSUFFICIENT",
                q,
                [
                    E(
                        "P",
                        f"{code}_obligation.txt",
                        "PERMISSION",
                        f"Someone is permitted to {action}.",
                    ),
                    E(
                        "S",
                        f"{code}_status.txt",
                        "PROJECT STATUS",
                        f"{c1.capitalize()} is active and {c2} is confirmed.",
                    ),
                ],
                "requirement",
                [],
                "Permission to perform the action does not establish whether the action is mandatory.",
                ["permission_vs_obligation", "evidence_boundary"],
            ),
        ],
    )


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


def build(builders, variants):
    result = []

    for builder in builders:
        for variant in variants:
            result.append(
                builder(variant)
            )

    return result


TRAIN = build(
    TRAIN_BUILDERS,
    TRAIN_VARIANTS,
)

VALIDATION = build(
    VALIDATION_BUILDERS,
    VALIDATION_VARIANTS,
)

TEST = build(
    TEST_BUILDERS,
    TEST_VARIANTS,
)

FAMILIES = {
    "train": TRAIN,
    "validation": VALIDATION,
    "test": TEST,
}
