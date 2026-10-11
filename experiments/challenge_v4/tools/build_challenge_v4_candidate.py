from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import random
import re


BASE = Path("/workspace")

SEED = 20260817

CONTRACT = (
    BASE
    / "experiments/challenge_v4/"
      "CHALLENGE_V4_DESIGN_CONTRACT.txt"
)

EXPECTED_CONTRACT_SHA = (
    "51201f70ef4b8299d87f447fd39fb425"
    "a4f6927534816709a74aea276dd2a1ec"
)

OUT = (
    BASE
    / "experiments/challenge_v4/candidate/"
      "sentinel_challenge_v4_candidate.json"
)

HIDDEN = (
    BASE
    / "experiments/challenge_v4/candidate/audit_only/"
      "sentinel_challenge_v4_hidden_semantics_candidate.jsonl"
)

MANIFEST = (
    BASE
    / "experiments/challenge_v4/candidate/"
      "CHALLENGE_V4_CANDIDATE_MANIFEST.json"
)


LABELS = (
    "YES",
    "CONDITIONAL",
    "NO",
    "INSUFFICIENT",
)


INSTRUCTION = """You are Sentinel, a policy-governance decision model.

Use ONLY the supplied evidence.

Classify the exact question as one of:

YES
NO
CONDITIONAL
INSUFFICIENT

Apply these rules:

INSUFFICIENT:
Choose INSUFFICIENT when the supplied evidence does not directly govern
the exact issue asked about. Related evidence is not enough. Do not invent
missing approvals, technologies, durations, thresholds, locations,
recipients, purposes, authorities, statuses, or rules.

NO:
Choose NO when the evidence directly prohibits the exact action or directly
establishes that the proposition in the question is false.

CONDITIONAL:
Choose CONDITIONAL when relevant governing evidence exists and supports
possible permission, but one or more material requirements remain unresolved.

YES:
Choose YES only when relevant governing evidence directly permits the exact
action and every material condition is already established as satisfied.

Silence is not prohibition.
Semantic similarity is not evidence.

Return valid JSON only with exactly these keys:

decision
citations
conditions
rationale
""".strip()


# ====================================================================
# BENCHMARK-LOCAL COMPOSITIONS
#
# These names and compositions are Challenge-V4-local.
# They are not V6 family names.
# ====================================================================

COMPOSITIONS = [
    {
        "name": "charter_scope_resolution",
        "question_type": "relational_counterfactual",
        "mode": "permission_condition",
        "action": "release",
        "adjacent_action": "archive",
        "object": "research package",
        "condition": "charter authorization",
        "purpose": "external replication",
    },
    {
        "name": "custodian_role_binding",
        "question_type": "relational_counterfactual",
        "mode": "scope_membership",
        "action": "transfer",
        "adjacent_action": "catalog",
        "object": "curated dataset",
        "condition": "controlled-class membership",
        "purpose": "collaborative analysis",
    },
    {
        "name": "amendment_route_control",
        "question_type": "relational_counterfactual",
        "mode": "precedence_current_rule",
        "action": "send",
        "adjacent_action": "retain",
        "object": "prototype record",
        "condition": "route authorization",
        "purpose": "technical validation",
    },
    {
        "name": "exception_trigger_resolution",
        "question_type": "relational_counterfactual",
        "mode": "exception_activation",
        "action": "disclose",
        "adjacent_action": "index",
        "object": "study extract",
        "condition": "exception trigger",
        "purpose": "independent review",
    },
    {
        "name": "delegated_signoff_chain",
        "question_type": "relational_counterfactual",
        "mode": "delegated_authority",
        "action": "export",
        "adjacent_action": "store",
        "object": "analysis bundle",
        "condition": "delegated approval",
        "purpose": "cross-site research",
    },
    {
        "name": "defined_class_release",
        "question_type": "relational_counterfactual",
        "mode": "definition_binding",
        "action": "publish",
        "adjacent_action": "summarize",
        "object": "derived corpus",
        "condition": "defined-class qualification",
        "purpose": "research dissemination",
    },
    {
        "name": "temporal_authorization_window",
        "question_type": "relational_counterfactual",
        "mode": "temporal_validity",
        "action": "provide",
        "adjacent_action": "prepare",
        "object": "evaluation artifact",
        "condition": "active authorization window",
        "purpose": "external assessment",
    },
    {
        "name": "workflow_completion_gate",
        "question_type": "relational_counterfactual",
        "mode": "workflow_dependency",
        "action": "forward",
        "adjacent_action": "stage",
        "object": "research submission",
        "condition": "workflow completion",
        "purpose": "partner review",
    },
    {
        "name": "destination_scope_reconciliation",
        "question_type": "relational_counterfactual",
        "mode": "scope_membership",
        "action": "route",
        "adjacent_action": "register",
        "object": "restricted collection",
        "condition": "approved-destination membership",
        "purpose": "joint investigation",
    },
    {
        "name": "specific_revision_override",
        "question_type": "relational_counterfactual",
        "mode": "precedence_current_rule",
        "action": "deliver",
        "adjacent_action": "preserve",
        "object": "technical archive",
        "condition": "current-revision authorization",
        "purpose": "sponsor verification",
    },
    {
        "name": "exception_authority_bridge",
        "question_type": "relational_counterfactual",
        "mode": "exception_activation",
        "action": "share",
        "adjacent_action": "classify",
        "object": "measurement set",
        "condition": "exception authorization",
        "purpose": "method verification",
    },
    {
        "name": "multi_stage_release_basis",
        "question_type": "relational_counterfactual",
        "mode": "workflow_dependency",
        "action": "release",
        "adjacent_action": "queue",
        "object": "research output",
        "condition": "release-workflow completion",
        "purpose": "external validation",
    },

    # ------------------------------------------------------------
    # Stress compositions
    # ------------------------------------------------------------

    {
        "name": "dense_distractor_authority",
        "question_type": "stress",
        "mode": "delegated_authority",
        "stress": "dense_distractor",
        "action": "transmit",
        "adjacent_action": "inventory",
        "object": "experimental package",
        "condition": "authorized-reviewer approval",
        "purpose": "outside consultation",
    },
    {
        "name": "near_match_scope_boundary",
        "question_type": "stress",
        "mode": "scope_membership",
        "stress": "near_match",
        "action": "release",
        "adjacent_action": "release internally",
        "object": "research dataset",
        "condition": "destination-class qualification",
        "purpose": "external collaboration",
    },
    {
        "name": "negation_precedence_trap",
        "question_type": "stress",
        "mode": "precedence_current_rule",
        "stress": "negation",
        "action": "transfer",
        "adjacent_action": "retain",
        "object": "investigation file",
        "condition": "current-rule authorization",
        "purpose": "independent evaluation",
    },
    {
        "name": "long_chain_evidence_sufficiency",
        "question_type": "stress",
        "mode": "workflow_dependency",
        "stress": "long_chain",
        "action": "provide",
        "adjacent_action": "prepare",
        "object": "research record",
        "condition": "multi-stage clearance",
        "purpose": "external audit",
    },
]


ACTORS = [
    "Aster Lab",
    "Beacon Group",
    "Cobalt Center",
    "Drift Institute",
    "Ember Lab",
    "Fjord Group",
    "Grove Center",
    "Harbor Institute",
    "Ion Lab",
    "Juniper Group",
    "Kestrel Center",
    "Lumen Institute",
    "Mesa Lab",
    "Nimbus Group",
    "Orchid Center",
    "Pine Institute",
    "Quartz Lab",
    "Ridge Group",
    "Solace Center",
    "Tundra Institute",
    "Umber Lab",
    "Vale Group",
    "Willow Center",
    "Xenon Institute",
    "Yarrow Lab",
    "Zephyr Group",
    "Axiom Center",
    "Birch Institute",
    "Cirrus Lab",
    "Delta Group",
    "Elm Center",
    "Flint Institute",
]

ASSETS = [
    "Atlas collection",
    "Boreal corpus",
    "Cinder archive",
    "Dawn dataset",
    "Echo records",
    "Fable package",
    "Glacier files",
    "Helix collection",
    "Indigo corpus",
    "Jade archive",
    "Keystone dataset",
    "Lantern records",
    "Monarch package",
    "Nova files",
    "Opal collection",
    "Prairie corpus",
]

ROUTES = [
    "Arbor Gateway",
    "Bluebird Exchange",
    "Canyon Portal",
    "Drift Channel",
    "Evergreen Gateway",
    "Falcon Exchange",
    "Granite Portal",
    "Horizon Channel",
    "Ivory Gateway",
    "Juniper Exchange",
    "Keystone Portal",
    "Lantern Channel",
]

PURPOSES = [
    "replication review",
    "method comparison",
    "external validation",
    "joint analysis",
    "technical verification",
    "independent assessment",
    "cross-site study",
    "partner evaluation",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


def slug(text: str) -> str:
    x = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return x[:48]


def verify_contract():
    actual = sha(CONTRACT)
    if actual != EXPECTED_CONTRACT_SHA:
        raise SystemExit(
            "Challenge V4 contract identity mismatch:\n"
            f"expected={EXPECTED_CONTRACT_SHA}\n"
            f"actual={actual}"
        )


def context_for(comp_idx, group_idx, target_idx):
    actor = ACTORS[
        (comp_idx * 7 + group_idx * 3 + target_idx * 5)
        % len(ACTORS)
    ]

    asset = ASSETS[
        (comp_idx * 5 + group_idx * 7 + target_idx * 3)
        % len(ASSETS)
    ]

    route = ROUTES[
        (comp_idx * 3 + group_idx * 5 + target_idx * 7)
        % len(ROUTES)
    ]

    purpose = PURPOSES[
        (comp_idx * 2 + group_idx * 3 + target_idx)
        % len(PURPOSES)
    ]

    return {
        "actor": actor,
        "asset": asset,
        "route": route,
        "purpose": purpose,
        "target_key": slug(f"{actor}_{asset}_{route}"),
    }


def question_for(comp, ctx):
    return (
        f"Under the supplied evidence, may {ctx['actor']} "
        f"{comp['action']} the {ctx['asset']} through "
        f"{ctx['route']} for {ctx['purpose']}?"
    )


def permission_condition(comp, state, ctx):
    sid1 = f"{ctx['target_key']}_governance.txt"
    sid2 = f"{ctx['target_key']}_status.txt"

    action = comp["action"]
    adjacent = comp["adjacent_action"]
    condition = comp["condition"]

    if state == "YES":
        docs = [
            (
                sid1,
                f"{ctx['actor']} may {action} the {ctx['asset']} through "
                f"{ctx['route']} for {ctx['purpose']} when {condition} is established."
            ),
            (
                sid2,
                f"For the {ctx['asset']} and {ctx['route']}, the record states that "
                f"{condition} is established for {ctx['actor']}."
            ),
        ]
        citations = [sid1, sid2]
        conditions = []
        rationale = (
            "The governing permission applies to the exact action, asset, route, "
            "and purpose, and its material condition is established."
        )

    elif state == "CONDITIONAL":
        docs = [
            (
                sid1,
                f"{ctx['actor']} may {action} the {ctx['asset']} through "
                f"{ctx['route']} for {ctx['purpose']} when {condition} is established."
            ),
            (
                sid2,
                f"The supplied record for {ctx['actor']} does not establish whether "
                f"{condition} has been completed for the {ctx['asset']}."
            ),
        ]
        citations = [sid1, sid2]
        conditions = [condition]
        rationale = (
            "A governing permission exists, but its material prerequisite remains unresolved."
        )

    elif state == "NO":
        docs = [
            (
                sid1,
                f"{ctx['actor']} must not {action} the {ctx['asset']} through "
                f"{ctx['route']} for {ctx['purpose']}."
            ),
            (
                sid2,
                f"The status record identifies the {ctx['asset']} and "
                f"{ctx['route']} as the active request for {ctx['actor']}."
            ),
        ]
        citations = [sid1]
        conditions = []
        rationale = (
            "The supplied evidence directly prohibits the exact requested action."
        )

    else:
        docs = [
            (
                sid1,
                f"{ctx['actor']} may {adjacent} the {ctx['asset']} through "
                f"{ctx['route']} for {ctx['purpose']} when {condition} is established."
            ),
            (
                sid2,
                f"For the {ctx['asset']}, the record states that {condition} is "
                f"established for {ctx['actor']}."
            ),
        ]
        citations = [sid1]
        conditions = []
        rationale = (
            "The supplied rule governs a different action and does not directly govern "
            "the exact action asked about."
        )

    return docs, citations, conditions, rationale


def scope_membership(comp, state, ctx):
    sid1 = f"{ctx['target_key']}_scope_policy.txt"
    sid2 = f"{ctx['target_key']}_classification.txt"

    action = comp["action"]
    adjacent = comp["adjacent_action"]
    condition = comp["condition"]

    scope_name = f"{ctx['route']} approved class"
    blocked_name = f"{ctx['route']} restricted class"

    if state == "YES":
        docs = [
            (
                sid1,
                f"Members of the {scope_name} may {action} records through "
                f"{ctx['route']} for {ctx['purpose']}."
            ),
            (
                sid2,
                f"The {ctx['asset']} held by {ctx['actor']} is classified as a "
                f"member of the {scope_name}; {condition} is complete."
            ),
        ]
        citations = [sid1, sid2]
        conditions = []
        rationale = (
            "The target is established to fall within the class directly permitted "
            "by the governing rule."
        )

    elif state == "CONDITIONAL":
        docs = [
            (
                sid1,
                f"Members of the {scope_name} may {action} records through "
                f"{ctx['route']} for {ctx['purpose']}."
            ),
            (
                sid2,
                f"The classification record for the {ctx['asset']} does not establish "
                f"whether it is a member of the {scope_name}; {condition} remains unresolved."
            ),
        ]
        citations = [sid1, sid2]
        conditions = [condition]
        rationale = (
            "The permission is applicable if the required class membership is established, "
            "but that membership remains unresolved."
        )

    elif state == "NO":
        docs = [
            (
                sid1,
                f"Members of the {blocked_name} must not {action} records through "
                f"{ctx['route']} for {ctx['purpose']}."
            ),
            (
                sid2,
                f"The {ctx['asset']} held by {ctx['actor']} is classified as a "
                f"member of the {blocked_name}."
            ),
        ]
        citations = [sid1, sid2]
        conditions = []
        rationale = (
            "The target is established to belong to a class for which the exact action "
            "is directly prohibited."
        )

    else:
        docs = [
            (
                sid1,
                f"Members of the {scope_name} may {adjacent} records through "
                f"{ctx['route']} for {ctx['purpose']}."
            ),
            (
                sid2,
                f"The {ctx['asset']} held by {ctx['actor']} is a member of the "
                f"{scope_name}."
            ),
        ]
        citations = [sid1]
        conditions = []
        rationale = (
            "The scope evidence concerns a different action and therefore does not "
            "govern the exact action asked about."
        )

    return docs, citations, conditions, rationale


def precedence_current_rule(comp, state, ctx):
    old_sid = f"{ctx['target_key']}_former_rule.txt"
    cur_sid = f"{ctx['target_key']}_current_rule.txt"
    stat_sid = f"{ctx['target_key']}_route_status.txt"

    action = comp["action"]
    adjacent = comp["adjacent_action"]
    condition = comp["condition"]

    if state == "YES":
        docs = [
            (
                old_sid,
                f"A superseded rule formerly prohibited {ctx['actor']} from "
                f"{action}ing the {ctx['asset']} through {ctx['route']}."
            ),
            (
                cur_sid,
                f"The current rule replaces the former rule and permits {ctx['actor']} "
                f"to {action} the {ctx['asset']} through {ctx['route']} for "
                f"{ctx['purpose']} when {condition} is established."
            ),
            (
                stat_sid,
                f"The current route record establishes {condition} for "
                f"{ctx['actor']} and the {ctx['asset']}."
            ),
        ]
        citations = [cur_sid, stat_sid]
        conditions = []
        rationale = (
            "The operative current rule permits the exact action and the required "
            "current-state condition is satisfied."
        )

    elif state == "CONDITIONAL":
        docs = [
            (
                old_sid,
                f"A superseded rule formerly prohibited {ctx['actor']} from "
                f"{action}ing the {ctx['asset']} through {ctx['route']}."
            ),
            (
                cur_sid,
                f"The current rule replaces the former rule and permits {ctx['actor']} "
                f"to {action} the {ctx['asset']} through {ctx['route']} for "
                f"{ctx['purpose']} when {condition} is established."
            ),
            (
                stat_sid,
                f"The supplied route record does not establish whether "
                f"{condition} is satisfied for the {ctx['asset']}."
            ),
        ]
        citations = [cur_sid, stat_sid]
        conditions = [condition]
        rationale = (
            "The operative rule permits the action, but a material requirement remains unresolved."
        )

    elif state == "NO":
        docs = [
            (
                old_sid,
                f"A superseded rule formerly permitted {ctx['actor']} to "
                f"{action} the {ctx['asset']} through {ctx['route']}."
            ),
            (
                cur_sid,
                f"The current rule replaces the former rule and prohibits "
                f"{ctx['actor']} from {action}ing the {ctx['asset']} through "
                f"{ctx['route']} for {ctx['purpose']}."
            ),
            (
                stat_sid,
                f"The route record identifies {ctx['route']} as the route requested "
                f"for the {ctx['asset']}."
            ),
        ]
        citations = [cur_sid]
        conditions = []
        rationale = (
            "The operative current rule directly prohibits the exact requested action."
        )

    else:
        docs = [
            (
                old_sid,
                f"A superseded rule concerned whether {ctx['actor']} could "
                f"{adjacent} the {ctx['asset']}."
            ),
            (
                cur_sid,
                f"The current revision permits {ctx['actor']} to {adjacent} the "
                f"{ctx['asset']} through {ctx['route']} when {condition} is established."
            ),
            (
                stat_sid,
                f"The record establishes {condition} for the adjacent activity."
            ),
        ]
        citations = [cur_sid]
        conditions = []
        rationale = (
            "The current-rule evidence governs a different activity and does not establish "
            "a rule for the exact requested action."
        )

    return docs, citations, conditions, rationale


def exception_activation(comp, state, ctx):
    base_sid = f"{ctx['target_key']}_base_rule.txt"
    exc_sid = f"{ctx['target_key']}_exception.txt"
    stat_sid = f"{ctx['target_key']}_exception_status.txt"

    action = comp["action"]
    adjacent = comp["adjacent_action"]
    condition = comp["condition"]

    if state == "YES":
        docs = [
            (
                base_sid,
                f"The general rule prohibits {ctx['actor']} from {action}ing the "
                f"{ctx['asset']} through {ctx['route']}."
            ),
            (
                exc_sid,
                f"An explicit exception overrides that prohibition for "
                f"{ctx['purpose']} when {condition} is established."
            ),
            (
                stat_sid,
                f"For this request, {condition} is established and the stated purpose "
                f"is {ctx['purpose']}."
            ),
        ]
        citations = [base_sid, exc_sid, stat_sid]
        conditions = []
        rationale = (
            "An applicable explicit exception overrides the general prohibition and "
            "its trigger is established."
        )

    elif state == "CONDITIONAL":
        docs = [
            (
                base_sid,
                f"The general rule prohibits {ctx['actor']} from {action}ing the "
                f"{ctx['asset']} through {ctx['route']}."
            ),
            (
                exc_sid,
                f"An explicit exception overrides that prohibition for "
                f"{ctx['purpose']} when {condition} is established."
            ),
            (
                stat_sid,
                f"The request identifies the purpose as {ctx['purpose']}, but the "
                f"supplied record does not establish whether {condition} is satisfied."
            ),
        ]
        citations = [base_sid, exc_sid, stat_sid]
        conditions = [condition]
        rationale = (
            "The exception could permit the action, but its material activation condition "
            "remains unresolved."
        )

    elif state == "NO":
        docs = [
            (
                base_sid,
                f"The governing rule prohibits {ctx['actor']} from {action}ing the "
                f"{ctx['asset']} through {ctx['route']} for {ctx['purpose']}."
            ),
            (
                exc_sid,
                f"The listed exception applies only to internal calibration activity "
                f"and does not override the prohibition for {ctx['purpose']}."
            ),
            (
                stat_sid,
                f"The request is for {ctx['purpose']}."
            ),
        ]
        citations = [base_sid, exc_sid]
        conditions = []
        rationale = (
            "The exact action is prohibited and the stated exception does not apply."
        )

    else:
        docs = [
            (
                base_sid,
                f"The general rule prohibits {ctx['actor']} from {adjacent}ing the "
                f"{ctx['asset']} through {ctx['route']}."
            ),
            (
                exc_sid,
                f"An exception permits that adjacent activity when {condition} is established."
            ),
            (
                stat_sid,
                f"The record establishes {condition} for the adjacent activity."
            ),
        ]
        citations = [base_sid, exc_sid]
        conditions = []
        rationale = (
            "The prohibition and exception govern a different activity, leaving the exact "
            "action in the question ungoverned."
        )

    return docs, citations, conditions, rationale


def delegated_authority(comp, state, ctx):
    deleg_sid = f"{ctx['target_key']}_delegation.txt"
    decision_sid = f"{ctx['target_key']}_reviewer_decision.txt"
    scope_sid = f"{ctx['target_key']}_delegation_scope.txt"

    action = comp["action"]
    adjacent = comp["adjacent_action"]
    condition = comp["condition"]

    reviewer = f"{ctx['route']} Research Steward"

    if state == "YES":
        docs = [
            (
                deleg_sid,
                f"The governance charter delegates authority to the {reviewer} to approve "
                f"{ctx['actor']} to {action} the {ctx['asset']} through {ctx['route']} "
                f"for {ctx['purpose']}."
            ),
            (
                decision_sid,
                f"The {reviewer} approved this exact request; {condition} is recorded as complete."
            ),
            (
                scope_sid,
                f"The delegation covers the {ctx['asset']}, {ctx['route']}, and "
                f"{ctx['purpose']}."
            ),
        ]
        citations = [deleg_sid, decision_sid, scope_sid]
        conditions = []
        rationale = (
            "The authorized delegate has authority over the exact request and has approved it."
        )

    elif state == "CONDITIONAL":
        docs = [
            (
                deleg_sid,
                f"The governance charter delegates authority to the {reviewer} to approve "
                f"{ctx['actor']} to {action} the {ctx['asset']} through {ctx['route']} "
                f"for {ctx['purpose']}."
            ),
            (
                decision_sid,
                f"No completed decision by the {reviewer} is established in the supplied record."
            ),
            (
                scope_sid,
                f"The delegation covers the {ctx['asset']}, {ctx['route']}, and "
                f"{ctx['purpose']}."
            ),
        ]
        citations = [deleg_sid, decision_sid, scope_sid]
        conditions = [condition]
        rationale = (
            "The delegated authority exists and covers the request, but the required "
            "delegate approval remains unresolved."
        )

    elif state == "NO":
        docs = [
            (
                deleg_sid,
                f"The governance charter assigns the {reviewer} authority over the "
                f"requested transfer."
            ),
            (
                decision_sid,
                f"The {reviewer} denied the request and the charter states that a denied "
                f"request must not proceed."
            ),
            (
                scope_sid,
                f"The denial covers the {ctx['asset']}, {ctx['route']}, and "
                f"{ctx['purpose']} requested by {ctx['actor']}."
            ),
        ]
        citations = [deleg_sid, decision_sid, scope_sid]
        conditions = []
        rationale = (
            "The authorized decision-maker denied the exact request and the governing rule "
            "prohibits proceeding after denial."
        )

    else:
        docs = [
            (
                deleg_sid,
                f"The charter delegates authority to the {reviewer} to approve "
                f"{ctx['actor']} to {adjacent} the {ctx['asset']}."
            ),
            (
                decision_sid,
                f"The {reviewer} approved the adjacent activity."
            ),
            (
                scope_sid,
                f"The delegation scope describes the adjacent activity but does not state "
                f"authority over {comp['action']} through {ctx['route']}."
            ),
        ]
        citations = [deleg_sid, scope_sid]
        conditions = []
        rationale = (
            "The delegation governs another activity and does not establish authority "
            "over the exact requested action."
        )

    return docs, citations, conditions, rationale


def definition_binding(comp, state, ctx):
    def_sid = f"{ctx['target_key']}_definitions.txt"
    rule_sid = f"{ctx['target_key']}_class_rule.txt"
    fact_sid = f"{ctx['target_key']}_classification_fact.txt"

    action = comp["action"]
    adjacent = comp["adjacent_action"]
    condition = comp["condition"]

    permitted_class = "Verified Derived Material"
    blocked_class = "Restricted Source Material"

    if state == "YES":
        docs = [
            (
                def_sid,
                f'"{permitted_class}" means a derived research object whose source-level '
                f"identifiers have been removed and whose derivation review is complete."
            ),
            (
                rule_sid,
                f"{permitted_class} may be {action}ed through {ctx['route']} for "
                f"{ctx['purpose']}."
            ),
            (
                fact_sid,
                f"The {ctx['asset']} held by {ctx['actor']} satisfies the definition of "
                f"{permitted_class}; {condition} is complete."
            ),
        ]
        citations = [def_sid, rule_sid, fact_sid]
        conditions = []
        rationale = (
            "The evidence binds the target to the class directly permitted by the governing rule."
        )

    elif state == "CONDITIONAL":
        docs = [
            (
                def_sid,
                f'"{permitted_class}" means a derived research object whose source-level '
                f"identifiers have been removed and whose derivation review is complete."
            ),
            (
                rule_sid,
                f"{permitted_class} may be {action}ed through {ctx['route']} for "
                f"{ctx['purpose']}."
            ),
            (
                fact_sid,
                f"The record does not establish whether the {ctx['asset']} satisfies all "
                f"elements of the {permitted_class} definition; {condition} remains unresolved."
            ),
        ]
        citations = [def_sid, rule_sid, fact_sid]
        conditions = [condition]
        rationale = (
            "Permission depends on membership in a defined class, but membership remains unresolved."
        )

    elif state == "NO":
        docs = [
            (
                def_sid,
                f'"{blocked_class}" means source material retaining restricted source-level fields.'
            ),
            (
                rule_sid,
                f"{blocked_class} must not be {action}ed through {ctx['route']} for "
                f"{ctx['purpose']}."
            ),
            (
                fact_sid,
                f"The {ctx['asset']} held by {ctx['actor']} satisfies the definition of "
                f"{blocked_class}."
            ),
        ]
        citations = [def_sid, rule_sid, fact_sid]
        conditions = []
        rationale = (
            "The evidence establishes that the target belongs to a class for which the "
            "exact action is prohibited."
        )

    else:
        docs = [
            (
                def_sid,
                f'"{permitted_class}" means a derived research object whose derivation review is complete.'
            ),
            (
                rule_sid,
                f"{permitted_class} may be {adjacent}ed through {ctx['route']}."
            ),
            (
                fact_sid,
                f"The {ctx['asset']} held by {ctx['actor']} satisfies that definition."
            ),
        ]
        citations = [def_sid, rule_sid]
        conditions = []
        rationale = (
            "The definition and rule govern a different activity and do not govern the "
            "exact action asked about."
        )

    return docs, citations, conditions, rationale


def temporal_validity(comp, state, ctx):
    rule_sid = f"{ctx['target_key']}_temporal_rule.txt"
    status_sid = f"{ctx['target_key']}_authorization_status.txt"

    action = comp["action"]
    adjacent = comp["adjacent_action"]
    condition = comp["condition"]

    if state == "YES":
        docs = [
            (
                rule_sid,
                f"{ctx['actor']} may {action} the {ctx['asset']} through {ctx['route']} "
                f"for {ctx['purpose']} while its authorization window is active."
            ),
            (
                status_sid,
                f"The authorization window for this exact request is active; "
                f"{condition} is established."
            ),
        ]
        citations = [rule_sid, status_sid]
        conditions = []
        rationale = (
            "The exact request is permitted during an active authorization window, "
            "which the evidence establishes."
        )

    elif state == "CONDITIONAL":
        docs = [
            (
                rule_sid,
                f"{ctx['actor']} may {action} the {ctx['asset']} through {ctx['route']} "
                f"for {ctx['purpose']} while its authorization window is active."
            ),
            (
                status_sid,
                f"The supplied status record does not establish whether the authorization "
                f"window for this exact request is currently active."
            ),
        ]
        citations = [rule_sid, status_sid]
        conditions = [condition]
        rationale = (
            "A governing permission exists, but current temporal validity remains unresolved."
        )

    elif state == "NO":
        docs = [
            (
                rule_sid,
                f"{ctx['actor']} must not {action} the {ctx['asset']} through "
                f"{ctx['route']} after the authorization window expires."
            ),
            (
                status_sid,
                f"The authorization window for the exact request has expired."
            ),
        ]
        citations = [rule_sid, status_sid]
        conditions = []
        rationale = (
            "The governing rule prohibits the action after expiration and the evidence "
            "establishes that expiration occurred."
        )

    else:
        docs = [
            (
                rule_sid,
                f"{ctx['actor']} may {adjacent} the {ctx['asset']} while an authorization "
                f"window is active."
            ),
            (
                status_sid,
                f"The authorization window for that adjacent activity is active."
            ),
        ]
        citations = [rule_sid]
        conditions = []
        rationale = (
            "The temporal rule governs another activity rather than the exact action asked about."
        )

    return docs, citations, conditions, rationale


def workflow_dependency(comp, state, ctx):
    rule_sid = f"{ctx['target_key']}_workflow_rule.txt"
    stage_sid = f"{ctx['target_key']}_workflow_stage.txt"
    release_sid = f"{ctx['target_key']}_release_status.txt"

    action = comp["action"]
    adjacent = comp["adjacent_action"]
    condition = comp["condition"]

    if state == "YES":
        docs = [
            (
                rule_sid,
                f"{ctx['actor']} may {action} the {ctx['asset']} through {ctx['route']} "
                f"for {ctx['purpose']} only after intake review and release review are complete."
            ),
            (
                stage_sid,
                f"Intake review for the {ctx['asset']} is complete."
            ),
            (
                release_sid,
                f"Release review is complete; {condition} is established for the exact request."
            ),
        ]
        citations = [rule_sid, stage_sid, release_sid]
        conditions = []
        rationale = (
            "Every workflow prerequisite required by the governing permission is established."
        )

    elif state == "CONDITIONAL":
        docs = [
            (
                rule_sid,
                f"{ctx['actor']} may {action} the {ctx['asset']} through {ctx['route']} "
                f"for {ctx['purpose']} only after intake review and release review are complete."
            ),
            (
                stage_sid,
                f"Intake review for the {ctx['asset']} is complete."
            ),
            (
                release_sid,
                f"The supplied record does not establish completion of release review."
            ),
        ]
        citations = [rule_sid, stage_sid, release_sid]
        conditions = ["release review completion"]
        rationale = (
            "A governing permission exists, but one required workflow stage remains unresolved."
        )

    elif state == "NO":
        docs = [
            (
                rule_sid,
                f"If a release review records a blocking finding, {ctx['actor']} must not "
                f"{action} the {ctx['asset']} through {ctx['route']} for {ctx['purpose']}."
            ),
            (
                stage_sid,
                f"Intake review for the {ctx['asset']} is complete."
            ),
            (
                release_sid,
                f"Release review records a blocking finding for this exact request."
            ),
        ]
        citations = [rule_sid, release_sid]
        conditions = []
        rationale = (
            "The workflow rule directly prohibits proceeding when the blocking state is present, "
            "and that state is established."
        )

    else:
        docs = [
            (
                rule_sid,
                f"{ctx['actor']} may {adjacent} the {ctx['asset']} after intake review "
                f"and staging review are complete."
            ),
            (
                stage_sid,
                f"Intake review is complete."
            ),
            (
                release_sid,
                f"Staging review is complete for the adjacent activity."
            ),
        ]
        citations = [rule_sid]
        conditions = []
        rationale = (
            "The workflow governs a different activity and does not establish a rule "
            "for the exact requested action."
        )

    return docs, citations, conditions, rationale


RENDERERS = {
    "permission_condition": permission_condition,
    "scope_membership": scope_membership,
    "precedence_current_rule": precedence_current_rule,
    "exception_activation": exception_activation,
    "delegated_authority": delegated_authority,
    "definition_binding": definition_binding,
    "temporal_validity": temporal_validity,
    "workflow_dependency": workflow_dependency,
}


def stress_distractors(comp, contexts, stress):
    if not stress:
        return []

    target = contexts[0]

    common = [
        (
            f"{slug(comp['name'])}_inventory_policy.txt",
            f"Research assets routed through institutional systems must remain listed "
            f"in the central inventory. This policy does not define release authority."
        ),
        (
            f"{slug(comp['name'])}_account_security.txt",
            f"Remote access to research systems requires multifactor authentication. "
            f"Authentication status does not establish permission for the requested research action."
        ),
    ]

    if stress == "dense_distractor":
        common.extend([
            (
                f"{slug(comp['name'])}_retention.txt",
                f"The {target['asset']} may be subject to retention scheduling after project closure."
            ),
            (
                f"{slug(comp['name'])}_accessibility.txt",
                "Published institutional web resources must satisfy accessibility requirements."
            ),
            (
                f"{slug(comp['name'])}_device.txt",
                "Institution-owned workstations must remain in the device inventory."
            ),
            (
                f"{slug(comp['name'])}_training.txt",
                "Annual security training is required for remote institutional accounts."
            ),
        ])

    elif stress == "near_match":
        common.extend([
            (
                f"{slug(comp['name'])}_near_route.txt",
                f"A policy for a similarly named internal route permits internal handling "
                f"but does not define authority for the external route in the question."
            ),
            (
                f"{slug(comp['name'])}_near_purpose.txt",
                f"Internal replication support and external collaboration are distinct purposes "
                f"under the routing glossary."
            ),
        ])

    elif stress == "negation":
        common.extend([
            (
                f"{slug(comp['name'])}_negation_note.txt",
                "The absence of an inventory annotation does not itself revoke an otherwise valid authorization."
            ),
            (
                f"{slug(comp['name'])}_non_equivalence.txt",
                "A rule that is not current is not treated as operative merely because its text remains archived."
            ),
        ])

    elif stress == "long_chain":
        common.extend([
            (
                f"{slug(comp['name'])}_dependency_glossary.txt",
                "Workflow completion and record retention are separate governance concepts."
            ),
            (
                f"{slug(comp['name'])}_routing_glossary.txt",
                "A routing record identifies a destination but does not by itself establish release authority."
            ),
            (
                f"{slug(comp['name'])}_review_glossary.txt",
                "An intake review and a release review are distinct workflow stages."
            ),
        ])

    return common


def make_prompt(evidence, question, input_order):
    evidence_text = "\n\n".join(
        f"[{d['source_id']}]\n{d['text']}"
        for d in evidence
    )

    if input_order == "evidence_first":
        return (
            "EVIDENCE\n"
            "========\n"
            f"{evidence_text}\n\n"
            "QUESTION\n"
            "========\n"
            f"{question}"
        )

    return (
        "QUESTION\n"
        "========\n"
        f"{question}\n\n"
        "EVIDENCE\n"
        "========\n"
        f"{evidence_text}"
    )


def build():
    verify_contract()

    if OUT.exists() or HIDDEN.exists() or MANIFEST.exists():
        raise SystemExit(
            "Candidate outputs already exist. Refusing to overwrite.\n"
            "Remove them explicitly only if this candidate has NOT been consumed."
        )

    public_cases = []
    hidden_rows = []

    global_group_index = 0
    case_serial = 0

    for comp_idx, comp in enumerate(COMPOSITIONS):
        renderer = RENDERERS[comp["mode"]]

        for group_idx in range(5):
            relational_system = (
                f"ch4::{comp['name']}::system_{group_idx + 1:02d}"
            )

            # --------------------------------------------------------
            # Coordinate schedule:
            #
            # Across 80 relational systems:
            # every (target_offset, label_shift) pair occurs 5 times.
            #
            # This forces target/world coordinate-majority accuracy
            # to exactly 0.25 globally.
            # --------------------------------------------------------

            target_offset = global_group_index % 4
            label_shift = (global_group_index // 4) % 4

            contexts = [
                context_for(comp_idx, group_idx, target_idx)
                for target_idx in range(4)
            ]

            for world_index in range(4):
                target_index = (world_index + target_offset) % 4

                evidence = []
                annotations = {}

                # Every world contains four target tracks.
                # The state rotation gives one instance of every label
                # in the evidence pack, preventing evidence-only label
                # prediction from simply reading global pack polarity.
                for t_idx, ctx in enumerate(contexts):
                    state_index = (
                        t_idx
                        + 2 * world_index
                        + label_shift
                    ) % 4

                    state = LABELS[state_index]

                    docs, citations, conditions, rationale = renderer(
                        comp,
                        state,
                        ctx,
                    )

                    for source_id, text in docs:
                        evidence.append({
                            "source_id": source_id,
                            "text": text,
                        })

                    annotations[t_idx] = {
                        "state": state,
                        "decision": state,
                        "citations": citations,
                        "conditions": conditions,
                        "rationale": rationale,
                    }

                extra = stress_distractors(
                    comp,
                    contexts,
                    comp.get("stress"),
                )

                for source_id, text in extra:
                    evidence.append({
                        "source_id": source_id,
                        "text": text,
                    })

                case_serial += 1

                case_id = f"CH4-{case_serial:04d}"

                # Deterministic but case-specific evidence shuffle.
                local_rng = random.Random(
                    SEED
                    + comp_idx * 100_000
                    + group_idx * 10_000
                    + world_index * 101
                )

                local_rng.shuffle(evidence)

                query_ctx = contexts[target_index]
                question = question_for(comp, query_ctx)

                gold_annotation = annotations[target_index]
                gold = gold_annotation["decision"]

                input_order = (
                    "evidence_first"
                    if case_serial % 2
                    else "question_first"
                )

                user_prompt = make_prompt(
                    evidence,
                    question,
                    input_order,
                )

                public_cases.append({
                    "case_id": case_id,
                    "question_type": comp["question_type"],
                    "composition": comp["name"],
                    "input_order": input_order,
                    "evidence": evidence,
                    "question": question,
                    "messages": [
                        {
                            "role": "system",
                            "content": INSTRUCTION,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    "gold": gold,
                })

                hidden_rows.append({
                    "case_id": case_id,
                    "composition": comp["name"],
                    "question_type": comp["question_type"],
                    "relational_system": relational_system,
                    "world_index": world_index,
                    "target_index": target_index,
                    "target_id": (
                        f"{relational_system}::target_{target_index}"
                    ),
                    "target_offset": target_offset,
                    "label_shift": label_shift,
                    "oracle_mode": comp["mode"],
                    "semantic_state": gold,
                    "gold_annotation": gold_annotation,
                })

            global_group_index += 1

    # ------------------------------------------------------------
    # Builder-local hard gates
    # ------------------------------------------------------------

    if len(public_cases) != 320:
        raise AssertionError(len(public_cases))

    counts = Counter(x["gold"] for x in public_cases)

    if counts != Counter({
        "YES": 80,
        "NO": 80,
        "CONDITIONAL": 80,
        "INSUFFICIENT": 80,
    }):
        raise AssertionError(counts)

    qtypes = Counter(
        x["question_type"]
        for x in public_cases
    )

    if qtypes != Counter({
        "relational_counterfactual": 240,
        "stress": 80,
    }):
        raise AssertionError(qtypes)

    comp_counts = Counter(
        x["composition"]
        for x in public_cases
    )

    if any(v != 20 for v in comp_counts.values()):
        raise AssertionError(comp_counts)

    ids = [x["case_id"] for x in public_cases]

    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate public case ids")

    hidden_ids = [x["case_id"] for x in hidden_rows]

    if ids != hidden_ids:
        raise AssertionError(
            "public/hidden identity ordering mismatch"
        )

    payload = {
        "benchmark": "Sentinel Challenge V4",
        "version": "candidate_v1",
        "status": "NON_CANONICAL_UNCONSUMED",
        "seed": SEED,
        "contract": str(CONTRACT.relative_to(BASE)),
        "contract_sha256": EXPECTED_CONTRACT_SHA,
        "case_count": len(public_cases),
        "cases": public_cases,
    }

    OUT.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with HIDDEN.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in hidden_rows:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "benchmark": "Sentinel Challenge V4",
        "candidate_version": 1,
        "status": "NON_CANONICAL_UNCONSUMED",
        "seed": SEED,
        "case_count": len(public_cases),
        "label_counts": dict(sorted(counts.items())),
        "question_type_counts": dict(sorted(qtypes.items())),
        "composition_counts": dict(sorted(comp_counts.items())),
        "relational_system_count": global_group_index,
        "public_candidate": {
            "path": str(OUT.relative_to(BASE)),
            "sha256": sha(OUT),
        },
        "hidden_candidate": {
            "path": str(HIDDEN.relative_to(BASE)),
            "sha256": sha(HIDDEN),
        },
        "contract": {
            "path": str(CONTRACT.relative_to(BASE)),
            "sha256": sha(CONTRACT),
        },
        "model_loads": 0,
        "inference_calls": 0,
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("============================================================")
    print("CHALLENGE V4 CANDIDATE GENERATED")
    print("============================================================")
    print("Cases:", len(public_cases))
    print("Relational systems:", global_group_index)
    print("Labels:", dict(sorted(counts.items())))
    print("Question types:", dict(sorted(qtypes.items())))
    print("Compositions:", len(comp_counts))
    print("Each composition:", sorted(set(comp_counts.values())))
    print()
    print("Public candidate SHA256:")
    print(" ", sha(OUT))
    print()
    print("Hidden semantics candidate SHA256:")
    print(" ", sha(HIDDEN))
    print()
    print("Manifest SHA256:")
    print(" ", sha(MANIFEST))
    print()
    print("Model loads: 0")
    print("Inference: 0")
    print("Individual cases printed: 0")


if __name__ == "__main__":
    build()
