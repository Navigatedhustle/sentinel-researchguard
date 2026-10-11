"""
Sentinel V4.4 gate-transfer pilot.

Purpose:
Teach hierarchical / short-circuit rule selection without copying the
V4.3 temporal_override validation grammar.

Core reasoning order:

    1. Determine whether the alternate rule is operative.
    2. If not operative, the base rule controls.
    3. Only if operative should downstream requirements be evaluated.

The pilot is fully relational:

    decision = f(question, evidence)

Neither question identity nor evidence identity alone determines the label.

This file intentionally DOES NOT import:
- v4_family_specs_v42
- V4.3 validation/test families
- Challenge V1
"""

from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import re


BASE = Path("/workspace")
OUT = BASE / "training/datasets"

SEED = 20260816

LABELS = [
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
]


# ---------------------------------------------------------------------
# Same relational Latin-square invariant used in V4.3.
# ---------------------------------------------------------------------

MATRIX = [
    [
        "YES",
        "NO",
        "CONDITIONAL",
        "INSUFFICIENT",
    ],
    [
        "NO",
        "CONDITIONAL",
        "INSUFFICIENT",
        "YES",
    ],
    [
        "CONDITIONAL",
        "INSUFFICIENT",
        "YES",
        "NO",
    ],
    [
        "INSUFFICIENT",
        "YES",
        "NO",
        "CONDITIONAL",
    ],
]


INSTRUCTION = """
You are Sentinel, a policy-governance decision model.

Use ONLY the supplied evidence.

Classify the exact question as one of:

YES
NO
CONDITIONAL
INSUFFICIENT

Apply these rules in order:

INSUFFICIENT:
Choose INSUFFICIENT when the supplied evidence does not directly govern
the exact issue asked about. Related evidence is not enough. Do not invent
missing approvers, technologies, durations, thresholds, locations, or rules.

NO:
Choose NO when the evidence directly prohibits the exact action or directly
establishes that the proposition in the question is false.

CONDITIONAL:
Choose CONDITIONAL when the action may be permitted, but one or more
material requirements remain unresolved.

YES:
Choose YES only when the evidence directly permits the exact action and
every material condition is already established as satisfied by the question
and supplied evidence.

Silence is not prohibition.
Semantic similarity is not evidence.

Return valid JSON only.
""".strip()


TARGETS = {
    "train": [
        "Arbor",
        "Basin",
        "Cobalt",
        "Delta",
    ],
    "validation": [
        "Ember",
        "Fable",
        "Garnet",
        "Hollow",
    ],
    "test": [
        "Indigo",
        "Juniper",
        "Keystone",
        "Lagoon",
    ],
}


VARIANTS = {
    "train": [
        {
            "code": "atlas",
            "action": "move the project packet into the partner workspace",
            "c1": "release clearance",
            "c2": "recipient attestation",
        },
        {
            "code": "beacon",
            "action": "publish the analysis bundle to the collaboration repository",
            "c1": "distribution clearance",
            "c2": "repository verification",
        },
        {
            "code": "cinder",
            "action": "send the review package to the downstream service",
            "c1": "service approval",
            "c2": "destination confirmation",
        },
        {
            "code": "dynamo",
            "action": "release the evaluation artifact to the collaboration zone",
            "c1": "release authorization",
            "c2": "zone attestation",
        },
    ],

    "validation": [
        {
            "code": "ember",
            "action": "transfer the assessment packet to the external workspace",
            "c1": "transfer clearance",
            "c2": "workspace confirmation",
        },
        {
            "code": "fjord",
            "action": "publish the project bundle to the shared repository",
            "c1": "publication approval",
            "c2": "repository attestation",
        },
    ],

    "test": [
        {
            "code": "grove",
            "action": "release the audit package to the partner environment",
            "c1": "release approval",
            "c2": "partner verification",
        },
        {
            "code": "harbor",
            "action": "send the evaluation bundle to the collaboration service",
            "c1": "transfer authorization",
            "c2": "service confirmation",
        },
    ],
}


COMPOSITIONS = [
    "stage_gate_precedence",
    "event_gate_precedence",
]


def stable_seed(*parts):

    raw = "||".join(
        str(x)
        for x in parts
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:16],
        16,
    )


def slug(text):

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text.lower(),
    ).strip("_")


def E(
    eid,
    filename,
    section,
    text,
):

    return {
        "id": eid,
        "file": filename,
        "section": section,
        "text": text,
    }


def questions_for(
    composition,
    target,
    action,
):

    if composition == "stage_gate_precedence":

        return [
            (
                f"For request {target}, under the recorded workflow stage, "
                f"may someone {action}?"
            ),
            (
                f"Does the controlling stage rule permit someone to "
                f"{action} for request {target}?"
            ),
            (
                f"What decision follows for request {target} on whether "
                f"someone may {action}, given its workflow stage?"
            ),
        ]

    if composition == "event_gate_precedence":

        return [
            (
                f"For request {target}, after considering the activation "
                f"event in the record, may someone {action}?"
            ),
            (
                f"Does the event-triggered rule permit someone to "
                f"{action} for request {target}?"
            ),
            (
                f"What decision follows for request {target} on whether "
                f"someone may {action}, given the recorded trigger event?"
            ),
        ]

    raise ValueError(
        f"Unknown composition: {composition}"
    )


def stage_case(
    target,
    code,
    action,
    c1,
    c2,
    label,
):

    prefix = slug(target)

    B = f"{prefix}__B"
    G = f"{prefix}__G"
    S = f"{prefix}__S"

    if label == "YES":

        evidence = [
            E(
                B,
                f"{code}_stage_rules.txt",
                f"{target} | BASE RULE",
                (
                    f"Unless request {target} has entered the "
                    f"release-authorized stage, someone may not {action}."
                ),
            ),
            E(
                G,
                f"{code}_stage_rules.txt",
                f"{target} | ALTERNATE RULE",
                (
                    f"Once request {target} enters the release-authorized "
                    f"stage, someone may {action} when {c1} is recorded "
                    f"and {c2} is confirmed."
                ),
            ),
            E(
                S,
                f"{code}_stage_status.txt",
                f"{target} | STATUS",
                (
                    f"Request {target} has entered the release-authorized "
                    f"stage. {c1.capitalize()} is recorded and "
                    f"{c2} is confirmed."
                ),
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [B, G, S],
            "conditions": [],
            "rationale": (
                "The alternate stage rule is operative and every "
                "downstream requirement is satisfied."
            ),
            "gate_state": "open",
            "downstream_state": "satisfied",
        }

    if label == "CONDITIONAL":

        evidence = [
            E(
                B,
                f"{code}_stage_rules.txt",
                f"{target} | BASE RULE",
                (
                    f"Unless request {target} has entered the "
                    f"release-authorized stage, someone may not {action}."
                ),
            ),
            E(
                G,
                f"{code}_stage_rules.txt",
                f"{target} | ALTERNATE RULE",
                (
                    f"Once request {target} enters the release-authorized "
                    f"stage, someone may {action} when {c1} is recorded "
                    f"and {c2} is confirmed."
                ),
            ),
            E(
                S,
                f"{code}_stage_status.txt",
                f"{target} | STATUS",
                (
                    f"Request {target} has entered the release-authorized "
                    f"stage and {c1} is recorded. "
                    f"The record does not establish {c2}."
                ),
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [B, G, S],
            "conditions": [
                f"{c2} must be confirmed"
            ],
            "rationale": (
                "The alternate stage rule is operative, but one "
                "downstream requirement remains unresolved."
            ),
            "gate_state": "open",
            "downstream_state": "unresolved",
        }

    if label == "NO":

        # IMPORTANT:
        # Downstream requirements are intentionally satisfied.
        # They are irrelevant because the gate itself is closed.
        evidence = [
            E(
                B,
                f"{code}_stage_rules.txt",
                f"{target} | BASE RULE",
                (
                    f"Unless request {target} has entered the "
                    f"release-authorized stage, someone may not {action}."
                ),
            ),
            E(
                G,
                f"{code}_stage_rules.txt",
                f"{target} | ALTERNATE RULE",
                (
                    f"Once request {target} enters the release-authorized "
                    f"stage, someone may {action} when {c1} is recorded "
                    f"and {c2} is confirmed."
                ),
            ),
            E(
                S,
                f"{code}_stage_status.txt",
                f"{target} | STATUS",
                (
                    f"Request {target} remains in internal review and has "
                    f"not entered the release-authorized stage. "
                    f"{c1.capitalize()} is recorded and {c2} is confirmed."
                ),
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [B, G, S],
            "conditions": [],
            "rationale": (
                "The alternate stage rule is not operative, so the "
                "base prohibition controls despite satisfied downstream "
                "requirements."
            ),
            "gate_state": "closed",
            "downstream_state": "satisfied",
        }

    if label == "INSUFFICIENT":

        evidence = [
            E(
                G,
                f"{code}_stage_admin.txt",
                f"{target} | STAGE ADMINISTRATION",
                (
                    f"Transitions into the release-authorized stage for "
                    f"request {target} must be entered in the workflow ledger."
                ),
            ),
            E(
                S,
                f"{code}_stage_status.txt",
                f"{target} | STATUS",
                (
                    f"Request {target} is listed in the release-authorized "
                    f"stage. {c1.capitalize()} is recorded and "
                    f"{c2} is confirmed."
                ),
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [],
            "conditions": [],
            "rationale": (
                "The evidence establishes workflow administration and "
                "status but supplies no rule governing the requested action."
            ),
            "gate_state": "undetermined",
            "downstream_state": "satisfied",
        }

    raise ValueError(label)


def event_case(
    target,
    code,
    action,
    c1,
    c2,
    label,
):

    prefix = slug(target)

    B = f"{prefix}__B"
    G = f"{prefix}__G"
    S = f"{prefix}__S"

    if label == "YES":

        evidence = [
            E(
                B,
                f"{code}_event_rules.txt",
                f"{target} | BASE RULE",
                (
                    f"Until an activation certificate is recorded for "
                    f"request {target}, someone may not {action}."
                ),
            ),
            E(
                G,
                f"{code}_event_rules.txt",
                f"{target} | TRIGGERED RULE",
                (
                    f"A recorded activation certificate for request {target} "
                    f"activates an alternate rule permitting someone to "
                    f"{action} when {c1} is recorded and {c2} is confirmed."
                ),
            ),
            E(
                S,
                f"{code}_event_status.txt",
                f"{target} | EVENT STATUS",
                (
                    f"The activation certificate for request {target} has "
                    f"been recorded. {c1.capitalize()} is recorded and "
                    f"{c2} is confirmed."
                ),
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [B, G, S],
            "conditions": [],
            "rationale": (
                "The activating event occurred and every downstream "
                "requirement of the alternate rule is satisfied."
            ),
            "gate_state": "open",
            "downstream_state": "satisfied",
        }

    if label == "CONDITIONAL":

        evidence = [
            E(
                B,
                f"{code}_event_rules.txt",
                f"{target} | BASE RULE",
                (
                    f"Until an activation certificate is recorded for "
                    f"request {target}, someone may not {action}."
                ),
            ),
            E(
                G,
                f"{code}_event_rules.txt",
                f"{target} | TRIGGERED RULE",
                (
                    f"A recorded activation certificate for request {target} "
                    f"activates an alternate rule permitting someone to "
                    f"{action} when {c1} is recorded and {c2} is confirmed."
                ),
            ),
            E(
                S,
                f"{code}_event_status.txt",
                f"{target} | EVENT STATUS",
                (
                    f"The activation certificate for request {target} has "
                    f"been recorded and {c1} is recorded. "
                    f"The record does not establish {c2}."
                ),
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [B, G, S],
            "conditions": [
                f"{c2} must be confirmed"
            ],
            "rationale": (
                "The activating event occurred, but one downstream "
                "requirement remains unresolved."
            ),
            "gate_state": "open",
            "downstream_state": "unresolved",
        }

    if label == "NO":

        # Again, downstream requirements are satisfied on purpose.
        # The model must short-circuit on the closed gate.
        evidence = [
            E(
                B,
                f"{code}_event_rules.txt",
                f"{target} | BASE RULE",
                (
                    f"Until an activation certificate is recorded for "
                    f"request {target}, someone may not {action}."
                ),
            ),
            E(
                G,
                f"{code}_event_rules.txt",
                f"{target} | TRIGGERED RULE",
                (
                    f"A recorded activation certificate for request {target} "
                    f"activates an alternate rule permitting someone to "
                    f"{action} when {c1} is recorded and {c2} is confirmed."
                ),
            ),
            E(
                S,
                f"{code}_event_status.txt",
                f"{target} | EVENT STATUS",
                (
                    f"No activation certificate has been recorded for "
                    f"request {target}. {c1.capitalize()} is recorded and "
                    f"{c2} is confirmed."
                ),
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [B, G, S],
            "conditions": [],
            "rationale": (
                "The activating event has not occurred, so the base "
                "prohibition controls regardless of downstream status."
            ),
            "gate_state": "closed",
            "downstream_state": "satisfied",
        }

    if label == "INSUFFICIENT":

        evidence = [
            E(
                G,
                f"{code}_event_admin.txt",
                f"{target} | EVENT ADMINISTRATION",
                (
                    f"Activation certificates for request {target} must be "
                    f"retained in the event ledger."
                ),
            ),
            E(
                S,
                f"{code}_event_status.txt",
                f"{target} | EVENT STATUS",
                (
                    f"An activation certificate for request {target} is "
                    f"present in the ledger. {c1.capitalize()} is recorded "
                    f"and {c2} is confirmed."
                ),
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [],
            "conditions": [],
            "rationale": (
                "The evidence establishes event administration and status "
                "but supplies no rule governing the requested action."
            ),
            "gate_state": "undetermined",
            "downstream_state": "satisfied",
        }

    raise ValueError(label)


def make_case(
    composition,
    target,
    variant,
    label,
):

    kwargs = {
        "target": target,
        "code": variant["code"],
        "action": variant["action"],
        "c1": variant["c1"],
        "c2": variant["c2"],
        "label": label,
    }

    if composition == "stage_gate_precedence":
        return stage_case(**kwargs)

    if composition == "event_gate_precedence":
        return event_case(**kwargs)

    raise ValueError(composition)


def render_evidence(chunks):

    parts = [
        "SUPPLIED EVIDENCE",
        "=================",
    ]

    for chunk in chunks:

        parts.extend([
            "",
            f"[{chunk['id']}]",
            f"Source: {chunk['file']}",
            f"Section: {chunk['section']}",
            f"Text: {chunk['text']}",
        ])

    return "\n".join(parts)


def build_split(split):

    rows = []

    targets = TARGETS[split]

    for composition in COMPOSITIONS:

        for variant in VARIANTS[split]:

            family = (
                f"{composition}__"
                f"{variant['code']}"
            )

            questions = {
                target: questions_for(
                    composition,
                    target,
                    variant["action"],
                )
                for target in targets
            }

            for world_index, world_labels in enumerate(MATRIX):

                world_chunks = []
                target_cases = {}

                for target_index, target in enumerate(targets):

                    label = world_labels[
                        target_index
                    ]

                    case = make_case(
                        composition,
                        target,
                        variant,
                        label,
                    )

                    target_cases[target] = (
                        label,
                        case,
                    )

                    world_chunks.extend(
                        case["evidence"]
                    )

                rng = random.Random(
                    stable_seed(
                        SEED,
                        split,
                        family,
                        world_index,
                    )
                )

                rng.shuffle(
                    world_chunks
                )

                evidence_text = render_evidence(
                    world_chunks
                )

                evidence_hash = hashlib.sha256(
                    evidence_text.encode("utf-8")
                ).hexdigest()

                for target in targets:

                    label, case = (
                        target_cases[
                            target
                        ]
                    )

                    for paraphrase_index, question in enumerate(
                        questions[target]
                    ):

                        scenario_id = (
                            f"{split}__{family}__"
                            f"world_{world_index}__"
                            f"target_{slug(target)}"
                        )

                        input_text = (
                            f"{evidence_text}\n\n"
                            f"QUESTION\n"
                            f"========\n"
                            f"{question}"
                        )

                        output = {
                            "citations":
                                case["citations"],

                            "conditions":
                                case["conditions"],

                            "decision":
                                label,

                            "rationale":
                                case["rationale"],
                        }

                        rows.append({
                            "instruction":
                                INSTRUCTION,

                            "input":
                                input_text,

                            "output":
                                json.dumps(
                                    output,
                                    sort_keys=True,
                                ),

                            "split":
                                split,

                            "composition":
                                composition,

                            "family":
                                family,

                            "scenario_id":
                                scenario_id,

                            "world_index":
                                world_index,

                            "target":
                                target,

                            "paraphrase_index":
                                paraphrase_index,

                            "question":
                                question,

                            "evidence":
                                evidence_text,

                            "evidence_hash":
                                evidence_hash,

                            "gate_state":
                                case["gate_state"],

                            "downstream_state":
                                case[
                                    "downstream_state"
                                ],
                        })

    return rows


def write_jsonl(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
                + "\n"
            )


def main():

    print("=" * 80)
    print("SENTINEL V4.4 GATE-TRANSFER PILOT BUILD")
    print("=" * 80)

    outputs = {
        "train":
            OUT / "sentinel_train_v44_gate_pilot.jsonl",

        "validation":
            OUT / "sentinel_validation_v44_gate_pilot.jsonl",

        "test":
            OUT / "sentinel_test_v44_gate_pilot.jsonl",
    }

    expected = {
        "train": 384,
        "validation": 192,
        "test": 192,
    }

    for split in [
        "train",
        "validation",
        "test",
    ]:

        rows = build_split(
            split
        )

        if len(rows) != expected[split]:
            raise AssertionError(
                (
                    split,
                    len(rows),
                    expected[split],
                )
            )

        counts = Counter(
            row["output"]
            for row in rows
        )

        label_counts = Counter(
            json.loads(
                row["output"]
            )["decision"]
            for row in rows
        )

        write_jsonl(
            outputs[split],
            rows,
        )

        print()
        print(split.upper())
        print("-" * 80)
        print("rows:", len(rows))
        print(
            "scenarios:",
            len({
                row["scenario_id"]
                for row in rows
            }),
        )
        print(
            "families:",
            len({
                row["family"]
                for row in rows
            }),
        )
        print(
            "labels:",
            dict(
                sorted(
                    label_counts.items()
                )
            ),
        )

    print()
    print("Pilot build complete.")


if __name__ == "__main__":
    main()
