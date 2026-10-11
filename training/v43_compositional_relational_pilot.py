from collections import Counter
from copy import deepcopy
import hashlib
import json
import random


LABELS = [
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
]

MATRIX = [
    ["YES",          "NO",           "CONDITIONAL", "INSUFFICIENT"],
    ["NO",           "CONDITIONAL",  "INSUFFICIENT","YES"],
    ["CONDITIONAL",  "INSUFFICIENT", "YES",         "NO"],
    ["INSUFFICIENT", "YES",          "NO",          "CONDITIONAL"],
]


TARGET_SETS = [
    ["Alder", "Birch", "Cedar", "Dune"],
    ["Echo", "Flint", "Grove", "Haven"],
    ["Iris", "Jade", "Kite", "Lumen"],
    ["Moss", "Noble", "Opal", "Pine"],
]


PERMUTATIONS = [
    (
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ),
    (
        [1, 3, 0, 2],
        [2, 0, 3, 1],
    ),
    (
        [2, 0, 3, 1],
        [1, 3, 0, 2],
    ),
    (
        [3, 2, 1, 0],
        [3, 1, 2, 0],
    ),
]


def E(eid, source, section, text):
    return {
        "id": eid,
        "source": source,
        "section": section,
        "text": text,
    }


def stable_seed(*parts):
    text = "||".join(
        str(x)
        for x in parts
    )

    digest = hashlib.sha256(
        text.encode()
    ).hexdigest()

    return int(
        digest[:16],
        16,
    )


def permission_questions(target, action):
    return [
        f"For request {target}, may someone {action}?",
        f"Does the supplied evidence permit someone to {action} "
        f"for request {target}?",
        f"What decision follows on whether someone may {action} "
        f"for request {target}?",
    ]


def requirement_questions(target, action):
    return [
        f"For request {target}, is someone required to {action}?",
        f"Does the supplied policy make it mandatory to {action} "
        f"for request {target}?",
        f"What decision follows on whether someone must {action} "
        f"for request {target}?",
    ]


# =====================================================================
# CONDITION LEDGER — TRAIN
# =====================================================================

def ledger_chunk(
    target,
    action,
    label,
    c1,
    c2,
):
    p = target.lower()

    if label == "YES":
        evidence = [
            E(
                f"{p}_R",
                "operating_policy.txt",
                f"{target} RULE",
                f"For request {target}, someone may {action} "
                f"when {c1} is active and {c2} is confirmed.",
            ),
            E(
                f"{p}_S",
                "request_status.txt",
                f"{target} STATUS",
                f"For request {target}, {c1} is active "
                f"and {c2} is confirmed.",
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [
                f"{p}_R",
                f"{p}_S",
            ],
            "conditions": [],
            "rationale":
                "Every material condition in the operative rule is satisfied.",
        }

    if label == "NO":
        evidence = [
            E(
                f"{p}_R",
                "operating_policy.txt",
                f"{target} RULE",
                f"For request {target}, someone may {action} only when "
                f"{c1} is active and {c2} is confirmed.",
            ),
            E(
                f"{p}_F",
                "operating_policy.txt",
                f"{target} FAILED CONTROL",
                f"If {c2} is denied for request {target}, "
                f"the action is not permitted.",
            ),
            E(
                f"{p}_S",
                "request_status.txt",
                f"{target} STATUS",
                f"For request {target}, {c1} is active "
                f"and {c2} is denied.",
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [
                f"{p}_R",
                f"{p}_F",
                f"{p}_S",
            ],
            "conditions": [],
            "rationale":
                "A material requirement is directly failed.",
        }

    if label == "CONDITIONAL":
        evidence = [
            E(
                f"{p}_R",
                "operating_policy.txt",
                f"{target} RULE",
                f"For request {target}, someone may {action} "
                f"when {c1} is active and {c2} is confirmed.",
            ),
            E(
                f"{p}_S",
                "request_status.txt",
                f"{target} STATUS",
                f"For request {target}, {c1} is active. "
                f"The record contains no determination of {c2}.",
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [
                f"{p}_R",
                f"{p}_S",
            ],
            "conditions": [
                f"{c2} must be confirmed",
            ],
            "rationale":
                "The action may be permitted, but a material "
                "condition remains unresolved.",
        }

    evidence = [
        E(
            f"{p}_I",
            "recordkeeping_policy.txt",
            f"{target} RECORDKEEPING",
            f"For request {target}, records concerning {c1} "
            f"and {c2} must remain available for audit.",
        ),
        E(
            f"{p}_S",
            "request_status.txt",
            f"{target} STATUS",
            f"A current record exists for request {target}.",
        ),
    ]

    return {
        "evidence": evidence,
        "citations": [],
        "conditions": [],
        "rationale":
            "The evidence discusses related controls but does not "
            "govern whether the requested action is permitted.",
    }


# =====================================================================
# SUPERSESSION — VALIDATION
# =====================================================================

def supersession_chunk(
    target,
    action,
    label,
    c1,
    c2,
):
    p = target.lower()

    if label == "YES":
        evidence = [
            E(
                f"{p}_O",
                "policy_2025.txt",
                f"{target} OLDER RULE",
                f"For request {target}, someone may not {action}.",
            ),
            E(
                f"{p}_N",
                "policy_2026.txt",
                f"{target} CURRENT RULE",
                f"For request {target}, someone may {action} "
                f"when {c1} is active and {c2} is confirmed.",
            ),
            E(
                f"{p}_M",
                "policy_2026.txt",
                f"{target} SUPERSESSION",
                f"For request {target}, the current policy "
                f"supersedes the prior rule.",
            ),
            E(
                f"{p}_S",
                "request_status.txt",
                f"{target} STATUS",
                f"{c1.capitalize()} is active and {c2} is confirmed.",
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [
                f"{p}_O",
                f"{p}_N",
                f"{p}_M",
                f"{p}_S",
            ],
            "conditions": [],
            "rationale":
                "The current permission supersedes the older rule "
                "and its requirements are satisfied.",
        }

    if label == "NO":
        evidence = [
            E(
                f"{p}_O",
                "policy_2025.txt",
                f"{target} OLDER RULE",
                f"For request {target}, someone may {action} "
                f"when {c1} is active.",
            ),
            E(
                f"{p}_N",
                "policy_2026.txt",
                f"{target} CURRENT RULE",
                f"For request {target}, someone may not {action}.",
            ),
            E(
                f"{p}_M",
                "policy_2026.txt",
                f"{target} SUPERSESSION",
                f"For request {target}, the current rule supersedes "
                f"the prior permission.",
            ),
            E(
                f"{p}_S",
                "request_status.txt",
                f"{target} STATUS",
                f"{c1.capitalize()} is active.",
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [
                f"{p}_O",
                f"{p}_N",
                f"{p}_M",
                f"{p}_S",
            ],
            "conditions": [],
            "rationale":
                "The current prohibition supersedes the older permission.",
        }

    if label == "CONDITIONAL":
        evidence = [
            E(
                f"{p}_O",
                "policy_2025.txt",
                f"{target} OLDER RULE",
                f"For request {target}, someone may {action} "
                f"when {c1} is active.",
            ),
            E(
                f"{p}_N",
                "policy_2026.txt",
                f"{target} CURRENT RULE",
                f"For request {target}, someone may {action} only when "
                f"{c1} is active and {c2} is confirmed.",
            ),
            E(
                f"{p}_M",
                "policy_2026.txt",
                f"{target} SUPERSESSION",
                f"For request {target}, the current rule replaces "
                f"the prior rule.",
            ),
            E(
                f"{p}_S",
                "request_status.txt",
                f"{target} STATUS",
                f"{c1.capitalize()} is active. "
                f"The record does not establish {c2}.",
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [
                f"{p}_O",
                f"{p}_N",
                f"{p}_M",
                f"{p}_S",
            ],
            "conditions": [
                f"{c2} must be confirmed",
            ],
            "rationale":
                "The current rule controls, but one current "
                "requirement is unresolved.",
        }

    evidence = [
        E(
            f"{p}_O",
            "policy_2025.txt",
            f"{target} OLDER RULE",
            f"For request {target}, someone may {action} "
            f"when {c1} is active.",
        ),
        E(
            f"{p}_N",
            "notice_2026.txt",
            f"{target} LATER NOTICE",
            f"A later notice for request {target} discusses "
            f"documentation for {c2}, but does not say whether "
            f"it replaces the earlier rule.",
        ),
    ]

    return {
        "evidence": evidence,
        "citations": [],
        "conditions": [],
        "rationale":
            "The evidence does not establish which rule currently "
            "controls the requested action.",
    }


# =====================================================================
# PERMISSION VS OBLIGATION — TEST
# =====================================================================

def obligation_chunk(
    target,
    action,
    label,
    c1,
    c2,
):
    p = target.lower()

    if label == "YES":
        evidence = [
            E(
                f"{p}_P",
                "obligation_policy.txt",
                f"{target} PERMISSION",
                f"For request {target}, someone is permitted to {action}.",
            ),
            E(
                f"{p}_D",
                "obligation_policy.txt",
                f"{target} SCOPE",
                f"Request {target} is high-risk when {c1} is active "
                f"and {c2} is confirmed.",
            ),
            E(
                f"{p}_O",
                "obligation_policy.txt",
                f"{target} MANDATORY RULE",
                f"For a high-risk request {target}, someone is "
                f"required to {action}.",
            ),
            E(
                f"{p}_S",
                "request_status.txt",
                f"{target} STATUS",
                f"{c1.capitalize()} is active and {c2} is confirmed.",
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [
                f"{p}_P",
                f"{p}_D",
                f"{p}_O",
                f"{p}_S",
            ],
            "conditions": [],
            "rationale":
                "The request falls within the mandatory scope.",
        }

    if label == "NO":
        evidence = [
            E(
                f"{p}_P",
                "obligation_policy.txt",
                f"{target} PERMISSION",
                f"For request {target}, someone is permitted to {action}.",
            ),
            E(
                f"{p}_D",
                "obligation_policy.txt",
                f"{target} STANDARD SCOPE",
                f"Request {target} is standard-risk when {c2} is denied.",
            ),
            E(
                f"{p}_O",
                "obligation_policy.txt",
                f"{target} MANDATORY RULE",
                f"The requirement to {action} applies only to "
                f"high-risk requests, not standard-risk requests.",
            ),
            E(
                f"{p}_S",
                "request_status.txt",
                f"{target} STATUS",
                f"{c2.capitalize()} is denied, so request {target} "
                f"is standard-risk.",
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [
                f"{p}_P",
                f"{p}_D",
                f"{p}_O",
                f"{p}_S",
            ],
            "conditions": [],
            "rationale":
                "The action is permitted but is not mandatory "
                "for the established scope.",
        }

    if label == "CONDITIONAL":
        evidence = [
            E(
                f"{p}_P",
                "obligation_policy.txt",
                f"{target} PERMISSION",
                f"For request {target}, someone is permitted to {action}.",
            ),
            E(
                f"{p}_D",
                "obligation_policy.txt",
                f"{target} SCOPE",
                f"Request {target} is high-risk when {c1} is active "
                f"and {c2} is confirmed.",
            ),
            E(
                f"{p}_O",
                "obligation_policy.txt",
                f"{target} MANDATORY RULE",
                f"For a high-risk request {target}, someone is "
                f"required to {action}.",
            ),
            E(
                f"{p}_S",
                "request_status.txt",
                f"{target} STATUS",
                f"{c1.capitalize()} is active. "
                f"The record does not establish {c2}.",
            ),
        ]

        return {
            "evidence": evidence,
            "citations": [
                f"{p}_P",
                f"{p}_D",
                f"{p}_O",
                f"{p}_S",
            ],
            "conditions": [
                f"{c2} must be confirmed to establish mandatory scope",
            ],
            "rationale":
                "Permission is established, but whether the mandatory "
                "scope applies remains unresolved.",
        }

    evidence = [
        E(
            f"{p}_P",
            "obligation_policy.txt",
            f"{target} PERMISSION",
            f"For request {target}, someone is permitted to {action}.",
        ),
        E(
            f"{p}_S",
            "request_status.txt",
            f"{target} STATUS",
            f"{c1.capitalize()} is active and {c2} is confirmed.",
        ),
    ]

    return {
        "evidence": evidence,
        "citations": [],
        "conditions": [],
        "rationale":
            "Permission to perform the action does not establish "
            "whether the action is mandatory.",
    }


SPECS = {
    "train": [
        (
            "ledger_orion",
            "condition_ledger",
            "analyze {target} records inside the Orion workspace",
            "workspace clearance",
            "data-steward confirmation",
        ),
        (
            "ledger_atlas",
            "condition_ledger",
            "release {target} records into the Atlas collaboration",
            "release clearance",
            "recipient assurance",
        ),
        (
            "ledger_nova",
            "condition_ledger",
            "use {target} records in the Nova workflow",
            "workflow authorization",
            "security attestation",
        ),
        (
            "ledger_helix",
            "condition_ledger",
            "retain {target} records in the Helix archive",
            "archive approval",
            "preservation confirmation",
        ),
    ],

    "validation": [
        (
            "supersession_apogee",
            "supersession_current_rule",
            "review {target} records in the Apogee enclave",
            "enclave authorization",
            "custodian acknowledgment",
        ),
        (
            "supersession_harbor",
            "supersession_current_rule",
            "stage {target} records for the Harbor exchange",
            "exchange approval",
            "counterparty verification",
        ),
        (
            "supersession_prism",
            "supersession_current_rule",
            "process {target} records through the Prism pipeline",
            "processing clearance",
            "control attestation",
        ),
        (
            "supersession_summit",
            "supersession_current_rule",
            "place {target} records in the Summit repository",
            "repository approval",
            "repository enrollment",
        ),
    ],

    "test": [
        (
            "obligation_juniper",
            "permission_obligation_scope",
            "inspect {target} records in the Juniper domain",
            "domain approval",
            "steward verification",
        ),
        (
            "obligation_cobalt",
            "permission_obligation_scope",
            "provide {target} records through the Cobalt exchange",
            "sharing approval",
            "receiver verification",
        ),
        (
            "obligation_aurora",
            "permission_obligation_scope",
            "operate on {target} records in the Aurora workflow",
            "workflow clearance",
            "safety verification",
        ),
        (
            "obligation_delta",
            "permission_obligation_scope",
            "transfer {target} records to the Delta enclave",
            "transfer clearance",
            "destination verification",
        ),
    ],
}


CHUNK_BUILDERS = {
    "train": ledger_chunk,
    "validation": supersession_chunk,
    "test": obligation_chunk,
}


def build_family(
    split,
    family_index,
    spec,
):
    (
        family,
        composition,
        action_template,
        c1,
        c2,
    ) = spec

    targets = TARGET_SETS[
        family_index
    ]

    row_perm, col_perm = PERMUTATIONS[
        family_index
    ]

    rows = []

    for world in range(4):

        evidence = []
        metadata = {}

        for target_index, target in enumerate(targets):

            action = action_template.format(
                target=target
            )

            label = MATRIX[
                row_perm[world]
            ][
                col_perm[target_index]
            ]

            chunk = CHUNK_BUILDERS[
                split
            ](
                target,
                action,
                label,
                c1,
                c2,
            )

            evidence.extend(
                chunk["evidence"]
            )

            metadata[target] = {
                **chunk,
                "label": label,
                "action": action,
            }

        rng = random.Random(
            stable_seed(
                family,
                world,
                "evidence",
            )
        )

        rng.shuffle(
            evidence
        )

        for target in targets:

            meta = metadata[target]

            if split == "test":
                questions = requirement_questions(
                    target,
                    meta["action"],
                )
                question_type = "requirement"
            else:
                questions = permission_questions(
                    target,
                    meta["action"],
                )
                question_type = "permission"

            for surface, question in enumerate(
                questions
            ):
                rows.append(
                    {
                        "family": family,
                        "composition": composition,
                        "world": world,
                        "target": target,
                        "surface_variant": surface,
                        "question_type": question_type,
                        "question": question,
                        "evidence": deepcopy(evidence),
                        "decision": meta["label"],
                        "citations": list(
                            meta["citations"]
                        ),
                        "conditions": list(
                            meta["conditions"]
                        ),
                        "rationale": meta["rationale"],
                    }
                )

    return rows


ROWS = {}

for split, specs in SPECS.items():
    rows = []

    for i, spec in enumerate(specs):
        rows.extend(
            build_family(
                split,
                i,
                spec,
            )
        )

    ROWS[split] = rows


if __name__ == "__main__":

    print("=" * 72)
    print("SENTINEL V4.3 COMPOSITIONAL RELATIONAL PILOT")
    print("=" * 72)

    for split, rows in ROWS.items():

        print()
        print(split.upper())
        print("-" * 72)

        print(
            "rows:",
            len(rows),
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
                Counter(
                    row["decision"]
                    for row in rows
                )
            ),
        )
