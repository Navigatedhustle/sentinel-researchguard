"""
Sentinel V4.3 relational pilot.

Goal:
Neither QUESTION alone nor EVIDENCE alone should identify the label.
The correct decision must depend on their relationship.

This is a proof-of-architecture pilot only.
Challenge V1 is never imported or referenced.
"""

from copy import deepcopy
import json


LABELS = [
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
]


# Latin-square relation matrix.
#
# Rows = evidence worlds
# Cols = question targets
#
# Each row contains every label once.
# Each column contains every label once.
MATRIX = [
    ["YES",          "NO",           "CONDITIONAL", "INSUFFICIENT"],
    ["NO",           "CONDITIONAL",  "INSUFFICIENT","YES"],
    ["CONDITIONAL",  "INSUFFICIENT", "YES",         "NO"],
    ["INSUFFICIENT", "YES",          "NO",          "CONDITIONAL"],
]


def E(eid, source, section, text):
    return {
        "id": eid,
        "source": source,
        "section": section,
        "text": text,
    }


def make_targets(prefix, action):
    return [
        {
            "id": f"{prefix}_alder",
            "name": "Alder",
            "action": action.format(target="Alder"),
        },
        {
            "id": f"{prefix}_birch",
            "name": "Birch",
            "action": action.format(target="Birch"),
        },
        {
            "id": f"{prefix}_cedar",
            "name": "Cedar",
            "action": action.format(target="Cedar"),
        },
        {
            "id": f"{prefix}_dune",
            "name": "Dune",
            "action": action.format(target="Dune"),
        },
    ]


def permission_evidence(target, action):
    return [
        E(
            f"{target}_R",
            "governance_rules.txt",
            f"{target} RULE",
            f"For request {target}, someone may {action} "
            f"when approval is active.",
        ),
        E(
            f"{target}_S",
            "request_status.txt",
            f"{target} STATUS",
            f"For request {target}, approval is active.",
        ),
    ]


def prohibition_evidence(target, action):
    return [
        E(
            f"{target}_R",
            "governance_rules.txt",
            f"{target} RULE",
            f"For request {target}, someone may not {action}.",
        ),
        E(
            f"{target}_S",
            "request_status.txt",
            f"{target} STATUS",
            f"Request {target} is currently under review.",
        ),
    ]


def conditional_evidence(target, action):
    return [
        E(
            f"{target}_R",
            "governance_rules.txt",
            f"{target} RULE",
            f"For request {target}, someone may {action} only when "
            f"approval is active and steward confirmation is recorded.",
        ),
        E(
            f"{target}_S",
            "request_status.txt",
            f"{target} STATUS",
            f"For request {target}, approval is active. "
            f"The record does not establish steward confirmation.",
        ),
    ]


def insufficient_evidence(target, action):
    return [
        E(
            f"{target}_I",
            "recordkeeping_rules.txt",
            f"{target} RECORDKEEPING",
            f"Records concerning request {target} and the associated "
            f"approval process must be retained for audit.",
        ),
        E(
            f"{target}_S",
            "request_status.txt",
            f"{target} STATUS",
            f"Request {target} has an active record.",
        ),
    ]


EVIDENCE_BUILDERS = {
    "YES": permission_evidence,
    "NO": prohibition_evidence,
    "CONDITIONAL": conditional_evidence,
    "INSUFFICIENT": insufficient_evidence,
}


def question(target):
    return (
        f"For request {target['name']}, "
        f"may someone {target['action']}?"
    )


def rationale(label):
    return {
        "YES":
            "The supplied evidence directly permits the action and "
            "establishes its material requirement.",
        "NO":
            "The supplied evidence directly prohibits the requested action.",
        "CONDITIONAL":
            "The action may be permitted, but a material requirement "
            "remains unresolved.",
        "INSUFFICIENT":
            "The supplied evidence is related to the request but does not "
            "govern whether the requested action is permitted.",
    }[label]


def citations_for(label, target):
    if label == "INSUFFICIENT":
        return []

    return [
        f"{target}_R",
        f"{target}_S",
    ]


def conditions_for(label):
    if label == "CONDITIONAL":
        return [
            "steward confirmation must be recorded",
        ]

    return []


def build_family(
    family_name,
    action_template,
    row_shift=0,
    col_shift=0,
):
    """
    Build one dual-counterfactual family.

    row_shift / col_shift rotate the Latin square so families do not share
    one globally fixed world/question/label mapping.
    """

    targets = make_targets(
        family_name,
        action_template,
    )

    rows = []

    for world_idx in range(4):

        # Build ONE evidence pack containing governance for all four targets.
        evidence = []

        for target_idx, target in enumerate(targets):

            matrix_row = (
                world_idx + row_shift
            ) % 4

            matrix_col = (
                target_idx + col_shift
            ) % 4

            label = MATRIX[
                matrix_row
            ][
                matrix_col
            ]

            builder = EVIDENCE_BUILDERS[
                label
            ]

            evidence.extend(
                builder(
                    target["name"],
                    target["action"],
                )
            )

        # Ask all four questions against the SAME evidence pack.
        for target_idx, target in enumerate(targets):

            matrix_row = (
                world_idx + row_shift
            ) % 4

            matrix_col = (
                target_idx + col_shift
            ) % 4

            label = MATRIX[
                matrix_row
            ][
                matrix_col
            ]

            rows.append(
                {
                    "family": family_name,
                    "world": world_idx,
                    "target": target["name"],
                    "question": question(target),
                    "evidence": deepcopy(evidence),
                    "decision": label,
                    "citations": citations_for(
                        label,
                        target["name"],
                    ),
                    "conditions": conditions_for(
                        label
                    ),
                    "rationale": rationale(label),
                }
            )

    return rows


FAMILIES = {
    "train": build_family(
        "pilot_orion",
        "analyze {target} records inside the Orion workspace",
        row_shift=0,
        col_shift=0,
    ),

    "validation": build_family(
        "pilot_apogee",
        "review {target} records within the Apogee enclave",
        row_shift=1,
        col_shift=2,
    ),

    "test": build_family(
        "pilot_juniper",
        "inspect {target} records inside the Juniper domain",
        row_shift=1,
        col_shift=0,
    ),
}


if __name__ == "__main__":
    for split, rows in FAMILIES.items():

        print()
        print("=" * 72)
        print(split.upper())
        print("=" * 72)

        counts = {
            label: sum(
                1
                for row in rows
                if row["decision"] == label
            )
            for label in LABELS
        }

        print("rows:", len(rows))
        print("labels:", counts)

        worlds = {}

        for row in rows:
            worlds.setdefault(
                row["world"],
                []
            ).append(
                (
                    row["target"],
                    row["decision"],
                )
            )

        for world, items in sorted(worlds.items()):
            print(
                "world",
                world,
                "->",
                items,
            )
