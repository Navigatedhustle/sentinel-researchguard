"""
Sentinel V4.3 full relational curriculum.

Core invariant:
    decision = f(question, evidence)

Neither question identity nor evidence identity alone determines the label.

V4.3 reuses the policy reasoning semantics already defined in V4.2,
but reorganizes them into dual-counterfactual relational evidence worlds.

Challenge V1 is never imported or referenced.
"""

from copy import deepcopy
from collections import Counter
from pathlib import Path
import hashlib
import json
import random
import re

from training.v4_family_specs_v42 import (
    FAMILIES as BASE_FAMILIES,
)


BASE = Path("/workspace")
OUT = BASE / "training/datasets"

SEED = 20260814

LABELS = [
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
]


# ---------------------------------------------------------------------
# Four-by-four relational design.
#
# Each row = one evidence world.
# Each column = one question target.
#
# Every evidence world contains all four labels.
# Every target traverses all four labels.
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


ASSETS = [
    "human-subject interview transcripts",
    "student advising records",
    "genomic sequencing files",
    "unpublished research manuscripts",
    "sponsor-confidential prototype designs",
]


# Two assets per family keeps V4.3 around the same order of magnitude
# as earlier curricula while rotating all five assets globally.
ASSETS_PER_FAMILY = 2


TARGET_POOLS = {
    "train": [
        "Alder",
        "Birch",
        "Cedar",
        "Dune",
        "Echo",
        "Flint",
        "Grove",
        "Haven",
        "Iris",
        "Jade",
        "Kite",
        "Lumen",
    ],

    "validation": [
        "Maple",
        "Noble",
        "Opal",
        "Pine",
        "Quartz",
        "Ridge",
        "Spruce",
        "Talon",
        "Umber",
        "Vale",
        "Willow",
        "Xenia",
    ],

    "test": [
        "Yarrow",
        "Zephyr",
        "Acorn",
        "Brook",
        "Crest",
        "Drift",
        "Elm",
        "Fjord",
        "Glade",
        "Heath",
        "Ivory",
        "Jetty",
    ],
}


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


def fill_asset(value, asset):
    if isinstance(value, str):
        return value.replace(
            "{asset}",
            asset,
        )

    if isinstance(value, list):
        return [
            fill_asset(x, asset)
            for x in value
        ]

    if isinstance(value, dict):
        return {
            k: fill_asset(v, asset)
            for k, v in value.items()
        }

    return value


def extract_action(question):
    """
    Recover the governed action from V4.2's canonical question.

    All ten V4.2 composition builders use one of these forms.
    """

    q = question.strip().rstrip("?")

    markers = [
        "may someone ",
        "for someone to ",
        "required to ",
    ]

    for marker in markers:
        if marker in q.lower():

            idx = q.lower().index(
                marker
            )

            return q[
                idx + len(marker):
            ].strip()

    raise ValueError(
        f"Unable to extract action from question: {question}"
    )


def questions_for(
    composition,
    target,
    action,
):
    """
    Three real syntactic realizations of the same exact issue.
    """

    if composition == "condition_ledger":
        return [
            (
                f"For request {target}, based on the recorded statuses, "
                f"may someone {action}?"
            ),
            (
                f"Do the supplied status records permit someone to "
                f"{action} for request {target}?"
            ),
            (
                f"What decision follows for request {target} on whether "
                f"someone may {action}, given the recorded controls?"
            ),
        ]

    if composition == "specific_rule_precedence":
        return [
            (
                f"For restricted request {target}, may someone {action}?"
            ),
            (
                f"Does the specific rule governing request {target} "
                f"permit someone to {action}?"
            ),
            (
                f"What decision follows for restricted request {target} "
                f"on whether someone may {action}?"
            ),
        ]

    if composition == "prohibition_exception":
        return [
            (
                f"Under the exception framework for request {target}, "
                f"may someone {action}?"
            ),
            (
                f"Does the stated exception allow someone to {action} "
                f"for request {target}?"
            ),
            (
                f"What decision follows for request {target} on whether "
                f"the exception permits someone to {action}?"
            ),
        ]

    if composition == "definition_chain":
        return [
            (
                f"For the environment associated with request {target}, "
                f"may someone {action}?"
            ),
            (
                f"Does the environment definition for request {target} "
                f"permit someone to {action}?"
            ),
            (
                f"What decision follows for request {target} on whether "
                f"the defined environment allows someone to {action}?"
            ),
        ]

    if composition == "delegated_authority_status":
        return [
            (
                f"For request {target}, is there valid authority for "
                f"someone to {action}?"
            ),
            (
                f"Does the supplied delegation record validly authorize "
                f"someone to {action} for request {target}?"
            ),
            (
                f"What decision follows for request {target} about "
                f"whether valid delegated authority exists to {action}?"
            ),
        ]

    if composition == "purpose_scope_definition":
        return [
            (
                f"For the use described by request {target}, "
                f"may someone {action}?"
            ),
            (
                f"Does the governing purpose rule permit someone to "
                f"{action} for request {target}?"
            ),
            (
                f"What decision follows for request {target} on whether "
                f"the proposed purpose allows someone to {action}?"
            ),
        ]

    if composition == "supersession_current_rule":
        return [
            (
                f"Under the rule currently in force for request {target}, "
                f"may someone {action}?"
            ),
            (
                f"Does the current policy permit someone to {action} "
                f"for request {target}?"
            ),
            (
                f"What decision follows for request {target} on whether "
                f"the currently controlling rule allows someone to {action}?"
            ),
        ]

    if composition == "temporal_override":
        return [
            (
                f"At the time described for request {target}, "
                f"may someone {action}?"
            ),
            (
                f"Does the applicable timing rule permit someone to "
                f"{action} for request {target}?"
            ),
            (
                f"What decision follows for request {target} on whether "
                f"someone may {action} at the stated time?"
            ),
        ]

    if composition == "jurisdiction_definition_authorization":
        return [
            (
                f"For the destination associated with request {target}, "
                f"may someone {action}?"
            ),
            (
                f"Does the governing jurisdiction permit someone to "
                f"{action} for request {target}?"
            ),
            (
                f"What decision follows for request {target} on whether "
                f"the destination falls within a jurisdiction allowing "
                f"someone to {action}?"
            ),
        ]

    if composition == "permission_obligation_scope":
        return [
            (
                f"For request {target}, is someone required to {action}?"
            ),
            (
                f"Does the supplied policy make it mandatory to "
                f"{action} for request {target}?"
            ),
            (
                f"What decision follows for request {target} on whether "
                f"someone must {action}?"
            ),
        ]

    raise ValueError(
        f"Unknown composition: {composition}"
    )


def targetize_evidence(
    evidence,
    target,
):
    """
    Make each passage explicitly belong to one request.

    Four request-specific policy chunks will coexist in every world.
    """

    prefix = slug(target)

    result = []

    for item in deepcopy(evidence):

        result.append(
            {
                "id":
                    f"{prefix}__{item['id']}",

                "source":
                    item["source"],

                "section":
                    f"{target} | {item['section']}",

                "text":
                    f"For request {target}: {item['text']}",
            }
        )

    return result


def targetize_citations(
    citations,
    target,
):
    prefix = slug(target)

    return [
        f"{prefix}__{cid}"
        for cid in citations
    ]


def family_targets(
    split,
    family_index,
):
    pool = TARGET_POOLS[split]

    start = (
        family_index * 4
    ) % len(pool)

    return [
        pool[
            (start + i) % len(pool)
        ]
        for i in range(4)
    ]


def family_assets(
    family_index,
):
    return [
        ASSETS[
            family_index % len(ASSETS)
        ],
        ASSETS[
            (family_index + 2) % len(ASSETS)
        ],
    ]


def matrix_permutations(
    family_name,
):
    rng = random.Random(
        stable_seed(
            SEED,
            family_name,
            "matrix",
        )
    )

    row_perm = list(range(4))
    col_perm = list(range(4))

    rng.shuffle(row_perm)
    rng.shuffle(col_perm)

    return (
        row_perm,
        col_perm,
    )


def render_evidence(evidence):
    blocks = []

    for item in evidence:

        blocks.append(
            "\n".join([
                f"[{item['id']}]",
                f"Source: {item['source']}",
                f"Section: {item['section']}",
                f"Text: {item['text']}",
            ])
        )

    return (
        "SUPPLIED EVIDENCE\n"
        "=================\n"
        + "\n\n".join(blocks)
    )


def render_input(
    evidence,
    question,
):
    return (
        render_evidence(evidence)
        + "\n\n"
        + "QUESTION\n"
        + "========\n"
        + question
    )


def render_output(
    decision,
    rationale,
    citations,
    conditions,
):
    obj = {
        "decision": decision,
        "rationale": rationale,
        "citations": citations,
        "conditions": conditions,
    }

    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
    )


def validate_base_family(
    split,
    family,
):
    cases = family["cases"]

    if set(cases) != set(LABELS):
        raise ValueError(
            f"{split}/{family['name']}: "
            f"expected four labels, got {sorted(cases)}"
        )

    # V4.2's causal architecture deliberately uses one exact
    # question across the four evidence worlds.
    questions = {
        case["question"]
        for case in cases.values()
    }

    if len(questions) != 1:
        raise ValueError(
            f"{split}/{family['name']}: "
            "base V4.2 cases do not share one question"
        )

    for label, case in cases.items():

        ids = {
            item["id"]
            for item in case["evidence"]
        }

        invalid = (
            set(case["citations"])
            - ids
        )

        if invalid:
            raise ValueError(
                f"{split}/{family['name']}/{label}: "
                f"invalid citations {sorted(invalid)}"
            )


def build_family(
    split,
    family_index,
    family,
):
    validate_base_family(
        split,
        family,
    )

    name = family["name"]
    composition = family["composition"]
    cases = family["cases"]

    targets = family_targets(
        split,
        family_index,
    )

    row_perm, col_perm = (
        matrix_permutations(
            name
        )
    )

    canonical_question = (
        cases["YES"]["question"]
    )

    action = extract_action(
        canonical_question
    )

    rows = []

    for asset_index, asset in enumerate(
        family_assets(
            family_index
        )
    ):

        for world in range(4):

            evidence_world = []
            target_meta = {}

            for target_index, target in enumerate(
                targets
            ):

                label = MATRIX[
                    row_perm[world]
                ][
                    col_perm[target_index]
                ]

                base_case = fill_asset(
                    cases[label],
                    asset,
                )

                target_evidence = (
                    targetize_evidence(
                        base_case["evidence"],
                        target,
                    )
                )

                evidence_world.extend(
                    target_evidence
                )

                target_meta[target] = {
                    "decision":
                        label,

                    "citations":
                        targetize_citations(
                            base_case["citations"],
                            target,
                        ),

                    "conditions":
                        deepcopy(
                            base_case["conditions"]
                        ),

                    "rationale":
                        base_case["rationale"],

                    "question_type":
                        base_case["question_type"],

                    "reasoning_tags":
                        deepcopy(
                            base_case[
                                "reasoning_tags"
                            ]
                        ),
                }

            # Evidence order is independent of label/target position.
            rng = random.Random(
                stable_seed(
                    SEED,
                    split,
                    name,
                    asset_index,
                    world,
                    "evidence-order",
                )
            )

            rng.shuffle(
                evidence_world
            )

            populated_action = (
                action.replace(
                    "{asset}",
                    asset,
                )
            )

            for target in targets:

                meta = target_meta[
                    target
                ]

                questions = questions_for(
                    composition,
                    target,
                    populated_action,
                )

                scenario_id = (
                    f"{split}__{name}"
                    f"__world_{world}"
                    f"__target_{slug(target)}"
                    f"__asset_{asset_index}"
                )

                for surface, question in enumerate(
                    questions
                ):

                    row = {
                        "split":
                            split,

                        "family":
                            name,

                        "composition":
                            composition,

                        "scenario_id":
                            scenario_id,

                        "world":
                            world,

                        "target":
                            target,

                        "asset":
                            asset,

                        "surface_variant":
                            surface,

                        "question_type":
                            meta["question_type"],

                        "reasoning_tags":
                            meta["reasoning_tags"],

                        "instruction":
                            INSTRUCTION,

                        "input":
                            render_input(
                                evidence_world,
                                question,
                            ),

                        "output":
                            render_output(
                                meta["decision"],
                                meta["rationale"],
                                meta["citations"],
                                meta["conditions"],
                            ),
                    }

                    rows.append(row)

    return rows


def build_split(
    split,
    families,
):
    rows = []

    for family_index, family in enumerate(
        families
    ):
        rows.extend(
            build_family(
                split,
                family_index,
                family,
            )
        )

    rng = random.Random(
        stable_seed(
            SEED,
            split,
            "row-order",
        )
    )

    rng.shuffle(rows)

    return rows


def audit_split(
    split,
    rows,
):
    decisions = Counter(
        json.loads(
            row["output"]
        )["decision"]
        for row in rows
    )

    compositions = Counter(
        row["composition"]
        for row in rows
    )

    families = {
        row["family"]
        for row in rows
    }

    scenarios = {
        row["scenario_id"]
        for row in rows
    }

    exact_inputs = [
        row["input"]
        for row in rows
    ]

    print()
    print(split.upper())
    print("-" * 72)

    print(
        "rows:",
        len(rows),
    )

    print(
        "scenarios:",
        len(scenarios),
    )

    print(
        "families:",
        len(families),
    )

    print(
        "decisions:",
        dict(
            sorted(
                decisions.items()
            )
        ),
    )

    print(
        "compositions:",
        dict(
            sorted(
                compositions.items()
            )
        ),
    )

    print(
        "duplicate exact inputs:",
        len(exact_inputs)
        - len(set(exact_inputs)),
    )


def overlap(
    a,
    b,
    key,
):
    aa = {
        row[key]
        for row in a
    }

    bb = {
        row[key]
        for row in b
    }

    return len(
        aa & bb
    )


def write_jsonl(
    path,
    rows,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():

    print("=" * 72)
    print(
        "SENTINEL V4.3 FULL RELATIONAL CURRICULUM"
    )
    print("=" * 72)

    datasets = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        datasets[split] = build_split(
            split,
            BASE_FAMILIES[split],
        )

        audit_split(
            split,
            datasets[split],
        )


    print()
    print("CROSS-SPLIT EXACT LEAKAGE")
    print("-" * 72)

    pairs = [
        (
            "train",
            "validation",
        ),
        (
            "train",
            "test",
        ),
        (
            "validation",
            "test",
        ),
    ]

    failed = False

    for left, right in pairs:

        family_overlap = overlap(
            datasets[left],
            datasets[right],
            "family",
        )

        scenario_overlap = overlap(
            datasets[left],
            datasets[right],
            "scenario_id",
        )

        input_overlap = overlap(
            datasets[left],
            datasets[right],
            "input",
        )

        print(
            f"{left} vs {right}: "
            f"family={family_overlap} "
            f"scenario={scenario_overlap} "
            f"exact_input={input_overlap}"
        )

        if (
            family_overlap
            or scenario_overlap
            or input_overlap
        ):
            failed = True


    expected = {
        "train": {
            "families": 30,
            "rows": 2880,
            "per_label": 720,
        },

        "validation": {
            "families": 10,
            "rows": 960,
            "per_label": 240,
        },

        "test": {
            "families": 10,
            "rows": 960,
            "per_label": 240,
        },
    }


    print()
    print("EXPECTED SHAPE AUDIT")
    print("-" * 72)

    for split, target in expected.items():

        rows = datasets[split]

        families = {
            row["family"]
            for row in rows
        }

        decisions = Counter(
            json.loads(
                row["output"]
            )["decision"]
            for row in rows
        )

        ok = True

        if len(rows) != target["rows"]:
            ok = False

        if len(families) != target["families"]:
            ok = False

        for label in LABELS:
            if (
                decisions[label]
                != target["per_label"]
            ):
                ok = False

        print(
            split,
            "PASS" if ok else "FAIL",
        )

        if not ok:
            failed = True


    if failed:
        raise SystemExit(
            "V4.3 BUILD AUDIT FAILED"
        )


    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "train":
            OUT / "sentinel_train_v43.jsonl",

        "validation":
            OUT / "sentinel_validation_v43.jsonl",

        "test":
            OUT / "sentinel_test_v43.jsonl",
    }

    for split, path in paths.items():

        write_jsonl(
            path,
            datasets[split],
        )

        print(
            f"WROTE {split}: "
            f"{path} "
            f"({len(datasets[split])} rows)"
        )


    print()
    print("=" * 72)
    print(
        "V4.3 FULL RELATIONAL BUILD PASSED"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
