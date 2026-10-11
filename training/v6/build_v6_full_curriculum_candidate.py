from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import sys


BASE = Path("/workspace")

sys.path.insert(
    0,
    str(BASE),
)

from training.v6.v6_family_specs import (
    FAMILY_SPECS,
    validate_family_specs,
)


TRAIN_OUT = (
    BASE
    / "training/v6/full_candidate/"
      "sentinel_v6_train_candidate.jsonl"
)

DEV_OUT = (
    BASE
    / "training/v6/full_candidate/"
      "sentinel_v6_development_candidate.jsonl"
)

MANIFEST_OUT = (
    BASE
    / "experiments/v6/full_candidate/"
      "V6_FULL_CURRICULUM_CANDIDATE_MANIFEST.json"
)


LABELS = (
    "YES",
    "CONDITIONAL",
    "INSUFFICIENT",
    "NO",
)


INSTRUCTION = """You are Sentinel, a policy-governance decision model.

Use ONLY the supplied evidence.

Evaluate the exact proposition in causal order:

1. bind facts to the exact target
2. determine whether a governing rule applies
3. identify the operative rule
4. resolve exception state when relevant
5. resolve each material requirement as SATISFIED, FAILED, or UNRESOLVED
6. return the decision implied by the completed state

Choose exactly one:

YES
NO
CONDITIONAL
INSUFFICIENT

INSUFFICIENT:
The evidence does not establish that a governing rule applies to the exact proposition.

NO:
An applicable controlling prohibition or explicitly failed required state prevents the action.

CONDITIONAL:
Applicability is established, no controlling requirement has failed, and at least one material requirement remains unresolved.

YES:
Applicability is established and every material requirement is satisfied.

Return valid JSON only with exactly these keys:

decision
citations
conditions
rationale

conditions is a complete material state ledger using:

SATISFIED | requirement
FAILED | requirement
UNRESOLVED | requirement""".strip()


ADJECTIVES = [
    "Amber", "Aster", "Aurora", "Beacon",
    "Birch", "Bronze", "Cedar", "Cipher",
    "Cobalt", "Copper", "Delta", "Drift",
    "Ember", "Finch", "Grove", "Harbor",
    "Helix", "Hollow", "Indigo", "Ion",
    "Ivory", "Jade", "Juncture", "Keystone",
    "Lantern", "Lark", "Loom", "Meridian",
    "Mosaic", "Moss", "Northstar", "Opal",
]

NOUNS = [
    "archive", "bundle", "collection", "corpus",
    "dataset", "dossier", "files", "ledger",
    "notebook", "packet", "package", "records",
    "register", "repository", "sample", "workspace",
    "catalog", "index", "matrix", "portfolio",
    "library", "registry", "vault", "journal",
    "brief", "recordset", "inventory", "compendium",
    "document", "artifact", "resource", "material",
    "study", "project", "volume", "series",
    "record", "folio", "casebook", "memorandum",
]


ACTIONS = [
    "release",
    "transfer",
    "process",
    "access",
    "publish",
    "connect",
    "use",
    "export",
]


PURPOSES = [
    "controlled analysis",
    "approved research",
    "internal validation",
    "authorized review",
    "documented testing",
    "restricted collaboration",
    "governed processing",
    "supervised evaluation",
]


REQUIREMENTS = {
    "target_entity_binding": (
        "identity confirmation",
        "request linkage review",
    ),

    "applicability_scope_resolution": (
        "operational approval",
        "recording control",
    ),

    "operative_rule_precedence": (
        "implementation approval",
        "audit logging",
    ),

    "exception_state_propagation": (
        "exception authorization",
        "exception activity logging",
    ),

    "requirement_ledger_composition": (
        "primary authorization",
        "secondary control validation",
    ),

    "final_decision_finalization": (
        "final review",
        "audit record",
    ),

    "temporal_authority_state": (
        "current-authority confirmation",
        "execution notice",
    ),

    "cross_gate_state_composition": (
        "downstream approval",
        "release logging",
    ),
}


TARGET_IDS = (
    "A",
    "B",
    "C",
    "D",
)


def text_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def target_name(index: int) -> str:
    # 32 * 40 = 1280 unique combinations,
    # exactly the number needed for:
    #
    # 40 families * 8 groups * 4 targets.
    adjective = ADJECTIVES[
        index // len(NOUNS)
    ]

    noun = NOUNS[
        index % len(NOUNS)
    ]

    return f"{adjective} {noun}"


def question_variants(
    action: str,
    target: str,
    purpose: str,
):
    return (
        (
            "Under the supplied evidence, may the requesting team "
            f"{action} the {target} for {purpose}?"
        ),
        (
            "Based only on the supplied material, is the requesting "
            f"team permitted to {action} the {target} for {purpose}?"
        ),
    )


def applicability_context(
    operator: str,
    target: str,
):
    if operator == "target_entity_binding":
        return (
            "The identifier and cross-reference bind the stated "
            f"governance facts specifically to the {target}; "
            "neighboring entities remain separate."
        )

    if operator == "applicability_scope_resolution":
        return (
            "The scope definition expressly includes this target "
            "and the stated use."
        )

    if operator == "operative_rule_precedence":
        return (
            "The current request-specific rule is identified as "
            "operative and displaces the competing general rule."
        )

    if operator == "exception_state_propagation":
        return (
            "The base rule governs this exact request and the stated "
            "exception trigger is active and valid for this request."
        )

    if operator == "requirement_ledger_composition":
        return (
            "The governing permission identifies two independent "
            "material requirements that must be resolved separately."
        )

    if operator == "final_decision_finalization":
        return (
            "Applicability and the operative permission state are "
            "established for this exact request."
        )

    if operator == "temporal_authority_state":
        return (
            "The authority record is current and operative at the "
            "time of this exact request."
        )

    if operator == "cross_gate_state_composition":
        return (
            "Target binding, applicability, and operative-rule "
            "selection are established in causal order for this request."
        )

    raise ValueError(
        f"unknown operator: {operator}"
    )


def insufficient_context(
    operator: str,
    target: str,
):
    if operator == "target_entity_binding":
        return (
            "A nearby record concerns a similar entity, but the "
            "supplied material does not bind a rule to this exact "
            f"target ({target})."
        )

    if operator == "applicability_scope_resolution":
        return (
            "The material describes a neighboring scope, but does "
            "not establish membership of this exact request."
        )

    if operator == "operative_rule_precedence":
        return (
            "Several candidate rules are mentioned, but the supplied "
            "material does not establish which rule governs this exact request."
        )

    if operator == "exception_state_propagation":
        return (
            "An exception is described for related activity, but the "
            "supplied material does not establish a governing rule for "
            "this exact request."
        )

    if operator == "requirement_ledger_composition":
        return (
            "Possible controls are listed for related requests, but "
            "no governing rule for this exact request is established."
        )

    if operator == "final_decision_finalization":
        return (
            "Related decision criteria are supplied, but applicability "
            "of a governing rule to this exact request is not established."
        )

    if operator == "temporal_authority_state":
        return (
            "Historical authority is mentioned, but the supplied "
            "material does not establish governance and current authority "
            "for this exact request."
        )

    if operator == "cross_gate_state_composition":
        return (
            "Several neighboring governance facts are supplied, but "
            "the causal chain never establishes a governing rule for "
            "this exact request."
        )

    raise ValueError(
        f"unknown operator: {operator}"
    )


def phrasing(variant: int):
    if variant == 0:
        return {
            "no_governance":
                "The supplied material does not establish a governing "
                "rule for this exact request.",

            "permit":
                "The controlling rule permits the requested action.",

            "prohibit":
                "The controlling rule prohibits the requested action.",

            "satisfied":
                "The material requirement '{r}' is documented as satisfied.",

            "unresolved":
                "The supplied evidence does not establish whether the "
                "material requirement '{r}' is satisfied.",

            "failed":
                "The material requirement '{r}' is documented as failed.",
        }

    return {
        "no_governance":
            "Governance for this exact request is not established by "
            "the supplied evidence.",

        "permit":
            "Under the operative rule, the requested action is allowed.",

        "prohibit":
            "Under the operative rule, the requested action is barred.",

        "satisfied":
            "The record confirms completion of the required control '{r}'.",

        "unresolved":
            "The record leaves the required control '{r}' unresolved.",

        "failed":
            "The record confirms that the required control '{r}' failed.",
    }


def build_track(
    *,
    operator: str,
    target: str,
    action: str,
    purpose: str,
    decision: str,
    variant: int,
):
    p = phrasing(
        variant
    )

    req1, req2 = REQUIREMENTS[
        operator
    ]

    head = (
        f"Target record: {target}. "
        f"Requested operation: {action} for {purpose}. "
    )

    if decision == "INSUFFICIENT":
        text = (
            head
            + insufficient_context(
                operator,
                target,
            )
            + " "
            + p["no_governance"]
        )

        conditions = []

        rationale = (
            "The supplied evidence does not establish governance "
            "for the exact target proposition."
        )

        return text, conditions, rationale


    context = applicability_context(
        operator,
        target,
    )


    if decision == "YES":
        text = (
            head
            + context
            + " "
            + p["permit"]
            + " "
            + p["satisfied"].format(
                r=req1
            )
            + " "
            + p["satisfied"].format(
                r=req2
            )
        )

        conditions = [
            f"SATISFIED | {req1}",
            f"SATISFIED | {req2}",
        ]

        rationale = (
            "The exact request is governed and permitted, and every "
            "material requirement is established as satisfied."
        )

        return text, conditions, rationale


    if decision == "CONDITIONAL":
        text = (
            head
            + context
            + " "
            + p["permit"]
            + " "
            + p["satisfied"].format(
                r=req1
            )
            + " "
            + p["unresolved"].format(
                r=req2
            )
        )

        conditions = [
            f"SATISFIED | {req1}",
            f"UNRESOLVED | {req2}",
        ]

        rationale = (
            "Applicability and permission are established, no "
            "material requirement has failed, and one material "
            "requirement remains unresolved."
        )

        return text, conditions, rationale


    if decision == "NO":
        # Half of the relational groups realize NO as a direct
        # controlling prohibition; the other half realize NO through
        # an explicitly failed required state.
        if variant == 0:
            if operator == "exception_state_propagation":
                context = (
                    "The base rule governs this exact request and no "
                    "active exception displaces that base rule."
                )

            text = (
                head
                + context
                + " "
                + p["prohibit"]
            )

            conditions = []

            rationale = (
                "A controlling rule applies to the exact request "
                "and directly prohibits the requested action."
            )

        else:
            text = (
                head
                + context
                + " "
                + p["permit"]
                + " "
                + p["satisfied"].format(
                    r=req1
                )
                + " "
                + p["failed"].format(
                    r=req2
                )
            )

            conditions = [
                f"SATISFIED | {req1}",
                f"FAILED | {req2}",
            ]

            rationale = (
                "The exact request is governed, but an explicitly "
                "failed material requirement prevents the action."
            )

        return text, conditions, rationale

    raise ValueError(
        f"unknown decision: {decision}"
    )


def evidence_order(
    surface_variant: int,
):
    # IMPORTANT:
    #
    # Evidence-ref assignment must NOT depend on world_index or
    # label_shift. Otherwise evidence position could become a label
    # shortcut after the question identifies the target.
    #
    # Both surface variants contain all four label shifts.
    if surface_variant == 0:
        return (
            0,
            1,
            2,
            3,
        )

    return (
        2,
        0,
        3,
        1,
    )


def render_evidence(
    tracks,
    surface_variant,
):
    order = evidence_order(
        surface_variant
    )

    ref_for_target = {}

    pieces = []

    for position, target_index in enumerate(
        order,
        1,
    ):
        ref = f"E{position}"

        ref_for_target[
            target_index
        ] = ref

        pieces.append(
            f"[{ref}] "
            + tracks[
                target_index
            ]["text"]
        )

    return (
        "\n\n".join(
            pieces
        ),
        ref_for_target,
    )


def render_input(
    question,
    evidence,
    input_order,
):
    if input_order == "question_first":
        return (
            "QUESTION\n"
            "========\n"
            f"{question}\n\n"
            "EVIDENCE\n"
            "========\n"
            f"{evidence}"
        )

    return (
        "EVIDENCE\n"
        "========\n"
        f"{evidence}\n\n"
        "QUESTION\n"
        "========\n"
        f"{question}"
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
                    sort_keys=True,
                )
                + "\n"
            )


def build_split(
    split_name,
    family_key,
    family_sequence_start,
):
    rows = []

    family_sequence = (
        family_sequence_start
    )

    for operator, spec in FAMILY_SPECS.items():
        families = spec[
            family_key
        ]

        for family in families:
            family_name = family[
                "family"
            ]

            for group_index in range(8):

                # Four shifts appear exactly once inside each surface /
                # group variant:
                #
                # groups 0..3 -> variant 0, shifts 0..3
                # groups 4..7 -> variant 1, shifts 0..3
                label_shift = (
                    group_index
                    % 4
                )

                surface_variant = (
                    group_index
                    // 4
                )

                group_variant = (
                    group_index
                    // 4
                )

                action = ACTIONS[
                    (
                        family_sequence
                        + group_index
                    )
                    % len(ACTIONS)
                ]

                purpose = PURPOSES[
                    (
                        2 * family_sequence
                        + group_index
                    )
                    % len(PURPOSES)
                ]

                targets = []

                for target_index in range(4):
                    global_target_index = (
                        family_sequence
                        * 32
                        + group_index
                        * 4
                        + target_index
                    )

                    name = target_name(
                        global_target_index
                    )

                    q_variants = question_variants(
                        action,
                        name,
                        purpose,
                    )

                    targets.append({
                        "target_index":
                            target_index,

                        "target_id":
                            TARGET_IDS[
                                target_index
                            ],

                        "target_name":
                            name,

                        "question":
                            q_variants[
                                surface_variant
                            ],
                    })


                relational_group = (
                    f"{family_name}__"
                    f"{split_name}__g"
                    f"{group_index:02d}"
                )

                scenario_id = (
                    f"v6full__"
                    f"{split_name}__"
                    f"{family_name}__"
                    f"g{group_index:02d}"
                )


                for world_index in range(4):

                    tracks = []

                    for target_index in range(4):
                        decision = LABELS[
                            (
                                target_index
                                + world_index
                                + label_shift
                            )
                            % 4
                        ]

                        text, conditions, rationale = (
                            build_track(
                                operator=operator,
                                target=targets[
                                    target_index
                                ]["target_name"],
                                action=action,
                                purpose=purpose,
                                decision=decision,
                                variant=group_variant,
                            )
                        )

                        tracks.append({
                            "decision":
                                decision,

                            "text":
                                text,

                            "conditions":
                                conditions,

                            "rationale":
                                rationale,
                        })


                    evidence, ref_for_target = (
                        render_evidence(
                            tracks,
                            surface_variant,
                        )
                    )

                    evidence_hash = text_hash(
                        evidence
                    )


                    for target_index in range(4):

                        target = targets[
                            target_index
                        ]

                        track = tracks[
                            target_index
                        ]

                        ref = ref_for_target[
                            target_index
                        ]

                        # Input order is based only on target identity,
                        # not world or label state.
                        input_order = (
                            "question_first"
                            if target_index
                            in {
                                0,
                                2,
                            }
                            else
                            "evidence_first"
                        )

                        output_obj = {
                            "decision":
                                track[
                                    "decision"
                                ],

                            "citations":
                                [ref],

                            "conditions":
                                track[
                                    "conditions"
                                ],

                            "rationale":
                                track[
                                    "rationale"
                                ],
                        }

                        row_id = (
                            f"{scenario_id}__"
                            f"w{world_index}__"
                            f"t{target_index}"
                        )

                        row = {
                            "evidence":
                                evidence,

                            "evidence_hash":
                                evidence_hash,

                            "family":
                                family_name,

                            "group_variant":
                                group_variant,

                            "input":
                                render_input(
                                    target[
                                        "question"
                                    ],
                                    evidence,
                                    input_order,
                                ),

                            "input_order":
                                input_order,

                            "instruction":
                                INSTRUCTION,

                            "operator":
                                operator,

                            "output":
                                json.dumps(
                                    output_obj,
                                    ensure_ascii=False,
                                    separators=(
                                        ",",
                                        ":",
                                    ),
                                ),

                            "question":
                                target[
                                    "question"
                                ],

                            "relational_group":
                                relational_group,

                            "row_id":
                                row_id,

                            "scenario_id":
                                scenario_id,

                            "split":
                                split_name,

                            "surface_variant":
                                surface_variant,

                            "target":
                                track[
                                    "decision"
                                ],

                            "target_evidence_ref":
                                ref,

                            "target_id":
                                target[
                                    "target_id"
                                ],

                            "target_index":
                                target_index,

                            "target_name":
                                target[
                                    "target_name"
                                ],

                            "world":
                                f"world_{world_index}",

                            "world_index":
                                world_index,
                        }

                        rows.append(
                            row
                        )

            family_sequence += 1

    return (
        rows,
        family_sequence,
    )


def summarize(
    name,
    rows,
):
    print()
    print(name)
    print("-" * 72)

    print(
        "rows:",
        len(rows),
    )

    print(
        "families:",
        len(
            {
                r["family"]
                for r in rows
            }
        ),
    )

    print(
        "relational groups:",
        len(
            {
                r["relational_group"]
                for r in rows
            }
        ),
    )

    print(
        "labels:",
        dict(
            sorted(
                Counter(
                    r["target"]
                    for r in rows
                ).items()
            )
        ),
    )

    print(
        "operators:",
        dict(
            sorted(
                Counter(
                    r["operator"]
                    for r in rows
                ).items()
            )
        ),
    )

    print(
        "input order:",
        dict(
            sorted(
                Counter(
                    r["input_order"]
                    for r in rows
                ).items()
            )
        ),
    )

    print(
        "surface variant:",
        dict(
            sorted(
                Counter(
                    r["surface_variant"]
                    for r in rows
                ).items()
            )
        ),
    )


def main():
    validation = (
        validate_family_specs()
    )

    if validation["errors"]:
        raise RuntimeError(
            "family-spec validation failed: "
            + repr(
                validation[
                    "errors"
                ]
            )
        )


    train_rows, next_sequence = (
        build_split(
            "train",
            "train",
            0,
        )
    )

    dev_rows, final_sequence = (
        build_split(
            "development",
            "development",
            next_sequence,
        )
    )


    if len(train_rows) != 4096:
        raise RuntimeError(
            f"TRAIN rows != 4096: "
            f"{len(train_rows)}"
        )

    if len(dev_rows) != 1024:
        raise RuntimeError(
            f"DEV rows != 1024: "
            f"{len(dev_rows)}"
        )

    if final_sequence != 40:
        raise RuntimeError(
            f"family sequence != 40: "
            f"{final_sequence}"
        )


    write_jsonl(
        TRAIN_OUT,
        train_rows,
    )

    write_jsonl(
        DEV_OUT,
        dev_rows,
    )


    manifest = {
        "experiment":
            "Sentinel V6 full curriculum candidate",

        "status":
            "NON_FROZEN_CANDIDATE",

        "geometry": {
            "operators": 8,
            "train_families": 32,
            "development_families": 8,
            "groups_per_family": 8,
            "rows_per_group": 16,
            "rows_per_family": 128,
            "train_rows": 4096,
            "development_rows": 1024,
            "total_rows": 5120,
        },

        "generation": {
            "semantic_state_formula":
                "(target_index + world_index + label_shift) mod 4",

            "label_shift_schedule":
                "group_index mod 4",

            "surface_variant_schedule":
                "group_index // 4",

            "group_variant_schedule":
                "group_index // 4",

            "input_order":
                "target-index parity only",

            "pilot_rows_reused":
                0,

            "challenge_v4_files_read":
                0,

            "model_loads":
                0,

            "inference_calls":
                0,
        },

        "artifacts": {
            "train": {
                "path":
                    str(
                        TRAIN_OUT.relative_to(
                            BASE
                        )
                    ),

                "sha256":
                    hashlib.sha256(
                        TRAIN_OUT.read_bytes()
                    ).hexdigest(),
            },

            "development": {
                "path":
                    str(
                        DEV_OUT.relative_to(
                            BASE
                        )
                    ),

                "sha256":
                    hashlib.sha256(
                        DEV_OUT.read_bytes()
                    ).hexdigest(),
            },
        },
    }


    MANIFEST_OUT.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    print()
    print("=" * 72)
    print("SENTINEL V6 FULL CURRICULUM CANDIDATE GENERATED")
    print("=" * 72)

    summarize(
        "TRAIN",
        train_rows,
    )

    summarize(
        "DEVELOPMENT",
        dev_rows,
    )

    print()
    print(
        "TRAIN SHA256:",
        hashlib.sha256(
            TRAIN_OUT.read_bytes()
        ).hexdigest(),
    )

    print(
        "DEVELOPMENT SHA256:",
        hashlib.sha256(
            DEV_OUT.read_bytes()
        ).hexdigest(),
    )

    print(
        "MANIFEST SHA256:",
        hashlib.sha256(
            MANIFEST_OUT.read_bytes()
        ).hexdigest(),
    )

    print()
    print("Model loads: 0")
    print("Inference: 0")
    print("Challenge V4 files read: 0")
    print("Individual rows printed: 0")


if __name__ == "__main__":
    main()
