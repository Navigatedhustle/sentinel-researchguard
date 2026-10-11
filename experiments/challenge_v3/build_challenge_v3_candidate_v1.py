"""
Build Challenge V3 Candidate V1.

Prints aggregate counts only.
Never prints individual benchmark content.
"""

from collections import Counter
from pathlib import Path
import hashlib
import json


from experiments.challenge_v3.challenge_v3_blueprint import (
    COMPOSITIONS,
    GOLD_KEYS,
    INPUT_ORDERS,
    LABELS,
    PUBLIC_CASE_KEYS,
    RELATIONAL_TARGETS,
    RELATIONAL_WORLDS,
    relational_label,
)

from experiments.challenge_v3.challenge_v3_oracle import (
    ExceptionState,
    Requirement,
    SemanticState,
    decide,
    state_to_dict,
)

from experiments.challenge_v3.challenge_v3_realizers import (
    DISTRACTORS,
    INSTRUCTION,
    RELATIONAL_ENTITIES,
    SPECS,
    STRESS_ENTITY,
    build_input,
    question_for,
    rationale_for,
    render_target_clause,
)


BASE = Path("/workspace")

OUT_DIR = (
    BASE
    / "experiments/challenge_v3/candidates"
)

CANDIDATE = (
    OUT_DIR
    / "sentinel_challenge_v3_candidate_v1.json"
)

HIDDEN = (
    OUT_DIR
    / "sentinel_challenge_v3_candidate_v1_hidden_semantics.jsonl"
)


ORDER_MATRIX = (
    (0, 0, 1, 1),
    (1, 0, 0, 1),
    (1, 1, 0, 0),
    (0, 1, 1, 0),
)


def text_hash(
    text,
):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def requirement_tuple(
    names,
    states,
):

    return tuple(
        Requirement(
            name,
            state,
        )
        for name, state
        in zip(
            names,
            states,
        )
    )


def semantic_state_for(
    composition,
    label,
    variant,
):

    spec = SPECS[
        composition
    ]


    if label == "INSUFFICIENT":

        return SemanticState(
            applicability="UNRESOLVED",
            base_rule="NONE",
        )


    mode = spec[
        "mode"
    ]


    if mode == "exception":

        trigger_name = spec[
            "trigger"
        ]

        requirement_names = spec[
            "requirements"
        ]


        if label == "YES":

            trigger_state = "SATISFIED"

            req_states = [
                "SATISFIED"
                for _ in requirement_names
            ]


        elif label == "NO":

            if (
                variant
                % (
                    len(
                        requirement_names
                    )
                    + 1
                )
                == 0
            ):

                trigger_state = "FAILED"

                req_states = [
                    "SATISFIED"
                    for _ in requirement_names
                ]

            else:

                trigger_state = "SATISFIED"

                req_states = [
                    "SATISFIED"
                    for _ in requirement_names
                ]

                req_states[
                    (
                        variant
                        - 1
                    )
                    % len(
                        requirement_names
                    )
                ] = "FAILED"


        elif label == "CONDITIONAL":

            if (
                variant
                % (
                    len(
                        requirement_names
                    )
                    + 1
                )
                == 0
            ):

                trigger_state = "UNRESOLVED"

                req_states = [
                    "SATISFIED"
                    for _ in requirement_names
                ]

            else:

                trigger_state = "SATISFIED"

                req_states = [
                    "SATISFIED"
                    for _ in requirement_names
                ]

                req_states[
                    (
                        variant
                        - 1
                    )
                    % len(
                        requirement_names
                    )
                ] = "UNRESOLVED"


        else:

            raise ValueError(
                label
            )


        return SemanticState(

            applicability="ESTABLISHED",

            base_rule="PROHIBITION",

            exception=ExceptionState(

                trigger=Requirement(
                    trigger_name,
                    trigger_state,
                ),

                requirements=requirement_tuple(
                    requirement_names,
                    req_states,
                ),
            ),
        )


    requirement_names = spec[
        "requirements"
    ]


    if label == "YES":

        return SemanticState(

            applicability="ESTABLISHED",

            base_rule="PERMISSION",

            requirements=requirement_tuple(
                requirement_names,
                [
                    "SATISFIED"
                    for _ in requirement_names
                ],
            ),
        )


    if label == "NO":

        if variant % 3 == 0:

            return SemanticState(
                applicability="ESTABLISHED",
                base_rule="PROHIBITION",
            )


        states = [
            "SATISFIED"
            for _ in requirement_names
        ]

        states[
            variant
            % len(
                requirement_names
            )
        ] = "FAILED"


        return SemanticState(

            applicability="ESTABLISHED",

            base_rule="PERMISSION",

            requirements=requirement_tuple(
                requirement_names,
                states,
            ),
        )


    if label == "CONDITIONAL":

        states = [
            "SATISFIED"
            for _ in requirement_names
        ]

        states[
            variant
            % len(
                requirement_names
            )
        ] = "UNRESOLVED"


        return SemanticState(

            applicability="ESTABLISHED",

            base_rule="PERMISSION",

            requirements=requirement_tuple(
                requirement_names,
                states,
            ),
        )


    raise ValueError(
        label
    )


def input_order_relational(
    composition_index,
    world_index,
    target_index,
):

    bit = (
        ORDER_MATRIX[
            world_index
        ][
            target_index
        ]
        ^
        (
            composition_index
            % 2
        )
    )


    return (
        "evidence_first"
        if bit == 0
        else "question_first"
    )


def input_order_stress(
    composition_index,
    label_index,
):

    return (
        "evidence_first"
        if (
            composition_index
            + label_index
        )
        % 2 == 0
        else "question_first"
    )


def pack_relational_evidence(
    *,
    composition_index,
    world_index,
    clauses,
):

    desired_count = (
        1
        +
        (
            composition_index
            + world_index
        )
        % 5
    )


    target_ids = list(
        RELATIONAL_TARGETS
    )


    rotation = (
        composition_index
        + world_index
    ) % 4


    target_ids = (
        target_ids[
            rotation:
        ]
        +
        target_ids[
            :rotation
        ]
    )


    refs = {}


    blocks = []


    if desired_count == 1:

        text = " ".join(
            clauses[
                target_id
            ]
            for target_id
            in target_ids
        )

        text += " " + DISTRACTORS[
            composition_index
            % len(
                DISTRACTORS
            )
        ]


        blocks = [
            text
        ]


        for target_id in target_ids:

            refs[
                target_id
            ] = "E1"


    elif desired_count == 2:

        groups = (
            target_ids[
                :2
            ],
            target_ids[
                2:
            ],
        )


        for block_index, group in enumerate(
            groups,
            start=1,
        ):

            text = " ".join(
                clauses[
                    target_id
                ]
                for target_id
                in group
            )

            text += " " + DISTRACTORS[
                (
                    composition_index
                    + block_index
                )
                % len(
                    DISTRACTORS
                )
            ]


            blocks.append(
                text
            )


            for target_id in group:

                refs[
                    target_id
                ] = (
                    f"E{block_index}"
                )


    elif desired_count == 3:

        groups = (
            target_ids[
                :2
            ],
            (
                target_ids[
                    2
                ],
            ),
            (
                target_ids[
                    3
                ],
            ),
        )


        for block_index, group in enumerate(
            groups,
            start=1,
        ):

            text = " ".join(
                clauses[
                    target_id
                ]
                for target_id
                in group
            )


            if block_index == 3:

                text += " " + DISTRACTORS[
                    (
                        composition_index
                        + world_index
                    )
                    % len(
                        DISTRACTORS
                    )
                ]


            blocks.append(
                text
            )


            for target_id in group:

                refs[
                    target_id
                ] = (
                    f"E{block_index}"
                )


    else:

        for block_index, target_id in enumerate(
            target_ids,
            start=1,
        ):

            blocks.append(
                clauses[
                    target_id
                ]
            )

            refs[
                target_id
            ] = (
                f"E{block_index}"
            )


        if desired_count == 5:

            blocks.append(
                DISTRACTORS[
                    (
                        composition_index
                        + world_index
                    )
                    % len(
                        DISTRACTORS
                    )
                ]
            )


    evidence = "\n\n".join(
        f"[E{index}] {text}"
        for index, text
        in enumerate(
            blocks,
            start=1,
        )
    )


    return (
        evidence,
        refs,
        desired_count,
    )


def pack_stress_evidence(
    *,
    composition_index,
    label_index,
    target_clause,
):

    desired_count = (
        1
        +
        (
            composition_index
            % 5
        )
    )


    target_block = (
        (
            composition_index
            + label_index
        )
        % desired_count
    )


    blocks = []


    for block_index in range(
        desired_count
    ):

        if block_index == target_block:

            blocks.append(
                target_clause
            )

        else:

            blocks.append(
                DISTRACTORS[
                    (
                        composition_index
                        + label_index
                        + block_index
                    )
                    % len(
                        DISTRACTORS
                    )
                ]
            )


    evidence = "\n\n".join(
        f"[E{index}] {text}"
        for index, text
        in enumerate(
            blocks,
            start=1,
        )
    )


    return (
        evidence,
        f"E{target_block + 1}",
        desired_count,
    )


def gold_object(
    *,
    result,
    citation_ref,
):

    return {

        "decision":
            result.decision,

        "citations": [
            citation_ref
        ],

        "conditions":
            list(
                result.conditions
            ),

        "rationale":
            rationale_for(
                result
            ),
    }


def main():

    if CANDIDATE.exists() or HIDDEN.exists():

        raise RuntimeError(
            "Candidate outputs already exist."
        )


    public_cases = []

    hidden_rows = []


    # ================================================================
    # 192 RELATIONAL CASES
    # ================================================================

    for composition_index, composition in enumerate(
        COMPOSITIONS
    ):

        entities = RELATIONAL_ENTITIES[
            composition
        ]


        for world_index, world in enumerate(
            RELATIONAL_WORLDS
        ):

            states = {}

            results = {}

            clauses = {}


            for target_index, target_id in enumerate(
                RELATIONAL_TARGETS
            ):

                label = relational_label(
                    world_index,
                    target_index,
                )


                state = semantic_state_for(
                    composition,
                    label,
                    (
                        composition_index
                        + target_index
                    ),
                )


                result = decide(
                    state
                )


                if result.decision != label:

                    raise RuntimeError(
                        "Benchmark-local oracle disagrees with relational rotation."
                    )


                states[
                    target_id
                ] = state

                results[
                    target_id
                ] = result


                clauses[
                    target_id
                ] = render_target_clause(
                    composition,
                    entities[
                        target_index
                    ],
                    state,
                )


            evidence, refs, evidence_count = pack_relational_evidence(

                composition_index=composition_index,

                world_index=world_index,

                clauses=clauses,
            )


            evidence_hash = text_hash(
                evidence
            )


            for target_index, target_id in enumerate(
                RELATIONAL_TARGETS
            ):

                entity = entities[
                    target_index
                ]


                question = question_for(
                    composition,
                    entity,
                )


                order = input_order_relational(
                    composition_index,
                    world_index,
                    target_index,
                )


                result = results[
                    target_id
                ]


                case_id = (
                    f"cv3_{len(public_cases) + 1:04d}"
                )


                public_cases.append({

                    "case_id":
                        case_id,

                    "composition":
                        composition,

                    "question_type":
                        "relational_counterfactual",

                    "input_order":
                        order,

                    "instruction":
                        INSTRUCTION,

                    "question":
                        question,

                    "evidence_text":
                        evidence,

                    "input":
                        build_input(
                            evidence,
                            question,
                            order,
                        ),

                    "gold":
                        gold_object(
                            result=result,
                            citation_ref=refs[
                                target_id
                            ],
                        ),
                })


                hidden_rows.append({

                    "case_id":
                        case_id,

                    "composition":
                        composition,

                    "question_type":
                        "relational_counterfactual",

                    "relational_system":
                        (
                            f"{composition}__rel"
                        ),

                    "world_index":
                        world_index,

                    "target_index":
                        target_index,

                    "target_id":
                        target_id,

                    "target_entity":
                        entity,

                    "semantic_state":
                        state_to_dict(
                            states[
                                target_id
                            ]
                        ),

                    "target_evidence_ref":
                        refs[
                            target_id
                        ],

                    "question_hash":
                        text_hash(
                            question
                        ),

                    "evidence_hash":
                        evidence_hash,

                    "evidence_count":
                        evidence_count,
                })


    # ================================================================
    # 48 STRESS CASES
    # ================================================================

    for composition_index, composition in enumerate(
        COMPOSITIONS
    ):

        entity = STRESS_ENTITY[
            composition
        ]


        question = question_for(
            composition,
            entity,
        )


        for label_index, label in enumerate(
            LABELS
        ):

            state = semantic_state_for(
                composition,
                label,
                (
                    composition_index
                    + label_index
                    + 7
                ),
            )


            result = decide(
                state
            )


            if result.decision != label:

                raise RuntimeError(
                    "Benchmark-local oracle disagrees with stress label."
                )


            target_clause = render_target_clause(
                composition,
                entity,
                state,
            )


            evidence, citation_ref, evidence_count = pack_stress_evidence(

                composition_index=composition_index,

                label_index=label_index,

                target_clause=target_clause,
            )


            order = input_order_stress(
                composition_index,
                label_index,
            )


            case_id = (
                f"cv3_{len(public_cases) + 1:04d}"
            )


            public_cases.append({

                "case_id":
                    case_id,

                "composition":
                    composition,

                "question_type":
                    "stress",

                "input_order":
                    order,

                "instruction":
                    INSTRUCTION,

                "question":
                    question,

                "evidence_text":
                    evidence,

                "input":
                    build_input(
                        evidence,
                        question,
                        order,
                    ),

                "gold":
                    gold_object(
                        result=result,
                        citation_ref=citation_ref,
                    ),
            })


            hidden_rows.append({

                "case_id":
                    case_id,

                "composition":
                    composition,

                "question_type":
                    "stress",

                "stress_label_index":
                    label_index,

                "semantic_state":
                    state_to_dict(
                        state
                    ),

                "target_evidence_ref":
                    citation_ref,

                "question_hash":
                    text_hash(
                        question
                    ),

                "evidence_hash":
                    text_hash(
                        evidence
                    ),

                "evidence_count":
                    evidence_count,
            })


    # ================================================================
    # HARD PRE-WRITE STRUCTURAL CHECK
    # ================================================================

    if len(
        public_cases
    ) != 240:

        raise RuntimeError(
            f"Expected 240 cases, found {len(public_cases)}."
        )


    labels = Counter(
        case[
            "gold"
        ][
            "decision"
        ]
        for case
        in public_cases
    )


    if labels != Counter({
        "YES": 60,
        "NO": 60,
        "CONDITIONAL": 60,
        "INSUFFICIENT": 60,
    }):

        raise RuntimeError(
            f"Label imbalance: {labels}"
        )


    question_types = Counter(
        case[
            "question_type"
        ]
        for case
        in public_cases
    )


    if question_types != Counter({
        "relational_counterfactual": 192,
        "stress": 48,
    }):

        raise RuntimeError(
            f"Question-type mismatch: {question_types}"
        )


    input_orders = Counter(
        case[
            "input_order"
        ]
        for case
        in public_cases
    )


    if input_orders != Counter({
        "evidence_first": 120,
        "question_first": 120,
    }):

        raise RuntimeError(
            f"Input-order mismatch: {input_orders}"
        )


    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    CANDIDATE.write_text(
        json.dumps(
            public_cases,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


    with HIDDEN.open(
        "x"
    ) as f:

        for row in hidden_rows:

            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
            )

            f.write(
                "\n"
            )


    # Aggregate output only.
    print("=" * 92)
    print("SENTINEL CHALLENGE V3 CANDIDATE V1 GENERATION")
    print("=" * 92)

    print(
        "cases:",
        len(
            public_cases
        )
    )

    print(
        "compositions:",
        len(
            set(
                case[
                    "composition"
                ]
                for case
                in public_cases
            )
        )
    )

    print(
        "question types:",
        dict(
            sorted(
                question_types.items()
            )
        )
    )

    print(
        "labels:",
        dict(
            sorted(
                labels.items()
            )
        )
    )

    print(
        "input order:",
        dict(
            sorted(
                input_orders.items()
            )
        )
    )

    print(
        "individual cases printed:",
        0
    )

    print()
    print(
        "CHALLENGE V3 CANDIDATE V1 GENERATION: PASS"
    )


if __name__ == "__main__":
    main()
