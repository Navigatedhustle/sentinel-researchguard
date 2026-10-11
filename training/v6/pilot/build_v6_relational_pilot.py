"""
Build Sentinel V6 128-row relational language pilot.

Gold decisions are produced only by the frozen V6 semantic engine.
"""

from collections import Counter
from pathlib import Path

import hashlib
import json


from training.v6.v6_relational_state_engine import (
    decide,
)

from training.v6.pilot.v6_relational_pilot_spec import (
    FAMILY_TO_OPERATOR,
    GROUP_VARIANTS,
    PILOT_FAMILIES,
    TARGET_IDS,
    WORLD_IDS,
    build_state,
    input_order,
    intended_label,
)

from training.v6.pilot.v6_pilot_realizers import (
    INSTRUCTION,
    build_input,
    output_object,
    question,
    render_world,
    serialize_output,
)


BASE = Path(
    "/workspace"
)

OUTPUT = (
    BASE
    / "training/v6/pilot/"
      "sentinel_v6_relational_pilot.jsonl"
)


def text_hash(
    value,
):

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


def main():

    if OUTPUT.exists():

        raise RuntimeError(
            "Pilot output already exists."
        )


    rows = []


    for family_index, family in enumerate(
        PILOT_FAMILIES
    ):

        for variant in GROUP_VARIANTS:

            group = (
                f"{family}"
                f"__pilot_g{variant}"
            )


            for world_index, world_id in enumerate(
                WORLD_IDS
            ):

                states = {}

                results = {}


                for target_index, target_id in enumerate(
                    TARGET_IDS
                ):

                    label = intended_label(
                        world_index,
                        target_index,
                    )


                    state = build_state(
                        family,
                        label,
                        variant,
                    )


                    result = decide(
                        state
                    )


                    if (
                        result.decision.value
                        != label
                    ):

                        raise RuntimeError(
                            "Frozen semantic spec disagrees "
                            "with frozen engine."
                        )


                    states[
                        target_id
                    ] = state

                    results[
                        target_id
                    ] = result


                evidence, refs, entities = render_world(
                    family=family,
                    variant=variant,
                    world_index=world_index,
                    states=states,
                )


                evidence_hash = text_hash(
                    evidence
                )


                for target_index, target_id in enumerate(
                    TARGET_IDS
                ):

                    target_name = entities[
                        target_id
                    ]


                    q = question(
                        family,
                        target_name,
                        variant,
                    )


                    order = input_order(
                        family_index=family_index,
                        group_variant=variant,
                        world_index=world_index,
                        target_index=target_index,
                    )


                    model_input = build_input(
                        evidence=evidence,
                        question_text=q,
                        input_order=order,
                    )


                    result = results[
                        target_id
                    ]

                    state = states[
                        target_id
                    ]


                    output = output_object(
                        family=family,
                        state=state,
                        result=result,
                        citation_ref=refs[
                            target_id
                        ],
                    )


                    row_id = (
                        f"v6pilot__"
                        f"{family}__"
                        f"g{variant}__"
                        f"w{world_index}__"
                        f"t{target_index}"
                    )


                    rows.append({

                        "row_id":
                            row_id,

                        "split":
                            "pilot",

                        "operator":
                            FAMILY_TO_OPERATOR[
                                family
                            ],

                        "family":
                            family,

                        "scenario_id":
                            (
                                f"v6pilot__"
                                f"{family}__"
                                f"g{variant}"
                            ),

                        "relational_group":
                            group,

                        "group_variant":
                            variant,

                        "world":
                            world_id,

                        "world_index":
                            world_index,

                        "target_id":
                            target_id,

                        "target_index":
                            target_index,

                        "target_name":
                            target_name,

                        "surface_variant":
                            variant,

                        "input_order":
                            order,

                        "instruction":
                            INSTRUCTION,

                        "question":
                            q,

                        "evidence":
                            evidence,

                        "evidence_hash":
                            evidence_hash,

                        "target_evidence_ref":
                            refs[
                                target_id
                            ],

                        "input":
                            model_input,

                        "target":
                            result.decision.value,

                        "output":
                            serialize_output(
                                output
                            ),
                    })


    if len(
        rows
    ) != 128:

        raise RuntimeError(
            f"Expected 128 rows, found {len(rows)}."
        )


    labels = Counter(
        row[
            "target"
        ]
        for row in rows
    )


    if labels != Counter({
        "YES": 32,
        "NO": 32,
        "CONDITIONAL": 32,
        "INSUFFICIENT": 32,
    }):

        raise RuntimeError(
            f"Label imbalance: {labels}"
        )


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with OUTPUT.open(
        "x"
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
            )

            f.write(
                "\n"
            )


    print("=" * 88)
    print("SENTINEL V6 RELATIONAL PILOT GENERATION")
    print("=" * 88)

    print(
        "rows:",
        len(
            rows
        )
    )

    print(
        "families:",
        len(
            set(
                row[
                    "family"
                ]
                for row in rows
            )
        )
    )

    print(
        "relational groups:",
        len(
            set(
                row[
                    "relational_group"
                ]
                for row in rows
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
        "output:",
        OUTPUT
    )

    print()
    print(
        "GENERATION: PASS"
    )


if __name__ == "__main__":
    main()
