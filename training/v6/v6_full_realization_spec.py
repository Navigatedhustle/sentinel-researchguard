"""
Sentinel V6 full realization geometry.

Defines structure only.

NO generated rows.
NO family language realizers.
NO model code.
NO benchmark content.
"""

from training.v6.v6_operator_ontology import (
    DEVELOPMENT_FAMILIES,
    FAMILY_TO_OPERATOR,
    OPERATOR_ORDER,
    TRAIN_FAMILIES,
)


LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)


TARGET_IDS = (
    "A",
    "B",
    "C",
    "D",
)


WORLD_IDS = (
    "world_0",
    "world_1",
    "world_2",
    "world_3",
)


SEMANTIC_GROUPS = (
    "group_0_canonical",
    "group_1_alternate_causal_source",
    "group_2_binding_distractor_stress",
    "group_3_dependency_composition_stress",
)


TRAIN_SURFACES = (
    "train_surface_0",
    "train_surface_1",
    "train_surface_2",
)


DEVELOPMENT_SURFACES = (
    "development_surface_0",
    "development_surface_1",
)


SURFACE_ROLE = {

    "train_surface_0":
        "direct_compact",

    "train_surface_1":
        "distributed_multiclause",

    "train_surface_2":
        "dense_distractor_rich",

    "development_surface_0":
        "heldout_reframed",

    "development_surface_1":
        "heldout_compact_reordered",
}


ORDER_MATRIX = (

    (0, 0, 1, 1),

    (1, 0, 0, 1),

    (1, 1, 0, 0),

    (0, 1, 1, 0),
)


EXPECTED = {

    "train": {
        "families":
            32,

        "semantic_groups":
            4,

        "surfaces_per_group":
            3,

        "worlds_per_relational_group":
            4,

        "targets_per_world":
            4,

        "relational_groups":
            384,

        "evidence_worlds":
            1536,

        "question_groups":
            1536,

        "rows":
            6144,

        "rows_per_label":
            1536,
    },


    "development": {
        "families":
            8,

        "semantic_groups":
            4,

        "surfaces_per_group":
            2,

        "worlds_per_relational_group":
            4,

        "targets_per_world":
            4,

        "relational_groups":
            64,

        "evidence_worlds":
            256,

        "question_groups":
            256,

        "rows":
            1024,

        "rows_per_label":
            256,
    },
}


def family_split(
    family,
):

    if family in TRAIN_FAMILIES:
        return "train"

    if family in DEVELOPMENT_FAMILIES:
        return "development"

    raise KeyError(
        family
    )


def surfaces_for_split(
    split,
):

    if split == "train":
        return TRAIN_SURFACES

    if split == "development":
        return DEVELOPMENT_SURFACES

    raise KeyError(
        split
    )


def intended_label(
    world_index,
    target_index,
):

    return LABELS[
        (
            world_index
            + target_index
        )
        % 4
    ]


def input_order(
    *,
    family_index,
    semantic_group_index,
    surface_index,
    world_index,
    target_index,
):

    base = ORDER_MATRIX[
        world_index
    ][
        target_index
    ]


    rotation = (
        family_index
        + semantic_group_index
        + surface_index
    ) % 2


    bit = (
        base
        ^ rotation
    )


    return (
        "evidence_first"
        if bit == 0
        else "question_first"
    )


def semantic_group_id(
    *,
    split,
    family,
    semantic_group_index,
):

    return (
        f"{split}__"
        f"{family}__"
        f"sg{semantic_group_index}"
    )


def relational_group_id(
    *,
    split,
    family,
    semantic_group_index,
    surface,
):

    return (
        f"{split}__"
        f"{family}__"
        f"sg{semantic_group_index}__"
        f"{surface}"
    )


def iter_virtual_rows(
    split,
):

    if split == "train":

        families = TRAIN_FAMILIES

    elif split == "development":

        families = DEVELOPMENT_FAMILIES

    else:

        raise KeyError(
            split
        )


    surfaces = surfaces_for_split(
        split
    )


    for family_index, family in enumerate(
        families
    ):

        for semantic_group_index, semantic_group in enumerate(
            SEMANTIC_GROUPS
        ):

            semantic_id = semantic_group_id(
                split=split,
                family=family,
                semantic_group_index=semantic_group_index,
            )


            for surface_index, surface in enumerate(
                surfaces
            ):

                relational_group = relational_group_id(
                    split=split,
                    family=family,
                    semantic_group_index=semantic_group_index,
                    surface=surface,
                )


                for world_index, world in enumerate(
                    WORLD_IDS
                ):

                    for target_index, target_id in enumerate(
                        TARGET_IDS
                    ):

                        yield {

                            "split":
                                split,

                            "operator":
                                FAMILY_TO_OPERATOR[
                                    family
                                ],

                            "family":
                                family,

                            "family_index":
                                family_index,

                            "semantic_group":
                                semantic_group,

                            "semantic_group_index":
                                semantic_group_index,

                            "semantic_group_id":
                                semantic_id,

                            "surface":
                                surface,

                            "surface_index":
                                surface_index,

                            "surface_role":
                                SURFACE_ROLE[
                                    surface
                                ],

                            "relational_group":
                                relational_group,

                            "world":
                                world,

                            "world_index":
                                world_index,

                            "target_id":
                                target_id,

                            "target_index":
                                target_index,

                            "label":
                                intended_label(
                                    world_index,
                                    target_index,
                                ),

                            "input_order":
                                input_order(
                                    family_index=family_index,
                                    semantic_group_index=semantic_group_index,
                                    surface_index=surface_index,
                                    world_index=world_index,
                                    target_index=target_index,
                                ),
                        }
