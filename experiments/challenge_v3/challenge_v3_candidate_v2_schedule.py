"""
Challenge V3 Candidate V2 relational label schedule.

Purpose
-------
Candidate V1 used one global Latin square:

    (target_index + world_index) mod 4

across all 12 compositions.

That caused the hidden pair:

    (target_index, world_index)

to predict the gold label perfectly.

Candidate V2 instead uses a deterministic family of 12 Latin squares.

Properties
----------
1. Every composition is individually Latin:
   - every fixed target sees all four labels across worlds
   - every fixed world sees all four labels across targets

2. Every absolute (target_index, world_index) coordinate is perfectly
   balanced across the 12 compositions:
   - YES          3
   - NO           3
   - CONDITIONAL  3
   - INSUFFICIENT 3

3. Therefore no deterministic predictor using only the absolute
   target/world coordinate can exceed 25% accuracy.

No randomness is required for these guarantees.
"""

from experiments.challenge_v3.challenge_v3_blueprint import (
    COMPOSITIONS,
    LABELS,
)


# Each transform is:
#
#     label_index =
#         (
#             a * target_index
#             + b * world_index
#             + offset
#         ) % 4
#
# a and b must be units modulo 4 (1 or 3), which guarantees that
# each row and each column remains a permutation of 0..3.
#
# The schedule is divided into three groups of four compositions.
# Within each group, all four offsets occur exactly once.
#
# Therefore, for ANY fixed coordinate (target_index, world_index),
# each group contributes all four labels exactly once.
# Across three groups, every label occurs exactly three times.


LATIN_TRANSFORMS = (
    # Block 1: ordinary orientation; scrambled offsets.
    (1, 1, 0),
    (1, 1, 2),
    (1, 1, 1),
    (1, 1, 3),

    # Block 2: reverse world orientation.
    (1, 3, 3),
    (1, 3, 1),
    (1, 3, 0),
    (1, 3, 2),

    # Block 3: reverse target orientation.
    (3, 1, 2),
    (3, 1, 0),
    (3, 1, 3),
    (3, 1, 1),
)


if len(COMPOSITIONS) != 12:
    raise RuntimeError(
        "Candidate V2 schedule requires exactly 12 compositions; "
        f"found {len(COMPOSITIONS)}."
    )


if len(LATIN_TRANSFORMS) != len(COMPOSITIONS):
    raise RuntimeError(
        "Transform count does not match composition count."
    )


def relational_label_v2(
    composition_index,
    world_index,
    target_index,
):
    """
    Return the Candidate V2 relational label.

    The composition-specific Latin-square transform is benchmark
    generator state only. It is not included in the model input.
    """

    if not 0 <= composition_index < len(LATIN_TRANSFORMS):
        raise ValueError(
            f"Invalid composition_index: {composition_index}"
        )

    if world_index not in (0, 1, 2, 3):
        raise ValueError(
            f"Invalid world_index: {world_index}"
        )

    if target_index not in (0, 1, 2, 3):
        raise ValueError(
            f"Invalid target_index: {target_index}"
        )

    a, b, offset = LATIN_TRANSFORMS[
        composition_index
    ]

    label_index = (
        a * target_index
        + b * world_index
        + offset
    ) % 4

    return LABELS[
        label_index
    ]
