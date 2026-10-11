"""
Sentinel V6 family realization interface.

This module defines semantic construction interfaces only.

No training examples are generated here.
"""

from dataclasses import dataclass
from typing import Mapping, Tuple

from training.v6.v6_relational_state_engine import (
    DecisionResult,
    PolicyState,
    decide,
)


MODEL_VISIBLE_FIELDS = (
    "instruction",
    "input",
)


ROW_METADATA_FIELDS = (
    "split",
    "operator",
    "family",
    "scenario_id",
    "world",
    "target_id",
    "relational_group",
    "surface_variant",
    "input_order",
)


@dataclass(frozen=True)
class TargetSemanticState:

    target_id: str

    policy_state: PolicyState

    governing_fact_ids: Tuple[
        str,
        ...
    ] = ()

    explanatory_fact_ids: Tuple[
        str,
        ...
    ] = ()


@dataclass(frozen=True)
class SemanticWorld:

    world_id: str

    targets: Mapping[
        str,
        TargetSemanticState,
    ]


@dataclass(frozen=True)
class SemanticCase:

    operator: str

    family: str

    split: str

    scenario_id: str

    world_id: str

    target_id: str

    relational_group: str

    target_state: TargetSemanticState

    result: DecisionResult


def bind_target(
    world: SemanticWorld,
    target_id: str,
) -> TargetSemanticState:

    if target_id not in world.targets:

        raise KeyError(
            f"Unknown target_id: {target_id}"
        )


    target = world.targets[
        target_id
    ]


    if target.target_id != target_id:

        raise ValueError(
            "Target mapping key does not match target semantic identity."
        )


    return target


def build_semantic_case(
    *,
    operator: str,
    family: str,
    split: str,
    scenario_id: str,
    relational_group: str,
    world: SemanticWorld,
    target_id: str,
) -> SemanticCase:

    target_state = bind_target(
        world,
        target_id,
    )


    result = decide(
        target_state.policy_state
    )


    return SemanticCase(
        operator=operator,
        family=family,
        split=split,
        scenario_id=scenario_id,
        world_id=world.world_id,
        target_id=target_id,
        relational_group=relational_group,
        target_state=target_state,
        result=result,
    )
