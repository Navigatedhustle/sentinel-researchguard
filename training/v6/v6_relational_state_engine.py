"""
Sentinel V6 frozen relational state engine.

Pure semantic oracle.

No text generation.
No model code.
No benchmark cases.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class Applicability(str, Enum):

    ESTABLISHED = "ESTABLISHED"
    ABSENT = "ABSENT"
    UNRESOLVED = "UNRESOLVED"


class BaseRule(str, Enum):

    PERMISSION = "PERMISSION"
    PROHIBITION = "PROHIBITION"
    NONE = "NONE"


class RequirementState(str, Enum):

    SATISFIED = "SATISFIED"
    FAILED = "FAILED"
    UNRESOLVED = "UNRESOLVED"


class Decision(str, Enum):

    YES = "YES"
    NO = "NO"
    CONDITIONAL = "CONDITIONAL"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class Requirement:

    name: str
    state: RequirementState


@dataclass(frozen=True)
class ExceptionState:

    trigger: Requirement

    requirements: Tuple[
        Requirement,
        ...
    ] = ()


@dataclass(frozen=True)
class PolicyState:

    applicability: Applicability

    base_rule: BaseRule

    requirements: Tuple[
        Requirement,
        ...
    ] = ()

    exception: Optional[
        ExceptionState
    ] = None


@dataclass(frozen=True)
class DecisionResult:

    decision: Decision

    conditions: Tuple[
        str,
        ...
    ]

    reason_code: str


def condition_text(
    requirement: Requirement,
) -> str:

    return (
        f"{requirement.state.value} | "
        f"{requirement.name}"
    )


def ledger_states(
    requirements,
):

    return tuple(
        requirement.state
        for requirement
        in requirements
    )


def finalize_ledger(
    requirements,
    *,
    yes_reason,
    failed_reason,
    unresolved_reason,
):

    requirements = tuple(
        requirements
    )

    states = ledger_states(
        requirements
    )

    conditions = tuple(
        condition_text(
            requirement
        )
        for requirement
        in requirements
    )


    if RequirementState.FAILED in states:

        return DecisionResult(
            decision=Decision.NO,
            conditions=conditions,
            reason_code=failed_reason,
        )


    if RequirementState.UNRESOLVED in states:

        return DecisionResult(
            decision=Decision.CONDITIONAL,
            conditions=conditions,
            reason_code=unresolved_reason,
        )


    return DecisionResult(
        decision=Decision.YES,
        conditions=conditions,
        reason_code=yes_reason,
    )


def validate_policy_state(
    state: PolicyState,
):

    if (
        state.applicability
        == Applicability.ESTABLISHED
        and state.base_rule
        == BaseRule.NONE
    ):

        raise ValueError(
            "ESTABLISHED applicability cannot use BaseRule.NONE."
        )


    if (
        state.base_rule
        == BaseRule.PERMISSION
        and state.exception
        is not None
    ):

        raise ValueError(
            "Permission state may not contain a prohibition exception."
        )


    if (
        state.base_rule
        == BaseRule.PROHIBITION
        and state.requirements
    ):

        raise ValueError(
            "Base prohibition requirements belong in exception state."
        )


    if state.exception is not None:

        if state.base_rule != BaseRule.PROHIBITION:

            raise ValueError(
                "Exception state requires BaseRule.PROHIBITION."
            )


        if not state.exception.trigger.name.strip():

            raise ValueError(
                "Exception trigger must have a name."
            )


    for requirement in state.requirements:

        if not requirement.name.strip():

            raise ValueError(
                "Requirement name must be nonempty."
            )


    if state.exception is not None:

        for requirement in state.exception.requirements:

            if not requirement.name.strip():

                raise ValueError(
                    "Exception requirement name must be nonempty."
                )


def decide(
    state: PolicyState,
) -> DecisionResult:

    validate_policy_state(
        state
    )


    # ---------------------------------------------------------------
    # Applicability gate
    # ---------------------------------------------------------------

    if (
        state.applicability
        == Applicability.ABSENT
    ):

        return DecisionResult(
            decision=Decision.INSUFFICIENT,
            conditions=(),
            reason_code="NO_GOVERNING_RULE",
        )


    if (
        state.applicability
        == Applicability.UNRESOLVED
    ):

        return DecisionResult(
            decision=Decision.INSUFFICIENT,
            conditions=(),
            reason_code="APPLICABILITY_UNRESOLVED",
        )


    # ---------------------------------------------------------------
    # Established permission
    # ---------------------------------------------------------------

    if (
        state.base_rule
        == BaseRule.PERMISSION
    ):

        return finalize_ledger(
            state.requirements,
            yes_reason="PERMISSION_ALL_REQUIREMENTS_SATISFIED",
            failed_reason="PERMISSION_REQUIREMENT_FAILED",
            unresolved_reason="PERMISSION_REQUIREMENT_UNRESOLVED",
        )


    # ---------------------------------------------------------------
    # Established prohibition
    # ---------------------------------------------------------------

    if (
        state.base_rule
        == BaseRule.PROHIBITION
    ):

        if state.exception is None:

            return DecisionResult(
                decision=Decision.NO,
                conditions=(),
                reason_code="DIRECT_PROHIBITION",
            )


        ledger = (
            state.exception.trigger,
            *state.exception.requirements,
        )


        return finalize_ledger(
            ledger,
            yes_reason="EXCEPTION_ACTIVE_ALL_REQUIREMENTS_SATISFIED",
            failed_reason="EXCEPTION_TRIGGER_OR_REQUIREMENT_FAILED",
            unresolved_reason="EXCEPTION_TRIGGER_OR_REQUIREMENT_UNRESOLVED",
        )


    raise AssertionError(
        "Unreachable policy state."
    )
