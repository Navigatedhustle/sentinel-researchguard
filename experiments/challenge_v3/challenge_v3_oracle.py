"""
Challenge V3 benchmark-local semantic oracle.

Independent implementation.

No imports from training.v6.
No model code.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


APPLICABILITY = (
    "ESTABLISHED",
    "ABSENT",
    "UNRESOLVED",
)

BASE_RULES = (
    "PERMISSION",
    "PROHIBITION",
    "NONE",
)

REQUIREMENT_STATES = (
    "SATISFIED",
    "FAILED",
    "UNRESOLVED",
)

DECISIONS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)


@dataclass(frozen=True)
class Requirement:

    name: str

    state: str


@dataclass(frozen=True)
class ExceptionState:

    trigger: Requirement

    requirements: Tuple[
        Requirement,
        ...
    ] = ()


@dataclass(frozen=True)
class SemanticState:

    applicability: str

    base_rule: str

    requirements: Tuple[
        Requirement,
        ...
    ] = ()

    exception: Optional[
        ExceptionState
    ] = None


@dataclass(frozen=True)
class OracleResult:

    decision: str

    conditions: Tuple[
        str,
        ...
    ]

    reason_code: str


def _validate_requirement(
    requirement,
):

    if not requirement.name.strip():

        raise ValueError(
            "Requirement name must be nonempty."
        )


    if requirement.state not in REQUIREMENT_STATES:

        raise ValueError(
            "Unknown requirement state."
        )


def validate_state(
    state,
):

    if state.applicability not in APPLICABILITY:

        raise ValueError(
            "Unknown applicability state."
        )


    if state.base_rule not in BASE_RULES:

        raise ValueError(
            "Unknown base rule."
        )


    if (
        state.applicability == "ESTABLISHED"
        and state.base_rule == "NONE"
    ):

        raise ValueError(
            "Established applicability cannot use NONE."
        )


    if (
        state.base_rule == "PERMISSION"
        and state.exception is not None
    ):

        raise ValueError(
            "Permission may not carry prohibition exception."
        )


    if (
        state.base_rule == "PROHIBITION"
        and state.requirements
    ):

        raise ValueError(
            "Prohibition requirements belong in exception state."
        )


    if (
        state.exception is not None
        and state.base_rule != "PROHIBITION"
    ):

        raise ValueError(
            "Exception requires base prohibition."
        )


    for requirement in state.requirements:

        _validate_requirement(
            requirement
        )


    if state.exception is not None:

        _validate_requirement(
            state.exception.trigger
        )


        for requirement in state.exception.requirements:

            _validate_requirement(
                requirement
            )


def condition_text(
    requirement,
):

    return (
        f"{requirement.state} | "
        f"{requirement.name}"
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


    conditions = tuple(
        condition_text(
            requirement
        )
        for requirement
        in requirements
    )


    states = {
        requirement.state
        for requirement
        in requirements
    }


    if "FAILED" in states:

        return OracleResult(
            decision="NO",
            conditions=conditions,
            reason_code=failed_reason,
        )


    if "UNRESOLVED" in states:

        return OracleResult(
            decision="CONDITIONAL",
            conditions=conditions,
            reason_code=unresolved_reason,
        )


    return OracleResult(
        decision="YES",
        conditions=conditions,
        reason_code=yes_reason,
    )


def decide(
    state,
):

    validate_state(
        state
    )


    if state.applicability == "ABSENT":

        return OracleResult(
            decision="INSUFFICIENT",
            conditions=(),
            reason_code="NO_APPLICABLE_RULE",
        )


    if state.applicability == "UNRESOLVED":

        return OracleResult(
            decision="INSUFFICIENT",
            conditions=(),
            reason_code="APPLICABILITY_UNRESOLVED",
        )


    if state.base_rule == "PERMISSION":

        return finalize_ledger(
            state.requirements,
            yes_reason="PERMISSION_COMPLETE",
            failed_reason="PERMISSION_REQUIREMENT_FAILED",
            unresolved_reason="PERMISSION_REQUIREMENT_UNRESOLVED",
        )


    if state.base_rule == "PROHIBITION":

        if state.exception is None:

            return OracleResult(
                decision="NO",
                conditions=(),
                reason_code="DIRECT_PROHIBITION",
            )


        ledger = (
            state.exception.trigger,
            *state.exception.requirements,
        )


        return finalize_ledger(
            ledger,
            yes_reason="EXCEPTION_COMPLETE",
            failed_reason="EXCEPTION_STATE_FAILED",
            unresolved_reason="EXCEPTION_STATE_UNRESOLVED",
        )


    raise AssertionError(
        "Unreachable semantic state."
    )


def state_to_dict(
    state,
):

    return {

        "applicability":
            state.applicability,

        "base_rule":
            state.base_rule,

        "requirements": [
            {
                "name":
                    requirement.name,

                "state":
                    requirement.state,
            }
            for requirement
            in state.requirements
        ],

        "exception":
            None
            if state.exception is None
            else {

                "trigger": {
                    "name":
                        state.exception.trigger.name,

                    "state":
                        state.exception.trigger.state,
                },

                "requirements": [
                    {
                        "name":
                            requirement.name,

                        "state":
                            requirement.state,
                    }
                    for requirement
                    in state.exception.requirements
                ],
            },
    }


def state_from_dict(
    value,
):

    exception = value[
        "exception"
    ]


    return SemanticState(

        applicability=value[
            "applicability"
        ],

        base_rule=value[
            "base_rule"
        ],

        requirements=tuple(
            Requirement(
                item[
                    "name"
                ],
                item[
                    "state"
                ],
            )
            for item
            in value[
                "requirements"
            ]
        ),

        exception=(
            None
            if exception is None
            else ExceptionState(

                trigger=Requirement(
                    exception[
                        "trigger"
                    ][
                        "name"
                    ],
                    exception[
                        "trigger"
                    ][
                        "state"
                    ],
                ),

                requirements=tuple(
                    Requirement(
                        item[
                            "name"
                        ],
                        item[
                            "state"
                        ],
                    )
                    for item
                    in exception[
                        "requirements"
                    ]
                ),
            )
        ),
    )
