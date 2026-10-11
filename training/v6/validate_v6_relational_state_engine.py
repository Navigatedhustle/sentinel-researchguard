from collections import Counter
from itertools import product
from pathlib import Path
import sys


from training.v6.v6_relational_state_engine import (
    Applicability,
    BaseRule,
    Decision,
    ExceptionState,
    PolicyState,
    Requirement,
    RequirementState,
    decide,
)

from training.v6.v6_family_interface import (
    SemanticWorld,
    TargetSemanticState,
    bind_target,
)


errors = []
tested = 0
decision_counts = Counter()


def check(
    condition,
    message,
):

    if not condition:

        errors.append(
            message
        )


def run(
    state,
    expected,
    description,
):

    global tested

    tested += 1


    try:

        result = decide(
            state
        )

    except Exception as exc:

        errors.append(
            f"{description}: unexpected exception: {exc}"
        )

        return None


    decision_counts[
        result.decision.value
    ] += 1


    if result.decision != expected:

        errors.append(
            f"{description}: expected {expected.value}, "
            f"got {result.decision.value}"
        )


    # ---------------------------------------------------------------
    # Ledger/decision invariants
    # ---------------------------------------------------------------

    conditions = result.conditions


    if result.decision == Decision.INSUFFICIENT:

        check(
            conditions == (),
            f"{description}: INSUFFICIENT ledger must be empty",
        )


    if result.decision == Decision.YES:

        check(
            not any(
                item.startswith(
                    "FAILED | "
                )
                for item
                in conditions
            ),
            f"{description}: YES contains FAILED ledger state",
        )

        check(
            not any(
                item.startswith(
                    "UNRESOLVED | "
                )
                for item
                in conditions
            ),
            f"{description}: YES contains UNRESOLVED ledger state",
        )


    if result.decision == Decision.CONDITIONAL:

        check(
            any(
                item.startswith(
                    "UNRESOLVED | "
                )
                for item
                in conditions
            ),
            f"{description}: CONDITIONAL lacks UNRESOLVED state",
        )

        check(
            not any(
                item.startswith(
                    "FAILED | "
                )
                for item
                in conditions
            ),
            f"{description}: CONDITIONAL contains FAILED state",
        )


    return result


# ====================================================================
# 1. Applicability must dominate everything downstream
# ====================================================================

for applicability in (
    Applicability.ABSENT,
    Applicability.UNRESOLVED,
):

    for base_rule in (
        BaseRule.PERMISSION,
        BaseRule.PROHIBITION,
        BaseRule.NONE,
    ):

        state = PolicyState(
            applicability=applicability,
            base_rule=base_rule,
        )


        run(
            state,
            Decision.INSUFFICIENT,
            f"applicability dominance {applicability.value}/{base_rule.value}",
        )


# ====================================================================
# 2. Permission ledger exhaustive two-requirement states
# ====================================================================

for first, second in product(
    RequirementState,
    repeat=2,
):

    requirements = (
        Requirement(
            "requirement A",
            first,
        ),
        Requirement(
            "requirement B",
            second,
        ),
    )


    if RequirementState.FAILED in (
        first,
        second,
    ):

        expected = Decision.NO


    elif RequirementState.UNRESOLVED in (
        first,
        second,
    ):

        expected = Decision.CONDITIONAL


    else:

        expected = Decision.YES


    run(
        PolicyState(
            applicability=Applicability.ESTABLISHED,
            base_rule=BaseRule.PERMISSION,
            requirements=requirements,
        ),
        expected,
        f"permission ledger {first.value}/{second.value}",
    )


# Permission with no requirements
run(
    PolicyState(
        applicability=Applicability.ESTABLISHED,
        base_rule=BaseRule.PERMISSION,
    ),
    Decision.YES,
    "unconditional permission",
)


# ====================================================================
# 3. Direct prohibition
# ====================================================================

result = run(
    PolicyState(
        applicability=Applicability.ESTABLISHED,
        base_rule=BaseRule.PROHIBITION,
    ),
    Decision.NO,
    "direct prohibition",
)

if result is not None:

    check(
        result.conditions == (),
        "direct prohibition must have empty condition ledger",
    )


# ====================================================================
# 4. Exception trigger + two requirements exhaustive
# ====================================================================

for trigger, first, second in product(
    RequirementState,
    repeat=3,
):

    ledger_states = (
        trigger,
        first,
        second,
    )


    if RequirementState.FAILED in ledger_states:

        expected = Decision.NO


    elif RequirementState.UNRESOLVED in ledger_states:

        expected = Decision.CONDITIONAL


    else:

        expected = Decision.YES


    result = run(
        PolicyState(
            applicability=Applicability.ESTABLISHED,
            base_rule=BaseRule.PROHIBITION,
            exception=ExceptionState(
                trigger=Requirement(
                    "exception trigger active",
                    trigger,
                ),
                requirements=(
                    Requirement(
                        "exception requirement A",
                        first,
                    ),
                    Requirement(
                        "exception requirement B",
                        second,
                    ),
                ),
            ),
        ),
        expected,
        (
            "exception ledger "
            f"{trigger.value}/{first.value}/{second.value}"
        ),
    )


    if result is not None:

        check(
            len(
                result.conditions
            )
            == 3,
            "exception ledger must contain trigger + both requirements",
        )


# ====================================================================
# 5. Failed must dominate unresolved
# ====================================================================

result = run(
    PolicyState(
        applicability=Applicability.ESTABLISHED,
        base_rule=BaseRule.PERMISSION,
        requirements=(
            Requirement(
                "approved",
                RequirementState.SATISFIED,
            ),
            Requirement(
                "notice sent",
                RequirementState.FAILED,
            ),
            Requirement(
                "review complete",
                RequirementState.UNRESOLVED,
            ),
        ),
    ),
    Decision.NO,
    "FAILED dominates UNRESOLVED",
)

if result is not None:

    check(
        any(
            item.startswith(
                "FAILED | "
            )
            for item
            in result.conditions
        ),
        "FAILED precedence case did not preserve FAILED ledger entry",
    )


# ====================================================================
# 6. Explicit V5 failure-pattern regression semantics
# ====================================================================

# satisfied exception must displace prohibition
run(
    PolicyState(
        applicability=Applicability.ESTABLISHED,
        base_rule=BaseRule.PROHIBITION,
        exception=ExceptionState(
            trigger=Requirement(
                "recovery event declared",
                RequirementState.SATISFIED,
            ),
            requirements=(
                Requirement(
                    "written security clearance recorded",
                    RequirementState.SATISFIED,
                ),
            ),
        ),
    ),
    Decision.YES,
    "satisfied exception overrides base prohibition",
)


# unknown applicability must NOT become conditional
run(
    PolicyState(
        applicability=Applicability.UNRESOLVED,
        base_rule=BaseRule.PERMISSION,
        requirements=(
            Requirement(
                "ethics review approved",
                RequirementState.UNRESOLVED,
            ),
        ),
    ),
    Decision.INSUFFICIENT,
    "unknown applicability stays INSUFFICIENT",
)


# satisfied ledger must finalize YES
run(
    PolicyState(
        applicability=Applicability.ESTABLISHED,
        base_rule=BaseRule.PERMISSION,
        requirements=(
            Requirement(
                "authorization active",
                RequirementState.SATISFIED,
            ),
            Requirement(
                "annual review complete",
                RequirementState.SATISFIED,
            ),
        ),
    ),
    Decision.YES,
    "all satisfied ledger finalizes YES",
)


# ====================================================================
# 7. Target binding world
# ====================================================================

world = SemanticWorld(
    world_id="validation_world",
    targets={
        "alpha": TargetSemanticState(
            target_id="alpha",
            policy_state=PolicyState(
                applicability=Applicability.ESTABLISHED,
                base_rule=BaseRule.PERMISSION,
            ),
        ),

        "beta": TargetSemanticState(
            target_id="beta",
            policy_state=PolicyState(
                applicability=Applicability.ESTABLISHED,
                base_rule=BaseRule.PROHIBITION,
            ),
        ),

        "gamma": TargetSemanticState(
            target_id="gamma",
            policy_state=PolicyState(
                applicability=Applicability.UNRESOLVED,
                base_rule=BaseRule.NONE,
            ),
        ),
    },
)


alpha = bind_target(
    world,
    "alpha",
)

beta = bind_target(
    world,
    "beta",
)

gamma = bind_target(
    world,
    "gamma",
)


check(
    decide(
        alpha.policy_state
    ).decision
    == Decision.YES,
    "target binding alpha should be YES",
)

check(
    decide(
        beta.policy_state
    ).decision
    == Decision.NO,
    "target binding beta should be NO",
)

check(
    decide(
        gamma.policy_state
    ).decision
    == Decision.INSUFFICIENT,
    "target binding gamma should be INSUFFICIENT",
)


# ====================================================================
# 8. Invalid semantic states must be rejected
# ====================================================================

invalid_states = [

    (
        "established NONE",
        PolicyState(
            applicability=Applicability.ESTABLISHED,
            base_rule=BaseRule.NONE,
        ),
    ),

    (
        "permission with exception",
        PolicyState(
            applicability=Applicability.ESTABLISHED,
            base_rule=BaseRule.PERMISSION,
            exception=ExceptionState(
                trigger=Requirement(
                    "trigger",
                    RequirementState.SATISFIED,
                ),
            ),
        ),
    ),

    (
        "prohibition base requirements",
        PolicyState(
            applicability=Applicability.ESTABLISHED,
            base_rule=BaseRule.PROHIBITION,
            requirements=(
                Requirement(
                    "invalid requirement",
                    RequirementState.SATISFIED,
                ),
            ),
        ),
    ),
]


for description, state in invalid_states:

    try:

        decide(
            state
        )

    except ValueError:

        pass

    else:

        errors.append(
            f"invalid semantic state accepted: {description}"
        )


# ====================================================================
# REPORT
# ====================================================================

print("=" * 88)
print("SENTINEL V6 RELATIONAL STATE ENGINE VALIDATION")
print("=" * 88)

print(
    "semantic test cases:",
    tested
)

print(
    "decision distribution:",
    dict(
        sorted(
            decision_counts.items()
        )
    )
)

print(
    "errors:",
    len(
        errors
    )
)


if errors:

    print()

    for error in errors:

        print(
            "ERROR:",
            error
        )


    print()
    print(
        "SENTINEL V6 RELATIONAL STATE ENGINE: FAIL"
    )

    sys.exit(
        1
    )


for required_label in (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
):

    if decision_counts[
        required_label
    ] == 0:

        print(
            "ERROR: validator did not exercise label",
            required_label
        )

        sys.exit(
            1
        )


print()
print(
    "SENTINEL V6 RELATIONAL STATE ENGINE: PASS"
)
