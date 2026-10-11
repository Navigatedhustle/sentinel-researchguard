from collections import Counter
from pathlib import Path
import importlib.util
import sys


BASE = Path("/workspace")

ONTOLOGY = (
    BASE
    / "training/v6/"
      "v6_operator_ontology.py"
)


spec = importlib.util.spec_from_file_location(
    "v6_operator_ontology",
    ONTOLOGY,
)

mod = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    mod
)


errors = []


# --------------------------------------------------------------------
# Operator count
# --------------------------------------------------------------------

if len(
    mod.OPERATORS
) != mod.EXPECTED_OPERATOR_COUNT:

    errors.append(
        "operator count mismatch"
    )


if tuple(
    mod.OPERATORS.keys()
) != mod.OPERATOR_ORDER:

    errors.append(
        "operator ordering mismatch"
    )


# --------------------------------------------------------------------
# Family cardinality
# --------------------------------------------------------------------

if len(
    mod.TRAIN_FAMILIES
) != mod.EXPECTED_TRAIN_FAMILY_COUNT:

    errors.append(
        "train family count mismatch"
    )


if len(
    mod.DEVELOPMENT_FAMILIES
) != mod.EXPECTED_DEVELOPMENT_FAMILY_COUNT:

    errors.append(
        "development family count mismatch"
    )


if len(
    mod.ALL_FAMILIES
) != mod.EXPECTED_TOTAL_FAMILY_COUNT:

    errors.append(
        "total family count mismatch"
    )


# --------------------------------------------------------------------
# Family uniqueness
# --------------------------------------------------------------------

counts = Counter(
    mod.ALL_FAMILIES
)


duplicates = sorted(
    family
    for family, count
    in counts.items()
    if count != 1
)


if duplicates:

    errors.append(
        "duplicate family identities: "
        + ", ".join(
            duplicates
        )
    )


# --------------------------------------------------------------------
# Train/dev isolation
# --------------------------------------------------------------------

overlap = (
    set(
        mod.TRAIN_FAMILIES
    )
    &
    set(
        mod.DEVELOPMENT_FAMILIES
    )
)


if overlap:

    errors.append(
        "TRAIN/DEVELOPMENT family overlap"
    )


# --------------------------------------------------------------------
# Per-operator structure
# --------------------------------------------------------------------

for operator in mod.OPERATOR_ORDER:

    spec_data = mod.OPERATORS[
        operator
    ]


    if len(
        spec_data[
            "train_families"
        ]
    ) != 4:

        errors.append(
            f"{operator}: expected 4 train families"
        )


    if len(
        spec_data[
            "development_families"
        ]
    ) != 1:

        errors.append(
            f"{operator}: expected 1 development family"
        )


    if tuple(
        spec_data[
            "required_labels"
        ]
    ) != mod.LABELS:

        errors.append(
            f"{operator}: label vocabulary mismatch"
        )


# --------------------------------------------------------------------
# Dependency validity
# --------------------------------------------------------------------

known = set(
    mod.OPERATOR_ORDER
)


for operator in mod.OPERATOR_ORDER:

    dependencies = mod.OPERATORS[
        operator
    ][
        "depends_on"
    ]


    for dependency in dependencies:

        if dependency not in known:

            errors.append(
                f"{operator}: unknown dependency {dependency}"
            )


        if dependency == operator:

            errors.append(
                f"{operator}: self dependency"
            )


# --------------------------------------------------------------------
# Dependency cycles
# --------------------------------------------------------------------

visiting = set()
visited = set()


def visit(node):

    if node in visited:
        return


    if node in visiting:

        errors.append(
            f"dependency cycle involving {node}"
        )

        return


    visiting.add(
        node
    )


    for dep in mod.OPERATORS[
        node
    ][
        "depends_on"
    ]:

        visit(
            dep
        )


    visiting.remove(
        node
    )

    visited.add(
        node
    )


for operator in mod.OPERATOR_ORDER:

    visit(
        operator
    )


# --------------------------------------------------------------------
# Family maps
# --------------------------------------------------------------------

if set(
    mod.FAMILY_TO_OPERATOR
) != set(
    mod.ALL_FAMILIES
):

    errors.append(
        "FAMILY_TO_OPERATOR coverage mismatch"
    )


if set(
    mod.FAMILY_TO_SPLIT
) != set(
    mod.ALL_FAMILIES
):

    errors.append(
        "FAMILY_TO_SPLIT coverage mismatch"
    )


for family in mod.TRAIN_FAMILIES:

    if mod.FAMILY_TO_SPLIT[
        family
    ] != "train":

        errors.append(
            f"{family}: incorrect split"
        )


for family in mod.DEVELOPMENT_FAMILIES:

    if mod.FAMILY_TO_SPLIT[
        family
    ] != "development":

        errors.append(
            f"{family}: incorrect split"
        )


# --------------------------------------------------------------------
# Forbidden benchmark identifiers/content hints
# --------------------------------------------------------------------

text = ONTOLOGY.read_text().lower()


for forbidden in (
    "challenge_v2__",
    "challenge_v3__",
):

    if forbidden in text:

        errors.append(
            f"forbidden benchmark case identifier found: {forbidden}"
        )


# --------------------------------------------------------------------
# Report
# --------------------------------------------------------------------

print("=" * 88)
print("SENTINEL V6 OPERATOR ONTOLOGY VALIDATION")
print("=" * 88)

print(
    "operators:",
    len(
        mod.OPERATORS
    )
)

print(
    "train families:",
    len(
        mod.TRAIN_FAMILIES
    )
)

print(
    "development families:",
    len(
        mod.DEVELOPMENT_FAMILIES
    )
)

print(
    "total families:",
    len(
        mod.ALL_FAMILIES
    )
)

print()

for operator in mod.OPERATOR_ORDER:

    print(
        operator
    )

    print(
        "  train:",
        ", ".join(
            mod.OPERATORS[
                operator
            ][
                "train_families"
            ]
        )
    )

    print(
        "  development:",
        ", ".join(
            mod.OPERATORS[
                operator
            ][
                "development_families"
            ]
        )
    )


print()
print(
    "errors:",
    len(
        errors
    )
)


if errors:

    for error in errors:

        print(
            "ERROR:",
            error
        )


    print()
    print(
        "SENTINEL V6 OPERATOR ONTOLOGY: FAIL"
    )

    sys.exit(
        1
    )


print()
print(
    "SENTINEL V6 OPERATOR ONTOLOGY: PASS"
)
