from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import re
import sys


BASE = Path("/workspace")

ONTOLOGY = BASE / "training/v6/v6_operator_ontology.py"

PILOT = (
    BASE
    / "training/v6/pilot/"
      "sentinel_v6_relational_pilot.jsonl"
)

CONTRACT = BASE / "experiments/v6/V6_EXPERIMENT_CONTRACT.txt"

ANNOTATION = (
    BASE
    / "experiments/v6/"
      "V6_EXPLANATORY_ANNOTATION_CONTRACT.txt"
)

REPAIR = (
    BASE
    / "experiments/v6/"
      "V6_RESIDUAL_REPAIR_SPEC.txt"
)

CH4_ROOT = BASE / "experiments/challenge_v4"


EXPECTED = {
    ONTOLOGY:
        "ac8613c2217d9c052c5555f0a72b5f2649332c9a0542cb824f1a1c351a40f926",

    PILOT:
        "77fd17759cd11b05793b57c76767f04bdbf5a1638f3316c312d4c72bbe085fea",

    CONTRACT:
        "ed37078c2925ec0d529254120a427c17ed051d78f4317ab1f3bb3b70b6f01106",

    ANNOTATION:
        "e750f99695e48872a8aceeea7d4f1aa563782f5d1d845bef7b0b57fc7fb48848",

    REPAIR:
        "630e662e08aea517a4636fb0f67564a52d1ee5c702adb817fc94972332fa199c",
}


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_jsonl(path):
    rows = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            rows.append(
                json.loads(line)
            )

    return rows


def type_shape(value):
    if isinstance(value, dict):
        return {
            k: type_shape(v)
            for k, v in sorted(value.items())
        }

    if isinstance(value, list):
        if not value:
            return ["EMPTY"]

        seen = []

        for item in value:
            shape = type_shape(item)

            if shape not in seen:
                seen.append(shape)

        return seen[:4]

    return type(value).__name__


def scalar_metadata(row):
    out = {}

    for key, value in row.items():
        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                type(None),
            ),
        ):
            out[key] = value

    return out


def load_ontology():
    spec = importlib.util.spec_from_file_location(
        "sentinel_v6_operator_ontology",
        ONTOLOGY,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "could not create ontology module spec"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    candidates = []

    for name in (
        "OPERATORS",
        "ONTOLOGY",
        "FAMILIES",
        "V6_OPERATORS",
    ):
        value = getattr(
            module,
            name,
            None,
        )

        if isinstance(value, dict):
            candidates.append(
                (
                    name,
                    value,
                )
            )

    if not candidates:
        raise RuntimeError(
            "no ontology dictionary found"
        )

    # Prefer an object that visibly contains train/development
    # family structures.
    for name, value in candidates:
        if any(
            isinstance(v, dict)
            and (
                "train" in v
                or "development" in v
                or "dev" in v
            )
            for v in value.values()
        ):
            return name, value

    return candidates[0]


def family_name(family):
    if isinstance(family, dict):
        return (
            family.get("family")
            or family.get("name")
            or family.get("id")
            or "<unnamed>"
        )

    return str(family)


# ====================================================================
# RUNTIME CHALLENGE V4 FILESYSTEM QUARANTINE
#
# The previous implementation scanned this file's own source for
# forbidden strings and therefore detected the strings contained in
# its own safety check.
#
# This replacement guards actual filesystem access instead.
#
# Any attempt to open/read a path anywhere beneath:
#
#     /workspace/experiments/challenge_v4
#
# raises immediately.
# ====================================================================

import builtins
import os


CH4_QUARANTINE_ROOT = (
    BASE
    / "experiments/challenge_v4"
).resolve()


_ORIGINAL_PATH_READ_TEXT = Path.read_text
_ORIGINAL_PATH_READ_BYTES = Path.read_bytes
_ORIGINAL_OPEN = builtins.open


def _assert_not_challenge_v4_path(path):
    try:
        resolved = Path(path).resolve()
    except Exception:
        return

    try:
        resolved.relative_to(
            CH4_QUARANTINE_ROOT
        )
    except ValueError:
        return

    raise RuntimeError(
        "Challenge V4 quarantine violation: "
        f"attempted filesystem read: {resolved}"
    )


def _guarded_read_text(self, *args, **kwargs):
    _assert_not_challenge_v4_path(self)

    return _ORIGINAL_PATH_READ_TEXT(
        self,
        *args,
        **kwargs,
    )


def _guarded_read_bytes(self, *args, **kwargs):
    _assert_not_challenge_v4_path(self)

    return _ORIGINAL_PATH_READ_BYTES(
        self,
        *args,
        **kwargs,
    )


def _guarded_open(file, *args, **kwargs):
    if isinstance(
        file,
        (
            str,
            bytes,
            os.PathLike,
        ),
    ):
        _assert_not_challenge_v4_path(
            file
        )

    return _ORIGINAL_OPEN(
        file,
        *args,
        **kwargs,
    )


Path.read_text = _guarded_read_text
Path.read_bytes = _guarded_read_bytes
builtins.open = _guarded_open


print(
    "PASS: runtime Challenge V4 filesystem "
    "quarantine installed"
)

# ====================================================================
# HASH GATE
# ====================================================================

for path, expected in EXPECTED.items():
    actual = sha(path)

    if actual != expected:
        raise RuntimeError(
            f"identity mismatch: {path}\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )


# ====================================================================
# ONTOLOGY INSPECTION
# ====================================================================

ontology_symbol, ontology = load_ontology()

print()
print("=" * 88)
print("V6 ONTOLOGY")
print("=" * 88)

print("ontology symbol:", ontology_symbol)
print("operators:", len(ontology))

train_families = []
development_families = []

operator_summary = {}

for operator, spec in ontology.items():

    if not isinstance(spec, dict):
        print()
        print(
            "WARNING: non-dict operator entry:",
            operator,
            type(spec).__name__,
        )
        continue

    train = spec.get(
        "train",
        [],
    )

    development = spec.get(
        "development",
        spec.get(
            "dev",
            [],
        ),
    )

    if not isinstance(train, list):
        train = []

    if not isinstance(
        development,
        list,
    ):
        development = []

    train_families.extend(
        (
            operator,
            family,
        )
        for family in train
    )

    development_families.extend(
        (
            operator,
            family,
        )
        for family in development
    )

    operator_summary[operator] = {
        "train": len(train),
        "development": len(development),
    }

    print()
    print("#" * 88)
    print("OPERATOR:", operator)

    if "description" in spec:
        print(
            "DESCRIPTION:",
            spec["description"],
        )

    print(
        "TRAIN FAMILY COUNT:",
        len(train),
    )

    for i, family in enumerate(
        train,
        1,
    ):
        print()
        print(
            f"  TRAIN [{i}]",
            family_name(family),
        )

        if isinstance(
            family,
            dict,
        ):
            print(
                "    fields:",
                sorted(family.keys()),
            )

            # Print ontology design attributes.
            # These are frozen specification text,
            # not generated examples.
            for key, value in family.items():
                if key in {
                    "family",
                    "name",
                    "id",
                }:
                    continue

                print(
                    f"    {key}:",
                    value,
                )

    print()
    print(
        "DEVELOPMENT FAMILY COUNT:",
        len(development),
    )

    for i, family in enumerate(
        development,
        1,
    ):
        print()
        print(
            f"  DEVELOPMENT [{i}]",
            family_name(family),
        )

        if isinstance(
            family,
            dict,
        ):
            print(
                "    fields:",
                sorted(family.keys()),
            )

            for key, value in family.items():
                if key in {
                    "family",
                    "name",
                    "id",
                }:
                    continue

                print(
                    f"    {key}:",
                    value,
                )


print()
print("=" * 88)
print("ONTOLOGY TOTALS")
print("=" * 88)

print(
    "train families:",
    len(train_families),
)

print(
    "development families:",
    len(development_families),
)

print(
    "all families:",
    len(train_families)
    + len(development_families),
)

print()
print(
    "operator family counts:",
    operator_summary,
)


# ====================================================================
# FAMILY FIELD SCHEMAS
# ====================================================================

print()
print("=" * 88)
print("FAMILY FIELD-SCHEMA DISTRIBUTION")
print("=" * 88)

schema_counts = Counter()

for split, entries in (
    (
        "train",
        train_families,
    ),
    (
        "development",
        development_families,
    ),
):
    for operator, family in entries:
        if isinstance(
            family,
            dict,
        ):
            fields = tuple(
                sorted(
                    k
                    for k in family
                    if k not in {
                        "family",
                        "name",
                        "id",
                    }
                )
            )
        else:
            fields = tuple()

        schema_counts[
            (
                operator,
                split,
                fields,
            )
        ] += 1

for key, count in sorted(
    schema_counts.items(),
    key=lambda x: str(x[0]),
):
    operator, split, fields = key

    print(
        f"{operator:36s} "
        f"{split:12s} "
        f"count={count:<3d} "
        f"fields={fields}"
    )


# ====================================================================
# PILOT SCHEMA
# ====================================================================

rows = load_jsonl(
    PILOT
)

if len(rows) != 128:
    raise RuntimeError(
        f"expected frozen pilot 128 rows, "
        f"found {len(rows)}"
    )

print()
print("=" * 88)
print("FROZEN V6 PILOT SCHEMA")
print("=" * 88)

print("rows:", len(rows))

all_keys = sorted(
    set().union(
        *(
            set(row.keys())
            for row in rows
        )
    )
)

print(
    "top-level keys:",
    all_keys,
)

print()
print("TYPE SHAPES BY FIELD")

for key in all_keys:
    shapes = []

    for row in rows:
        if key not in row:
            continue

        shape = type_shape(
            row[key]
        )

        if shape not in shapes:
            shapes.append(shape)

    print(
        f"  {key}:",
        shapes,
    )


# ====================================================================
# PILOT METADATA DISTRIBUTIONS
#
# Only low-cardinality metadata is printed.
# No questions / evidence / answers / rationales are printed.
# ====================================================================

print()
print("=" * 88)
print("PILOT LOW-CARDINALITY METADATA")
print("=" * 88)

candidate_scalar_keys = sorted(
    {
        key
        for row in rows
        for key, value in row.items()
        if isinstance(
            value,
            (
                str,
                int,
                bool,
            ),
        )
    }
)

SENSITIVE_TEXT_KEYS = {
    "question",
    "input",
    "raw_input",
    "prompt",
    "evidence",
    "answer",
    "assistant",
    "output",
    "rationale",
    "explanation",
    "response",
}

for key in candidate_scalar_keys:

    if key.lower() in SENSITIVE_TEXT_KEYS:
        continue

    values = [
        row[key]
        for row in rows
        if key in row
    ]

    unique = set(
        map(
            str,
            values,
        )
    )

    if len(unique) <= 40:
        counts = Counter(
            map(
                str,
                values,
            )
        )

        print()
        print(
            key,
            "categories=",
            len(counts),
        )

        print(
            " ",
            dict(
                sorted(
                    counts.items()
                )
            ),
        )


# ====================================================================
# PILOT STRUCTURAL IDENTITY ANALYSIS
# ====================================================================

print()
print("=" * 88)
print("PILOT STRUCTURAL COUNTS")
print("=" * 88)

for key in (
    "decision",
    "label",
    "gold",
    "family",
    "operator",
    "relational_group",
    "input_order",
    "evidence_count",
    "surface_variant",
    "world",
    "world_index",
    "target",
    "target_index",
):
    if any(
        key in row
        for row in rows
    ):
        counts = Counter(
            str(row[key])
            for row in rows
            if key in row
        )

        print(
            f"{key}: "
            f"categories={len(counts)} "
            f"counts={dict(sorted(counts.items()))}"
        )


# ====================================================================
# NESTED OUTPUT / ANNOTATION KEY DISCOVERY
#
# Schema only. No values.
# ====================================================================

print()
print("=" * 88)
print("NESTED DICTIONARY SCHEMAS")
print("=" * 88)

nested = defaultdict(
    Counter
)

for row in rows:
    for key, value in row.items():
        if isinstance(
            value,
            dict,
        ):
            nested[key][
                tuple(
                    sorted(
                        value.keys()
                    )
                )
            ] += 1

for key, shapes in sorted(
    nested.items()
):
    print()
    print(key)

    for shape, count in shapes.items():
        print(
            " ",
            f"count={count}",
            "keys=",
            shape,
        )


# ====================================================================
# LIST FIELD STRUCTURES
# ====================================================================

print()
print("=" * 88)
print("LIST FIELD SCHEMAS")
print("=" * 88)

list_fields = defaultdict(
    Counter
)

for row in rows:
    for key, value in row.items():

        if not isinstance(
            value,
            list,
        ):
            continue

        if not value:
            list_fields[key][
                "EMPTY"
            ] += 1
            continue

        first = value[0]

        if isinstance(
            first,
            dict,
        ):
            shape = (
                "dict",
                tuple(
                    sorted(
                        first.keys()
                    )
                ),
            )

        else:
            shape = (
                type(first).__name__,
            )

        list_fields[key][
            shape
        ] += 1

for key, shapes in sorted(
    list_fields.items()
):
    print()
    print(key)

    for shape, count in shapes.items():
        print(
            " ",
            f"count={count}",
            "shape=",
            shape,
        )


# ====================================================================
# RELATIONAL GROUP GEOMETRY
# ====================================================================

print()
print("=" * 88)
print("RELATIONAL GROUP GEOMETRY")
print("=" * 88)

group_key = None

for candidate in (
    "relational_group",
    "relational_system",
    "group",
):
    if all(
        candidate in row
        for row in rows
    ):
        group_key = candidate
        break

if group_key is None:
    print(
        "No universal relational group "
        "field discovered."
    )

else:
    groups = defaultdict(
        list
    )

    for row in rows:
        groups[
            str(
                row[group_key]
            )
        ].append(row)

    print(
        "group field:",
        group_key,
    )

    print(
        "groups:",
        len(groups),
    )

    print(
        "rows/group distribution:",
        dict(
            Counter(
                len(v)
                for v in groups.values()
            )
        ),
    )

    for field in (
        "world",
        "world_index",
        "target",
        "target_index",
        "surface_variant",
        "input_order",
        "decision",
        "label",
        "gold",
    ):
        if all(
            field in row
            for row in rows
        ):
            per_group_cardinality = Counter()

            for group_rows in groups.values():
                per_group_cardinality[
                    len(
                        {
                            str(r[field])
                            for r in group_rows
                        }
                    )
                ] += 1

            print(
                f"{field} unique-values/group:",
                dict(
                    sorted(
                        per_group_cardinality.items()
                    )
                ),
            )


# ====================================================================
# CONTRACT HEADINGS ONLY
#
# We print headings, not benchmark or generated example contents.
# ====================================================================

def print_headings(
    title,
    path,
):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)

    text = path.read_text(
        encoding="utf-8"
    )

    headings = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        # Uppercase contract headings.
        letters = [
            c
            for c in stripped
            if c.isalpha()
        ]

        if (
            letters
            and
            stripped == stripped.upper()
            and
            len(stripped) <= 100
            and
            not set(stripped) <= {
                "=",
                "-",
                "#",
            }
        ):
            headings.append(
                stripped
            )

    for heading in headings:
        print(
            " ",
            heading,
        )


print_headings(
    "V6 EXPERIMENT CONTRACT HEADINGS",
    CONTRACT,
)

print_headings(
    "V6 ANNOTATION CONTRACT HEADINGS",
    ANNOTATION,
)

print_headings(
    "V6 RESIDUAL REPAIR SPEC HEADINGS",
    REPAIR,
)


# ====================================================================
# SOURCE FUNCTION MAP
# ====================================================================

print()
print("=" * 88)
print("V6 SOURCE FUNCTION MAP")
print("=" * 88)

for path in (
    ONTOLOGY,
    BASE
    / "training/v6/pilot/"
      "build_v6_relational_pilot.py",
    BASE
    / "training/v6/pilot/"
      "semantic_audit_v6_relational_pilot.py",
):
    if not path.exists():
        print(
            path.relative_to(BASE),
            "MISSING",
        )
        continue

    tree = ast.parse(
        path.read_text(
            encoding="utf-8"
        )
    )

    print()
    print(
        path.relative_to(BASE)
    )

    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            end = getattr(
                node,
                "end_lineno",
                node.lineno,
            )

            print(
                f"  {type(node).__name__:16s} "
                f"{node.name:42s} "
                f"lines {node.lineno}-{end}"
            )


# ====================================================================
# FINAL
# ====================================================================

print()
print("=" * 88)
print("V6 PRODUCTION PREFLIGHT COMPLETE")
print("=" * 88)

print(
    "Ontology SHA:",
    sha(ONTOLOGY),
)

print(
    "Pilot SHA:",
    sha(PILOT),
)

print(
    "Pilot rows:",
    len(rows),
)

print(
    "Model loads: 0"
)

print(
    "Inference: 0"
)

print(
    "Data generated: 0"
)

print(
    "Challenge V4 files read: 0"
)

print()
print(
    "NEXT: construct full V6 curriculum builder "
    "directly from this frozen architecture."
)
