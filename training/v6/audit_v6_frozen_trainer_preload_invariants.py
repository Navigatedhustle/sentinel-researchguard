from __future__ import annotations

from collections import Counter
from pathlib import Path
import ast
import hashlib
import json
import re


BASE = Path("/workspace")

TRAIN = (
    BASE
    / "training/v6/frozen/"
      "sentinel_v6_train.jsonl"
)

DEV = (
    BASE
    / "training/v6/frozen/"
      "sentinel_v6_development.jsonl"
)

TRAINER = (
    BASE
    / "training/v6/frozen/"
      "train_sentinel_lora_v6.py"
)


EXPECTED_V6_TRAIN_SHA = (
    "9dae655425b391708beb26b01f72bc8f36cd2353c6092d8431307dacaeb6a9ff"
)

EXPECTED_V6_DEV_SHA = (
    "23163c4decdf3a4af4df85811b63538ac469582241b96a4f153c30c2725d3007"
)


OUT_JSON = (
    BASE
    / "experiments/v6/"
      "V6_FROZEN_TRAINER_PRELOAD_INVARIANT_AUDIT.json"
)

OUT_TXT = (
    BASE
    / "experiments/v6/"
      "V6_FROZEN_TRAINER_PRELOAD_INVARIANT_AUDIT.txt"
)


LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)


def sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_rows(path: Path):
    rows = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        rows.append(
            json.loads(
                line
            )
        )

    return rows


def output_decision(row):
    output = row.get(
        "output"
    )

    if not isinstance(
        output,
        str,
    ):
        return None

    try:
        parsed = json.loads(
            output
        )
    except Exception:
        return None

    return parsed.get(
        "decision"
    )


def source_segment(
    source: str,
    node,
):
    segment = ast.get_source_segment(
        source,
        node,
    )

    if segment is None:
        try:
            segment = ast.unparse(
                node
            )
        except Exception:
            segment = "<unavailable>"

    return segment


# ====================================================================
# ACTUAL FROZEN DATA FACTS
# ====================================================================

train_rows = load_rows(
    TRAIN
)

dev_rows = load_rows(
    DEV
)


train_target_counts = Counter(
    row.get(
        "target"
    )
    for row in train_rows
)

dev_target_counts = Counter(
    row.get(
        "target"
    )
    for row in dev_rows
)


train_decision_counts = Counter(
    output_decision(
        row
    )
    for row in train_rows
)

dev_decision_counts = Counter(
    output_decision(
        row
    )
    for row in dev_rows
)


actual = {
    "train_sha256":
        sha(
            TRAIN
        ),

    "development_sha256":
        sha(
            DEV
        ),

    "train_rows":
        len(
            train_rows
        ),

    "development_rows":
        len(
            dev_rows
        ),

    "train_target_counts":
        dict(
            sorted(
                train_target_counts.items(),
                key=lambda x:
                    str(
                        x[0]
                    ),
            )
        ),

    "development_target_counts":
        dict(
            sorted(
                dev_target_counts.items(),
                key=lambda x:
                    str(
                        x[0]
                    ),
            )
        ),

    "train_output_decision_counts":
        dict(
            sorted(
                train_decision_counts.items(),
                key=lambda x:
                    str(
                        x[0]
                    ),
            )
        ),

    "development_output_decision_counts":
        dict(
            sorted(
                dev_decision_counts.items(),
                key=lambda x:
                    str(
                        x[0]
                    ),
            )
        ),
}


if actual[
    "train_sha256"
] != EXPECTED_V6_TRAIN_SHA:
    raise RuntimeError(
        "Frozen V6 TRAIN changed."
    )

if actual[
    "development_sha256"
] != EXPECTED_V6_DEV_SHA:
    raise RuntimeError(
        "Frozen V6 DEVELOPMENT changed."
    )

if len(
    train_rows
) != 4096:
    raise RuntimeError(
        "V6 TRAIN row count changed."
    )

if len(
    dev_rows
) != 1024:
    raise RuntimeError(
        "V6 DEVELOPMENT row count changed."
    )


expected_train_balance = {
    label:
        1024
    for label in LABELS
}

expected_dev_balance = {
    label:
        256
    for label in LABELS
}


if dict(
    train_target_counts
) != expected_train_balance:
    raise RuntimeError(
        "Unexpected V6 TRAIN target balance."
    )

if dict(
    dev_target_counts
) != expected_dev_balance:
    raise RuntimeError(
        "Unexpected V6 DEVELOPMENT target balance."
    )

if dict(
    train_decision_counts
) != expected_train_balance:
    raise RuntimeError(
        "Unexpected V6 TRAIN output-decision balance."
    )

if dict(
    dev_decision_counts
) != expected_dev_balance:
    raise RuntimeError(
        "Unexpected V6 DEVELOPMENT output-decision balance."
    )


# ====================================================================
# STATIC TRAINER AST
# ====================================================================

source = TRAINER.read_text(
    encoding="utf-8"
)

tree = ast.parse(
    source
)


from_pretrained_lines = []


for node in ast.walk(
    tree
):

    if not isinstance(
        node,
        ast.Call,
    ):
        continue

    func = node.func

    if (
        isinstance(
            func,
            ast.Attribute,
        )
        and func.attr
        == "from_pretrained"
    ):
        from_pretrained_lines.append(
            node.lineno
        )


if not from_pretrained_lines:
    raise RuntimeError(
        "No from_pretrained calls located."
    )


first_load_line = min(
    from_pretrained_lines
)


# ====================================================================
# ASSIGNMENT INVENTORY BEFORE FIRST LOAD
# ====================================================================

assignments = []


for node in ast.walk(
    tree
):

    if node.lineno >= first_load_line:
        continue

    if isinstance(
        node,
        ast.Assign,
    ):
        targets = node.targets
        value = node.value

    elif isinstance(
        node,
        ast.AnnAssign,
    ):
        targets = [
            node.target
        ]
        value = node.value

    else:
        continue


    for target in targets:

        if not isinstance(
            target,
            ast.Name,
        ):
            continue

        name = target.id

        interesting = (
            name.startswith(
                "EXPECTED_"
            )
            or name.startswith(
                "expected_"
            )
            or name
            in {
                "TRAIN_FILE",
                "DEV_FILE",
                "train_sha",
                "dev_sha",
                "planned_optimizer_steps",
                "warmup_steps",
                "micro_batches_per_epoch",
                "optimizer_steps_per_epoch",
            }
        )


        if not interesting:
            continue


        try:
            expression = ast.unparse(
                value
            )
        except Exception:
            expression = "<unparseable>"


        literal = None

        try:
            literal = ast.literal_eval(
                value
            )
        except Exception:
            pass


        assignments.append({
            "name":
                name,

            "line":
                node.lineno,

            "end_line":
                getattr(
                    node,
                    "end_lineno",
                    node.lineno,
                ),

            "expression":
                expression,

            "literal":
                literal,
        })


assignments.sort(
    key=lambda x:
        (
            x[
                "line"
            ],
            x[
                "name"
            ],
        )
)


# ====================================================================
# ALL ASSERTS / IF GUARDS BEFORE MODEL LOAD
# ====================================================================

guards = []


for node in ast.walk(
    tree
):

    if node.lineno >= first_load_line:
        continue


    if isinstance(
        node,
        ast.Assert,
    ):
        guards.append({
            "kind":
                "assert",

            "line":
                node.lineno,

            "end_line":
                getattr(
                    node,
                    "end_lineno",
                    node.lineno,
                ),

            "condition":
                source_segment(
                    source,
                    node.test,
                ),
        })


    elif isinstance(
        node,
        ast.If,
    ):

        condition = source_segment(
            source,
            node.test,
        )

        names = {
            n.id
            for n in ast.walk(
                node.test
            )
            if isinstance(
                n,
                ast.Name,
            )
        }


        interesting_names = {
            name
            for name in names
            if (
                name.startswith(
                    "EXPECTED_"
                )
                or name.startswith(
                    "expected_"
                )
                or name
                in {
                    "train_sha",
                    "dev_sha",
                    "train_rows",
                    "dev_rows",
                    "train_labels",
                    "dev_labels",
                    "planned_optimizer_steps",
                    "warmup_steps",
                }
            )
        }


        if interesting_names:

            guards.append({
                "kind":
                    "if",

                "line":
                    node.lineno,

                "end_line":
                    getattr(
                        node,
                        "end_lineno",
                        node.lineno,
                    ),

                "condition":
                    condition,

                "referenced_symbols":
                    sorted(
                        interesting_names
                    ),
            })


guards.sort(
    key=lambda x:
        x[
            "line"
        ]
)


# ====================================================================
# RUNTIME STRINGS / SHA LITERALS BEFORE MODEL LOAD
# ====================================================================

sha_pattern = re.compile(
    r"^[0-9a-f]{64}$"
)


sha_literals = []

v5_runtime_strings = []


for node in ast.walk(
    tree
):

    if node.lineno >= first_load_line:
        continue

    if not (
        isinstance(
            node,
            ast.Constant,
        )
        and isinstance(
            node.value,
            str,
        )
    ):
        continue


    if sha_pattern.fullmatch(
        node.value
    ):

        sha_literals.append({
            "line":
                node.lineno,

            "value":
                node.value,
        })


    value_lower = node.value.lower()

    if (
        "training/v5/" in value_lower
        or "sentinel_v5" in value_lower
        or " v5 " in (
            " "
            + value_lower
            + " "
        )
    ):

        v5_runtime_strings.append({
            "line":
                node.lineno,

            "value":
                node.value,
        })


# ====================================================================
# KNOWN INVARIANT VALUES
# ====================================================================

assignment_map = {
    item[
        "name"
    ]:
        item
    for item in assignments
}


def literal_for(
    name,
):
    item = assignment_map.get(
        name
    )

    if item is None:
        return None

    return item[
        "literal"
    ]


known_checks = {
    "EXPECTED_TRAIN_SHA": {
        "observed":
            literal_for(
                "EXPECTED_TRAIN_SHA"
            ),

        "required":
            EXPECTED_V6_TRAIN_SHA,
    },

    "EXPECTED_DEV_SHA": {
        "observed":
            literal_for(
                "EXPECTED_DEV_SHA"
            ),

        "required":
            EXPECTED_V6_DEV_SHA,
    },

    "EXPECTED_TRAIN_ROWS": {
        "observed":
            literal_for(
                "EXPECTED_TRAIN_ROWS"
            ),

        "required":
            4096,
    },

    "EXPECTED_DEV_ROWS": {
        "observed":
            literal_for(
                "EXPECTED_DEV_ROWS"
            ),

        "required":
            1024,
    },
}


for item in known_checks.values():
    item[
        "matches"
    ] = (
        item[
            "observed"
        ]
        ==
        item[
            "required"
        ]
    )


# ====================================================================
# EXPLICITLY INSPECT EXPECTED LABEL EXPRESSIONS
# ====================================================================

label_invariant_checks = {}


for name, required_per_label in (
    (
        "expected_train_labels",
        1024,
    ),
    (
        "expected_dev_labels",
        256,
    ),
):

    item = assignment_map.get(
        name
    )

    if item is None:

        label_invariant_checks[
            name
        ] = {
            "found":
                False,

            "required_per_label":
                required_per_label,

            "matches":
                False,
        }

        continue


    expression = item[
        "expression"
    ]


    integers = [
        n.value
        for n in ast.walk(
            ast.parse(
                expression,
                mode="eval",
            )
        )
        if (
            isinstance(
                n,
                ast.Constant,
            )
            and type(
                n.value
            )
            is int
        )
    ]


    label_invariant_checks[
        name
    ] = {
        "found":
            True,

        "line":
            item[
                "line"
            ],

        "expression":
            expression,

        "integer_literals":
            integers,

        "required_per_label":
            required_per_label,

        "matches":
            (
                required_per_label
                in integers
            ),
    }


# ====================================================================
# FINAL STATUS
# ====================================================================

known_mismatches = [
    name
    for name, item in known_checks.items()
    if not item[
        "matches"
    ]
]


label_mismatches = [
    name
    for name, item
    in label_invariant_checks.items()
    if not item[
        "matches"
    ]
]


status = (
    "VERSIONED_RECIPE_REFREEZE_REQUIRED"
    if (
        known_mismatches
        or label_mismatches
        or v5_runtime_strings
    )
    else
    "NO_STATIC_PRELOAD_MISMATCH_FOUND"
)


report = {
    "status":
        status,

    "trainer_path":
        str(
            TRAINER.relative_to(
                BASE
            )
        ),

    "trainer_sha256":
        sha(
            TRAINER
        ),

    "first_from_pretrained_line":
        first_load_line,

    "all_from_pretrained_lines":
        sorted(
            from_pretrained_lines
        ),

    "actual_frozen_data":
        actual,

    "required_balance": {
        "train":
            expected_train_balance,

        "development":
            expected_dev_balance,
    },

    "preload_assignments":
        assignments,

    "preload_guards":
        guards,

    "sha_literals_before_model_load":
        sorted(
            sha_literals,
            key=lambda x:
                x[
                    "line"
                ],
        ),

    "v5_runtime_strings_before_model_load":
        sorted(
            v5_runtime_strings,
            key=lambda x:
                x[
                    "line"
                ],
        ),

    "known_invariant_checks":
        known_checks,

    "label_invariant_checks":
        label_invariant_checks,

    "known_mismatches":
        known_mismatches,

    "label_mismatches":
        label_mismatches,

    "safety": {
        "model_loads":
            0,

        "checkpoint_deserialization":
            0,

        "inference_calls":
            0,

        "optimizer_steps":
            0,

        "challenge_v4_files_read":
            0,

        "frozen_trainer_modified":
            False,
    },
}


OUT_JSON.write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


lines = [
    "SENTINEL V6 — FROZEN TRAINER PRE-MODEL INVARIANT AUDIT",
    "=" * 88,
    "",
    f"STATUS: {status}",
    "",
    "ACTUAL FROZEN V6 DATA",
    "-" * 88,
    f"TRAIN SHA: {actual['train_sha256']}",
    f"DEV SHA:   {actual['development_sha256']}",
    f"TRAIN rows: {actual['train_rows']}",
    f"DEV rows:   {actual['development_rows']}",
    f"TRAIN target counts: {actual['train_target_counts']}",
    f"DEV target counts:   {actual['development_target_counts']}",
    "",
    "FIRST MODEL LOAD",
    "-" * 88,
    f"first from_pretrained line: {first_load_line}",
    f"all from_pretrained lines: {sorted(from_pretrained_lines)}",
    "",
    "KNOWN INVARIANT CHECKS",
    "-" * 88,
]


for name, item in known_checks.items():

    lines.append(
        f"{name}: "
        f"observed={item['observed']!r} "
        f"required={item['required']!r} "
        f"{'PASS' if item['matches'] else 'MISMATCH'}"
    )


lines.extend([
    "",
    "LABEL INVARIANT CHECKS",
    "-" * 88,
])


for name, item in label_invariant_checks.items():

    lines.append(
        f"{name}: "
        f"{item}"
    )


lines.extend([
    "",
    "ALL INTERESTING PRE-MODEL ASSIGNMENTS",
    "-" * 88,
])


for item in assignments:

    lines.append(
        f"L{item['line']}-{item['end_line']} "
        f"{item['name']} = {item['expression']}"
    )


lines.extend([
    "",
    "ALL INTERESTING PRE-MODEL GUARDS",
    "-" * 88,
])


for item in guards:

    lines.append(
        f"L{item['line']}-{item['end_line']} "
        f"{item['kind']}: {item['condition']}"
    )


lines.extend([
    "",
    "SHA LITERALS BEFORE MODEL LOAD",
    "-" * 88,
])


for item in sha_literals:

    lines.append(
        f"L{item['line']}: {item['value']}"
    )


lines.extend([
    "",
    "V5 RUNTIME STRINGS BEFORE MODEL LOAD",
    "-" * 88,
])


if v5_runtime_strings:

    for item in v5_runtime_strings:

        lines.append(
            f"L{item['line']}: {item['value']!r}"
        )

else:

    lines.append(
        "NONE"
    )


lines.extend([
    "",
    "MISMATCH SUMMARY",
    "-" * 88,
    f"known mismatches: {known_mismatches}",
    f"label mismatches: {label_mismatches}",
    "",
    "SAFETY",
    "-" * 88,
    "Model loads: 0",
    "Checkpoint deserialization: 0",
    "Inference: 0",
    "Optimizer steps: 0",
    "Challenge V4 files read: 0",
    "Frozen trainer modified: NO",
])


OUT_TXT.write_text(
    "\n".join(
        lines
    )
    + "\n",
    encoding="utf-8",
)


print()
print("=" * 88)
print(
    "V6 FROZEN TRAINER PRE-MODEL INVARIANT AUDIT"
)
print("=" * 88)

print()
print(
    "STATUS:",
    status,
)

print()
print("ACTUAL V6 DATA")
print("-" * 88)

print(
    "TRAIN SHA:",
    actual[
        "train_sha256"
    ],
)

print(
    "DEV SHA:  ",
    actual[
        "development_sha256"
    ],
)

print(
    "TRAIN rows:",
    actual[
        "train_rows"
    ],
)

print(
    "DEV rows:  ",
    actual[
        "development_rows"
    ],
)

print(
    "TRAIN labels:",
    actual[
        "train_target_counts"
    ],
)

print(
    "DEV labels:  ",
    actual[
        "development_target_counts"
    ],
)

print()
print("KNOWN INVARIANTS")
print("-" * 88)

for name, item in known_checks.items():

    print(
        name,
        "observed=",
        item[
            "observed"
        ],
        "required=",
        item[
            "required"
        ],
        "=>",
        (
            "PASS"
            if item[
                "matches"
            ]
            else "MISMATCH"
        ),
    )


print()
print("LABEL INVARIANTS")
print("-" * 88)

for name, item in label_invariant_checks.items():

    print(
        name,
        "=>",
        item,
    )


print()
print("ALL PRE-MODEL EXPECTED / DERIVED ASSIGNMENTS")
print("-" * 88)

for item in assignments:

    print(
        f"L{item['line']}-{item['end_line']} "
        f"{item['name']} = {item['expression']}"
    )


print()
print("PRE-MODEL GUARDS")
print("-" * 88)

for item in guards:

    print(
        f"L{item['line']}-{item['end_line']} "
        f"{item['kind']}: {item['condition']}"
    )


print()
print(
    "Known mismatches:",
    known_mismatches,
)

print(
    "Label mismatches:",
    label_mismatches,
)

print(
    "V5 runtime strings before model load:",
    len(
        v5_runtime_strings
    ),
)

print()
print("Model loads: 0")
print("Checkpoint deserialization: 0")
print("Inference: 0")
print("Optimizer steps: 0")
print("Challenge V4 files read: 0")
print("Frozen trainer modified: NO")
