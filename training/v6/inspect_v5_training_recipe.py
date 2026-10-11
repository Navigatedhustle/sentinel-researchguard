from __future__ import annotations

from pathlib import Path
import ast
import hashlib
import json
import os
import re


BASE = Path("/workspace")

OUT_JSON = (
    BASE
    / "experiments/v6/"
      "V5_TRAINING_RECIPE_RECONSTRUCTION.json"
)

OUT_TXT = (
    BASE
    / "experiments/v6/"
      "V5_TRAINING_RECIPE_RECONSTRUCTION.txt"
)


# ====================================================================
# SAFETY
# ====================================================================

SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".cache",
    "challenge_v4",
}

TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".log",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".md",
}


RECIPE_TERMS = (
    "lora",
    "loraconfig",
    "trainingarguments",
    "sftconfig",
    "sfttrainer",
    "learning_rate",
    "learning-rate",
    "num_train_epochs",
    "epochs",
    "batch_size",
    "batch-size",
    "gradient_accumulation",
    "grad_accum",
    "grad-accum",
    "max_length",
    "max_seq_length",
    "warmup",
    "weight_decay",
    "lr_scheduler",
    "scheduler",
    "bf16",
    "fp16",
    "optim",
    "seed",
    "train examples",
    "validation examples",
    "trainable params",
)


KEY_NAMES = {
    "model_name",
    "model_name_or_path",
    "base_model",
    "base_model_name_or_path",

    "r",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "bias",
    "task_type",

    "epochs",
    "num_train_epochs",

    "batch_size",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",

    "gradient_accumulation_steps",
    "grad_accum",

    "learning_rate",
    "lr",

    "weight_decay",

    "warmup_steps",
    "warmup_ratio",

    "lr_scheduler_type",

    "optim",
    "optimizer",

    "max_length",
    "max_seq_length",

    "bf16",
    "fp16",
    "tf32",

    "gradient_checkpointing",

    "seed",
    "data_seed",

    "save_strategy",
    "save_steps",

    "eval_strategy",
    "evaluation_strategy",
    "eval_steps",

    "logging_steps",

    "load_best_model_at_end",

    "metric_for_best_model",
    "greater_is_better",

    "output_dir",

    "train_dataset",
    "validation_dataset",
    "eval_dataset",

    "train_file",
    "validation_file",
}


INTERESTING_JSON_KEYS = {
    "base_model_name_or_path",
    "peft_type",
    "task_type",
    "r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "bias",
    "modules_to_save",

    "epoch",
    "global_step",
    "max_steps",
    "best_metric",
    "best_model_checkpoint",
    "total_flos",

    "train_batch_size",
    "eval_batch_size",
    "gradient_accumulation_steps",

    "learning_rate",
    "num_train_epochs",
    "warmup_steps",
    "warmup_ratio",
    "weight_decay",
    "lr_scheduler_type",

    "seed",
    "data_seed",

    "max_length",
    "max_seq_length",

    "bf16",
    "fp16",
}


def sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def rel(path: Path) -> str:
    return str(
        path.relative_to(
            BASE
        )
    )


def safe_path(path: Path) -> bool:
    try:
        parts = set(
            path.relative_to(
                BASE
            ).parts
        )
    except ValueError:
        return False

    return not bool(
        parts & SKIP_PARTS
    )


def static_value(node):
    if isinstance(
        node,
        ast.Constant,
    ):
        return node.value

    if isinstance(
        node,
        ast.List,
    ):
        values = []

        for item in node.elts:
            value = static_value(
                item
            )

            if value is None and not (
                isinstance(
                    item,
                    ast.Constant,
                )
                and item.value is None
            ):
                return None

            values.append(
                value
            )

        return values

    if isinstance(
        node,
        ast.Tuple,
    ):
        values = []

        for item in node.elts:
            value = static_value(
                item
            )

            if value is None and not (
                isinstance(
                    item,
                    ast.Constant,
                )
                and item.value is None
            ):
                return None

            values.append(
                value
            )

        return tuple(
            values
        )

    if isinstance(
        node,
        ast.Dict,
    ):
        result = {}

        for k, v in zip(
            node.keys,
            node.values,
        ):
            key = static_value(
                k
            )

            value = static_value(
                v
            )

            if key is None:
                return None

            result[
                str(
                    key
                )
            ] = value

        return result

    if (
        isinstance(
            node,
            ast.UnaryOp,
        )
        and isinstance(
            node.op,
            ast.USub,
        )
    ):
        value = static_value(
            node.operand
        )

        if isinstance(
            value,
            (
                int,
                float,
            ),
        ):
            return -value

    return None


def call_name(node):
    if isinstance(
        node,
        ast.Name,
    ):
        return node.id

    if isinstance(
        node,
        ast.Attribute,
    ):
        return node.attr

    return ""


def python_recipe(path: Path):
    try:
        text = path.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(
            text
        )

    except Exception:
        return None


    assignments = {}

    argparse_defaults = {}

    calls = []


    for node in ast.walk(
        tree
    ):

        # ------------------------------------------------------------
        # Constant assignments.
        # ------------------------------------------------------------

        if isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        ):

            targets = []

            value_node = None

            if isinstance(
                node,
                ast.Assign,
            ):
                targets = node.targets
                value_node = node.value

            else:
                targets = [
                    node.target
                ]
                value_node = node.value


            value = static_value(
                value_node
            )

            for target in targets:

                if isinstance(
                    target,
                    ast.Name,
                ):
                    name = target.id

                    if (
                        name.lower()
                        in KEY_NAMES
                        or any(
                            term in name.lower()
                            for term in (
                                "lora",
                                "epoch",
                                "batch",
                                "learning",
                                "warmup",
                                "scheduler",
                                "length",
                                "seed",
                            )
                        )
                    ):
                        assignments[
                            name
                        ] = value


        # ------------------------------------------------------------
        # argparse defaults.
        # ------------------------------------------------------------

        if isinstance(
            node,
            ast.Call,
        ):
            name = call_name(
                node.func
            )

            if name == "add_argument":
                flags = [
                    static_value(
                        arg
                    )
                    for arg in node.args
                ]

                flags = [
                    x
                    for x in flags
                    if isinstance(
                        x,
                        str,
                    )
                ]

                default = None

                for kw in node.keywords:
                    if kw.arg == "default":
                        default = static_value(
                            kw.value
                        )

                for flag in flags:
                    if flag.startswith(
                        "--"
                    ):
                        argparse_defaults[
                            flag
                        ] = default


            if name in {
                "LoraConfig",
                "TrainingArguments",
                "SFTConfig",
                "SFTTrainer",
            }:

                kwargs = {}

                for kw in node.keywords:
                    if kw.arg:
                        kwargs[
                            kw.arg
                        ] = static_value(
                            kw.value
                        )

                calls.append({
                    "call":
                        name,

                    "line":
                        getattr(
                            node,
                            "lineno",
                            None,
                        ),

                    "kwargs":
                        kwargs,
                })


    if not (
        assignments
        or argparse_defaults
        or calls
    ):
        return None


    return {
        "path":
            rel(
                path
            ),

        "sha256":
            sha(
                path
            ),

        "assignments":
            assignments,

        "argparse_defaults":
            argparse_defaults,

        "configuration_calls":
            calls,
    }


def json_recipe(path: Path):
    try:
        obj = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return None


    extracted = {}


    def walk(value, prefix=""):
        if isinstance(
            value,
            dict,
        ):
            for key, item in value.items():
                key_lower = str(
                    key
                ).lower()

                name = (
                    prefix
                    + "."
                    + str(
                        key
                    )
                    if prefix
                    else str(
                        key
                    )
                )

                if key_lower in INTERESTING_JSON_KEYS:
                    extracted[
                        name
                    ] = item

                walk(
                    item,
                    name,
                )

        elif isinstance(
            value,
            list,
        ):
            # Avoid dumping full trainer log histories.
            return


    walk(
        obj
    )


    name = path.name.lower()

    particularly_relevant = (
        name
        in {
            "adapter_config.json",
            "trainer_state.json",
            "config.json",
        }
        or
        "v5" in rel(
            path
        ).lower()
    )


    if not extracted:
        return None

    if not particularly_relevant:
        return None


    return {
        "path":
            rel(
                path
            ),

        "sha256":
            sha(
                path
            ),

        "fields":
            extracted,
    }


def log_evidence(path: Path):
    try:
        if path.stat().st_size > 10_000_000:
            return None

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except Exception:
        return None


    lower = text.lower()

    if not any(
        term in lower
        for term in RECIPE_TERMS
    ):
        return None


    # Require stronger evidence for ordinary logs/text files.
    path_lower = rel(
        path
    ).lower()

    relevant = (
        "v5" in path_lower
        or "sentinel" in path_lower
        or "lora" in path_lower
        or "train" in path_lower
    )

    if not relevant:
        return None


    lines = []

    patterns = (
        r"epoch",
        r"batch",
        r"grad",
        r"learning[_ -]?rate",
        r"\blr\b",
        r"lora",
        r"trainable params",
        r"max[_ -]?(?:seq[_ -]?)?length",
        r"warmup",
        r"scheduler",
        r"weight[_ -]?decay",
        r"\bbf16\b",
        r"\bfp16\b",
        r"optimizer",
        r"\boptim\b",
        r"\bseed\b",
        r"train examples",
        r"validation examples",
        r"eval examples",
        r"save[_ -]?steps",
        r"eval[_ -]?steps",
        r"output[_ -]?dir",
    )

    regex = re.compile(
        "|".join(
            patterns
        ),
        re.IGNORECASE,
    )


    for number, line in enumerate(
        text.splitlines(),
        1,
    ):
        if regex.search(
            line
        ):
            line = line.strip()

            if len(
                line
            ) > 500:
                line = (
                    line[:500]
                    + "..."
                )

            lines.append({
                "line":
                    number,

                "text":
                    line,
            })


        if len(
            lines
        ) >= 200:
            break


    if not lines:
        return None


    return {
        "path":
            rel(
                path
            ),

        "sha256":
            sha(
                path
            ),

        "matched_lines":
            lines,
    }


def candidate_files():
    roots = [
        BASE
        / "training",

        BASE
        / "results",

        BASE
        / "experiments/v5",
    ]


    seen = set()

    files = []


    for root in roots:
        if not root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(
            root
        ):
            current = Path(
                dirpath
            )

            dirnames[:] = [
                name
                for name in dirnames
                if name not in SKIP_PARTS
            ]

            for filename in filenames:
                path = (
                    current
                    / filename
                )

                if not safe_path(
                    path
                ):
                    continue

                if path.suffix.lower() not in TEXT_SUFFIXES:
                    continue

                resolved = path.resolve()

                if resolved in seen:
                    continue

                seen.add(
                    resolved
                )

                files.append(
                    path
                )


    return sorted(
        files
    )


def checkpoint_inventory():
    roots = [
        BASE
        / "training",

        BASE
        / "results",
    ]

    entries = []


    for root in roots:
        if not root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(
            root
        ):
            current = Path(
                dirpath
            )

            dirnames[:] = [
                name
                for name in dirnames
                if name not in SKIP_PARTS
            ]

            names = set(
                filenames
            )

            signals = sorted(
                names
                & {
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "trainer_state.json",
                    "training_args.bin",
                    "optimizer.pt",
                    "scheduler.pt",
                }
            )

            if signals:
                entries.append({
                    "path":
                        rel(
                            current
                        ),

                    "artifacts":
                        signals,
                })


    return entries


def score_source(item):
    text = json.dumps(
        item,
        sort_keys=True,
    ).lower()

    score = 0

    if "v5" in text:
        score += 8

    if "sentinel_v5" in text:
        score += 8

    if "lora" in text:
        score += 4

    if "train" in text:
        score += 2

    if "adapter_config" in text:
        score += 4

    if "trainer_state" in text:
        score += 4

    if "learning_rate" in text:
        score += 2

    if "num_train_epochs" in text:
        score += 2

    return score


def main():

    files = candidate_files()


    python_sources = []

    json_sources = []

    text_sources = []


    for path in files:

        if path.suffix.lower() == ".py":
            item = python_recipe(
                path
            )

            if item:
                python_sources.append(
                    item
                )


        if path.suffix.lower() == ".json":
            item = json_recipe(
                path
            )

            if item:
                json_sources.append(
                    item
                )


        if path.suffix.lower() in {
            ".log",
            ".txt",
            ".md",
            ".sh",
        }:
            item = log_evidence(
                path
            )

            if item:
                text_sources.append(
                    item
                )


    python_sources.sort(
        key=lambda x: (
            -score_source(
                x
            ),
            x["path"],
        )
    )

    json_sources.sort(
        key=lambda x: (
            -score_source(
                x
            ),
            x["path"],
        )
    )

    text_sources.sort(
        key=lambda x: (
            -score_source(
                x
            ),
            x["path"],
        )
    )


    checkpoints = checkpoint_inventory()


    report = {
        "status":
            "PREFLIGHT_ONLY",

        "purpose":
            "Reconstruct actual V5 Sentinel training recipe before V6 recipe freeze.",

        "python_recipe_sources":
            python_sources,

        "json_recipe_sources":
            json_sources,

        "text_log_sources":
            text_sources,

        "checkpoint_inventory":
            checkpoints,

        "safety": {
            "challenge_v4_files_read":
                0,

            "model_loads":
                0,

            "inference_calls":
                0,

            "torch_checkpoint_deserialization":
                0,

            "training_started":
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
        "SENTINEL V6 — V5 TRAINING RECIPE RECONSTRUCTION PREFLIGHT",
        "=" * 80,
        "",
        "STATUS: PREFLIGHT_ONLY",
        "",
        "PYTHON RECIPE SOURCES",
        "-" * 80,
    ]


    for item in python_sources[:20]:
        lines.append(
            item[
                "path"
            ]
        )

        lines.append(
            "  SHA256: "
            + item[
                "sha256"
            ]
        )

        for key, value in sorted(
            item[
                "assignments"
            ].items()
        ):
            lines.append(
                f"  assignment {key} = {value!r}"
            )

        for key, value in sorted(
            item[
                "argparse_defaults"
            ].items()
        ):
            lines.append(
                f"  argparse {key} = {value!r}"
            )

        for call in item[
            "configuration_calls"
        ]:
            lines.append(
                f"  {call['call']} line "
                f"{call['line']}: "
                f"{call['kwargs']!r}"
            )

        lines.append(
            ""
        )


    lines.extend([
        "",
        "JSON / CHECKPOINT METADATA SOURCES",
        "-" * 80,
    ])


    for item in json_sources[:30]:
        lines.append(
            item[
                "path"
            ]
        )

        lines.append(
            "  SHA256: "
            + item[
                "sha256"
            ]
        )

        for key, value in sorted(
            item[
                "fields"
            ].items()
        ):
            lines.append(
                f"  {key} = {value!r}"
            )

        lines.append(
            ""
        )


    lines.extend([
        "",
        "TEXT / LOG EVIDENCE",
        "-" * 80,
    ])


    for item in text_sources[:20]:
        lines.append(
            item[
                "path"
            ]
        )

        lines.append(
            "  SHA256: "
            + item[
                "sha256"
            ]
        )

        for match in item[
            "matched_lines"
        ][:80]:
            lines.append(
                f"  L{match['line']}: "
                f"{match['text']}"
            )

        lines.append(
            ""
        )


    lines.extend([
        "",
        "CHECKPOINT / ADAPTER INVENTORY",
        "-" * 80,
    ])


    for item in checkpoints:
        lines.append(
            item[
                "path"
            ]
            + ": "
            + ", ".join(
                item[
                    "artifacts"
                ]
            )
        )


    lines.extend([
        "",
        "SAFETY",
        "-" * 80,
        "Challenge V4 files read: 0",
        "Model loads: 0",
        "Inference calls: 0",
        "Torch checkpoint deserialization: 0",
        "Training started: NO",
    ])


    OUT_TXT.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


    print()
    print("=" * 80)
    print("V5 TRAINING RECIPE RECONSTRUCTION PREFLIGHT")
    print("=" * 80)

    print()
    print(
        "Candidate Python recipe sources:",
        len(
            python_sources
        ),
    )

    for item in python_sources[:12]:
        print(
            "  ",
            item[
                "path"
            ],
        )


    print()
    print(
        "Candidate JSON/config sources:",
        len(
            json_sources
        ),
    )

    for item in json_sources[:12]:
        print(
            "  ",
            item[
                "path"
            ],
        )


    print()
    print(
        "Candidate log/text sources:",
        len(
            text_sources
        ),
    )

    for item in text_sources[:12]:
        print(
            "  ",
            item[
                "path"
            ],
        )


    print()
    print(
        "Checkpoint / adapter directories:",
        len(
            checkpoints
        ),
    )

    for item in checkpoints[:20]:
        print(
            "  ",
            item[
                "path"
            ],
            "->",
            ", ".join(
                item[
                    "artifacts"
                ]
            ),
        )


    print()
    print(
        "Report JSON SHA256:",
        sha(
            OUT_JSON
        ),
    )

    print(
        "Report TXT SHA256:",
        sha(
            OUT_TXT
        ),
    )

    print()
    print(
        "Challenge V4 files read: 0"
    )

    print(
        "Model loads: 0"
    )

    print(
        "Inference: 0"
    )

    print(
        "Torch checkpoint deserialization: 0"
    )

    print(
        "Training started: NO"
    )


if __name__ == "__main__":
    main()
