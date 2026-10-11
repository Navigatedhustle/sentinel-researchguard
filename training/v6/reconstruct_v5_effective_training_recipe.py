from __future__ import annotations

from pathlib import Path
import ast
import hashlib
import json
import re


BASE = Path("/workspace")

ARCHIVE = (
    BASE
    / "training/archive/"
      "v5_run01_epoch2_candidate_frozen_20260817"
)

ARCH_SCRIPT = (
    ARCHIVE
    / "provenance/"
      "train_sentinel_lora_v5.py"
)

ARCH_LOG = (
    ARCHIVE
    / "provenance/"
      "v5_run01_train.log"
)

ARCH_RECORD = (
    ARCHIVE
    / "provenance/"
      "V5_RUN01_FINAL_CANDIDATE_RECORD.txt"
)

ARCH_MODEL = (
    ARCHIVE
    / "model"
)

LIVE_SCRIPT = (
    BASE
    / "training/v5/"
      "train_sentinel_lora_v5.py"
)

RUN_ROOT = (
    BASE
    / "training/output/"
      "llama-sentinel-8b-lora-v5-run01"
)

RUN_LOG = (
    BASE
    / "experiments/v5/"
      "v5_run01_train.log"
)

RUN_RECORD = (
    BASE
    / "experiments/v5/"
      "V5_RUN01_FINAL_CANDIDATE_RECORD.txt"
)

RUN_CHECKSUMS = (
    BASE
    / "experiments/v5/"
      "V5_RUN01_SHA256SUMS.txt"
)

ARCH_CHECKSUMS = (
    ARCHIVE
    / "CANDIDATE_SHA256SUMS.txt"
)

OUT_JSON = (
    BASE
    / "experiments/v6/"
      "V5_EFFECTIVE_TRAINING_RECIPE.json"
)

OUT_TXT = (
    BASE
    / "experiments/v6/"
      "V5_EFFECTIVE_TRAINING_RECIPE.txt"
)


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


# ====================================================================
# STATIC PYTHON EXTRACTION
# ====================================================================

def simple_value(
    node,
    env,
):
    if isinstance(
        node,
        ast.Constant,
    ):
        return node.value

    if isinstance(
        node,
        ast.Name,
    ):
        return env.get(
            node.id
        )

    if isinstance(
        node,
        ast.List,
    ):
        return [
            simple_value(
                item,
                env,
            )
            for item in node.elts
        ]

    if isinstance(
        node,
        ast.Tuple,
    ):
        return tuple(
            simple_value(
                item,
                env,
            )
            for item in node.elts
        )

    if isinstance(
        node,
        ast.Dict,
    ):
        result = {}

        for key, value in zip(
            node.keys,
            node.values,
        ):
            k = simple_value(
                key,
                env,
            )

            v = simple_value(
                value,
                env,
            )

            result[
                str(
                    k
                )
            ] = v

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
        value = simple_value(
            node.operand,
            env,
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


def expression(node):
    try:
        return ast.unparse(
            node
        )
    except Exception:
        return "<unparseable>"


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


def parse_training_script(
    path: Path,
):
    text = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        text
    )

    env = {}


    # First pass: simple module constants.
    for node in tree.body:
        if isinstance(
            node,
            ast.Assign,
        ):
            value = simple_value(
                node.value,
                env,
            )

            for target in node.targets:
                if isinstance(
                    target,
                    ast.Name,
                ):
                    env[
                        target.id
                    ] = value


    argparse = {}

    config_calls = []

    dataset_strings = set()


    for node in ast.walk(
        tree
    ):

        if isinstance(
            node,
            ast.Constant,
        ) and isinstance(
            node.value,
            str,
        ):
            value = node.value

            if (
                ".jsonl" in value
                or "/datasets/" in value
                or "dataset" in value.lower()
            ):
                if len(
                    value
                ) < 500:
                    dataset_strings.add(
                        value
                    )


        if not isinstance(
            node,
            ast.Call,
        ):
            continue


        name = call_name(
            node.func
        )


        if name == "add_argument":

            flags = []

            for arg in node.args:
                value = simple_value(
                    arg,
                    env,
                )

                if isinstance(
                    value,
                    str,
                ):
                    flags.append(
                        value
                    )


            default_value = None
            default_expr = None


            for kw in node.keywords:
                if kw.arg == "default":
                    default_value = simple_value(
                        kw.value,
                        env,
                    )

                    default_expr = expression(
                        kw.value
                    )


            for flag in flags:
                if flag.startswith(
                    "--"
                ):
                    argparse[
                        flag
                    ] = {
                        "default":
                            default_value,

                        "default_expression":
                            default_expr,
                    }


        if name in {
            "LoraConfig",
            "TrainingArguments",
            "SFTConfig",
            "SFTTrainer",
            "Trainer",
            "from_pretrained",
        }:

            kwargs = {}

            for kw in node.keywords:
                if not kw.arg:
                    continue

                kwargs[
                    kw.arg
                ] = {
                    "value":
                        simple_value(
                            kw.value,
                            env,
                        ),

                    "expression":
                        expression(
                            kw.value
                        ),
                }


            config_calls.append({
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


    return {
        "path":
            rel(
                path
            ),

        "sha256":
            sha(
                path
            ),

        "argparse":
            argparse,

        "config_calls":
            config_calls,

        "dataset_string_literals":
            sorted(
                dataset_strings
            ),
    }


# ====================================================================
# ADAPTER CONFIG
# ====================================================================

def adapter_config():
    path = (
        ARCH_MODEL
        / "adapter_config.json"
    )

    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    keys = (
        "base_model_name_or_path",
        "revision",
        "peft_type",
        "task_type",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "bias",
        "modules_to_save",
        "use_dora",
        "use_rslora",
    )

    return {
        "path":
            rel(
                path
            ),

        "sha256":
            sha(
                path
            ),

        "fields": {
            key:
                obj.get(
                    key
                )
            for key in keys
            if key in obj
        },
    }


# ====================================================================
# TRAINER STATE
# ====================================================================

def trainer_state(
    path: Path,
):
    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    summary = {
        "path":
            rel(
                path
            ),

        "sha256":
            sha(
                path
            ),

        "epoch":
            obj.get(
                "epoch"
            ),

        "global_step":
            obj.get(
                "global_step"
            ),

        "max_steps":
            obj.get(
                "max_steps"
            ),

        "best_metric":
            obj.get(
                "best_metric"
            ),

        "best_model_checkpoint":
            obj.get(
                "best_model_checkpoint"
            ),

        "total_flos":
            obj.get(
                "total_flos"
            ),

        "is_local_process_zero":
            obj.get(
                "is_local_process_zero"
            ),

        "is_world_process_zero":
            obj.get(
                "is_world_process_zero"
            ),
    }


    history = obj.get(
        "log_history",
        []
    )

    eval_entries = []

    epoch_entries = []


    for item in history:
        if not isinstance(
            item,
            dict,
        ):
            continue

        if any(
            key.startswith(
                "eval_"
            )
            for key in item
        ):
            eval_entries.append(
                item
            )

        if "epoch" in item:
            epoch_entries.append(
                item
            )


    summary[
        "eval_history"
    ] = eval_entries

    summary[
        "last_history_entries"
    ] = epoch_entries[
        -8:
    ]


    return summary


# ====================================================================
# TEXT PROVENANCE
# ====================================================================

def matching_lines(
    path: Path,
    patterns,
    max_lines=200,
):
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    regex = re.compile(
        "|".join(
            patterns
        ),
        re.IGNORECASE,
    )

    output = []


    for number, line in enumerate(
        text.splitlines(),
        1,
    ):
        if regex.search(
            line
        ):
            cleaned = line.strip()

            if len(
                cleaned
            ) > 700:
                cleaned = (
                    cleaned[:700]
                    + "..."
                )

            output.append({
                "line":
                    number,

                "text":
                    cleaned,
            })


        if len(
            output
        ) >= max_lines:
            break


    return output


# ====================================================================
# CHECKPOINT IDENTITY
# ====================================================================

def hash_if_exists(
    path: Path,
):
    if not path.is_file():
        return None

    return sha(
        path
    )


def checkpoint_inventory():

    archive_hashes = {
        filename:
            hash_if_exists(
                ARCH_MODEL
                / filename
            )
        for filename in (
            "adapter_model.safetensors",
            "adapter_config.json",
            "trainer_state.json",
        )
    }


    dirs = []

    if RUN_ROOT.is_dir():

        for path in sorted(
            RUN_ROOT.glob(
                "checkpoint-*"
            ),
            key=lambda p:
                int(
                    p.name.split(
                        "-"
                    )[-1]
                ),
        ):
            if path.is_dir():
                dirs.append(
                    path
                )


        # Also compare the top-level exported adapter.
        dirs.append(
            RUN_ROOT
        )


    rows = []


    for directory in dirs:

        hashes = {
            filename:
                hash_if_exists(
                    directory
                    / filename
                )
            for filename in (
                "adapter_model.safetensors",
                "adapter_config.json",
                "trainer_state.json",
            )
        }


        matches = {
            filename:
                (
                    hashes[
                        filename
                    ]
                    is not None
                    and
                    hashes[
                        filename
                    ]
                    ==
                    archive_hashes[
                        filename
                    ]
                )
            for filename in hashes
        }


        state = None

        state_path = (
            directory
            / "trainer_state.json"
        )

        if state_path.is_file():
            state = trainer_state(
                state_path
            )


        rows.append({
            "path":
                rel(
                    directory
                ),

            "hashes":
                hashes,

            "matches_archived_candidate":
                matches,

            "trainer_state":
                state,
        })


    exact_adapter_matches = [
        row[
            "path"
        ]
        for row in rows
        if row[
            "matches_archived_candidate"
        ][
            "adapter_model.safetensors"
        ]
    ]


    return {
        "archive_model_hashes":
            archive_hashes,

        "run_candidates":
            rows,

        "exact_adapter_weight_matches":
            exact_adapter_matches,
    }


# ====================================================================
# CHECKSUM MANIFESTS
# ====================================================================

def checksum_manifest(
    path: Path,
):
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    parsed = []


    for line in lines:
        match = re.match(
            r"^([0-9a-fA-F]{64})\s+[* ]?(.+?)\s*$",
            line,
        )

        if match:
            parsed.append({
                "sha256":
                    match.group(1).lower(),

                "path":
                    match.group(2),
            })


    return {
        "path":
            rel(
                path
            ),

        "sha256":
            sha(
                path
            ),

        "entries":
            parsed,
    }


# ====================================================================
# MAIN
# ====================================================================

def main():

    archived_script = parse_training_script(
        ARCH_SCRIPT
    )

    live_script = parse_training_script(
        LIVE_SCRIPT
    )


    adapter = adapter_config()


    selected_state = trainer_state(
        ARCH_MODEL
        / "trainer_state.json"
    )


    checkpoint_data = checkpoint_inventory()


    invocation_patterns = (
        r"train_sentinel_lora_v5\.py",
        r"--epochs",
        r"--batch-size",
        r"--grad-accum",
        r"--learning-rate",
        r"--max-length",
        r"python\s+.*v5",
    )


    training_patterns = (
        r"train examples",
        r"validation examples",
        r"development",
        r"epoch",
        r"global[_ ]?step",
        r"learning[_ -]?rate",
        r"batch",
        r"grad",
        r"lora",
        r"trainable params",
        r"warmup",
        r"scheduler",
        r"optimizer",
        r"max[_ -]?(?:seq[_ -]?)?length",
        r"\bbf16\b",
        r"\bfp16\b",
        r"seed",
        r"checkpoint",
        r"candidate",
        r"selected",
    )


    candidate_patterns = (
        r"candidate",
        r"epoch",
        r"checkpoint",
        r"selected",
        r"development",
        r"challenge",
        r"sha256",
        r"adapter",
    )


    invocation_evidence = {
        "archive_log":
            matching_lines(
                ARCH_LOG,
                invocation_patterns,
                80,
            ),

        "current_log":
            matching_lines(
                RUN_LOG,
                invocation_patterns,
                80,
            ),

        "archive_record":
            matching_lines(
                ARCH_RECORD,
                invocation_patterns,
                80,
            ),

        "current_record":
            matching_lines(
                RUN_RECORD,
                invocation_patterns,
                80,
            ),
    }


    training_evidence = {
        "archive_log":
            matching_lines(
                ARCH_LOG,
                training_patterns,
                200,
            ),

        "archive_final_candidate_record":
            matching_lines(
                ARCH_RECORD,
                candidate_patterns,
                160,
            ),

        "current_final_candidate_record":
            matching_lines(
                RUN_RECORD,
                candidate_patterns,
                160,
            ),
    }


    report = {
        "status":
            "AUTHORITATIVE_V5_RECONSTRUCTION",

        "authority": {
            "primary_archive":
                rel(
                    ARCHIVE
                ),

            "reason":
                (
                    "Frozen V5 run01 epoch-2 candidate archive "
                    "contains model plus provenance."
                ),

            "archived_script_equals_live_script":
                (
                    sha(
                        ARCH_SCRIPT
                    )
                    ==
                    sha(
                        LIVE_SCRIPT
                    )
                ),

            "archived_log_equals_current_log":
                (
                    sha(
                        ARCH_LOG
                    )
                    ==
                    sha(
                        RUN_LOG
                    )
                ),

            "archived_record_equals_current_record":
                (
                    sha(
                        ARCH_RECORD
                    )
                    ==
                    sha(
                        RUN_RECORD
                    )
                ),
        },

        "scripts": {
            "archived":
                archived_script,

            "live":
                live_script,
        },

        "adapter_config":
            adapter,

        "archived_candidate_trainer_state":
            selected_state,

        "checkpoint_identity":
            checkpoint_data,

        "invocation_evidence":
            invocation_evidence,

        "training_log_evidence":
            training_evidence,

        "checksum_manifests": {
            "archived":
                checksum_manifest(
                    ARCH_CHECKSUMS
                ),

            "run":
                checksum_manifest(
                    RUN_CHECKSUMS
                ),
        },

        "binary_artifacts_hash_only": {
            filename: {
                "sha256":
                    sha(
                        ARCH_MODEL
                        / filename
                    ),

                "bytes":
                    (
                        ARCH_MODEL
                        / filename
                    ).stat().st_size,
            }
            for filename in (
                "adapter_model.safetensors",
                "training_args.bin",
                "optimizer.pt",
                "scheduler.pt",
            )
        },

        "safety": {
            "model_loads":
                0,

            "inference_calls":
                0,

            "checkpoint_deserialization":
                0,

            "challenge_v4_files_read":
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


    # =================================================================
    # HUMAN REPORT
    # =================================================================

    lines = [
        "SENTINEL V5 — EFFECTIVE TRAINING RECIPE RECONSTRUCTION",
        "=" * 82,
        "",
        "STATUS: AUTHORITATIVE_V5_RECONSTRUCTION",
        "",
        "AUTHORITY",
        "-" * 82,
        f"Primary archive: {rel(ARCHIVE)}",
        (
            "Archived script == live script: "
            + str(
                report[
                    "authority"
                ][
                    "archived_script_equals_live_script"
                ]
            )
        ),
        (
            "Archived log == current log: "
            + str(
                report[
                    "authority"
                ][
                    "archived_log_equals_current_log"
                ]
            )
        ),
        (
            "Archived candidate record == current record: "
            + str(
                report[
                    "authority"
                ][
                    "archived_record_equals_current_record"
                ]
            )
        ),
        "",
        "ADAPTER / LORA CONFIG",
        "-" * 82,
    ]


    for key, value in adapter[
        "fields"
    ].items():
        lines.append(
            f"{key}: {value!r}"
        )


    lines.extend([
        "",
        "ARCHIVED CANDIDATE TRAINER STATE",
        "-" * 82,
        f"epoch: {selected_state['epoch']!r}",
        f"global_step: {selected_state['global_step']!r}",
        f"max_steps: {selected_state['max_steps']!r}",
        f"best_metric: {selected_state['best_metric']!r}",
        (
            "best_model_checkpoint: "
            f"{selected_state['best_model_checkpoint']!r}"
        ),
        "",
        "EXACT ARCHIVED ADAPTER-WEIGHT MATCHES",
        "-" * 82,
    ])


    for path in checkpoint_data[
        "exact_adapter_weight_matches"
    ]:
        lines.append(
            path
        )


    lines.extend([
        "",
        "ARCHIVED SCRIPT ARGUMENT DEFAULTS",
        "-" * 82,
    ])


    for flag, info in sorted(
        archived_script[
            "argparse"
        ].items()
    ):
        lines.append(
            f"{flag}: {info!r}"
        )


    lines.extend([
        "",
        "ARCHIVED SCRIPT CONFIGURATION CALLS",
        "-" * 82,
    ])


    for call in archived_script[
        "config_calls"
    ]:
        lines.append(
            (
                f"{call['call']} "
                f"line {call['line']}"
            )
        )

        for key, value in sorted(
            call[
                "kwargs"
            ].items()
        ):
            lines.append(
                f"  {key}: {value!r}"
            )


    lines.extend([
        "",
        "INVOCATION EVIDENCE",
        "-" * 82,
    ])


    for source, matches in invocation_evidence.items():
        lines.append(
            source
        )

        if not matches:
            lines.append(
                "  NONE FOUND"
            )

        for item in matches:
            lines.append(
                f"  L{item['line']}: "
                f"{item['text']}"
            )


    lines.extend([
        "",
        "FINAL-CANDIDATE / TRAINING EVIDENCE",
        "-" * 82,
    ])


    for source, matches in training_evidence.items():
        lines.append(
            source
        )

        for item in matches[
            :100
        ]:
            lines.append(
                f"  L{item['line']}: "
                f"{item['text']}"
            )


    lines.extend([
        "",
        "BINARY ARTIFACTS",
        "-" * 82,
        "HASHED ONLY — NOT DESERIALIZED",
    ])


    for name, info in report[
        "binary_artifacts_hash_only"
    ].items():
        lines.append(
            f"{name}: "
            f"{info['sha256']} "
            f"({info['bytes']} bytes)"
        )


    lines.extend([
        "",
        "SAFETY",
        "-" * 82,
        "Model loads: 0",
        "Inference: 0",
        "Checkpoint deserialization: 0",
        "Challenge V4 files read: 0",
        "Training started: NO",
    ])


    OUT_TXT.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


    # =================================================================
    # TERMINAL SUMMARY
    # =================================================================

    print()
    print("=" * 82)
    print("AUTHORITATIVE V5 TRAINING RECIPE RECONSTRUCTION")
    print("=" * 82)

    print()
    print(
        "Primary authority:",
        rel(
            ARCHIVE
        ),
    )

    print(
        "Archived script == live:",
        report[
            "authority"
        ][
            "archived_script_equals_live_script"
        ],
    )

    print(
        "Archived log == current:",
        report[
            "authority"
        ][
            "archived_log_equals_current_log"
        ],
    )

    print(
        "Archived candidate record == current:",
        report[
            "authority"
        ][
            "archived_record_equals_current_record"
        ],
    )


    print()
    print("LoRA / adapter configuration")
    print("-" * 82)

    for key, value in adapter[
        "fields"
    ].items():
        print(
            f"{key:28s}",
            repr(
                value
            ),
        )


    print()
    print("Archived candidate trainer state")
    print("-" * 82)

    for key in (
        "epoch",
        "global_step",
        "max_steps",
        "best_metric",
        "best_model_checkpoint",
    ):
        print(
            f"{key:28s}",
            repr(
                selected_state[
                    key
                ]
            ),
        )


    print()
    print("Exact archived adapter-weight matches")
    print("-" * 82)

    matches = checkpoint_data[
        "exact_adapter_weight_matches"
    ]

    if not matches:
        print(
            "NONE"
        )

    for path in matches:
        print(
            path
        )


    print()
    print("Archived script CLI defaults")
    print("-" * 82)

    for flag, info in sorted(
        archived_script[
            "argparse"
        ].items()
    ):
        print(
            f"{flag:28s}",
            repr(
                info[
                    "default"
                ]
            ),
            "expr=",
            repr(
                info[
                    "default_expression"
                ]
            ),
        )


    print()
    print("Configuration calls")
    print("-" * 82)

    for call in archived_script[
        "config_calls"
    ]:
        print(
            f"{call['call']} @ line "
            f"{call['line']}"
        )

        for key, info in sorted(
            call[
                "kwargs"
            ].items()
        ):
            print(
                f"    {key:26s}"
                f"value={info['value']!r} "
                f"expr={info['expression']!r}"
            )


    print()
    print("Invocation evidence")
    print("-" * 82)

    invocation_count = 0

    for source, items in invocation_evidence.items():
        for item in items:
            invocation_count += 1

            print(
                f"{source} "
                f"L{item['line']}: "
                f"{item['text']}"
            )


    if invocation_count == 0:
        print(
            "No explicit command line recovered "
            "from text provenance."
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
    print("Model loads: 0")
    print("Inference: 0")
    print("Checkpoint deserialization: 0")
    print("Challenge V4 files read: 0")
    print("Training started: NO")


if __name__ == "__main__":
    main()
