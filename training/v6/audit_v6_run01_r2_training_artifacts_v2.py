from __future__ import annotations

from pathlib import Path
import ast
import hashlib
import json
import math
import re


BASE = Path("/workspace")

RUN_DIR = (
    BASE
    / "training/output/"
      "llama-sentinel-8b-lora-v6-run01"
)

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
    / "training/v6/frozen_recipe_r2/"
      "train_sentinel_lora_v6_r2.py"
)

MANIFEST = (
    BASE
    / "experiments/v6/frozen_recipe_r2/"
      "V6_R2_TRAINING_RECIPE_FREEZE_MANIFEST.json"
)

EVALUATOR = (
    BASE
    / "training/v6/frozen/"
      "evaluate_v6_development_run01.py"
)

POLICY = (
    BASE
    / "experiments/v6/frozen/"
      "V6_RUN01_DEVELOPMENT_SELECTION_POLICY.txt"
)

LAUNCH = (
    BASE
    / "experiments/v6/"
      "V6_RUN01_R2_TRAINING_LAUNCH_RECORD.txt"
)

LOG = (
    BASE
    / "experiments/v6/"
      "v6_run01_r2_train.log"
)

COMPLETION = (
    BASE
    / "experiments/v6/"
      "V6_RUN01_R2_TRAINING_COMPLETION_RECORD.txt"
)


OUT_DIR = (
    BASE
    / "experiments/v6/"
      "run01_r2_training_artifact_audit"
)

OUT_JSON = (
    OUT_DIR
    / "V6_RUN01_R2_TRAINING_ARTIFACT_AUDIT_V2.json"
)

OUT_TXT = (
    OUT_DIR
    / "V6_RUN01_R2_TRAINING_ARTIFACT_AUDIT_V2.txt"
)

OUT_SUMS = (
    OUT_DIR
    / "V6_RUN01_R2_TRAINING_ARTIFACT_SHA256SUMS_V2.txt"
)


EXPECTED = {
    "train_sha256":
        "9dae655425b391708beb26b01f72bc8f36cd2353c6092d8431307dacaeb6a9ff",

    "development_sha256":
        "23163c4decdf3a4af4df85811b63538ac469582241b96a4f153c30c2725d3007",

    "trainer_sha256":
        "551283ae8cb999bef9a0517bea2fb32cc62e43e702611da26e0c20f6af4d52ab",

    "manifest_sha256":
        "6789f3535e66d339915ffda203f7170a658f75fc10f839816f752010d03db036",

    "evaluator_sha256":
        "5c5e7a12703ce688fe5d71dbddef0f442bfe4c8e8b2aa9c78d79344d32a4c89b",

    "policy_sha256":
        "e775f2c9d636f1c8c472187d654be6a3552ca6f1151e762b0a1f3efcd0442965",

    "launch_sha256":
        "c7f3d7871bfdff2832fea49a42b1e7a95d32b9b98bd53abdfe605373fddbfc6e",

    "log_sha256":
        "de680b1a28c22b2b3f5473fe997f41fa2bb716c91b3a13782a6130d68398fa40",

    "completion_sha256":
        "25789acd73da75f182b5ece0a507fbfaaf5163aa8b36be49df47fd8d3eb215a6",

    "final_adapter_sha256":
        "c80ba2f8a702940b8a69760dda5bb439020d95637185b1fee7fc346027841ce6",
}


EXPECTED_CHECKPOINTS = {
    512:
        1.0,

    1024:
        2.0,

    1536:
        3.0,
}


CORE_CHECKPOINT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "trainer_state.json",
}


EXPECTED_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def sha256(path: Path) -> str:

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def assert_sha(
    path: Path,
    expected: str,
):
    actual = sha256(
        path
    )

    if actual != expected:
        raise RuntimeError(
            f"SHA mismatch: {path}\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )

    return actual


def load_json(path: Path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def finite_walk(
    value,
    path="root",
):

    if isinstance(
        value,
        bool,
    ) or value is None:
        return

    if isinstance(
        value,
        (int, float),
    ):

        if isinstance(
            value,
            float,
        ) and not math.isfinite(
            value
        ):
            raise RuntimeError(
                f"Non-finite numeric value at {path}"
            )

        return

    if isinstance(
        value,
        dict,
    ):

        for key, item in value.items():
            finite_walk(
                item,
                f"{path}.{key}",
            )

        return

    if isinstance(
        value,
        list,
    ):

        for index, item in enumerate(
            value
        ):
            finite_walk(
                item,
                f"{path}[{index}]",
            )

        return


def file_inventory(
    directory: Path,
):

    result = {}

    for path in sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
    ):

        relative = str(
            path.relative_to(
                directory
            )
        )

        result[
            relative
        ] = {
            "size_bytes":
                path.stat().st_size,

            "sha256":
                sha256(
                    path
                ),
        }

    return result


def validate_adapter_config(
    config,
    name,
):

    if str(
        config.get(
            "peft_type",
            "",
        )
    ).upper() != "LORA":
        raise RuntimeError(
            f"{name}: PEFT type is not LORA"
        )

    if int(
        config.get(
            "r"
        )
    ) != 8:
        raise RuntimeError(
            f"{name}: LoRA r != 8"
        )

    if int(
        config.get(
            "lora_alpha"
        )
    ) != 16:
        raise RuntimeError(
            f"{name}: LoRA alpha != 16"
        )

    if float(
        config.get(
            "lora_dropout"
        )
    ) != 0.0:
        raise RuntimeError(
            f"{name}: LoRA dropout != 0"
        )

    if str(
        config.get(
            "bias",
            "",
        )
    ).lower() != "none":
        raise RuntimeError(
            f"{name}: LoRA bias != none"
        )

    targets = set(
        config.get(
            "target_modules",
            []
        )
    )

    if targets != EXPECTED_TARGETS:
        raise RuntimeError(
            f"{name}: target module mismatch\n"
            f"{targets}"
        )

    base_model = str(
        config.get(
            "base_model_name_or_path",
            ""
        )
    )

    if not base_model.endswith(
        "Llama-3.1-8B-Instruct"
    ):
        raise RuntimeError(
            f"{name}: unexpected base model "
            f"{base_model!r}"
        )


# ====================================================================
# PARENT IDENTITIES
# ====================================================================

parent_hashes = {
    "train":
        assert_sha(
            TRAIN,
            EXPECTED[
                "train_sha256"
            ],
        ),

    "development":
        assert_sha(
            DEV,
            EXPECTED[
                "development_sha256"
            ],
        ),

    "trainer":
        assert_sha(
            TRAINER,
            EXPECTED[
                "trainer_sha256"
            ],
        ),

    "manifest":
        assert_sha(
            MANIFEST,
            EXPECTED[
                "manifest_sha256"
            ],
        ),

    "evaluator":
        assert_sha(
            EVALUATOR,
            EXPECTED[
                "evaluator_sha256"
            ],
        ),

    "selection_policy":
        assert_sha(
            POLICY,
            EXPECTED[
                "policy_sha256"
            ],
        ),

    "launch_record":
        assert_sha(
            LAUNCH,
            EXPECTED[
                "launch_sha256"
            ],
        ),

    "training_log":
        assert_sha(
            LOG,
            EXPECTED[
                "log_sha256"
            ],
        ),

    "completion_record":
        assert_sha(
            COMPLETION,
            EXPECTED[
                "completion_sha256"
            ],
        ),
}


# ====================================================================
# LOG COMPLETION EVIDENCE
# ====================================================================

log_text = LOG.read_text(
    encoding="utf-8",
    errors="strict",
)


required_log_strings = [
    "TRAINING COMPLETE",
    "SENTINEL V6 RUN TRAINING: PASS",
    "1536/1536",
    EXPECTED[
        "final_adapter_sha256"
    ],
]


for token in required_log_strings:

    if token not in log_text:
        raise RuntimeError(
            "Training log missing completion evidence: "
            + token
        )


# ====================================================================
# CHECKPOINT SET
# ====================================================================

observed_checkpoint_dirs = sorted(
    (
        p
        for p in RUN_DIR.iterdir()
        if (
            p.is_dir()
            and re.fullmatch(
                r"checkpoint-\d+",
                p.name,
            )
        )
    ),
    key=lambda p:
        int(
            p.name.split(
                "-"
            )[-1]
        ),
)


observed_steps = [
    int(
        p.name.split(
            "-"
        )[-1]
    )
    for p in observed_checkpoint_dirs
]


if observed_steps != [
    512,
    1024,
    1536,
]:
    raise RuntimeError(
        "Unexpected checkpoint set: "
        + repr(
            observed_steps
        )
    )


# ====================================================================
# CHECKPOINT AUDITS
# ====================================================================

checkpoints = {}


for directory in observed_checkpoint_dirs:

    step = int(
        directory.name.split(
            "-"
        )[-1]
    )

    expected_epoch = (
        EXPECTED_CHECKPOINTS[
            step
        ]
    )


    files = {
        p.name
        for p in directory.iterdir()
        if p.is_file()
    }


    missing = (
        CORE_CHECKPOINT_FILES
        - files
    )

    if missing:
        raise RuntimeError(
            f"{directory.name}: "
            f"missing core checkpoint files: "
            f"{sorted(missing)}"
        )


    adapter_model = (
        directory
        / "adapter_model.safetensors"
    )

    adapter_config_path = (
        directory
        / "adapter_config.json"
    )

    state_path = (
        directory
        / "trainer_state.json"
    )


    if adapter_model.stat().st_size <= 0:
        raise RuntimeError(
            f"{directory.name}: "
            "empty adapter_model.safetensors"
        )


    adapter_config = load_json(
        adapter_config_path
    )

    validate_adapter_config(
        adapter_config,
        directory.name,
    )


    state = load_json(
        state_path
    )

    finite_walk(
        state,
        directory.name,
    )


    if int(
        state.get(
            "global_step",
            -1,
        )
    ) != step:
        raise RuntimeError(
            f"{directory.name}: "
            f"global_step mismatch "
            f"{state.get('global_step')}"
        )


    epoch = float(
        state.get(
            "epoch"
        )
    )

    if abs(
        epoch
        - expected_epoch
    ) > 0.01:
        raise RuntimeError(
            f"{directory.name}: "
            f"epoch={epoch}, "
            f"expected={expected_epoch}"
        )


    if (
        "max_steps" in state
        and int(
            state[
                "max_steps"
            ]
        )
        != 1536
    ):
        raise RuntimeError(
            f"{directory.name}: "
            "max_steps != 1536"
        )


    if (
        "num_train_epochs" in state
        and abs(
            float(
                state[
                    "num_train_epochs"
                ]
            )
            - 3.0
        ) > 1e-9
    ):
        raise RuntimeError(
            f"{directory.name}: "
            "num_train_epochs != 3"
        )


    history = state.get(
        "log_history",
        []
    )

    if not isinstance(
        history,
        list,
    ) or not history:
        raise RuntimeError(
            f"{directory.name}: "
            "missing log_history"
        )


    history_steps = []

    for entry in history:

        if not isinstance(
            entry,
            dict,
        ):
            raise RuntimeError(
                f"{directory.name}: "
                "invalid log_history entry"
            )

        finite_walk(
            entry,
            f"{directory.name}.log_history",
        )

        if "step" in entry:

            history_step = int(
                entry[
                    "step"
                ]
            )

            if (
                history_step < 0
                or history_step > step
            ):
                raise RuntimeError(
                    f"{directory.name}: "
                    f"invalid history step "
                    f"{history_step}"
                )

            history_steps.append(
                history_step
            )


    if history_steps != sorted(
        history_steps
    ):
        raise RuntimeError(
            f"{directory.name}: "
            "log_history steps not monotonic"
        )


    eval_entries = [
        entry
        for entry in history
        if "eval_loss" in entry
    ]


    checkpoints[
        directory.name
    ] = {
        "step":
            step,

        "epoch":
            epoch,

        "expected_epoch":
            expected_epoch,

        "global_step":
            int(
                state[
                    "global_step"
                ]
            ),

        "max_steps":
            state.get(
                "max_steps"
            ),

        "num_train_epochs":
            state.get(
                "num_train_epochs"
            ),

        "adapter_sha256":
            sha256(
                adapter_model
            ),

        "adapter_size_bytes":
            adapter_model.stat().st_size,

        "adapter_config":
            {
                "peft_type":
                    adapter_config.get(
                        "peft_type"
                    ),

                "r":
                    adapter_config.get(
                        "r"
                    ),

                "lora_alpha":
                    adapter_config.get(
                        "lora_alpha"
                    ),

                "lora_dropout":
                    adapter_config.get(
                        "lora_dropout"
                    ),

                "bias":
                    adapter_config.get(
                        "bias"
                    ),

                "target_modules":
                    sorted(
                        adapter_config.get(
                            "target_modules",
                            []
                        )
                    ),

                "base_model_name_or_path":
                    adapter_config.get(
                        "base_model_name_or_path"
                    ),
            },

        "log_history_entries":
            len(
                history
            ),

        "eval_history_entries":
            len(
                eval_entries
            ),

        "state_files_present":
            {
                name:
                    (
                        directory
                        / name
                    ).is_file()

                for name in (
                    "optimizer.pt",
                    "scheduler.pt",
                    "rng_state.pth",
                    "training_args.bin",
                )
            },

        "all_file_hashes":
            file_inventory(
                directory
            ),
    }


# ====================================================================
# FINAL ROOT ADAPTER
# ====================================================================

root_adapter = (
    RUN_DIR
    / "adapter_model.safetensors"
)

root_config_path = (
    RUN_DIR
    / "adapter_config.json"
)


if not root_adapter.is_file():
    raise RuntimeError(
        "Final root adapter_model.safetensors missing"
    )

if not root_config_path.is_file():
    raise RuntimeError(
        "Final root adapter_config.json missing"
    )


root_adapter_sha = sha256(
    root_adapter
)


if (
    root_adapter_sha
    != EXPECTED[
        "final_adapter_sha256"
    ]
):
    raise RuntimeError(
        "Final adapter SHA mismatch\n"
        f"expected="
        f"{EXPECTED['final_adapter_sha256']}\n"
        f"actual={root_adapter_sha}"
    )


root_config = load_json(
    root_config_path
)

validate_adapter_config(
    root_config,
    "final-root-adapter",
)


cp1536_sha = checkpoints[
    "checkpoint-1536"
][
    "adapter_sha256"
]


if root_adapter_sha != cp1536_sha:
    raise RuntimeError(
        "Final root adapter does not match "
        "checkpoint-1536 adapter bytes."
    )


cp1536_config = load_json(
    RUN_DIR
    / "checkpoint-1536"
    / "adapter_config.json"
)


if root_config != cp1536_config:
    raise RuntimeError(
        "Final root adapter_config differs "
        "from checkpoint-1536 config."
    )


# ====================================================================
# ROOT OUTPUT INVENTORY
# ====================================================================

root_files = {
    p.name:
        {
            "size_bytes":
                p.stat().st_size,

            "sha256":
                sha256(
                    p
                ),
        }

    for p in sorted(
        RUN_DIR.iterdir()
    )
    if p.is_file()
}


# ====================================================================
# TRAINER STATE SUMMARY / EXPECTED GEOMETRY
# ====================================================================

geometry = {
    "epochs":
        3,

    "effective_batch":
        8,

    "steps_per_epoch":
        512,

    "total_optimizer_steps":
        1536,

    "warmup_steps":
        77,

    "checkpoint_steps":
        [
            512,
            1024,
            1536,
        ],
}


# ====================================================================
# REPORT
# ====================================================================

report = {
    "status":
        "PASS",

    "experiment":
        "Sentinel V6",

    "run":
        "Run 01",

    "recipe":
        "R2",

    "frozen_parent_hashes":
        parent_hashes,

    "training_geometry":
        geometry,

    "observed_checkpoint_steps":
        observed_steps,

    "checkpoints":
        checkpoints,

    "final_adapter": {
        "path":
            str(
                root_adapter.relative_to(
                    BASE
                )
            ),

        "sha256":
            root_adapter_sha,

        "matches_checkpoint_1536":
            True,

        "adapter_config_matches_checkpoint_1536":
            True,

        "size_bytes":
            root_adapter.stat().st_size,
    },

    "run_root_files":
        root_files,

    "training_log_completion_evidence":
        required_log_strings,

    "scientific_interpretation": {
        "training_process_completed":
            True,

        "checkpoint_selection_performed":
            False,

        "development_inference_performed_by_this_audit":
            False,

        "teacher_forced_training_metrics_used_for_selection":
            False,

        "challenge_v4_consumed":
            False,
    },

    "safety": {
        "model_loads":
            0,

        "checkpoint_deserialization":
            0,

        "inference_calls":
            0,

        "development_rows_consumed":
            0,

        "challenge_v4_files_read":
            0,

        "training_output_mutations":
            0,
    },
}


# ====================================================================
# WRITE ONLY AFTER COMPLETE VALIDATION
# ====================================================================

if OUT_DIR.exists():
    raise RuntimeError(
        "Audit output directory already exists."
    )


OUT_DIR.mkdir(
    parents=False,
    exist_ok=False,
)


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
    "SENTINEL V6 — RUN 01 / R2 TRAINING ARTIFACT AUDIT",
    "=" * 88,
    "",
    "STATUS: PASS",
    "",
    "TRAINING",
    "-" * 88,
    "Run 01 / Recipe R2 completed successfully.",
    "Expected optimizer horizon: 1536",
    "Observed checkpoints: 512 / 1024 / 1536",
    "",
    "CHECKPOINTS",
    "-" * 88,
]


for name in (
    "checkpoint-512",
    "checkpoint-1024",
    "checkpoint-1536",
):

    c = checkpoints[
        name
    ]

    lines.extend([
        f"{name}",
        f"  global_step: {c['global_step']}",
        f"  epoch: {c['epoch']}",
        f"  adapter SHA256: {c['adapter_sha256']}",
        f"  adapter bytes: {c['adapter_size_bytes']}",
        f"  log history entries: {c['log_history_entries']}",
        f"  eval history entries: {c['eval_history_entries']}",
        f"  state files: {c['state_files_present']}",
        "",
    ])


lines.extend([
    "FINAL ADAPTER",
    "-" * 88,
    f"SHA256: {root_adapter_sha}",
    "Matches checkpoint-1536 adapter bytes: YES",
    "Matches checkpoint-1536 adapter config: YES",
    "",
    "SCIENTIFIC STATE",
    "-" * 88,
    "Checkpoint selected: NO",
    "DEVELOPMENT evaluation performed by this audit: NO",
    "Teacher-forced loss used for checkpoint selection: NO",
    "Challenge V4 consumed: NO",
    "",
    "SAFETY",
    "-" * 88,
    "Model loads: 0",
    "Checkpoint deserialization: 0",
    "Inference calls: 0",
    "DEVELOPMENT rows consumed: 0",
    "Challenge V4 files read: 0",
    "Training output mutations: 0",
    "",
    "NEXT",
    "-" * 88,
    "Freeze this audit evidence.",
    "Then evaluate ALL THREE checkpoints on frozen DEVELOPMENT",
    "using the already-frozen evaluator and selection policy.",
])


OUT_TXT.write_text(
    "\n".join(
        lines
    )
    + "\n",
    encoding="utf-8",
)


sum_lines = []


for checkpoint_name in (
    "checkpoint-512",
    "checkpoint-1024",
    "checkpoint-1536",
):

    directory = (
        RUN_DIR
        / checkpoint_name
    )

    for path in sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
    ):

        sum_lines.append(
            f"{sha256(path)}  "
            f"{path.relative_to(BASE)}"
        )


for path in sorted(
    p
    for p in RUN_DIR.iterdir()
    if p.is_file()
):

    sum_lines.append(
        f"{sha256(path)}  "
        f"{path.relative_to(BASE)}"
    )


OUT_SUMS.write_text(
    "\n".join(
        sum_lines
    )
    + "\n",
    encoding="utf-8",
)


print("=" * 88)
print("SENTINEL V6 RUN 01 / R2 TRAINING ARTIFACT AUDIT")
print("=" * 88)

print()
print("STATUS: PASS")

print()
print(
    "Checkpoint set:",
    observed_steps,
)

for name in (
    "checkpoint-512",
    "checkpoint-1024",
    "checkpoint-1536",
):

    c = checkpoints[
        name
    ]

    print()
    print(name)
    print(
        "  global_step:",
        c[
            "global_step"
        ],
    )
    print(
        "  epoch:",
        c[
            "epoch"
        ],
    )
    print(
        "  adapter SHA256:",
        c[
            "adapter_sha256"
        ],
    )
    print(
        "  state files:",
        c[
            "state_files_present"
        ],
    )


print()
print(
    "Final adapter SHA256:",
    root_adapter_sha,
)

print(
    "Final adapter == checkpoint-1536:",
    True,
)

print()
print("Model loads: 0")
print("Checkpoint deserialization: 0")
print("Inference: 0")
print("DEVELOPMENT rows consumed: 0")
print("Challenge V4 files read: 0")
print("Training output mutations: 0")
