from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import statistics


from transformers import AutoTokenizer


BASE = Path("/workspace")

MODEL_NAME = (
    "meta-llama/Llama-3.1-8B-Instruct"
)

V5_TRAIN = (
    BASE
    / "training/v5/datasets/"
      "sentinel_v5_train.jsonl"
)

V5_DEV = (
    BASE
    / "training/v5/datasets/"
      "sentinel_v5_development.jsonl"
)

V6_TRAIN = (
    BASE
    / "training/v6/frozen/"
      "sentinel_v6_train.jsonl"
)

V6_DEV = (
    BASE
    / "training/v6/frozen/"
      "sentinel_v6_development.jsonl"
)

OUT_JSON = (
    BASE
    / "experiments/v6/"
      "V6_TRAINING_RECIPE_COMPATIBILITY_PREFLIGHT.json"
)

OUT_TXT = (
    BASE
    / "experiments/v6/"
      "V6_TRAINING_RECIPE_COMPATIBILITY_PREFLIGHT.txt"
)


EXPECTED = {
    "v5_train":
        "945eef0841648911aaee5b88a5f7efd6c84bcad7399697f7f786f4774645d972",

    "v5_dev":
        "4e9fe2effa908d8994fdfb3be486661f9ef178860725fbd887d60b5181900124",

    "v6_train":
        "9dae655425b391708beb26b01f72bc8f36cd2353c6092d8431307dacaeb6a9ff",

    "v6_dev":
        "23163c4decdf3a4af4df85811b63538ac469582241b96a4f153c30c2725d3007",
}


MAX_LENGTH = 768

EPOCHS = 3

BATCH = 4

GRAD_ACCUM = 2

WORLD_SIZE = 1

EFFECTIVE_BATCH = (
    BATCH
    * GRAD_ACCUM
    * WORLD_SIZE
)


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def percentile(
    values,
    q,
):
    ordered = sorted(
        values
    )

    if not ordered:
        return None

    index = (
        len(ordered) - 1
    ) * q

    lower = math.floor(
        index
    )

    upper = math.ceil(
        index
    )

    if lower == upper:
        return ordered[
            lower
        ]

    fraction = (
        index
        - lower
    )

    return (
        ordered[lower]
        * (
            1 - fraction
        )
        +
        ordered[upper]
        * fraction
    )


def render_lengths(
    tokenizer,
    row,
):
    prompt_messages = [
        {
            "role":
                "system",

            "content":
                row[
                    "instruction"
                ],
        },
        {
            "role":
                "user",

            "content":
                row[
                    "input"
                ],
        },
    ]

    full_messages = (
        prompt_messages
        + [
            {
                "role":
                    "assistant",

                "content":
                    row[
                        "output"
                    ],
            }
        ]
    )


    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )


    # For the Llama conversational template, the generation prompt
    # should be a prefix of the full assistant-containing sequence.
    prefix_match = (
        list(
            full_ids[
                :len(
                    prompt_ids
                )
            ]
        )
        ==
        list(
            prompt_ids
        )
    )


    prompt_length = len(
        prompt_ids
    )

    total_length = len(
        full_ids
    )

    completion_length = max(
        0,
        total_length
        - prompt_length
    )


    retained_completion = max(
        0,
        MAX_LENGTH
        - prompt_length
    )

    completion_truncated = (
        total_length > MAX_LENGTH
        and
        retained_completion
        <
        completion_length
    )


    return {
        "prompt":
            prompt_length,

        "completion":
            completion_length,

        "total":
            total_length,

        "prefix_match":
            prefix_match,

        "over_limit":
            total_length
            > MAX_LENGTH,

        "prompt_at_or_over_limit":
            prompt_length
            >= MAX_LENGTH,

        "completion_truncated":
            completion_truncated,
    }


def summarize(
    name,
    rows,
    tokenizer,
):
    lengths = [
        render_lengths(
            tokenizer,
            row,
        )
        for row in rows
    ]


    totals = [
        item[
            "total"
        ]
        for item in lengths
    ]

    prompts = [
        item[
            "prompt"
        ]
        for item in lengths
    ]

    completions = [
        item[
            "completion"
        ]
        for item in lengths
    ]


    return {
        "name":
            name,

        "rows":
            len(
                rows
            ),

        "max_length_setting":
            MAX_LENGTH,

        "total_tokens": {
            "min":
                min(
                    totals
                ),

            "median":
                statistics.median(
                    totals
                ),

            "p95":
                percentile(
                    totals,
                    0.95,
                ),

            "p99":
                percentile(
                    totals,
                    0.99,
                ),

            "max":
                max(
                    totals
                ),
        },

        "prompt_tokens": {
            "min":
                min(
                    prompts
                ),

            "median":
                statistics.median(
                    prompts
                ),

            "p95":
                percentile(
                    prompts,
                    0.95,
                ),

            "p99":
                percentile(
                    prompts,
                    0.99,
                ),

            "max":
                max(
                    prompts
                ),
        },

        "completion_tokens": {
            "min":
                min(
                    completions
                ),

            "median":
                statistics.median(
                    completions
                ),

            "p95":
                percentile(
                    completions,
                    0.95,
                ),

            "p99":
                percentile(
                    completions,
                    0.99,
                ),

            "max":
                max(
                    completions
                ),
        },

        "rows_over_768":
            sum(
                item[
                    "over_limit"
                ]
                for item in lengths
            ),

        "prompts_at_or_over_768":
            sum(
                item[
                    "prompt_at_or_over_limit"
                ]
                for item in lengths
            ),

        "completion_truncations_at_768":
            sum(
                item[
                    "completion_truncated"
                ]
                for item in lengths
            ),

        "chat_template_prefix_failures":
            sum(
                not item[
                    "prefix_match"
                ]
                for item in lengths
            ),
    }


def package_versions():
    names = (
        "torch",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "accelerate",
    )

    result = {}

    for name in names:
        try:
            result[
                name
            ] = importlib.metadata.version(
                name
            )
        except Exception:
            result[
                name
            ] = None

    return result


def hf_cache_revision():
    roots = [
        Path.home()
        / ".cache/huggingface/hub/"
          "models--meta-llama--Llama-3.1-8B-Instruct",

        BASE
        / ".cache/huggingface/hub/"
          "models--meta-llama--Llama-3.1-8B-Instruct",
    ]


    for root in roots:
        ref = (
            root
            / "refs/main"
        )

        if ref.is_file():
            value = ref.read_text(
                encoding="utf-8"
            ).strip()

            if value:
                return {
                    "cache_root":
                        str(
                            root
                        ),

                    "main_revision":
                        value,
                }


    return {
        "cache_root":
            None,

        "main_revision":
            None,
    }


def main():

    paths = {
        "v5_train":
            V5_TRAIN,

        "v5_dev":
            V5_DEV,

        "v6_train":
            V6_TRAIN,

        "v6_dev":
            V6_DEV,
    }


    for name, path in paths.items():
        actual = sha(
            path
        )

        if actual != EXPECTED[
            name
        ]:
            raise RuntimeError(
                f"{name} SHA mismatch"
            )


    print(
        "Loading tokenizer only..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        use_fast=True,
    )


    datasets = {
        "v5_train":
            load_jsonl(
                V5_TRAIN
            ),

        "v5_development":
            load_jsonl(
                V5_DEV
            ),

        "v6_train":
            load_jsonl(
                V6_TRAIN
            ),

        "v6_development":
            load_jsonl(
                V6_DEV
            ),
    }


    summaries = {
        name:
            summarize(
                name,
                rows,
                tokenizer,
            )

        for name, rows
        in datasets.items()
    }


    v5_train_rows = len(
        datasets[
            "v5_train"
        ]
    )

    v6_train_rows = len(
        datasets[
            "v6_train"
        ]
    )


    v5_micro_batches = math.ceil(
        v5_train_rows
        / BATCH
    )

    v5_steps_per_epoch = math.ceil(
        v5_micro_batches
        / GRAD_ACCUM
    )

    v5_total_steps = (
        v5_steps_per_epoch
        * EPOCHS
    )

    v5_warmup = max(
        1,
        round(
            v5_total_steps
            * 0.05
        ),
    )


    v6_micro_batches = math.ceil(
        v6_train_rows
        / BATCH
    )

    v6_steps_per_epoch = math.ceil(
        v6_micro_batches
        / GRAD_ACCUM
    )

    v6_total_steps = (
        v6_steps_per_epoch
        * EPOCHS
    )

    v6_warmup = max(
        1,
        round(
            v6_total_steps
            * 0.05
        ),
    )


    v6_truncation_free = (
        summaries[
            "v6_train"
        ][
            "rows_over_768"
        ]
        == 0
        and
        summaries[
            "v6_development"
        ][
            "rows_over_768"
        ]
        == 0
    )


    if v6_truncation_free:
        status = (
            "V5_MAX_LENGTH_768_COMPATIBLE"
        )

        recommendation = (
            "Preserve max_length=768."
        )

    else:
        status = (
            "MAX_LENGTH_CHANGE_REQUIRES_ADJUDICATION"
        )

        recommendation = (
            "Do not freeze V6 max_length yet."
        )


    report = {
        "status":
            status,

        "base_model":
            MODEL_NAME,

        "tokenizer_only_load":
            True,

        "dataset_summaries":
            summaries,

        "geometry": {
            "epochs":
                EPOCHS,

            "per_device_batch":
                BATCH,

            "gradient_accumulation":
                GRAD_ACCUM,

            "world_size":
                WORLD_SIZE,

            "effective_batch":
                EFFECTIVE_BATCH,

            "v5_steps_per_epoch":
                v5_steps_per_epoch,

            "v5_total_steps":
                v5_total_steps,

            "v5_warmup_steps":
                v5_warmup,

            "v6_steps_per_epoch":
                v6_steps_per_epoch,

            "v6_total_steps":
                v6_total_steps,

            "v6_warmup_steps":
                v6_warmup,

            "v6_epoch_checkpoints": [
                v6_steps_per_epoch,
                v6_steps_per_epoch
                * 2,
                v6_steps_per_epoch
                * 3,
            ],
        },

        "checkpoint_selection_policy": {
            "eligibility_gates": {
                "decision_accuracy":
                    0.95,

                "worst_label_accuracy":
                    0.90,

                "worst_operator_accuracy":
                    0.90,

                "worst_family_accuracy":
                    0.80,

                "strict_json_validity":
                    0.99,

                "citation_reference_validity":
                    0.99,

                "exact_citation_set_accuracy":
                    0.90,
            },

            "ranking_order": [
                "decision_accuracy DESC",
                "exact_citation_set_accuracy DESC",
                "worst_family_accuracy DESC",
                "strict_json_validity DESC",
                "epoch ASC",
            ],

            "teacher_forced_eval_loss_used":
                False,
        },

        "proposed_preserved_recipe": {
            "learning_rate":
                0.0001,

            "epochs":
                3,

            "batch_size":
                4,

            "gradient_accumulation":
                2,

            "max_length":
                (
                    768
                    if v6_truncation_free
                    else None
                ),

            "lr_scheduler_type":
                "cosine",

            "warmup_fraction":
                0.05,

            "warmup_steps_v6":
                v6_warmup,

            "optimizer":
                "adamw_torch",

            "bf16":
                True,

            "fp16":
                False,

            "gradient_checkpointing":
                True,

            "completion_only_loss":
                True,

            "packing":
                False,

            "eval_strategy":
                "epoch",

            "save_strategy":
                "epoch",

            "load_best_model_at_end":
                False,

            "logging_steps":
                5,

            "save_total_limit":
                4,

            "seed":
                42,
        },

        "environment": {
            "package_versions":
                package_versions(),

            "hf_cache":
                hf_cache_revision(),
        },

        "recommendation":
            recommendation,

        "safety": {
            "model_loads":
                0,

            "model_inference":
                0,

            "checkpoint_deserialization":
                0,

            "tokenizer_loads":
                1,

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


    lines = [
        "SENTINEL V6 — TRAINING RECIPE COMPATIBILITY PREFLIGHT",
        "=" * 80,
        "",
        f"STATUS: {status}",
        "",
    ]


    for name in (
        "v5_train",
        "v5_development",
        "v6_train",
        "v6_development",
    ):
        item = summaries[
            name
        ]

        lines.extend([
            name.upper(),
            "-" * 80,
            f"rows: {item['rows']}",
            f"total max: {item['total_tokens']['max']}",
            f"total p95: {item['total_tokens']['p95']}",
            f"total p99: {item['total_tokens']['p99']}",
            f"prompt max: {item['prompt_tokens']['max']}",
            f"completion max: {item['completion_tokens']['max']}",
            f"rows > 768: {item['rows_over_768']}",
            f"prompts >= 768: {item['prompts_at_or_over_768']}",
            (
                "completion truncations @ 768: "
                f"{item['completion_truncations_at_768']}"
            ),
            (
                "chat-template prefix failures: "
                f"{item['chat_template_prefix_failures']}"
            ),
            "",
        ])


    lines.extend([
        "STEP GEOMETRY",
        "-" * 80,
        f"effective batch: {EFFECTIVE_BATCH}",
        f"V5 steps/epoch: {v5_steps_per_epoch}",
        f"V5 total steps: {v5_total_steps}",
        f"V5 warmup: {v5_warmup}",
        f"V6 steps/epoch: {v6_steps_per_epoch}",
        f"V6 total steps: {v6_total_steps}",
        f"V6 warmup: {v6_warmup}",
        (
            "V6 checkpoints: "
            f"{v6_steps_per_epoch}, "
            f"{v6_steps_per_epoch * 2}, "
            f"{v6_steps_per_epoch * 3}"
        ),
        "",
        "RECOMMENDATION",
        "-" * 80,
        recommendation,
        "",
        "SAFETY",
        "-" * 80,
        "Tokenizer loads: 1",
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


    print()
    print("=" * 80)
    print("V6 TRAINING RECIPE COMPATIBILITY PREFLIGHT")
    print("=" * 80)

    print()
    print(
        "STATUS:",
        status,
    )


    for name in (
        "v5_train",
        "v5_development",
        "v6_train",
        "v6_development",
    ):
        item = summaries[
            name
        ]

        print()
        print(name.upper())
        print("-" * 80)

        print(
            "rows:",
            item[
                "rows"
            ],
        )

        print(
            "total tokens:",
            "p95=",
            item[
                "total_tokens"
            ][
                "p95"
            ],
            "p99=",
            item[
                "total_tokens"
            ][
                "p99"
            ],
            "max=",
            item[
                "total_tokens"
            ][
                "max"
            ],
        )

        print(
            "prompt max:",
            item[
                "prompt_tokens"
            ][
                "max"
            ],
        )

        print(
            "completion max:",
            item[
                "completion_tokens"
            ][
                "max"
            ],
        )

        print(
            "rows > 768:",
            item[
                "rows_over_768"
            ],
        )

        print(
            "completion truncations @ 768:",
            item[
                "completion_truncations_at_768"
            ],
        )

        print(
            "chat-template prefix failures:",
            item[
                "chat_template_prefix_failures"
            ],
        )


    print()
    print("GEOMETRY")
    print("-" * 80)

    print(
        "effective batch:",
        EFFECTIVE_BATCH,
    )

    print(
        "V5:",
        v5_steps_per_epoch,
        "steps/epoch |",
        v5_total_steps,
        "total | warmup",
        v5_warmup,
    )

    print(
        "V6:",
        v6_steps_per_epoch,
        "steps/epoch |",
        v6_total_steps,
        "total | warmup",
        v6_warmup,
    )

    print(
        "V6 checkpoints:",
        v6_steps_per_epoch,
        v6_steps_per_epoch * 2,
        v6_steps_per_epoch * 3,
    )


    print()
    print("SELECTION POLICY")
    print("-" * 80)

    print(
        "gates:",
        report[
            "checkpoint_selection_policy"
        ][
            "eligibility_gates"
        ],
    )

    print(
        "ranking:",
        " -> ".join(
            report[
                "checkpoint_selection_policy"
            ][
                "ranking_order"
            ]
        ),
    )

    print(
        "teacher-forced eval_loss used:",
        False,
    )


    print()
    print("ENVIRONMENT")
    print("-" * 80)

    print(
        json.dumps(
            report[
                "environment"
            ],
            indent=2,
            sort_keys=True,
        )
    )


    print()
    print("RECOMMENDATION:")
    print(
        recommendation
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
    print("Tokenizer loads: 1")
    print("Model loads: 0")
    print("Inference: 0")
    print("Checkpoint deserialization: 0")
    print("Challenge V4 files read: 0")
    print("Training started: NO")


if __name__ == "__main__":
    main()
