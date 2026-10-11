from __future__ import annotations

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

MAX_LENGTH = 768

EPOCHS = 3

BATCH_SIZE = 4

GRAD_ACCUM = 2

WORLD_SIZE = 1

EFFECTIVE_BATCH = (
    BATCH_SIZE
    * GRAD_ACCUM
    * WORLD_SIZE
)


DATASETS = {
    "v5_train": (
        BASE
        / "training/v5/datasets/"
          "sentinel_v5_train.jsonl"
    ),

    "v5_development": (
        BASE
        / "training/v5/datasets/"
          "sentinel_v5_development.jsonl"
    ),

    "v6_train": (
        BASE
        / "training/v6/frozen/"
          "sentinel_v6_train.jsonl"
    ),

    "v6_development": (
        BASE
        / "training/v6/frozen/"
          "sentinel_v6_development.jsonl"
    ),
}


EXPECTED_SHA = {
    "v5_train":
        "945eef0841648911aaee5b88a5f7efd6c84bcad7399697f7f786f4774645d972",

    "v5_development":
        "4e9fe2effa908d8994fdfb3be486661f9ef178860725fbd887d60b5181900124",

    "v6_train":
        "9dae655425b391708beb26b01f72bc8f36cd2353c6092d8431307dacaeb6a9ff",

    "v6_development":
        "23163c4decdf3a4af4df85811b63538ac469582241b96a4f153c30c2725d3007",
}


OUT_JSON = (
    BASE
    / "experiments/v6/"
      "V6_TRAINING_RECIPE_COMPATIBILITY_PREFLIGHT_V2.json"
)

OUT_TXT = (
    BASE
    / "experiments/v6/"
      "V6_TRAINING_RECIPE_COMPATIBILITY_PREFLIGHT_V2.txt"
)


def sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_jsonl(path: Path):
    return [
        json.loads(
            line
        )
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def percentile(
    values,
    q,
):
    values = sorted(
        values
    )

    if not values:
        return None

    index = (
        len(values) - 1
    ) * q

    lower = math.floor(
        index
    )

    upper = math.ceil(
        index
    )

    if lower == upper:
        return values[
            lower
        ]

    fraction = (
        index
        - lower
    )

    return (
        values[lower]
        * (
            1.0 - fraction
        )
        +
        values[upper]
        * fraction
    )


def tokenize_rendered(
    tokenizer,
    text: str,
):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    ids = encoded[
        "input_ids"
    ]


    # Defensive handling if a tokenizer ever returns
    # one nested sequence for a single string.
    if (
        isinstance(
            ids,
            list,
        )
        and ids
        and isinstance(
            ids[0],
            list,
        )
    ):
        if len(
            ids
        ) != 1:
            raise RuntimeError(
                "Unexpected batched tokenization."
            )

        ids = ids[
            0
        ]


    if not isinstance(
        ids,
        list,
    ):
        ids = list(
            ids
        )


    if (
        ids
        and not isinstance(
            ids[0],
            int,
        )
    ):
        raise RuntimeError(
            "input_ids did not resolve to integer token IDs."
        )


    return ids


def analyze_row(
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


    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )


    if not isinstance(
        prompt_text,
        str,
    ):
        raise RuntimeError(
            "Prompt template did not render text."
        )

    if not isinstance(
        full_text,
        str,
    ):
        raise RuntimeError(
            "Full template did not render text."
        )


    prompt_ids = tokenize_rendered(
        tokenizer,
        prompt_text,
    )

    full_ids = tokenize_rendered(
        tokenizer,
        full_text,
    )


    prompt_len = len(
        prompt_ids
    )

    total_len = len(
        full_ids
    )


    # String-prefix validity is more robust than demanding
    # token-prefix equality at a BPE boundary.
    string_prefix_match = full_text.startswith(
        prompt_text
    )


    token_prefix_match = (
        full_ids[
            :prompt_len
        ]
        ==
        prompt_ids
    )


    completion_delta = max(
        0,
        total_len
        - prompt_len
    )


    return {
        "prompt_tokens":
            prompt_len,

        "full_tokens":
            total_len,

        "completion_delta_tokens":
            completion_delta,

        "over_max_length":
            total_len
            > MAX_LENGTH,

        "prompt_at_or_over_max":
            prompt_len
            >= MAX_LENGTH,

        "string_prefix_match":
            string_prefix_match,

        "token_prefix_match":
            token_prefix_match,
    }


def summarize_dataset(
    tokenizer,
    name,
    rows,
):
    analyses = [
        analyze_row(
            tokenizer,
            row,
        )
        for row in rows
    ]


    totals = [
        x[
            "full_tokens"
        ]
        for x in analyses
    ]

    prompts = [
        x[
            "prompt_tokens"
        ]
        for x in analyses
    ]

    completions = [
        x[
            "completion_delta_tokens"
        ]
        for x in analyses
    ]


    return {
        "name":
            name,

        "rows":
            len(
                rows
            ),

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

        "completion_delta_tokens": {
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

        "rows_over_max_length":
            sum(
                x[
                    "over_max_length"
                ]
                for x in analyses
            ),

        "prompts_at_or_over_max":
            sum(
                x[
                    "prompt_at_or_over_max"
                ]
                for x in analyses
            ),

        "string_prefix_failures":
            sum(
                not x[
                    "string_prefix_match"
                ]
                for x in analyses
            ),

        # Diagnostic only; BPE boundary tokenization can cause
        # this to differ even if the rendered string prefix is valid.
        "token_prefix_failures":
            sum(
                not x[
                    "token_prefix_match"
                ]
                for x in analyses
            ),
    }


def package_versions():
    result = {}

    for name in (
        "torch",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "accelerate",
    ):
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
    root = (
        Path.home()
        / ".cache/huggingface/hub/"
          "models--meta-llama--Llama-3.1-8B-Instruct"
    )

    ref = (
        root
        / "refs/main"
    )

    if not ref.is_file():
        return None

    return ref.read_text(
        encoding="utf-8"
    ).strip()


def main():

    # ------------------------------------------------------------
    # Dataset identity.
    # ------------------------------------------------------------

    loaded = {}

    for name, path in DATASETS.items():

        actual = sha(
            path
        )

        if actual != EXPECTED_SHA[
            name
        ]:
            raise RuntimeError(
                f"{name} SHA mismatch."
            )

        loaded[
            name
        ] = load_jsonl(
            path
        )


    # ------------------------------------------------------------
    # Tokenizer only.
    # ------------------------------------------------------------

    print(
        "Loading tokenizer only..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        use_fast=True,
    )


    print(
        "Tokenizer class:",
        tokenizer.__class__.__name__,
    )

    print(
        "Tokenizer truncation_side:",
        tokenizer.truncation_side,
    )

    print(
        "Tokenizer padding_side:",
        tokenizer.padding_side,
    )


    summaries = {
        name:
            summarize_dataset(
                tokenizer,
                name,
                rows,
            )
        for name, rows
        in loaded.items()
    }


    # ------------------------------------------------------------
    # Hard sanity checks.
    #
    # These specifically prevent recurrence of the V1 defect.
    # ------------------------------------------------------------

    for name, summary in summaries.items():

        if summary[
            "total_tokens"
        ][
            "max"
        ] <= 2:
            raise RuntimeError(
                f"{name}: impossible token lengths; "
                "BatchEncoding defect may remain."
            )


        if summary[
            "total_tokens"
        ][
            "median"
        ] <= 2:
            raise RuntimeError(
                f"{name}: impossible median token length."
            )


        if summary[
            "string_prefix_failures"
        ] != 0:
            raise RuntimeError(
                f"{name}: rendered chat-template "
                "string-prefix failures."
            )


    # ------------------------------------------------------------
    # Training geometry.
    # ------------------------------------------------------------

    v5_train_rows = len(
        loaded[
            "v5_train"
        ]
    )

    v6_train_rows = len(
        loaded[
            "v6_train"
        ]
    )


    v5_steps_epoch = math.ceil(
        math.ceil(
            v5_train_rows
            / BATCH_SIZE
        )
        / GRAD_ACCUM
    )

    v6_steps_epoch = math.ceil(
        math.ceil(
            v6_train_rows
            / BATCH_SIZE
        )
        / GRAD_ACCUM
    )


    v5_total_steps = (
        v5_steps_epoch
        * EPOCHS
    )

    v6_total_steps = (
        v6_steps_epoch
        * EPOCHS
    )


    v5_warmup = max(
        1,
        round(
            v5_total_steps
            * 0.05
        ),
    )

    v6_warmup = max(
        1,
        round(
            v6_total_steps
            * 0.05
        ),
    )


    assert (
        v5_steps_epoch
        == 768
    )

    assert (
        v5_total_steps
        == 2304
    )

    assert (
        v5_warmup
        == 115
    )

    assert (
        v6_steps_epoch
        == 512
    )

    assert (
        v6_total_steps
        == 1536
    )

    assert (
        v6_warmup
        == 77
    )


    # ------------------------------------------------------------
    # Compatibility decision.
    # ------------------------------------------------------------

    v6_train_over = summaries[
        "v6_train"
    ][
        "rows_over_max_length"
    ]

    v6_dev_over = summaries[
        "v6_development"
    ][
        "rows_over_max_length"
    ]


    if (
        v6_train_over
        == 0
        and
        v6_dev_over
        == 0
    ):
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
            "Do not freeze max_length. "
            "V6 contains sequences above 768 tokens."
        )


    report = {
        "audit_version":
            2,

        "status":
            status,

        "v1_status":
            "INVALID_AUDIT_IMPLEMENTATION",

        "v1_defect":
            (
                "BatchEncoding length was mistaken "
                "for token-sequence length."
            ),

        "base_model":
            MODEL_NAME,

        "max_length_under_test":
            MAX_LENGTH,

        "dataset_summaries":
            summaries,

        "geometry": {
            "epochs":
                EPOCHS,

            "per_device_batch":
                BATCH_SIZE,

            "gradient_accumulation":
                GRAD_ACCUM,

            "world_size":
                WORLD_SIZE,

            "effective_batch":
                EFFECTIVE_BATCH,

            "v5_steps_per_epoch":
                v5_steps_epoch,

            "v5_total_steps":
                v5_total_steps,

            "v5_warmup_steps":
                v5_warmup,

            "v6_steps_per_epoch":
                v6_steps_epoch,

            "v6_total_steps":
                v6_total_steps,

            "v6_warmup_steps":
                v6_warmup,

            "v6_checkpoints": [
                512,
                1024,
                1536,
            ],
        },

        "selection_policy": {
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

            "ranking": [
                "decision_accuracy DESC",
                "exact_citation_set_accuracy DESC",
                "worst_family_accuracy DESC",
                "strict_json_validity DESC",
                "epoch ASC",
            ],

            "teacher_forced_eval_loss_used":
                False,
        },

        "environment": {
            "packages":
                package_versions(),

            "hf_main_revision":
                hf_cache_revision(),

            "tokenizer_class":
                tokenizer.__class__.__name__,

            "tokenizer_truncation_side":
                tokenizer.truncation_side,

            "tokenizer_padding_side":
                tokenizer.padding_side,
        },

        "recommendation":
            recommendation,

        "safety": {
            "tokenizer_loads":
                1,

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


    lines = [
        "SENTINEL V6 — TRAINING RECIPE COMPATIBILITY PREFLIGHT V2",
        "=" * 80,
        "",
        f"STATUS: {status}",
        "",
        "V1:",
        "INVALID_AUDIT_IMPLEMENTATION",
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
            (
                "total tokens: "
                f"min={item['total_tokens']['min']} "
                f"median={item['total_tokens']['median']} "
                f"p95={item['total_tokens']['p95']} "
                f"p99={item['total_tokens']['p99']} "
                f"max={item['total_tokens']['max']}"
            ),
            (
                "prompt tokens: "
                f"median={item['prompt_tokens']['median']} "
                f"p95={item['prompt_tokens']['p95']} "
                f"max={item['prompt_tokens']['max']}"
            ),
            (
                "completion delta: "
                f"median={item['completion_delta_tokens']['median']} "
                f"p95={item['completion_delta_tokens']['p95']} "
                f"max={item['completion_delta_tokens']['max']}"
            ),
            (
                "rows over 768: "
                f"{item['rows_over_max_length']}"
            ),
            (
                "prompts >= 768: "
                f"{item['prompts_at_or_over_max']}"
            ),
            (
                "string-prefix failures: "
                f"{item['string_prefix_failures']}"
            ),
            (
                "token-prefix failures (diagnostic): "
                f"{item['token_prefix_failures']}"
            ),
            "",
        ])


    lines.extend([
        "GEOMETRY",
        "-" * 80,
        f"effective batch: {EFFECTIVE_BATCH}",
        (
            f"V5: {v5_steps_epoch} steps/epoch | "
            f"{v5_total_steps} total | "
            f"{v5_warmup} warmup"
        ),
        (
            f"V6: {v6_steps_epoch} steps/epoch | "
            f"{v6_total_steps} total | "
            f"{v6_warmup} warmup"
        ),
        "V6 checkpoints: 512 / 1024 / 1536",
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
    print(
        "V6 TRAINING RECIPE COMPATIBILITY PREFLIGHT V2"
    )
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
        print(
            name.upper()
        )

        print(
            "-" * 80
        )

        print(
            "rows:",
            item[
                "rows"
            ],
        )

        print(
            "total:",
            "min",
            item[
                "total_tokens"
            ][
                "min"
            ],
            "| median",
            item[
                "total_tokens"
            ][
                "median"
            ],
            "| p95",
            item[
                "total_tokens"
            ][
                "p95"
            ],
            "| p99",
            item[
                "total_tokens"
            ][
                "p99"
            ],
            "| max",
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
            "completion delta max:",
            item[
                "completion_delta_tokens"
            ][
                "max"
            ],
        )

        print(
            "rows > 768:",
            item[
                "rows_over_max_length"
            ],
        )

        print(
            "prompts >= 768:",
            item[
                "prompts_at_or_over_max"
            ],
        )

        print(
            "string-prefix failures:",
            item[
                "string_prefix_failures"
            ],
        )

        print(
            "token-prefix failures (diagnostic):",
            item[
                "token_prefix_failures"
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
        v5_steps_epoch,
        "steps/epoch |",
        v5_total_steps,
        "total | warmup",
        v5_warmup,
    )

    print(
        "V6:",
        v6_steps_epoch,
        "steps/epoch |",
        v6_total_steps,
        "total | warmup",
        v6_warmup,
    )

    print(
        "V6 checkpoints:",
        "512 / 1024 / 1536",
    )


    print()
    print(
        "RECOMMENDATION:",
        recommendation,
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
