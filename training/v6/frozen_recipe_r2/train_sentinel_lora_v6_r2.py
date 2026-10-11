'\nSentinel V6 fresh Llama-3.1-8B LoRA trainer.\n\nScientific constraints\n----------------------\n\n- Base model is always meta-llama/Llama-3.1-8B-Instruct.\n- LoRA adapter is initialized FRESH with get_peft_model().\n- No prior Sentinel adapter is loaded.\n- Gradients use V6 TRAIN only.\n- V6 DEVELOPMENT is evaluation-only.\n- Challenge V2 is never read by this script.\n- Frozen dataset SHA256 values are verified before model loading.\n- Completion-only loss is used.\n- Checkpoints are NOT selected by eval_loss inside the trainer.\n  Development task metrics will be used later for checkpoint selection.\n'

import argparse
from collections import Counter
from pathlib import Path

import hashlib
import json
import math
import os
import platform
import sys

import torch
import transformers
import datasets
import peft
import trl

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from trl import (
    SFTConfig,
    SFTTrainer,
)


BASE = Path("/workspace")

MODEL_NAME = (
    "meta-llama/Llama-3.1-8B-Instruct"
)

TRAIN_FILE = (
    BASE / 'training/v6/frozen/sentinel_v6_train.jsonl'
)

DEV_FILE = (
    BASE / 'training/v6/frozen/sentinel_v6_development.jsonl'
)


EXPECTED_TRAIN_SHA = (
    "9dae655425b391708beb26b01f72bc8f"
    "36cd2353c6092d8431307dacaeb6a9ff"
)

EXPECTED_DEV_SHA = (
    "23163c4decdf3a4af4df85811b63538a"
    "c469582241b96a4f153c30c2725d3007"
)


EXPECTED_TRAIN_ROWS = 4096
EXPECTED_DEV_ROWS = 1024

LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)


DEFAULT_OUTPUT = (
    BASE / 'training/output/llama-sentinel-8b-lora-v6-run01'
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=768,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help=(
            "Positive value overrides epoch-derived step count. "
            "Use only for smoke testing."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a short training-only GPU smoke test. "
            "No development evaluation or epoch checkpointing."
        ),
    )

    return parser.parse_args()


def sha256(
    path,
):

    h = hashlib.sha256()

    with Path(
        path
    ).open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def count_jsonl(
    path,
):

    count = 0

    labels = Counter()

    with Path(
        path
    ).open() as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(
                line
            )

            count += 1

            labels[
                row[
                    "target"
                ]
            ] += 1

    return (
        count,
        labels,
    )


def convert_example(
    example,
):
    """
    TRL conversational prompt-completion representation.

    completion_only_loss=True means the model is optimized only on
    assistant output tokens rather than system/user prompt tokens.
    """

    prompt = [

        {
            "role":
                "system",

            "content":
                example[
                    "instruction"
                ],
        },

        {
            "role":
                "user",

            "content":
                example[
                    "input"
                ],
        },
    ]


    completion = [

        {
            "role":
                "assistant",

            "content":
                example[
                    "output"
                ],
        }
    ]


    return {
        "prompt":
            prompt,

        "completion":
            completion,
    }


def assert_clean_output_dir(
    output_dir,
):

    path = Path(
        output_dir
    )

    if not path.exists():
        return

    contents = list(
        path.iterdir()
    )

    if contents:

        raise RuntimeError(
            "Output directory already contains files:\n"
            f"{path}\n"
            "Refusing to overwrite an existing experiment."
        )


def main():

    args = parse_args()

    output_dir = Path(
        args.output_dir
    )


    print("=" * 88)
    print('SENTINEL V6 — FRESH LLAMA-3.1-8B LoRA')
    print("=" * 88)


    # ================================================================
    # MODE / ARGUMENT SAFETY
    # ================================================================

    if args.max_length < 768:

        raise RuntimeError(
            'V6 Run 01 requires max_length >= 768. Frozen token audit observed maximum sequence length 599.'
        )


    if args.smoke:

        if args.max_steps <= 0:

            raise RuntimeError(
                "--smoke requires positive --max-steps."
            )

    else:

        if args.max_steps > 0:

            raise RuntimeError(
                'Positive --max-steps is reserved for --smoke. Full V6 runs should use complete epochs.'
            )


    assert_clean_output_dir(
        output_dir
    )


    # ================================================================
    # FROZEN DATA HASH GATE
    # ================================================================

    print()
    print("=" * 88)
    print("FROZEN DATA HASH GATE")
    print("=" * 88)


    train_sha = sha256(
        TRAIN_FILE
    )

    dev_sha = sha256(
        DEV_FILE
    )

    # Frozen Sentinel V6 dataset identity gate.
    if train_sha != '9dae655425b391708beb26b01f72bc8f36cd2353c6092d8431307dacaeb6a9ff':
        raise RuntimeError('Frozen V6 TRAIN SHA mismatch.')

    if dev_sha != '23163c4decdf3a4af4df85811b63538ac469582241b96a4f153c30c2725d3007':
        raise RuntimeError('Frozen V6 DEVELOPMENT SHA mismatch.')


    print(
        "TRAIN:",
        train_sha,
    )

    print(
        "DEV:  ",
        dev_sha,
    )


    if train_sha != EXPECTED_TRAIN_SHA:

        raise RuntimeError(
            'FROZEN V6 TRAIN SHA256 MISMATCH.'
        )


    if dev_sha != EXPECTED_DEV_SHA:

        raise RuntimeError(
            'FROZEN V6 DEVELOPMENT SHA256 MISMATCH.'
        )


    print(
        'PASS: frozen V6 hashes verified.'
    )


    # ================================================================
    # STRUCTURE GATE
    # ================================================================

    train_rows, train_labels = count_jsonl(
        TRAIN_FILE
    )

    dev_rows, dev_labels = count_jsonl(
        DEV_FILE
    )


    expected_train_labels = Counter({
        label:
            1024
        for label
        in LABELS
    })

    expected_dev_labels = Counter({
        label:
            256
        for label
        in LABELS
    })


    assert train_rows == EXPECTED_TRAIN_ROWS
    assert dev_rows == EXPECTED_DEV_ROWS

    assert train_labels == expected_train_labels
    assert dev_labels == expected_dev_labels


    print()
    print(
        "TRAIN rows:",
        train_rows,
    )

    print(
        "DEVELOPMENT rows:",
        dev_rows,
    )

    print(
        "PASS: row counts and label balance verified."
    )


    # ================================================================
    # HARDWARE
    # ================================================================

    print()
    print("=" * 88)
    print("HARDWARE")
    print("=" * 88)


    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )


    if not torch.cuda.is_available():

        raise RuntimeError(
            'CUDA unavailable. Refusing V6 training.'
        )


    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported(),
    )


    if not torch.cuda.is_bf16_supported():

        raise RuntimeError(
            'BF16 unsupported. Refusing configured V6 run.'
        )


    # ================================================================
    # DATASET
    # ================================================================

    print()
    print("=" * 88)
    print("LOAD DATASET")
    print("=" * 88)


    raw_dataset = load_dataset(
        "json",
        data_files={
            "train":
                str(
                    TRAIN_FILE
                ),

            "development":
                str(
                    DEV_FILE
                ),
        },
    )


    train_columns = (
        raw_dataset[
            "train"
        ].column_names
    )

    dev_columns = (
        raw_dataset[
            "development"
        ].column_names
    )


    dataset = raw_dataset.map(
        convert_example,
        remove_columns={
            "train":
                train_columns,

            "development":
                dev_columns,
        }
        if False
        else None,
    )


    # datasets.DatasetDict.map does not accept split-specific
    # remove_columns mappings across all installed versions.
    # Remove original columns explicitly and reproducibly instead.

    converted = {}

    for split in (
        "train",
        "development",
    ):

        original_columns = (
            raw_dataset[
                split
            ].column_names
        )

        converted[
            split
        ] = raw_dataset[
            split
        ].map(
            convert_example,
            remove_columns=original_columns,
        )


    dataset = datasets.DatasetDict(
        converted
    )


    print(
        "Train examples:",
        len(
            dataset[
                "train"
            ]
        ),
    )

    print(
        "Development examples:",
        len(
            dataset[
                "development"
            ]
        ),
    )


    assert set(
        dataset[
            "train"
        ].column_names
    ) == {
        "prompt",
        "completion",
    }

    assert set(
        dataset[
            "development"
        ].column_names
    ) == {
        "prompt",
        "completion",
    }


    # ================================================================
    # STEP ACCOUNTING
    # ================================================================

    micro_batches_per_epoch = math.ceil(
        len(
            dataset[
                "train"
            ]
        )
        / args.batch_size
    )


    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch
        / args.grad_accum
    )


    if args.max_steps > 0:

        planned_optimizer_steps = (
            args.max_steps
        )

    else:

        planned_optimizer_steps = math.ceil(
            optimizer_steps_per_epoch
            * args.epochs
        )


    warmup_steps = max(
        1,
        round(
            planned_optimizer_steps
            * 0.05
        ),
    )


    effective_batch = (
        args.batch_size
        * args.grad_accum
    )


    print()
    print("=" * 88)
    print("TRAINING PLAN")
    print("=" * 88)


    print(
        "mode:",
        (
            "SMOKE"
            if args.smoke
            else "FULL"
        ),
    )

    print(
        "epochs:",
        args.epochs,
    )

    print(
        "micro batch:",
        args.batch_size,
    )

    print(
        "gradient accumulation:",
        args.grad_accum,
    )

    print(
        "effective batch:",
        effective_batch,
    )

    print(
        "max length:",
        args.max_length,
    )

    print(
        "learning rate:",
        args.learning_rate,
    )

    print(
        "optimizer steps / epoch:",
        optimizer_steps_per_epoch,
    )

    print(
        "planned optimizer steps:",
        planned_optimizer_steps,
    )

    print(
        "warmup steps:",
        warmup_steps,
    )

    print(
        "output:",
        output_dir,
    )


    # ================================================================
    # TOKENIZER
    # ================================================================

    print()
    print(
        "Loading tokenizer..."
    )


    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    tokenizer.padding_side = (
        "right"
    )


    if not tokenizer.chat_template:

        raise RuntimeError(
            "Tokenizer has no chat template."
        )


    # ================================================================
    # BASE MODEL — FRESH
    # ================================================================

    print()
    print("=" * 88)
    print("LOAD FRESH BASE MODEL")
    print("=" * 88)


    print(
        MODEL_NAME
    )


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map={
            "":
                0
        },
    )


    model.config.use_cache = False


    # ================================================================
    # FRESH LORA
    # ================================================================

    print()
    print("=" * 88)
    print("CREATE FRESH LoRA")
    print("=" * 88)


    lora_config = LoraConfig(

        r=8,

        lora_alpha=16,

        lora_dropout=0.0,

        bias="none",

        task_type=TaskType.CAUSAL_LM,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


    model = get_peft_model(
        model,
        lora_config,
    )


    print()
    print(
        "Trainable parameter check:"
    )

    model.print_trainable_parameters()


    # ================================================================
    # OUTPUT / PROVENANCE
    # ================================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


    run_record = {

        "experiment":
            'Sentinel V6',

        "mode":
            (
                "smoke"
                if args.smoke
                else "full"
            ),

        "base_model":
            MODEL_NAME,

        "adapter_initialization":
            "fresh_get_peft_model",

        "train_sha256":
            train_sha,

        "development_sha256":
            dev_sha,

        "train_rows":
            train_rows,

        "development_rows":
            dev_rows,

        "epochs":
            args.epochs,

        "batch_size":
            args.batch_size,

        "gradient_accumulation_steps":
            args.grad_accum,

        "effective_batch_size":
            effective_batch,

        "learning_rate":
            args.learning_rate,

        "max_length":
            args.max_length,

        "max_steps":
            args.max_steps,

        "planned_optimizer_steps":
            planned_optimizer_steps,

        "warmup_steps":
            warmup_steps,

        "lora_r":
            8,

        "lora_alpha":
            16,

        "lora_dropout":
            0.0,

        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],

        "completion_only_loss":
            True,

        "packing":
            False,

        "seed":
            args.seed,

        "torch":
            torch.__version__,

        "transformers":
            transformers.__version__,

        "datasets":
            datasets.__version__,

        "peft":
            peft.__version__,

        "trl":
            trl.__version__,

        "python":
            sys.version,

        "platform":
            platform.platform(),

        "gpu":
            torch.cuda.get_device_name(
                0
            ),

        "bf16":
            True,

        "challenge_v2_access":
            "none",
    }


    with (
        output_dir
        / "RUN_CONFIG.json"
    ).open(
        "w"
    ) as f:

        json.dump(
            run_record,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write(
            "\n"
        )


    # ================================================================
    # TRAINER CONFIG
    # ================================================================

    if args.smoke:

        eval_strategy = "no"
        save_strategy = "no"

    else:

        eval_strategy = "epoch"
        save_strategy = "epoch"


    training_args = SFTConfig(

        output_dir=str(
            output_dir
        ),

        num_train_epochs=args.epochs,

        max_steps=args.max_steps,

        per_device_train_batch_size=
            args.batch_size,

        per_device_eval_batch_size=
            args.batch_size,

        gradient_accumulation_steps=
            args.grad_accum,

        learning_rate=
            args.learning_rate,

        warmup_steps=
            warmup_steps,

        lr_scheduler_type=
            "cosine",

        optim=
            "adamw_torch",

        bf16=True,

        fp16=False,

        gradient_checkpointing=True,

        max_length=
            args.max_length,

        completion_only_loss=True,

        packing=False,

        eval_strategy=
            eval_strategy,

        save_strategy=
            save_strategy,

        # Deliberately disabled.
        #
        # We will select V5 checkpoints later using the complete
        # development decision/citation metric suite rather than
        # automatically selecting on language-model eval_loss.
        load_best_model_at_end=False,

        logging_steps=5,

        save_total_limit=4,

        report_to="none",

        seed=args.seed,
    )


    trainer = SFTTrainer(

        model=model,

        args=training_args,

        processing_class=
            tokenizer,

        train_dataset=
            dataset[
                "train"
            ],

        eval_dataset=(
            None
            if args.smoke
            else dataset[
                "development"
            ]
        ),
    )


    # ================================================================
    # TRAIN
    # ================================================================

    print()
    print("=" * 88)
    print("STARTING TRAINING")
    print("=" * 88)


    result = trainer.train()


    print()
    print("=" * 88)
    print("TRAINING COMPLETE")
    print("=" * 88)


    print(
        "Train loss:",
        result.metrics.get(
            "train_loss"
        ),
    )


    print(
        "Runtime metrics:",
        result.metrics,
    )


    # ================================================================
    # SAVE ADAPTER
    # ================================================================

    print()
    print(
        "Saving final adapter state..."
    )


    trainer.save_model(
        str(
            output_dir
        )
    )


    tokenizer.save_pretrained(
        str(
            output_dir
        )
    )


    adapter_file = (
        output_dir
        / "adapter_model.safetensors"
    )


    if not adapter_file.exists():

        raise RuntimeError(
            "Expected adapter_model.safetensors was not saved."
        )


    adapter_sha = sha256(
        adapter_file
    )


    print()
    print(
        "Adapter SHA256:",
        adapter_sha,
    )


    completion_record = {

        "adapter_sha256":
            adapter_sha,

        "train_loss":
            result.metrics.get(
                "train_loss"
            ),

        "global_step":
            trainer.state.global_step,

        "mode":
            (
                "smoke"
                if args.smoke
                else "full"
            ),
    }


    with (
        output_dir
        / "TRAINING_RESULT.json"
    ).open(
        "w"
    ) as f:

        json.dump(
            completion_record,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write(
            "\n"
        )


    print()
    print("=" * 88)

    if args.smoke:

        print(
            'SENTINEL V6 GPU SMOKE TRAINING: PASS'
        )

    else:

        print(
            'SENTINEL V6 RUN TRAINING: PASS'
        )

    print("=" * 88)

    print(
        "Adapter directory:",
        output_dir,
    )

    print(
        "Challenge V2 access: NONE"
    )


if __name__ == "__main__":
    main()
