import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer


BASE = Path(__file__).resolve().parent.parent

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

TRAIN_FILE = BASE / "training/datasets/sentinel_train_v2.jsonl"
VAL_FILE = BASE / "training/datasets/sentinel_validation_v2.jsonl"

OUTPUT_DIR = BASE / "training/output/llama-sentinel-8b-lora-v2"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Use a small positive number for a smoke test."
    )

    return parser.parse_args()


def convert_example(example):
    """
    Convert our instruction/input/output JSONL into
    conversational prompt-completion format.

    Loss will be computed ONLY on the assistant completion.
    """

    prompt = [
        {
            "role": "system",
            "content": example["instruction"]
        },
        {
            "role": "user",
            "content": example["input"]
        }
    ]

    completion = [
        {
            "role": "assistant",
            "content": example["output"]
        }
    ]

    return {
        "prompt": prompt,
        "completion": completion
    }


def main():
    args = parse_args()

    print("=" * 72)
    print("LLAMA-SENTINEL-8B LoRA V2 TRAINING")
    print("=" * 72)

    print("\nHardware:")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "BF16 supported:",
            torch.cuda.is_bf16_supported()
        )

    print("\nLoading dataset...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VAL_FILE)
        }
    )

    original_columns = dataset["train"].column_names

    dataset = dataset.map(
        convert_example,
        remove_columns=original_columns
    )

    print(
        "Train examples:",
        len(dataset["train"])
    )

    print(
        "Validation examples:",
        len(dataset["validation"])
    )

    print("\nExample prompt:")
    print(dataset["train"][0]["prompt"])

    print("\nExample completion:")
    print(dataset["train"][0]["completion"])

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("\nLoading Llama 3.1 8B in BF16...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map={"": 0}
    )

    model.config.use_cache = False

    print("\nCreating LoRA adapter...")

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
            "down_proj"
        ]
    )

    model = get_peft_model(
        model,
        lora_config
    )

    print("\nTrainable parameter check:")
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=args.epochs,
        max_steps=args.max_steps,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.learning_rate,

        warmup_steps=0.05,

        lr_scheduler_type="cosine",

        optim="adamw_torch",

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        max_length=args.max_length,

        completion_only_loss=True,

        packing=False,

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=5,

        save_total_limit=2,

        report_to="none",

        seed=42
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )

    print("\n" + "=" * 72)
    print("STARTING TRAINING")
    print("=" * 72)

    result = trainer.train()

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        "Train loss:",
        result.metrics.get("train_loss")
    )

    print(
        "Runtime:",
        result.metrics.get("train_runtime")
    )

    print("\nFinal validation evaluation:")

    metrics = trainer.evaluate()

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaving LoRA adapter...")

    trainer.model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print("\nAdapter saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
