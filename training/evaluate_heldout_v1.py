import json
import re
import time
from collections import defaultdict
from pathlib import Path
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE = Path(__file__).resolve().parent.parent

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

ADAPTER_PATH = (
    BASE /
    "training/output/llama-sentinel-8b-lora-v1"
)

TEST_FILE = (
    BASE /
    "training/datasets/sentinel_test_v1.jsonl"
)

RESULTS_FILE = (
    BASE /
    "results/heldout_lora_v1.json"
)

MAX_NEW_TOKENS = 220


def parse_json(text):
    text = text.strip()

    try:
        return json.loads(text), True
    except Exception:
        pass

    # Salvage a JSON object if the model added extra text.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(
                text[start:end + 1]
            ), False
        except Exception:
            pass

    # Last-resort decision extraction.
    match = re.search(
        r'"decision"\s*:\s*"'
        r'(YES|NO|CONDITIONAL|INSUFFICIENT)"',
        text
    )

    if match:
        return {
            "decision": match.group(1)
        }, False

    return {}, False


def generate(model, tokenizer, example, use_adapter=True):

    messages = [
        {
            "role": "system",
            "content": example["instruction"]
        },
        {
            "role": "user",
            "content": example["input"]
        }
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to("cuda")

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    context = (
        nullcontext()
        if use_adapter
        else model.disable_adapter()
    )

    torch.cuda.synchronize()
    start = time.perf_counter()

    with context:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

    torch.cuda.synchronize()

    latency = time.perf_counter() - start

    generated = output[
        0,
        input_ids.shape[1]:
    ]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return text, latency


def evaluate_mode(
    name,
    model,
    tokenizer,
    examples,
    use_adapter
):

    correct = 0
    sufficient_correct = 0
    valid_json = 0

    latencies = []

    by_class = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0
        }
    )

    rows = []

    print("\n" + "=" * 76)
    print(name)
    print("=" * 76)

    for i, example in enumerate(
        examples,
        start=1
    ):

        gold = json.loads(
            example["output"]
        )

        expected_decision = gold[
            "decision"
        ]

        expected_sufficient = gold[
            "evidence_sufficient"
        ]

        text, latency = generate(
            model,
            tokenizer,
            example,
            use_adapter=use_adapter
        )

        parsed, json_ok = parse_json(
            text
        )

        predicted = parsed.get(
            "decision",
            "PARSE_ERROR"
        )

        predicted_sufficient = (
            parsed.get(
                "evidence_sufficient",
                None
            )
        )

        decision_ok = (
            predicted == expected_decision
        )

        sufficiency_ok = (
            predicted_sufficient
            == expected_sufficient
        )

        correct += int(decision_ok)
        sufficient_correct += int(
            sufficiency_ok
        )
        valid_json += int(json_ok)

        latencies.append(latency)

        by_class[
            expected_decision
        ]["total"] += 1

        by_class[
            expected_decision
        ]["correct"] += int(
            decision_ok
        )

        rows.append({
            "index": i,
            "expected_decision":
                expected_decision,
            "predicted_decision":
                predicted,
            "expected_sufficient":
                expected_sufficient,
            "predicted_sufficient":
                predicted_sufficient,
            "valid_json":
                json_ok,
            "latency_seconds":
                latency,
            "raw_output":
                text
        })

        marker = (
            "✓"
            if decision_ok
            else "✗"
        )

        print(
            f"{marker} "
            f"{i:02d}/"
            f"{len(examples)} "
            f"expected="
            f"{expected_decision:12s} "
            f"predicted="
            f"{predicted:12s} "
            f"{latency:.2f}s"
        )

        if not decision_ok:
            question = (
                example["input"]
                .split(
                    "\n\nEVIDENCE:",
                    1
                )[0]
                .replace(
                    "QUESTION:\n",
                    ""
                )
            )

            print(
                f"    Q: {question}"
            )

            print(
                f"    OUTPUT: {text}"
            )

    total = len(examples)

    summary = {
        "mode": name,
        "examples": total,
        "decision_accuracy":
            correct / total,
        "sufficiency_accuracy":
            sufficient_correct / total,
        "valid_json_rate":
            valid_json / total,
        "average_latency_seconds":
            sum(latencies) /
            len(latencies),
        "per_class": dict(by_class),
        "rows": rows
    }

    print("\nSUMMARY")

    print(
        f"Decision accuracy:    "
        f"{summary['decision_accuracy']:.3f}"
    )

    print(
        f"Sufficiency accuracy: "
        f"{summary['sufficiency_accuracy']:.3f}"
    )

    print(
        f"Valid JSON:           "
        f"{summary['valid_json_rate']:.3f}"
    )

    print(
        f"Avg latency:          "
        f"{summary['average_latency_seconds']:.2f}s"
    )

    print("\nPER CLASS")

    for label in [
        "YES",
        "NO",
        "CONDITIONAL",
        "INSUFFICIENT"
    ]:

        stats = by_class[label]

        accuracy = (
            stats["correct"]
            / stats["total"]
        )

        print(
            f"{label:12s}: "
            f"{stats['correct']:2d}/"
            f"{stats['total']:2d} "
            f"({accuracy:.3f})"
        )

    return summary


def main():

    examples = []

    with TEST_FILE.open() as f:
        for line in f:
            examples.append(
                json.loads(line)
            )

    print(
        f"Loaded {len(examples)} "
        f"held-out examples."
    )

    print("\nLoading tokenizer...")

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        "\nLoading base "
        "Llama 3.1 8B..."
    )

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map={"": 0}
        )
    )

    base_model.eval()
    base_model.config.use_cache = True

    print("\nLoading Sentinel LoRA...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False
    )

    model.eval()

    # Warm up CUDA once so startup doesn't
    # distort our latency comparison.
    warmup_example = examples[0]

    print("\nWarming up model...")

    generate(
        model,
        tokenizer,
        warmup_example,
        use_adapter=True
    )

    print(
        "\nEvaluating BASE 8B "
        "(adapter disabled)..."
    )

    base_summary = evaluate_mode(
        "LLAMA 3.1 8B BASE",
        model,
        tokenizer,
        examples,
        use_adapter=False
    )

    print(
        "\nEvaluating SENTINEL LoRA..."
    )

    lora_summary = evaluate_mode(
        "LLAMA-SENTINEL-8B LoRA",
        model,
        tokenizer,
        examples,
        use_adapter=True
    )

    output = {
        "base": base_summary,
        "lora": lora_summary
    }

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_FILE.write_text(
        json.dumps(
            output,
            indent=2
        )
    )

    print("\n" + "=" * 76)
    print("HEAD-TO-HEAD")
    print("=" * 76)

    print(
        f"Base accuracy: "
        f"{base_summary['decision_accuracy']:.3f}"
    )

    print(
        f"LoRA accuracy: "
        f"{lora_summary['decision_accuracy']:.3f}"
    )

    delta = (
        lora_summary[
            "decision_accuracy"
        ]
        -
        base_summary[
            "decision_accuracy"
        ]
    )

    print(
        f"Accuracy delta: "
        f"{delta:+.3f}"
    )

    print(
        f"\nBase latency: "
        f"{base_summary['average_latency_seconds']:.2f}s"
    )

    print(
        f"LoRA latency: "
        f"{lora_summary['average_latency_seconds']:.2f}s"
    )

    print(
        f"\nSaved: {RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()
