import argparse
import json
import re
import time

from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftModel


BASE = Path("/workspace")

MODEL_NAME = (
    "meta-llama/Llama-3.1-8B-Instruct"
)

DEFAULT_ADAPTER = (
    BASE
    / "training/output/"
      "llama-sentinel-8b-lora-v43-run01"
)

DATASETS = {
    "train":
        BASE
        / "training/datasets/"
          "sentinel_train_v44_gate_pilot.jsonl",

    "validation":
        BASE
        / "training/datasets/"
          "sentinel_validation_v44_gate_pilot.jsonl",

    "test":
        BASE
        / "training/datasets/"
          "sentinel_test_v44_gate_pilot.jsonl",
}

ALLOWED = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
}

EXPECTED_SCHEMA = {
    "citations",
    "conditions",
    "decision",
    "rationale",
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=[
            "test",
        ],
        default="test",
    )

    parser.add_argument(
        "--adapter-path",
        default=str(DEFAULT_ADAPTER),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "lora",
            "base",
        ],
        default="lora",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help=(
            "0 means evaluate the whole split."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="",
    )

    return parser.parse_args()


def load_rows(path):

    rows = []

    with path.open() as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def extract_json(text):

    cleaned = text.strip()

    strict = False

    try:

        obj = json.loads(cleaned)

        if isinstance(obj, dict):
            return obj, True

    except Exception:
        pass

    # Fair fallback:
    # recover one JSON object if the model added
    # harmless text around it.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        try:

            obj = json.loads(
                cleaned[
                    start:end + 1
                ]
            )

            if isinstance(obj, dict):
                return obj, strict

        except Exception:
            pass

    return {}, False


def semantic_decision(
    obj,
    raw_text,
):

    value = obj.get(
        "decision"
    )

    if isinstance(value, str):

        value = value.strip().upper()

        if value in ALLOWED:
            return value

    # Fair parsing fallback only.
    match = re.search(
        r'"decision"\s*:\s*"'
        r'(YES|NO|CONDITIONAL|INSUFFICIENT)'
        r'"',
        raw_text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    return "PARSE_ERROR"


def schema_ok(
    obj,
    strict_json,
):

    if not strict_json:
        return False

    if set(obj.keys()) != EXPECTED_SCHEMA:
        return False

    if (
        not isinstance(
            obj.get("decision"),
            str,
        )
        or
        obj["decision"] not in ALLOWED
    ):
        return False

    if not isinstance(
        obj.get("rationale"),
        str,
    ):
        return False

    citations = obj.get(
        "citations"
    )

    if not isinstance(
        citations,
        list,
    ):
        return False

    if not all(
        isinstance(x, str)
        for x in citations
    ):
        return False

    conditions = obj.get(
        "conditions"
    )

    if not isinstance(
        conditions,
        list,
    ):
        return False

    if not all(
        isinstance(x, str)
        for x in conditions
    ):
        return False

    return True


def evidence_ids(text):

    return set(
        re.findall(
            r"^\[([^\]]+)\]\s*$",
            text,
            flags=re.MULTILINE,
        )
    )


def citation_refs_valid(
    obj,
    row,
):

    citations = obj.get(
        "citations"
    )

    if not isinstance(
        citations,
        list,
    ):
        return False

    if not all(
        isinstance(x, str)
        for x in citations
    ):
        return False

    valid_ids = evidence_ids(
        row["input"]
    )

    return all(
        citation in valid_ids
        for citation in citations
    )


def citation_set_match(
    obj,
    gold_obj,
):

    predicted = obj.get(
        "citations"
    )

    expected = gold_obj.get(
        "citations"
    )

    if (
        not isinstance(
            predicted,
            list,
        )
        or
        not isinstance(
            expected,
            list,
        )
    ):
        return False

    return (
        set(predicted)
        == set(expected)
    )


def build_messages(row):

    return [
        {
            "role": "system",
            "content":
                row["instruction"],
        },
        {
            "role": "user",
            "content":
                row["input"],
        },
    ]


def main():

    args = parse_args()

    dataset_path = DATASETS[
        args.split
    ]

    rows = load_rows(
        dataset_path
    )

    if args.max_cases > 0:

        rows = rows[
            :args.max_cases
        ]

    print("=" * 80)
    print(
        "SENTINEL V4.4 GATE TEST — FROZEN ONE-SHOT"
    )
    print("=" * 80)

    print(
        "split:",
        args.split,
    )

    print(
        "mode:",
        args.mode,
    )

    print(
        "cases:",
        len(rows),
    )

    print(
        "adapter:",
        args.adapter_path,
    )

    print(
        "batch size:",
        args.batch_size,
    )

    print()

    print("Loading tokenizer...")

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.adapter_path
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # Required for batched decoder-only
    # generation.
    tokenizer.padding_side = "left"

    print(
        "Loading base model in BF16..."
    )

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
    )

    print(
        "Loading LoRA adapter..."
    )

    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
    )

    model.eval()

    terminators = [
        tokenizer.eos_token_id,
    ]

    eot_id = (
        tokenizer.convert_tokens_to_ids(
            "<|eot_id|>"
        )
    )

    if (
        isinstance(eot_id, int)
        and eot_id >= 0
        and eot_id not in terminators
    ):
        terminators.append(
            eot_id
        )

    print(
        "EOS terminators:",
        terminators,
    )

    correct = 0
    strict_json_count = 0
    schema_count = 0
    citation_valid_count = 0
    citation_match_count = 0
    parse_errors = 0

    confusion = defaultdict(
        Counter
    )

    label_total = Counter()
    label_correct = Counter()

    composition_total = Counter()
    composition_correct = Counter()

    family_total = Counter()
    family_correct = Counter()

    predictions = []

    start_time = time.perf_counter()

    batch_size = args.batch_size

    for batch_start in range(
        0,
        len(rows),
        batch_size,
    ):

        batch_rows = rows[
            batch_start:
            batch_start + batch_size
        ]

        conversations = [
            build_messages(row)
            for row in batch_rows
        ]

        # Render chat template first so batch
        # tokenization is version-stable.
        rendered = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in conversations
        ]

        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        prompt_width = (
            encoded[
                "input_ids"
            ].shape[1]
        )

        context = (
            nullcontext()
            if args.mode == "lora"
            else model.disable_adapter()
        )

        with context:

            with torch.inference_mode():

                outputs = model.generate(
                    **encoded,

                    max_new_tokens=(
                        args.max_new_tokens
                    ),

                    do_sample=False,

                    pad_token_id=(
                        tokenizer.pad_token_id
                    ),

                    eos_token_id=terminators,

                    use_cache=True,
                )

        for j, row in enumerate(
            batch_rows
        ):

            generated_ids = outputs[
                j,
                prompt_width:
            ]

            raw_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            obj, strict_json = (
                extract_json(
                    raw_text
                )
            )

            predicted = (
                semantic_decision(
                    obj,
                    raw_text,
                )
            )

            gold_obj = json.loads(
                row["output"]
            )

            expected = gold_obj[
                "decision"
            ]

            decision_ok = (
                predicted
                == expected
            )

            strict_json_count += int(
                strict_json
            )

            schema_valid = schema_ok(
                obj,
                strict_json,
            )

            schema_count += int(
                schema_valid
            )

            citation_valid = (
                citation_refs_valid(
                    obj,
                    row,
                )
            )

            citation_valid_count += int(
                citation_valid
            )

            citations_match = (
                citation_set_match(
                    obj,
                    gold_obj,
                )
            )

            citation_match_count += int(
                citations_match
            )

            if (
                predicted
                == "PARSE_ERROR"
            ):
                parse_errors += 1

            correct += int(
                decision_ok
            )

            label_total[
                expected
            ] += 1

            label_correct[
                expected
            ] += int(
                decision_ok
            )

            composition = row.get(
                "composition",
                "UNKNOWN",
            )

            composition_total[
                composition
            ] += 1

            composition_correct[
                composition
            ] += int(
                decision_ok
            )

            family = row.get(
                "family",
                "UNKNOWN",
            )

            family_total[
                family
            ] += 1

            family_correct[
                family
            ] += int(
                decision_ok
            )

            confusion[
                expected
            ][
                predicted
            ] += 1

            predictions.append({
                "scenario_id":
                    row.get(
                        "scenario_id"
                    ),

                "family":
                    family,

                "composition":
                    composition,

                "expected":
                    expected,

                "predicted":
                    predicted,

                "decision_correct":
                    decision_ok,

                "strict_json":
                    strict_json,

                "schema_ok":
                    schema_valid,

                "citation_refs_valid":
                    citation_valid,

                "citation_set_match":
                    citations_match,

                "raw_model_output":
                    raw_text,
            })

        processed = min(
            batch_start
            + len(batch_rows),
            len(rows),
        )

        if (
            processed % 32 == 0
            or processed == len(rows)
        ):

            running_accuracy = (
                correct
                / processed
            )

            print(
                f"{processed:4}/{len(rows)}  "
                f"accuracy="
                f"{running_accuracy:.4f}"
            )

    runtime = (
        time.perf_counter()
        - start_time
    )

    total = len(rows)

    summary = {
        "split":
            args.split,

        "mode":
            args.mode,

        "cases":
            total,

        "decision_accuracy":
            (
                correct / total
                if total
                else 0.0
            ),

        "correct":
            correct,

        "strict_json_rate":
            (
                strict_json_count / total
                if total
                else 0.0
            ),

        "schema_rate":
            (
                schema_count / total
                if total
                else 0.0
            ),

        "citation_reference_valid_rate":
            (
                citation_valid_count / total
                if total
                else 0.0
            ),

        "citation_gold_set_match_rate":
            (
                citation_match_count / total
                if total
                else 0.0
            ),

        "parse_errors":
            parse_errors,

        "runtime_seconds":
            runtime,

        "cases_per_second":
            (
                total / runtime
                if runtime > 0
                else 0.0
            ),

        "per_label": {},

        "per_composition": {},

        "per_family": {},

        "confusion_matrix": {},
    }

    for label in sorted(
        ALLOWED
    ):

        n = label_total[
            label
        ]

        summary[
            "per_label"
        ][label] = {
            "n":
                n,

            "correct":
                label_correct[
                    label
                ],

            "accuracy":
                (
                    label_correct[
                        label
                    ] / n
                    if n
                    else 0.0
                ),
        }

    for composition in sorted(
        composition_total
    ):

        n = composition_total[
            composition
        ]

        summary[
            "per_composition"
        ][composition] = {
            "n":
                n,

            "correct":
                composition_correct[
                    composition
                ],

            "accuracy":
                (
                    composition_correct[
                        composition
                    ] / n
                    if n
                    else 0.0
                ),
        }

    for family in sorted(
        family_total
    ):

        n = family_total[
            family
        ]

        summary[
            "per_family"
        ][family] = {
            "n":
                n,

            "correct":
                family_correct[
                    family
                ],

            "accuracy":
                (
                    family_correct[
                        family
                    ] / n
                    if n
                    else 0.0
                ),
        }

    predicted_columns = sorted(
        ALLOWED
        | {"PARSE_ERROR"}
    )

    for expected in sorted(
        ALLOWED
    ):

        summary[
            "confusion_matrix"
        ][expected] = {
            predicted:
                confusion[
                    expected
                ][
                    predicted
                ]

            for predicted
            in predicted_columns
        }

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        "decision accuracy:",
        f"{correct}/{total}",
        f"({summary['decision_accuracy']:.4f})",
    )

    print(
        "strict JSON:",
        f"{summary['strict_json_rate']:.4f}",
    )

    print(
        "V4.3 schema:",
        f"{summary['schema_rate']:.4f}",
    )

    print(
        "valid citation references:",
        f"{summary['citation_reference_valid_rate']:.4f}",
    )

    print(
        "gold citation-set match:",
        f"{summary['citation_gold_set_match_rate']:.4f}",
    )

    print(
        "parse errors:",
        parse_errors,
    )

    print()
    print("PER LABEL")

    for label, stats in (
        summary[
            "per_label"
        ].items()
    ):

        print(
            f"{label:14} "
            f"{stats['correct']:4}/"
            f"{stats['n']:4} "
            f"{stats['accuracy']:.4f}"
        )

    print()
    print("PER COMPOSITION")

    for composition, stats in (
        summary[
            "per_composition"
        ].items()
    ):

        print(
            f"{composition:40} "
            f"{stats['correct']:4}/"
            f"{stats['n']:4} "
            f"{stats['accuracy']:.4f}"
        )

    print()
    print("CONFUSION MATRIX")

    print(
        json.dumps(
            summary[
                "confusion_matrix"
            ],
            indent=2,
        )
    )

    if args.output:

        output_path = Path(
            args.output
        )

    else:

        output_path = (
            BASE
            / "results"
            / (
                f"sentinel_v44_gate_"
                f"{args.split}_"
                f"{args.mode}.json"
            )
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "summary":
            summary,

        "predictions":
            predictions,
    }

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
