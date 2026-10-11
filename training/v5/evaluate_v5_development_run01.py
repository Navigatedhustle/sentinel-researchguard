"""
Sentinel V5 generative DEVELOPMENT evaluator.

Evaluates:
    BASE
    checkpoint-768
    checkpoint-1536
    checkpoint-2304

against ONLY the frozen 1,024-row V5 DEVELOPMENT set.

Challenge V2 is never opened by this script.

The base model is loaded once. All three LoRA adapters are then loaded
into one PEFT model and activated one at a time.

Metrics are computed from ACTUAL GENERATED TEXT, not teacher-forced
loss.
"""

from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import argparse
import hashlib
import json
import re
import sys
import time

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftModel


BASE = Path("/workspace")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

DEV_FILE = (
    BASE
    / "training/v5/datasets/"
      "sentinel_v5_development.jsonl"
)

RUN_DIR = (
    BASE
    / "training/output/"
      "llama-sentinel-8b-lora-v5-run01"
)

EXPECTED_DEV_SHA = (
    "4e9fe2effa908d8994fdfb3be486661f"
    "9ef178860725fbd887d60b5181900124"
)

LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)

EXPECTED_KEYS = {
    "decision",
    "citations",
    "conditions",
    "rationale",
}

MODELS = (
    ("base", None),
    ("epoch1", RUN_DIR / "checkpoint-768"),
    ("epoch2", RUN_DIR / "checkpoint-1536"),
    ("epoch3", RUN_DIR / "checkpoint-2304"),
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
    )

    return parser.parse_args()


def sha256(path):

    h = hashlib.sha256()

    with Path(path).open("rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_jsonl(path):

    rows = []

    with Path(path).open() as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def write_json(path, value):

    with Path(path).open("w") as f:

        json.dump(
            value,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")


def evidence_refs(evidence):

    return set(
        re.findall(
            r"\[(E\d+)\]",
            evidence,
        )
    )


def strict_parse(text):
    """
    Strict means the entire stripped generation must be one JSON object
    with exactly the expected fields and expected field types.
    """

    try:
        obj = json.loads(
            text.strip()
        )

    except Exception:
        return None


    if not isinstance(obj, dict):
        return None


    if set(obj) != EXPECTED_KEYS:
        return None


    if obj.get("decision") not in LABELS:
        return None


    if not isinstance(
        obj.get("citations"),
        list,
    ):
        return None


    if not all(
        isinstance(item, str)
        for item in obj["citations"]
    ):
        return None


    if not isinstance(
        obj.get("conditions"),
        list,
    ):
        return None


    if not all(
        isinstance(item, str)
        for item in obj["conditions"]
    ):
        return None


    if not isinstance(
        obj.get("rationale"),
        str,
    ):
        return None


    return obj


def lenient_decision(text):

    parsed = strict_parse(text)

    if parsed is not None:
        return parsed["decision"]


    match = re.search(
        r'"decision"\s*:\s*'
        r'"(YES|NO|CONDITIONAL|INSUFFICIENT)"',
        text,
        flags=re.I,
    )

    if match:
        return match.group(1).upper()


    match = re.search(
        r"\b(YES|NO|CONDITIONAL|INSUFFICIENT)\b",
        text,
        flags=re.I,
    )

    if match:
        return match.group(1).upper()


    return None


def rate(correct, total):

    if total == 0:
        return 0.0

    return correct / total


def group_accuracy(records, key):

    totals = Counter()
    correct = Counter()

    for record in records:

        group = record[key]

        totals[group] += 1

        if record["decision_correct"]:
            correct[group] += 1


    result = {}

    for group in sorted(totals):

        result[group] = {
            "correct":
                correct[group],

            "total":
                totals[group],

            "accuracy":
                rate(
                    correct[group],
                    totals[group],
                ),
        }

    return result


def confusion_matrix(records):

    matrix = {
        gold: {
            pred: 0
            for pred in (
                *LABELS,
                "INVALID",
            )
        }
        for gold in LABELS
    }


    for record in records:

        gold = record["gold_decision"]

        pred = (
            record["predicted_decision"]
            if record["predicted_decision"] in LABELS
            else "INVALID"
        )

        matrix[gold][pred] += 1


    return matrix


def summarize(name, records, adapter_sha):

    total = len(records)

    strict_json = sum(
        record["strict_json"]
        for record in records
    )

    decision_correct = sum(
        record["decision_correct"]
        for record in records
    )

    lenient_correct = sum(
        record["lenient_decision_correct"]
        for record in records
    )

    citation_valid = sum(
        record["citation_refs_valid"]
        for record in records
    )

    citation_set_exact = sum(
        record["citation_set_exact"]
        for record in records
    )

    citation_list_exact = sum(
        record["citation_list_exact"]
        for record in records
    )

    conditions_exact = sum(
        record["conditions_exact"]
        for record in records
    )


    labels = group_accuracy(
        records,
        "label",
    )

    operators = group_accuracy(
        records,
        "operator",
    )

    families = group_accuracy(
        records,
        "family",
    )


    worst_label = min(
        value["accuracy"]
        for value in labels.values()
    )

    worst_operator = min(
        value["accuracy"]
        for value in operators.values()
    )

    worst_family = min(
        value["accuracy"]
        for value in families.values()
    )


    metrics = {

        "model":
            name,

        "adapter_sha256":
            adapter_sha,

        "rows":
            total,

        "decision_accuracy":
            rate(
                decision_correct,
                total,
            ),

        "lenient_decision_accuracy":
            rate(
                lenient_correct,
                total,
            ),

        "strict_json_validity":
            rate(
                strict_json,
                total,
            ),

        "citation_reference_validity":
            rate(
                citation_valid,
                total,
            ),

        "exact_citation_set_accuracy":
            rate(
                citation_set_exact,
                total,
            ),

        "exact_citation_list_accuracy":
            rate(
                citation_list_exact,
                total,
            ),

        "exact_conditions_accuracy":
            rate(
                conditions_exact,
                total,
            ),

        "worst_label_accuracy":
            worst_label,

        "worst_operator_accuracy":
            worst_operator,

        "worst_family_accuracy":
            worst_family,

        "by_label":
            labels,

        "by_operator":
            operators,

        "by_family":
            families,

        "confusion_matrix":
            confusion_matrix(
                records
            ),
    }


    gates = {

        "decision_accuracy":
            metrics[
                "decision_accuracy"
            ] >= 0.95,

        "worst_label_accuracy":
            metrics[
                "worst_label_accuracy"
            ] >= 0.90,

        "worst_operator_accuracy":
            metrics[
                "worst_operator_accuracy"
            ] >= 0.90,

        "worst_family_accuracy":
            metrics[
                "worst_family_accuracy"
            ] >= 0.80,

        "strict_json_validity":
            metrics[
                "strict_json_validity"
            ] >= 0.99,

        "citation_reference_validity":
            metrics[
                "citation_reference_validity"
            ] >= 0.99,

        "exact_citation_set_accuracy":
            metrics[
                "exact_citation_set_accuracy"
            ] >= 0.90,
    }


    metrics["gates"] = gates

    metrics["all_primary_gates_pass"] = all(
        gates.values()
    )


    return metrics


def render_prompts(tokenizer, rows):

    rendered = []

    for row in rows:

        messages = [
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


        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


        if not isinstance(text, str):
            raise RuntimeError(
                "Chat template failed to return text."
            )


        if row["instruction"] not in text:
            raise RuntimeError(
                "Rendered prompt lost instruction."
            )


        if row["input"] not in text:
            raise RuntimeError(
                "Rendered prompt lost input."
            )


        rendered.append(text)


    return rendered


def generate_mode(
    model,
    tokenizer,
    rows,
    prompts,
    batch_size,
    max_new_tokens,
    mode,
):

    print()
    print("=" * 88)
    print("GENERATE:", mode)
    print("=" * 88)


    start = time.time()

    outputs = []


    context = (
        model.disable_adapter()
        if mode == "base"
        else nullcontext()
    )


    with context:

        with torch.inference_mode():

            for start_index in range(
                0,
                len(rows),
                batch_size,
            ):

                end_index = min(
                    start_index
                    + batch_size,
                    len(rows),
                )

                batch_prompts = prompts[
                    start_index:
                    end_index
                ]


                encoded = tokenizer(
                    batch_prompts,
                    padding=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                )


                encoded = {
                    key:
                        value.to("cuda")
                    for key, value
                    in encoded.items()
                }


                input_width = (
                    encoded[
                        "input_ids"
                    ].shape[1]
                )


                generated = model.generate(
                    **encoded,

                    max_new_tokens=
                        max_new_tokens,

                    do_sample=False,

                    use_cache=True,

                    pad_token_id=
                        tokenizer.pad_token_id,

                    eos_token_id=
                        tokenizer.eos_token_id,
                )


                suffix = generated[
                    :,
                    input_width:
                ]


                texts = tokenizer.batch_decode(
                    suffix,
                    skip_special_tokens=True,
                )


                outputs.extend(
                    text.strip()
                    for text in texts
                )


                if (
                    end_index % 128 == 0
                    or end_index == len(rows)
                ):

                    print(
                        f"  {end_index}/"
                        f"{len(rows)}"
                    )


    elapsed = time.time() - start


    print(
        "generation seconds:",
        round(
            elapsed,
            2,
        )
    )


    assert len(outputs) == len(rows)

    return outputs


def score_outputs(
    rows,
    texts,
):

    records = []


    for row, text in zip(
        rows,
        texts,
    ):

        gold = json.loads(
            row["output"]
        )

        parsed = strict_parse(
            text
        )

        strict = (
            parsed is not None
        )


        predicted = (
            parsed["decision"]
            if strict
            else None
        )


        lenient = lenient_decision(
            text
        )


        valid_refs = evidence_refs(
            row["evidence"]
        )


        if strict:

            citations = parsed[
                "citations"
            ]

            no_duplicates = (
                len(citations)
                == len(set(citations))
            )

            citation_refs_valid = (
                no_duplicates
                and all(
                    citation in valid_refs
                    for citation in citations
                )
            )


            citation_set_exact = (
                citation_refs_valid
                and len(citations)
                    == len(
                        gold["citations"]
                    )
                and set(citations)
                    == set(
                        gold["citations"]
                    )
            )


            citation_list_exact = (
                citations
                == gold[
                    "citations"
                ]
            )


            conditions_exact = (
                parsed[
                    "conditions"
                ]
                == gold[
                    "conditions"
                ]
            )


        else:

            citations = None

            citation_refs_valid = False
            citation_set_exact = False
            citation_list_exact = False
            conditions_exact = False


        records.append({

            "scenario_id":
                row[
                    "scenario_id"
                ],

            "family":
                row[
                    "family"
                ],

            "operator":
                row[
                    "operator"
                ],

            "label":
                row[
                    "target"
                ],

            "gold_decision":
                gold[
                    "decision"
                ],

            "gold_citations":
                gold[
                    "citations"
                ],

            "gold_conditions":
                gold[
                    "conditions"
                ],

            "generated_text":
                text,

            "strict_json":
                strict,

            "parsed":
                parsed,

            "predicted_decision":
                predicted,

            "lenient_decision":
                lenient,

            "decision_correct":
                (
                    predicted
                    == gold[
                        "decision"
                    ]
                ),

            "lenient_decision_correct":
                (
                    lenient
                    == gold[
                        "decision"
                    ]
                ),

            "citation_refs_valid":
                citation_refs_valid,

            "citation_set_exact":
                citation_set_exact,

            "citation_list_exact":
                citation_list_exact,

            "conditions_exact":
                conditions_exact,
        })


    return records


def write_jsonl(
    path,
    records,
):

    with Path(path).open(
        "w"
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    sort_keys=True,
                )
            )

            f.write("\n")


def print_metrics(
    metrics,
):

    print()
    print(
        "-" * 88
    )

    print(
        metrics[
            "model"
        ]
    )

    print(
        "-" * 88
    )


    for key in (
        "decision_accuracy",
        "lenient_decision_accuracy",
        "strict_json_validity",
        "citation_reference_validity",
        "exact_citation_set_accuracy",
        "exact_citation_list_accuracy",
        "exact_conditions_accuracy",
        "worst_label_accuracy",
        "worst_operator_accuracy",
        "worst_family_accuracy",
    ):

        print(
            f"{key:36s} "
            f"{metrics[key]:.4f}"
        )


    print(
        "all primary gates:",
        (
            "PASS"
            if metrics[
                "all_primary_gates_pass"
            ]
            else "FAIL"
        ),
    )


def main():

    args = parse_args()

    result_dir = Path(
        args.result_dir
    )


    if result_dir.exists():

        raise RuntimeError(
            "Result directory already exists; "
            "refusing overwrite:\n"
            f"{result_dir}"
        )


    result_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


    # ================================================================
    # FROZEN DEVELOPMENT HASH
    # ================================================================

    dev_sha = sha256(
        DEV_FILE
    )


    print(
        "DEVELOPMENT SHA256:",
        dev_sha,
    )


    if dev_sha != EXPECTED_DEV_SHA:

        raise RuntimeError(
            "Frozen DEVELOPMENT SHA mismatch."
        )


    rows = load_jsonl(
        DEV_FILE
    )


    if len(rows) != 1024:

        raise RuntimeError(
            "DEVELOPMENT row count mismatch."
        )


    print(
        "Development rows:",
        len(rows),
    )


    # ================================================================
    # ADAPTER HASHES
    # ================================================================

    adapter_hashes = {}


    for name, path in MODELS:

        if path is None:

            adapter_hashes[
                name
            ] = None

            continue


        adapter_file = (
            path
            / "adapter_model.safetensors"
        )


        if not adapter_file.exists():

            raise RuntimeError(
                f"Missing adapter: {adapter_file}"
            )


        adapter_hashes[
            name
        ] = sha256(
            adapter_file
        )


        print(
            name,
            adapter_hashes[
                name
            ],
        )


    # ================================================================
    # TOKENIZER
    # ================================================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    # Left padding is appropriate for batched decoder-only generation.
    tokenizer.padding_side = "left"


    prompts = render_prompts(
        tokenizer,
        rows,
    )


    # ================================================================
    # LOAD BASE ONCE
    # ================================================================

    print()
    print(
        "Loading fresh base model once..."
    )


    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
    )


    base_model.eval()


    # ================================================================
    # LOAD ALL THREE ADAPTERS
    # ================================================================

    print()
    print(
        "Loading epoch-1 adapter..."
    )


    model = PeftModel.from_pretrained(
        base_model,
        str(
            RUN_DIR
            / "checkpoint-768"
        ),
        adapter_name="epoch1",
        is_trainable=False,
    )


    print(
        "Loading epoch-2 adapter..."
    )


    model.load_adapter(
        str(
            RUN_DIR
            / "checkpoint-1536"
        ),
        adapter_name="epoch2",
        is_trainable=False,
    )


    print(
        "Loading epoch-3 adapter..."
    )


    model.load_adapter(
        str(
            RUN_DIR
            / "checkpoint-2304"
        ),
        adapter_name="epoch3",
        is_trainable=False,
    )


    model.eval()


    all_metrics = {}


    # ================================================================
    # BASE
    # ================================================================

    base_text = generate_mode(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        prompts=prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        mode="base",
    )


    base_records = score_outputs(
        rows,
        base_text,
    )


    base_metrics = summarize(
        "base",
        base_records,
        None,
    )


    all_metrics[
        "base"
    ] = base_metrics


    write_jsonl(
        result_dir
        / "base_predictions.jsonl",
        base_records,
    )


    write_json(
        result_dir
        / "base_metrics.json",
        base_metrics,
    )


    print_metrics(
        base_metrics
    )


    # ================================================================
    # THREE CHECKPOINTS
    # ================================================================

    for adapter_name in (
        "epoch1",
        "epoch2",
        "epoch3",
    ):

        model.set_adapter(
            adapter_name
        )


        texts = generate_mode(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            prompts=prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            mode=adapter_name,
        )


        records = score_outputs(
            rows,
            texts,
        )


        metrics = summarize(
            adapter_name,
            records,
            adapter_hashes[
                adapter_name
            ],
        )


        all_metrics[
            adapter_name
        ] = metrics


        write_jsonl(
            result_dir
            / f"{adapter_name}_predictions.jsonl",
            records,
        )


        write_json(
            result_dir
            / f"{adapter_name}_metrics.json",
            metrics,
        )


        print_metrics(
            metrics
        )


    # ================================================================
    # APPLY PRECOMMITTED SELECTION POLICY
    # ================================================================

    candidates = [
        all_metrics[
            name
        ]
        for name
        in (
            "epoch1",
            "epoch2",
            "epoch3",
        )
        if all_metrics[
            name
        ][
            "all_primary_gates_pass"
        ]
    ]


    if candidates:

        epoch_order = {
            "epoch1": 1,
            "epoch2": 2,
            "epoch3": 3,
        }


        candidates.sort(
            key=lambda m: (
                -m[
                    "decision_accuracy"
                ],

                -m[
                    "exact_citation_set_accuracy"
                ],

                -m[
                    "worst_family_accuracy"
                ],

                -m[
                    "strict_json_validity"
                ],

                epoch_order[
                    m[
                        "model"
                    ]
                ],
            )
        )


        selected = candidates[
            0
        ][
            "model"
        ]

        status = (
            "CANDIDATE_SELECTED"
        )


    else:

        selected = None

        status = (
            "NO_CHECKPOINT_PASSED_ALL_GATES"
        )


    selection = {

        "status":
            status,

        "selected":
            selected,

        "selection_policy":
            "V5_RUN01_DEVELOPMENT_SELECTION_POLICY.txt",

        "development_sha256":
            dev_sha,

        "models":
            all_metrics,
    }


    write_json(
        result_dir
        / "SUMMARY.json",
        selection,
    )


    # ================================================================
    # CONSOLE SUMMARY
    # ================================================================

    print()
    print("=" * 88)
    print("V5 RUN 01 DEVELOPMENT SUMMARY")
    print("=" * 88)


    print(
        f"{'model':10s} "
        f"{'decision':>9s} "
        f"{'json':>9s} "
        f"{'cite-set':>9s} "
        f"{'worst-L':>9s} "
        f"{'worst-O':>9s} "
        f"{'worst-F':>9s} "
        f"{'gates':>7s}"
    )


    for name in (
        "base",
        "epoch1",
        "epoch2",
        "epoch3",
    ):

        m = all_metrics[
            name
        ]

        print(
            f"{name:10s} "
            f"{m['decision_accuracy']:9.4f} "
            f"{m['strict_json_validity']:9.4f} "
            f"{m['exact_citation_set_accuracy']:9.4f} "
            f"{m['worst_label_accuracy']:9.4f} "
            f"{m['worst_operator_accuracy']:9.4f} "
            f"{m['worst_family_accuracy']:9.4f} "
            f"{('PASS' if m['all_primary_gates_pass'] else 'FAIL'):>7s}"
        )


    print()
    print(
        "Selection status:",
        status,
    )

    print(
        "Selected checkpoint:",
        selected,
    )


    print()
    print(
        "SENTINEL V5 DEVELOPMENT EVALUATION: PASS"
    )

    print(
        "Challenge V2 access: NONE"
    )


if __name__ == "__main__":
    main()
