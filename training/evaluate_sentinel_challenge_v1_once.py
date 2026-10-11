"""
Sentinel V4.4 Run 02 — Challenge V1 frozen one-shot evaluator.

This evaluator is intentionally hard-locked:

Benchmark:
    experiments/challenge_v1/sentinel_challenge_v1.json

Benchmark SHA256:
    02b49320e3b46b87af8841344044d1d51bfc0eb4af2d285e09cce6a43811894d

Candidate:
    training/output/llama-sentinel-8b-lora-v44-run02

Adapter SHA256:
    cd7790f8dfe16e31d002a3cdc12ad345dcd412885223114ffebb81b58384178e

No CLI arguments.
No alternate adapter.
No alternate benchmark.
No base-model comparison.
No training.

Primary endpoint:
    four-way decision accuracy.

Challenge failures must not be used for iterative tuning.
"""

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


BASE = Path("/workspace")

MODEL_NAME = (
    "meta-llama/Llama-3.1-8B-Instruct"
)

CHALLENGE_PATH = (
    BASE
    / "experiments/challenge_v1/"
      "sentinel_challenge_v1.json"
)

EXPECTED_CHALLENGE_SHA = (
    "02b49320e3b46b87af8841344044d1d51"
    "bfc0eb4af2d285e09cce6a43811894d"
)

ADAPTER_PATH = (
    BASE
    / "training/output/"
      "llama-sentinel-8b-lora-v44-run02"
)

ADAPTER_FILE = (
    ADAPTER_PATH
    / "adapter_model.safetensors"
)

EXPECTED_ADAPTER_SHA = (
    "cd7790f8dfe16e31d002a3cdc12ad345"
    "dcd412885223114ffebb81b58384178e"
)

RESULT_PATH = (
    BASE
    / "results/"
      "sentinel_v44_run02_challenge_v1.json"
)

MARKER_PATH = (
    BASE
    / "results/"
      "CHALLENGE_V1_CONSUMED.txt"
)

BATCH_SIZE = 4
MAX_NEW_TOKENS = 192

LABELS = [
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
]


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def verify_frozen_inputs():

    challenge_sha = sha256(
        CHALLENGE_PATH
    )

    adapter_sha = sha256(
        ADAPTER_FILE
    )

    print(
        "Challenge SHA:",
        challenge_sha,
    )

    print(
        "Adapter SHA:",
        adapter_sha,
    )

    if (
        challenge_sha
        != EXPECTED_CHALLENGE_SHA
    ):
        raise RuntimeError(
            "FROZEN CHALLENGE SHA MISMATCH"
        )

    if (
        adapter_sha
        != EXPECTED_ADAPTER_SHA
    ):
        raise RuntimeError(
            "FROZEN ADAPTER SHA MISMATCH"
        )


def extract_json(text):

    cleaned = text.strip()

    try:

        obj = json.loads(
            cleaned
        )

        if isinstance(obj, dict):
            return obj, True

    except Exception:
        pass

    # Fair fallback:
    # recover exactly one outer JSON object
    # if harmless surrounding text exists.
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
                return obj, False

        except Exception:
            pass

    return {}, False


def semantic_decision(obj):

    value = obj.get(
        "decision"
    )

    if value in LABELS:
        return value

    return "PARSE_ERROR"


def compatible_schema(obj):

    if (
        obj.get("decision")
        not in LABELS
    ):
        return False

    if not isinstance(
        obj.get("citations"),
        list,
    ):
        return False

    if not isinstance(
        obj.get("conditions"),
        list,
    ):
        return False

    return True


def gold(case):

    g = case["gold"]

    if not isinstance(g, dict):
        raise TypeError(
            "Challenge gold must be dict."
        )

    if g.get("decision") not in LABELS:
        raise ValueError(
            "Invalid Challenge decision."
        )

    return g


def prepare_batch(
    tokenizer,
    cases,
):

    conversations = []

    for case in cases:

        conversations.append([
            {
                "role": "system",
                "content":
                    case["instruction"],
            },
            {
                "role": "user",
                "content":
                    case["input"],
            },
        ])

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
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    return {
        key: value.to("cuda")
        for key, value
        in encoded.items()
    }


def generate_batch(
    model,
    tokenizer,
    cases,
):

    encoded = prepare_batch(
        tokenizer,
        cases,
    )

    input_length = (
        encoded[
            "input_ids"
        ].shape[1]
    )

    with torch.inference_mode():

        generated = model.generate(
            **encoded,
            max_new_tokens=
                MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=
                tokenizer.eos_token_id,
            eos_token_id=
                tokenizer.eos_token_id,
        )

    new_tokens = generated[
        :,
        input_length:
    ]

    return tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )


def main():

    print("=" * 88)
    print(
        "SENTINEL V4.4 RUN 02 — "
        "CHALLENGE V1 FROZEN ONE-SHOT"
    )
    print("=" * 88)

    if RESULT_PATH.exists():
        raise RuntimeError(
            "REFUSING TO RUN: result already exists."
        )

    if MARKER_PATH.exists():
        raise RuntimeError(
            "REFUSING TO RUN: Challenge already consumed."
        )

    verify_frozen_inputs()

    cases = json.loads(
        CHALLENGE_PATH.read_text()
    )

    if not isinstance(
        cases,
        list,
    ):
        raise TypeError(
            "Challenge must be a list."
        )

    if len(cases) != 80:
        raise RuntimeError(
            f"Expected 80 cases, found {len(cases)}"
        )

    gold_counts = Counter(
        gold(case)["decision"]
        for case in cases
    )

    expected_counts = {
        "YES": 20,
        "NO": 20,
        "CONDITIONAL": 20,
        "INSUFFICIENT": 20,
    }

    if dict(
        sorted(
            gold_counts.items()
        )
    ) != dict(
        sorted(
            expected_counts.items()
        )
    ):
        raise RuntimeError(
            "Challenge label balance mismatch."
        )

    print(
        "Cases:",
        len(cases),
    )

    print(
        "Gold labels:",
        dict(
            sorted(
                gold_counts.items()
            )
        ),
    )

    print()
    print("Loading tokenizer...")

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

    tokenizer.padding_side = "left"

    print(
        "EOS token:",
        tokenizer.eos_token_id,
    )

    print()
    print(
        "Loading Llama 3.1 8B in BF16..."
    )

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=
                torch.bfloat16,
            device_map={
                "": 0
            },
        )
    )

    base_model.eval()
    base_model.config.use_cache = True

    print(
        "Loading frozen Run 02 LoRA..."
    )

    model = (
        PeftModel
        .from_pretrained(
            base_model,
            ADAPTER_PATH,
            is_trainable=False,
        )
    )

    model.eval()

    # Warm-up does not score or print case content.
    print("Warming up...")

    _ = generate_batch(
        model,
        tokenizer,
        cases[:1],
    )

    total = 0
    correct = 0
    strict_json = 0
    schema_ok = 0
    citation_set_match = 0
    parse_errors = 0

    per_label = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
        }
    )

    confusion = {
        label: Counter()
        for label in LABELS
    }

    rows = []

    print()
    print("Evaluating frozen benchmark...")

    for start in range(
        0,
        len(cases),
        BATCH_SIZE,
    ):

        batch = cases[
            start:
            start + BATCH_SIZE
        ]

        outputs = generate_batch(
            model,
            tokenizer,
            batch,
        )

        for case, raw in zip(
            batch,
            outputs,
        ):

            g = gold(case)

            parsed, strict = (
                extract_json(raw)
            )

            predicted = (
                semantic_decision(
                    parsed
                )
            )

            expected = g[
                "decision"
            ]

            decision_ok = (
                predicted == expected
            )

            citations_pred = (
                parsed.get(
                    "citations"
                )
                if isinstance(
                    parsed.get(
                        "citations"
                    ),
                    list,
                )
                else []
            )

            citations_gold = (
                g.get(
                    "citations",
                    []
                )
            )

            citation_match = (
                set(citations_pred)
                == set(citations_gold)
            )

            total += 1
            correct += int(
                decision_ok
            )

            strict_json += int(
                strict
            )

            schema_ok += int(
                compatible_schema(
                    parsed
                )
            )

            citation_set_match += int(
                citation_match
            )

            parse_errors += int(
                predicted
                == "PARSE_ERROR"
            )

            per_label[
                expected
            ]["total"] += 1

            per_label[
                expected
            ]["correct"] += int(
                decision_ok
            )

            confusion[
                expected
            ][predicted] += 1

            # Save only benchmark identifiers and scoring
            # outcomes. Do NOT duplicate question/evidence
            # into the result file.
            rows.append({
                "case_id":
                    case.get(
                        "case_id"
                    ),

                "expected":
                    expected,

                "predicted":
                    predicted,

                "decision_correct":
                    decision_ok,

                "strict_json":
                    strict,

                "schema_ok":
                    compatible_schema(
                        parsed
                    ),

                "citation_set_match":
                    citation_match,

                "raw_model_output":
                    raw,
            })

        print(
            f"{min(start + BATCH_SIZE, len(cases)):3d}"
            f"/{len(cases)} "
            f"accuracy="
            f"{correct/total:.4f}"
        )

    accuracy = (
        correct / total
    )

    result = {
        "benchmark":
            "Challenge V1",

        "challenge_sha256":
            EXPECTED_CHALLENGE_SHA,

        "adapter_sha256":
            EXPECTED_ADAPTER_SHA,

        "cases":
            total,

        "decision_accuracy":
            accuracy,

        "correct":
            correct,

        "strict_json_rate":
            strict_json / total,

        "compatible_schema_rate":
            schema_ok / total,

        "gold_citation_set_match_rate":
            citation_set_match / total,

        "parse_errors":
            parse_errors,

        "per_label": {
            label: {
                "correct":
                    per_label[
                        label
                    ]["correct"],

                "total":
                    per_label[
                        label
                    ]["total"],

                "accuracy":
                    (
                        per_label[
                            label
                        ]["correct"]
                        /
                        per_label[
                            label
                        ]["total"]
                    ),
            }
            for label in LABELS
        },

        "confusion": {
            label:
                dict(
                    confusion[
                        label
                    ]
                )
            for label in LABELS
        },

        "predictions":
            rows,
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("=" * 88)
    print("RESULT")
    print("=" * 88)

    print(
        f"decision accuracy: "
        f"{correct}/{total} "
        f"({accuracy:.4f})"
    )

    print(
        "strict JSON:",
        f"{strict_json/total:.4f}",
    )

    print(
        "compatible schema:",
        f"{schema_ok/total:.4f}",
    )

    print(
        "gold citation-set match:",
        f"{citation_set_match/total:.4f}",
    )

    print(
        "parse errors:",
        parse_errors,
    )

    print()
    print("PER LABEL")

    for label in LABELS:

        stats = per_label[
            label
        ]

        print(
            f"{label:13s} "
            f"{stats['correct']:2d}/"
            f"{stats['total']:2d} "
            f"{stats['correct']/stats['total']:.4f}"
        )

    print()
    print("CONFUSION MATRIX")

    print(
        json.dumps(
            result["confusion"],
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print(
        "Saved:",
        RESULT_PATH,
    )

    # Verify neither frozen input changed.
    verify_frozen_inputs()

    MARKER_PATH.write_text(
        "\n".join([
            "CHALLENGE V1 CONSUMED",
            "",
            (
                "Challenge SHA256: "
                + EXPECTED_CHALLENGE_SHA
            ),
            (
                "Adapter SHA256: "
                + EXPECTED_ADAPTER_SHA
            ),
            (
                "Result: "
                + str(
                    RESULT_PATH.relative_to(
                        BASE
                    )
                )
            ),
            "",
            (
                "This benchmark must not be used "
                "for iterative tuning."
            ),
            "",
        ])
    )

    print()
    print(
        "CONSUMED MARKER WRITTEN:",
        MARKER_PATH,
    )


if __name__ == "__main__":
    main()
