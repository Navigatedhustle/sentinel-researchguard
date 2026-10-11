"""
Sentinel V5 — ONE-SHOT Challenge V2 evaluator.

This script evaluates exactly one frozen V5 adapter.

It performs schema/extraction validation BEFORE model generation.

It does not train.
It does not select checkpoints.
It does not repair outputs.
It does not retry generation.
"""

from collections import Counter
from pathlib import Path

import argparse
import hashlib
import json
import math
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

MODEL_NAME = (
    "meta-llama/Llama-3.1-8B-Instruct"
)

TRAIN_FILE = (
    BASE
    / "training/v5/datasets/"
      "sentinel_v5_train.jsonl"
)

CHALLENGE_FILE = (
    BASE
    / "experiments/challenge_v2/"
      "sentinel_challenge_v2.json"
)

CANDIDATE_MODEL = (
    BASE
    / "training/archive/"
      "v5_run01_epoch2_candidate_frozen_20260817/"
      "model"
)


EXPECTED_TRAIN_SHA = (
    "945eef0841648911aaee5b88a5f7efd6"
    "c84bcad7399697f7f786f4774645d972"
)

EXPECTED_CHALLENGE_SHA = (
    "30eb37ff39382f0d205adb7c17383b240"
    "25601ea3db1dd61f22b715af6c1e482"
)

EXPECTED_CANDIDATE_SHA = (
    "e228965de1d857cfe9cde1af073ce2e0136"
    "c14deff5ba52b923f87a3cfeda907"
)


LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)


EXPECTED_OUTPUT_KEYS = {
    "decision",
    "citations",
    "conditions",
    "rationale",
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
    )

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


def textify(value):

    if value is None:
        return ""


    if isinstance(value, str):
        return value


    if isinstance(
        value,
        (
            int,
            float,
            bool,
        ),
    ):
        return str(value)


    if isinstance(value, list):

        return "\n".join(
            part
            for item in value
            if (
                part := textify(
                    item
                ).strip()
            )
        )


    if isinstance(value, dict):

        preferred = []

        for key in (
            "text",
            "content",
            "passage",
            "body",
            "value",
        ):

            if key in value:

                candidate = textify(
                    value[key]
                ).strip()

                if candidate:
                    preferred.append(
                        candidate
                    )


        if preferred:

            return "\n".join(
                preferred
            )


        parts = []

        for key, item in value.items():

            if str(key).lower() in {
                "label",
                "target",
                "answer",
                "decision",
                "gold_label",
                "gold_decision",
                "output",
                "expected",
                "expected_output",
                "response",
                "rationale",
                "citations",
                "conditions",
                "id",
                "case_id",
                "scenario_id",
            }:
                continue


            candidate = textify(
                item
            ).strip()

            if candidate:
                parts.append(
                    candidate
                )


        return "\n".join(
            parts
        )


    return ""


def case_list_from_json(root):

    if isinstance(root, list):

        if all(
            isinstance(item, dict)
            for item in root
        ):
            return root


    if isinstance(root, dict):

        for key in (
            "cases",
            "examples",
            "items",
            "rows",
            "data",
            "benchmark",
        ):

            value = root.get(key)

            if (
                isinstance(value, list)
                and all(
                    isinstance(item, dict)
                    for item in value
                )
            ):
                return value


        candidates = [
            value
            for value in root.values()
            if (
                isinstance(value, list)
                and value
                and all(
                    isinstance(item, dict)
                    for item in value
                )
            )
        ]


        if len(candidates) == 1:
            return candidates[0]


    raise RuntimeError(
        "Unable to mechanically locate Challenge V2 cases."
    )


def direct_value(case, keys):

    lower_map = {
        str(key).lower():
            key
        for key in case
    }


    for wanted in keys:

        actual = lower_map.get(
            wanted.lower()
        )

        if actual is not None:

            return case[
                actual
            ]


    return None


def direct_text(case, keys):

    value = direct_value(
        case,
        keys,
    )

    return textify(
        value
    ).strip()


def normalize_label(value):

    if value is None:
        return None


    if isinstance(value, dict):

        for key in (
            "decision",
            "gold_decision",
            "label",
            "gold_label",
            "target",
            "answer",
        ):

            if key in value:

                result = normalize_label(
                    value[key]
                )

                if result is not None:
                    return result


        return None


    if isinstance(value, list):

        if len(value) == 1:

            return normalize_label(
                value[0]
            )

        return None


    text = str(
        value
    ).strip()


    if not text:
        return None


    upper = text.upper()


    if upper in LABELS:
        return upper


    try:

        parsed = json.loads(
            text
        )

    except Exception:

        parsed = None


    if parsed is not None:

        result = normalize_label(
            parsed
        )

        if result is not None:
            return result


    # Permit forms like:
    # "YES."
    # "Decision: YES"
    # but reject strings containing multiple label words.

    found = re.findall(
        r"\b(?:YES|NO|CONDITIONAL|INSUFFICIENT)\b",
        upper,
    )


    unique = list(
        dict.fromkeys(
            found
        )
    )


    if len(unique) == 1:
        return unique[0]


    return None


def extract_gold_container(case):

    for key in (
        "gold",
        "expected_output",
        "expected",
        "output",
    ):

        value = direct_value(
            case,
            (key,),
        )

        if value is None:
            continue


        if isinstance(value, dict):
            return value


        if isinstance(value, str):

            try:
                parsed = json.loads(
                    value
                )

            except Exception:
                continue


            if isinstance(parsed, dict):
                return parsed


    return None


def extract_gold_label(case):

    for key in (
        "gold_label",
        "gold_decision",
        "target",
        "label",
        "answer",
        "decision",
    ):

        value = direct_value(
            case,
            (key,),
        )

        label = normalize_label(
            value
        )

        if label is not None:
            return label


    container = extract_gold_container(
        case
    )


    if container is not None:

        label = normalize_label(
            container
        )

        if label is not None:
            return label


    return None


def parse_rendered_input(text):

    if not text:

        return (
            "",
            "",
        )


    match = re.search(
        r"(?is)"
        r"\bQUESTION\b\s*:?\s*"
        r"(.*?)"
        r"\s*\b(?:SUPPLIED\s+EVIDENCE|EVIDENCE)\b\s*:?\s*"
        r"(.*)",
        text,
    )


    if match:

        return (
            match.group(1).strip(),
            match.group(2).strip(),
        )


    match = re.search(
        r"(?is)"
        r"\b(?:SUPPLIED\s+EVIDENCE|EVIDENCE)\b\s*:?\s*"
        r"(.*?)"
        r"\s*\bQUESTION\b\s*:?\s*"
        r"(.*)",
        text,
    )


    if match:

        return (
            match.group(2).strip(),
            match.group(1).strip(),
        )


    return (
        "",
        "",
    )


def read_frozen_instruction():

    with TRAIN_FILE.open() as f:

        for line in f:

            line = line.strip()

            if line:

                row = json.loads(
                    line
                )

                instruction = row[
                    "instruction"
                ]

                if not isinstance(
                    instruction,
                    str,
                ):

                    raise RuntimeError(
                        "Frozen V5 instruction is not a string."
                    )


                return instruction


    raise RuntimeError(
        "Frozen TRAIN dataset is empty."
    )


def extract_case(
    case,
    index,
    frozen_instruction,
):

    case_id_value = direct_value(
        case,
        (
            "case_id",
            "scenario_id",
            "id",
        ),
    )


    if case_id_value is None:

        case_id = (
            f"challenge_v2_case_{index:04d}"
        )

    else:

        case_id = str(
            case_id_value
        )


    gold_label = extract_gold_label(
        case
    )


    benchmark_instruction = direct_text(
        case,
        (
            "instruction",
            "system_instruction",
            "system",
        ),
    )


    if benchmark_instruction:

        instruction = benchmark_instruction

        instruction_source = (
            "benchmark"
        )

    else:

        instruction = frozen_instruction

        instruction_source = (
            "frozen_v5"
        )


    rendered_input = direct_text(
        case,
        (
            "input",
            "user_input",
        ),
    )


    question = direct_text(
        case,
        (
            "question",
            "query",
            "user_question",
        ),
    )


    evidence = direct_text(
        case,
        (
            "evidence",
            "evidence_text",
            "supplied_evidence",
            "passages",
            "documents",
            "context",
            "policy_text",
            "policies",
        ),
    )


    if rendered_input:

        user_input = rendered_input

        input_source = (
            "benchmark_input"
        )


        if not question or not evidence:

            (
                parsed_question,
                parsed_evidence,
            ) = parse_rendered_input(
                rendered_input
            )


            if not question:
                question = parsed_question


            if not evidence:
                evidence = parsed_evidence


    else:

        if not question:

            raise RuntimeError(
                f"Case {index} has no extractable question."
            )


        if not evidence:

            raise RuntimeError(
                f"Case {index} has no extractable evidence."
            )


        input_order = direct_text(
            case,
            (
                "input_order",
                "order",
            ),
        ).lower()


        if input_order in {
            "question_first",
            "question-first",
            "question first",
        }:

            user_input = (
                "QUESTION:\n"
                f"{question}\n\n"
                "SUPPLIED EVIDENCE:\n"
                f"{evidence}"
            )

            input_source = (
                "constructed_question_first"
            )


        else:

            user_input = (
                "SUPPLIED EVIDENCE:\n"
                f"{evidence}\n\n"
                "QUESTION:\n"
                f"{question}"
            )

            input_source = (
                "constructed_evidence_first"
            )


    gold_container = extract_gold_container(
        case
    )


    gold_citations = None
    gold_conditions = None


    if gold_container is not None:

        citations = gold_container.get(
            "citations"
        )

        conditions = gold_container.get(
            "conditions"
        )


        if (
            isinstance(citations, list)
            and all(
                isinstance(item, str)
                for item in citations
            )
        ):
            gold_citations = citations


        if (
            isinstance(conditions, list)
            and all(
                isinstance(item, str)
                for item in conditions
            )
        ):
            gold_conditions = conditions


    return {

        "case_id":
            case_id,

        "gold_label":
            gold_label,

        "instruction":
            instruction,

        "instruction_source":
            instruction_source,

        "user_input":
            user_input,

        "input_source":
            input_source,

        "evidence":
            evidence,

        "gold_citations":
            gold_citations,

        "gold_conditions":
            gold_conditions,
    }


def strict_parse(text):

    try:

        obj = json.loads(
            text.strip()
        )

    except Exception:

        return None


    if not isinstance(
        obj,
        dict,
    ):
        return None


    if set(
        obj
    ) != EXPECTED_OUTPUT_KEYS:
        return None


    if obj.get(
        "decision"
    ) not in LABELS:
        return None


    if not isinstance(
        obj.get(
            "citations"
        ),
        list,
    ):
        return None


    if not all(
        isinstance(
            item,
            str,
        )
        for item
        in obj[
            "citations"
        ]
    ):
        return None


    if not isinstance(
        obj.get(
            "conditions"
        ),
        list,
    ):
        return None


    if not all(
        isinstance(
            item,
            str,
        )
        for item
        in obj[
            "conditions"
        ]
    ):
        return None


    if not isinstance(
        obj.get(
            "rationale"
        ),
        str,
    ):
        return None


    return obj


def lenient_decision(text):

    parsed = strict_parse(
        text
    )

    if parsed is not None:

        return parsed[
            "decision"
        ]


    match = re.search(
        r'"decision"\s*:\s*'
        r'"(YES|NO|CONDITIONAL|INSUFFICIENT)"',
        text,
        flags=re.I,
    )


    if match:

        return (
            match
            .group(1)
            .upper()
        )


    matches = re.findall(
        r"\b(?:YES|NO|CONDITIONAL|INSUFFICIENT)\b",
        text.upper(),
    )


    unique = list(
        dict.fromkeys(
            matches
        )
    )


    if len(unique) == 1:
        return unique[0]


    return None


def evidence_refs(text):

    return set(
        re.findall(
            r"\[(E\d+)\]",
            text or "",
        )
    )


def wilson_interval(
    correct,
    total,
    z=1.96,
):

    if total == 0:

        return (
            0.0,
            0.0,
        )


    p = (
        correct
        / total
    )


    denominator = (
        1
        + z * z / total
    )


    center = (
        p
        + z * z / (
            2 * total
        )
    ) / denominator


    margin = (
        z
        * math.sqrt(
            (
                p * (
                    1 - p
                )
                / total
            )
            + (
                z * z
                / (
                    4 * total * total
                )
            )
        )
        / denominator
    )


    return (
        center - margin,
        center + margin,
    )


def write_json(
    path,
    value,
):

    with Path(
        path
    ).open(
        "w"
    ) as f:

        json.dump(
            value,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write(
            "\n"
        )


def write_jsonl(
    path,
    values,
):

    with Path(
        path
    ).open(
        "w"
    ) as f:

        for value in values:

            f.write(
                json.dumps(
                    value,
                    sort_keys=True,
                )
            )

            f.write(
                "\n"
            )


def main():

    args = parse_args()

    result_dir = Path(
        args.result_dir
    )


    if result_dir.exists():

        raise RuntimeError(
            "Result directory already exists. "
            "Refusing a second one-shot run."
        )


    # ================================================================
    # HASH GATES — BEFORE READING BENCHMARK CONTENT
    # ================================================================

    train_sha = sha256(
        TRAIN_FILE
    )

    challenge_sha = sha256(
        CHALLENGE_FILE
    )

    candidate_sha = sha256(
        CANDIDATE_MODEL
        / "adapter_model.safetensors"
    )


    print(
        "TRAIN SHA:",
        train_sha,
    )

    print(
        "Challenge V2 SHA:",
        challenge_sha,
    )

    print(
        "Candidate SHA:",
        candidate_sha,
    )


    if train_sha != EXPECTED_TRAIN_SHA:

        raise RuntimeError(
            "Frozen TRAIN hash mismatch."
        )


    if (
        challenge_sha
        != EXPECTED_CHALLENGE_SHA
    ):

        raise RuntimeError(
            "Challenge V2 hash mismatch."
        )


    if (
        candidate_sha
        != EXPECTED_CANDIDATE_SHA
    ):

        raise RuntimeError(
            "Frozen candidate hash mismatch."
        )


    # ================================================================
    # SCHEMA / EXTRACTION PREFLIGHT
    #
    # No model prediction occurs in this section.
    # ================================================================

    frozen_instruction = (
        read_frozen_instruction()
    )


    with CHALLENGE_FILE.open() as f:

        challenge_root = json.load(
            f
        )


    raw_cases = case_list_from_json(
        challenge_root
    )


    if len(
        raw_cases
    ) != 240:

        raise RuntimeError(
            "Expected exactly 240 Challenge V2 cases; "
            f"found {len(raw_cases)}."
        )


    cases = []


    for index, case in enumerate(
        raw_cases,
        start=1,
    ):

        cases.append(
            extract_case(
                case,
                index,
                frozen_instruction,
            )
        )


    missing_labels = sum(
        case[
            "gold_label"
        ] is None
        for case in cases
    )


    missing_user_input = sum(
        not case[
            "user_input"
        ].strip()
        for case in cases
    )


    if missing_labels:

        raise RuntimeError(
            f"{missing_labels} Challenge V2 cases "
            "lack mechanically extractable gold labels. "
            "NO MODEL EVALUATION HAS OCCURRED."
        )


    if missing_user_input:

        raise RuntimeError(
            f"{missing_user_input} Challenge V2 cases "
            "lack mechanically extractable model input. "
            "NO MODEL EVALUATION HAS OCCURRED."
        )


    label_counts = Counter(
        case[
            "gold_label"
        ]
        for case in cases
    )


    instruction_sources = Counter(
        case[
            "instruction_source"
        ]
        for case in cases
    )


    input_sources = Counter(
        case[
            "input_source"
        ]
        for case in cases
    )


    citation_ref_cases = sum(
        bool(
            evidence_refs(
                case[
                    "evidence"
                ]
            )
        )
        for case in cases
    )


    gold_citation_cases = sum(
        case[
            "gold_citations"
        ] is not None
        for case in cases
    )


    gold_condition_cases = sum(
        case[
            "gold_conditions"
        ] is not None
        for case in cases
    )


    print()
    print("=" * 88)
    print("CHALLENGE V2 PREFLIGHT")
    print("=" * 88)

    print(
        "cases:",
        len(
            cases
        ),
    )

    print(
        "label distribution:",
        dict(
            sorted(
                label_counts.items()
            )
        ),
    )

    print(
        "instruction sources:",
        dict(
            sorted(
                instruction_sources.items()
            )
        ),
    )

    print(
        "input sources:",
        dict(
            sorted(
                input_sources.items()
            )
        ),
    )

    print(
        "citation-reference evaluable cases:",
        citation_ref_cases,
    )

    print(
        "gold citation-set cases:",
        gold_citation_cases,
    )

    print(
        "gold conditions cases:",
        gold_condition_cases,
    )

    print()
    print(
        "PREFLIGHT PASS"
    )

    print(
        "NO MODEL PREDICTIONS HAVE YET BEEN MADE."
    )


    # ================================================================
    # LOAD TOKENIZER + FROZEN MODEL
    #
    # Still no benchmark prediction.
    # ================================================================

    print()
    print(
        "Loading frozen candidate tokenizer..."
    )


    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            str(
                CANDIDATE_MODEL
            )
        )
    )


    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    tokenizer.padding_side = (
        "left"
    )


    print(
        "Loading fresh Llama-3.1-8B base..."
    )


    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map={
                "":
                    0
            },
        )
    )


    print(
        "Loading exact frozen V5 adapter..."
    )


    model = (
        PeftModel
        .from_pretrained(
            base_model,
            str(
                CANDIDATE_MODEL
            ),
            is_trainable=False,
        )
    )


    model.eval()


    # ================================================================
    # RENDER FIXED PROMPTS
    # ================================================================

    prompts = []


    for case in cases:

        messages = [

            {
                "role":
                    "system",

                "content":
                    case[
                        "instruction"
                    ],
            },

            {
                "role":
                    "user",

                "content":
                    case[
                        "user_input"
                    ],
            },
        ]


        rendered = (
            tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


        if not isinstance(
            rendered,
            str,
        ):

            raise RuntimeError(
                "Chat template did not produce text. "
                "NO MODEL EVALUATION HAS OCCURRED."
            )


        prompts.append(
            rendered
        )


    # ================================================================
    # ONE-SHOT CONSUMPTION BEGINS HERE
    # ================================================================

    result_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


    started = {

        "status":
            "ONE_SHOT_GENERATION_STARTED",

        "challenge_sha256":
            challenge_sha,

        "candidate_sha256":
            candidate_sha,

        "cases":
            len(
                cases
            ),

        "batch_size":
            args.batch_size,

        "max_new_tokens":
            args.max_new_tokens,

        "decoding":
            "greedy_do_sample_false",
    }


    write_json(
        result_dir
        / "ONE_SHOT_STARTED.json",
        started,
    )


    print()
    print("=" * 88)
    print("ONE-SHOT CHALLENGE V2 GENERATION STARTED")
    print("=" * 88)


    generated_texts = []

    start_time = time.time()


    with torch.inference_mode():

        for start_index in range(
            0,
            len(cases),
            args.batch_size,
        ):

            end_index = min(
                start_index
                + args.batch_size,
                len(cases),
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
                    value.to(
                        "cuda"
                    )
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
                    args.max_new_tokens,

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


            texts = (
                tokenizer
                .batch_decode(
                    suffix,
                    skip_special_tokens=True,
                )
            )


            generated_texts.extend(
                text.strip()
                for text in texts
            )


            print(
                f"  {end_index}/"
                f"{len(cases)}"
            )


    elapsed = (
        time.time()
        - start_time
    )


    if len(
        generated_texts
    ) != len(
        cases
    ):

        raise RuntimeError(
            "Generation count mismatch AFTER "
            "one-shot consumption began."
        )


    # ================================================================
    # SCORE
    # ================================================================

    records = []


    for index, (
        case,
        text,
    ) in enumerate(
        zip(
            cases,
            generated_texts,
        ),
        start=1,
    ):

        parsed = strict_parse(
            text
        )


        strict_json = (
            parsed is not None
        )


        strict_decision = (
            parsed[
                "decision"
            ]
            if parsed is not None
            else None
        )


        lenient = (
            lenient_decision(
                text
            )
        )


        valid_evidence_refs = (
            evidence_refs(
                case[
                    "evidence"
                ]
            )
        )


        citation_ref_evaluable = (
            bool(
                valid_evidence_refs
            )
        )


        if (
            citation_ref_evaluable
            and parsed is not None
        ):

            predicted_citations = (
                parsed[
                    "citations"
                ]
            )


            citation_refs_valid = (
                len(
                    predicted_citations
                )
                == len(
                    set(
                        predicted_citations
                    )
                )
                and all(
                    citation
                    in valid_evidence_refs
                    for citation
                    in predicted_citations
                )
            )


        elif citation_ref_evaluable:

            citation_refs_valid = False


        else:

            citation_refs_valid = None


        gold_citations = (
            case[
                "gold_citations"
            ]
        )


        if gold_citations is not None:

            if parsed is None:

                exact_gold_citation_set = (
                    False
                )

            else:

                predicted_citations = (
                    parsed[
                        "citations"
                    ]
                )


                exact_gold_citation_set = (
                    len(
                        predicted_citations
                    )
                    == len(
                        gold_citations
                    )
                    and set(
                        predicted_citations
                    )
                    == set(
                        gold_citations
                    )
                )


        else:

            exact_gold_citation_set = (
                None
            )


        gold_conditions = (
            case[
                "gold_conditions"
            ]
        )


        if gold_conditions is not None:

            exact_gold_conditions = (
                parsed is not None
                and parsed[
                    "conditions"
                ]
                == gold_conditions
            )


        else:

            exact_gold_conditions = (
                None
            )


        records.append({

            "index":
                index,

            "case_id":
                case[
                    "case_id"
                ],

            "gold_label":
                case[
                    "gold_label"
                ],

            "generated_text":
                text,

            "strict_json":
                strict_json,

            "strict_decision":
                strict_decision,

            "lenient_decision":
                lenient,

            "decision_correct":
                (
                    strict_decision
                    == case[
                        "gold_label"
                    ]
                ),

            "lenient_decision_correct":
                (
                    lenient
                    == case[
                        "gold_label"
                    ]
                ),

            "citation_ref_evaluable":
                citation_ref_evaluable,

            "citation_refs_valid":
                citation_refs_valid,

            "gold_citation_evaluable":
                (
                    gold_citations
                    is not None
                ),

            "exact_gold_citation_set":
                exact_gold_citation_set,

            "gold_conditions_evaluable":
                (
                    gold_conditions
                    is not None
                ),

            "exact_gold_conditions":
                exact_gold_conditions,
        })


    total = len(
        records
    )


    strict_json_correct = sum(
        record[
            "strict_json"
        ]
        for record in records
    )


    decision_correct = sum(
        record[
            "decision_correct"
        ]
        for record in records
    )


    lenient_correct = sum(
        record[
            "lenient_decision_correct"
        ]
        for record in records
    )


    label_total = Counter()

    label_correct = Counter()


    for record in records:

        label = record[
            "gold_label"
        ]

        label_total[
            label
        ] += 1


        if record[
            "decision_correct"
        ]:

            label_correct[
                label
            ] += 1


    by_label = {}


    for label in LABELS:

        total_label = (
            label_total[
                label
            ]
        )

        correct_label = (
            label_correct[
                label
            ]
        )


        by_label[
            label
        ] = {

            "correct":
                correct_label,

            "total":
                total_label,

            "accuracy":
                (
                    correct_label
                    / total_label
                    if total_label
                    else None
                ),
        }


    observed_label_accuracies = [
        value[
            "accuracy"
        ]
        for value
        in by_label.values()
        if value[
            "accuracy"
        ] is not None
    ]


    worst_label_accuracy = min(
        observed_label_accuracies
    )


    confusion = {

        gold: {

            pred:
                0

            for pred in (
                *LABELS,
                "INVALID",
            )

        }

        for gold in LABELS
    }


    for record in records:

        gold = record[
            "gold_label"
        ]


        pred = (
            record[
                "strict_decision"
            ]
            if record[
                "strict_decision"
            ] in LABELS
            else "INVALID"
        )


        confusion[
            gold
        ][
            pred
        ] += 1


    citation_records = [
        record
        for record in records
        if record[
            "citation_ref_evaluable"
        ]
    ]


    citation_valid_count = sum(
        bool(
            record[
                "citation_refs_valid"
            ]
        )
        for record in citation_records
    )


    gold_citation_records = [
        record
        for record in records
        if record[
            "gold_citation_evaluable"
        ]
    ]


    exact_gold_citation_count = sum(
        bool(
            record[
                "exact_gold_citation_set"
            ]
        )
        for record in gold_citation_records
    )


    condition_records = [
        record
        for record in records
        if record[
            "gold_conditions_evaluable"
        ]
    ]


    exact_condition_count = sum(
        bool(
            record[
                "exact_gold_conditions"
            ]
        )
        for record in condition_records
    )


    low_ci, high_ci = (
        wilson_interval(
            decision_correct,
            total,
        )
    )


    metrics = {

        "benchmark":
            "Challenge V2",

        "challenge_sha256":
            challenge_sha,

        "candidate_sha256":
            candidate_sha,

        "rows":
            total,

        "decision_correct":
            decision_correct,

        "decision_accuracy":
            decision_correct
            / total,

        "decision_accuracy_wilson_95": [
            low_ci,
            high_ci,
        ],

        "lenient_decision_correct":
            lenient_correct,

        "lenient_decision_accuracy":
            lenient_correct
            / total,

        "strict_json_correct":
            strict_json_correct,

        "strict_json_validity":
            strict_json_correct
            / total,

        "worst_label_accuracy":
            worst_label_accuracy,

        "by_label":
            by_label,

        "confusion_matrix":
            confusion,

        "citation_reference_coverage":
            len(
                citation_records
            ),

        "citation_reference_valid_count":
            citation_valid_count,

        "citation_reference_validity":
            (
                citation_valid_count
                / len(
                    citation_records
                )
                if citation_records
                else None
            ),

        "gold_citation_coverage":
            len(
                gold_citation_records
            ),

        "exact_gold_citation_count":
            exact_gold_citation_count,

        "exact_gold_citation_set_accuracy":
            (
                exact_gold_citation_count
                / len(
                    gold_citation_records
                )
                if gold_citation_records
                else None
            ),

        "gold_conditions_coverage":
            len(
                condition_records
            ),

        "exact_gold_conditions_count":
            exact_condition_count,

        "exact_gold_conditions_accuracy":
            (
                exact_condition_count
                / len(
                    condition_records
                )
                if condition_records
                else None
            ),

        "generation_seconds":
            elapsed,

        "generation_policy": {
            "do_sample":
                False,

            "max_new_tokens":
                args.max_new_tokens,

            "batch_size":
                args.batch_size,

            "retry_count":
                0,
        },
    }


    write_jsonl(
        result_dir
        / "predictions.jsonl",
        records,
    )


    write_json(
        result_dir
        / "METRICS.json",
        metrics,
    )


    completed = {

        "status":
            "ONE_SHOT_COMPLETE",

        "challenge_sha256":
            challenge_sha,

        "candidate_sha256":
            candidate_sha,

        "decision_accuracy":
            metrics[
                "decision_accuracy"
            ],
    }


    write_json(
        result_dir
        / "ONE_SHOT_COMPLETED.json",
        completed,
    )


    # ================================================================
    # CONSOLE RESULT
    # ================================================================

    print()
    print("=" * 88)
    print("SENTINEL V5 — CHALLENGE V2 RESULT")
    print("=" * 88)


    print(
        "cases:",
        total,
    )


    print(
        "decision:",
        f"{decision_correct}/{total}",
        f"= {metrics['decision_accuracy']:.4f}",
    )


    print(
        "95% Wilson CI:",
        f"[{low_ci:.4f}, {high_ci:.4f}]",
    )


    print(
        "lenient decision:",
        f"{lenient_correct}/{total}",
        f"= {metrics['lenient_decision_accuracy']:.4f}",
    )


    print(
        "strict JSON:",
        f"{strict_json_correct}/{total}",
        f"= {metrics['strict_json_validity']:.4f}",
    )


    print()
    print(
        "BY LABEL"
    )


    for label in LABELS:

        value = by_label[
            label
        ]

        print(
            f"  {label:12s} "
            f"{value['correct']:3d}/"
            f"{value['total']:3d} "
            f"= "
            f"{value['accuracy']:.4f}"
        )


    print()
    print(
        "worst label:",
        f"{worst_label_accuracy:.4f}",
    )


    print()
    print(
        "citation-reference coverage:",
        len(
            citation_records
        ),
    )


    if citation_records:

        print(
            "citation-reference validity:",
            f"{citation_valid_count}/"
            f"{len(citation_records)} "
            f"= "
            f"{metrics['citation_reference_validity']:.4f}"
        )


    print(
        "gold citation-set coverage:",
        len(
            gold_citation_records
        ),
    )


    if gold_citation_records:

        print(
            "exact gold citation-set:",
            f"{exact_gold_citation_count}/"
            f"{len(gold_citation_records)} "
            f"= "
            f"{metrics['exact_gold_citation_set_accuracy']:.4f}"
        )


    print(
        "gold conditions coverage:",
        len(
            condition_records
        ),
    )


    if condition_records:

        print(
            "exact gold conditions:",
            f"{exact_condition_count}/"
            f"{len(condition_records)} "
            f"= "
            f"{metrics['exact_gold_conditions_accuracy']:.4f}"
        )


    print()
    print(
        "generation seconds:",
        round(
            elapsed,
            2,
        ),
    )


    print()
    print(
        "CHALLENGE V2 ONE-SHOT EVALUATION: COMPLETE"
    )

    print(
        "The benchmark is now CONSUMED for this experiment."
    )


if __name__ == "__main__":
    main()
