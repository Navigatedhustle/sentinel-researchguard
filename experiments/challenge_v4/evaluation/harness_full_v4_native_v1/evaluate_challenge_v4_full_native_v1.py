from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import gc
import hashlib
import json
import os
import random
import time


from strict_json_parser_v1 import (
    parse_completion,
)


SCRIPT_DIR = Path(
    __file__
).resolve().parent

POLICY_PATH = (
    SCRIPT_DIR
    / "DECODING_POLICY_V1.json"
)

PARSER_PATH = (
    SCRIPT_DIR
    / "strict_json_parser_v1.py"
)

TARGETS_PATH = Path(
    "/workspace/experiments/challenge_v4/evaluation/"
    "frozen_targets_v1/CHALLENGE_V4_TARGETS_V1.json"
)

LEDGER_PATH = Path(
    "/workspace/experiments/challenge_v4/"
    "CHALLENGE_V4_CONSUMPTION_LEDGER.txt"
)

RUNS_ROOT = Path(
    "/workspace/experiments/challenge_v4/evaluation/runs"
)


EXPECTED_TARGETS_SHA = "19aac1f9f5133f19b920251cf46af5286d16735b21ddd2e50938da7e40851e88"

EXPECTED_CANONICAL_SHA = "03c1a24c3a6e988cc30278a129e6aad3e83a95c9d3f35fb29604f67219bda4b6"

EXPECTED_BASE_CONFIG_SHA = (
    "29e4c210b0d6ac178b16b2a255a568bd"
    "b23b581e50ca1ef6a6d071dd85704e6e"
)

EXPECTED_BASE_SHARDS = {
    "model-00001-of-00004.safetensors":
        "2b1879f356aed350030bb40eb45ad362"
        "c89d9891096f79a3ab323d3ba5607668",

    "model-00002-of-00004.safetensors":
        "09d433f650646834a83c580877bd60c6"
        "d1f88f7755305c12576b5c7058f9af15",

    "model-00003-of-00004.safetensors":
        "fc1cdddd6bfa91128d6e94ee73d0ce62"
        "bfcdb7af29e978ddcab30c66ae9ea7fa",

    "model-00004-of-00004.safetensors":
        "92ecfe1a2414458b4821ac8c13cf8cb"
        "70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}


def sha256_file(path):
    path = Path(path)

    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_json(path):
    return json.loads(
        Path(path).read_text()
    )


def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def atomic_json_write(path, value):
    path = Path(path)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )

    os.replace(
        temporary,
        path,
    )


# =====================================================================
# FROZEN FILE VERIFICATION
# =====================================================================

def verify_frozen_runtime_files():

    if sha256_file(
        TARGETS_PATH
    ) != EXPECTED_TARGETS_SHA:
        raise RuntimeError(
            "Frozen target configuration SHA mismatch."
        )


    targets = load_json(
        TARGETS_PATH
    )


    canonical = Path(
        targets[
            "canonical_benchmark"
        ][
            "path"
        ]
    )


    if sha256_file(
        canonical
    ) != EXPECTED_CANONICAL_SHA:
        raise RuntimeError(
            "Canonical benchmark SHA mismatch."
        )


    tokenizer_spec = targets[
        "shared_tokenizer"
    ]

    tokenizer_dir = Path(
        tokenizer_spec[
            "path"
        ]
    )


    expected_tokenizer_files = {
        "tokenizer.json":
            tokenizer_spec[
                "tokenizer_sha256"
            ],

        "tokenizer_config.json":
            tokenizer_spec[
                "tokenizer_config_sha256"
            ],

        "chat_template.jinja":
            tokenizer_spec[
                "chat_template_sha256"
            ],
    }


    for name, expected_sha in (
        expected_tokenizer_files.items()
    ):

        actual = sha256_file(
            tokenizer_dir
            / name
        )

        if actual != expected_sha:
            raise RuntimeError(
                f"Shared tokenizer SHA mismatch: {name}"
            )


    return (
        targets,
        canonical,
        tokenizer_dir,
    )


def verify_model_identity(target):

    base = Path(
        target[
            "base_snapshot"
        ]
    )


    if sha256_file(
        base
        / "config.json"
    ) != EXPECTED_BASE_CONFIG_SHA:

        raise RuntimeError(
            "Base config SHA mismatch."
        )


    for name, expected_sha in (
        EXPECTED_BASE_SHARDS.items()
    ):

        actual = sha256_file(
            base
            / name
        )

        if actual != expected_sha:
            raise RuntimeError(
                f"Base weight SHA mismatch: {name}"
            )


    adapter = target.get(
        "adapter"
    )


    if adapter is None:
        return


    adapter_path = Path(
        adapter
    )


    if sha256_file(
        adapter_path
        / "adapter_model.safetensors"
    ) != target[
        "adapter_model_sha256"
    ]:

        raise RuntimeError(
            "Adapter model SHA mismatch."
        )


    if sha256_file(
        adapter_path
        / "adapter_config.json"
    ) != target[
        "adapter_config_sha256"
    ]:

        raise RuntimeError(
            "Adapter config SHA mismatch."
        )


# =====================================================================
# BLIND MODEL-FACING VIEW
# =====================================================================

def load_blind_cases(canonical_path):
    """
    This function is called only during an authorized real run.

    The full public row is immediately reduced to:

      case_id
      instruction
      input

    No scoring information survives into the inference loop.
    """

    payload = load_json(
        canonical_path
    )
    public_rows = (
        payload.get("cases")
        if isinstance(payload, dict)
        else None
    )


    if (
        not isinstance(
            public_rows,
            list,
        )
        or len(
            public_rows
        ) != 320
    ):
        raise RuntimeError(
            "Unexpected canonical row count."
        )


    blind = []

    seen = set()


    for case in public_rows:

        case_id = case[
            "case_id"
        ]


        if case_id in seen:
            raise RuntimeError(
                "Duplicate case_id."
            )


        seen.add(
            case_id
        )


        blind.append({
            "case_id":
                case_id,

            "instruction":
                case["messages"][0]["content"],

            "input":
                case["messages"][1]["content"],
        })


    del public_rows

    return blind


def render_case(case, tokenizer):

    messages = [
        {
            "role": "system",
            "content":
                case[
                    "instruction"
                ],
        },
        {
            "role": "user",
            "content":
                case[
                    "input"
                ],
        },
    ]


    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# =====================================================================
# AUTHORIZATION TRIPWIRE
# =====================================================================

def verify_authorization(
    authorization_path,
    target_id,
):

    authorization_path = Path(
        authorization_path
    )


    authorization = load_json(
        authorization_path
    )


    required = {
        "version",
        "status",
        "run_id",
        "target_id",
        "output_dir",
        "canonical_sha256",
        "targets_config_sha256",
        "decoding_policy_sha256",
        "parser_sha256",
        "evaluator_sha256",
        "ledger_sha256",
    }


    if set(
        authorization.keys()
    ) != required:

        raise RuntimeError(
            "Authorization schema mismatch."
        )


    if authorization[
        "version"
    ] != 1:

        raise RuntimeError(
            "Authorization version mismatch."
        )


    if authorization[
        "status"
    ] != "AUTHORIZED_FOR_INFERENCE":

        raise RuntimeError(
            "Inference not authorized."
        )


    if authorization[
        "target_id"
    ] != target_id:

        raise RuntimeError(
            "Authorization target mismatch."
        )


    if authorization[
        "canonical_sha256"
    ] != EXPECTED_CANONICAL_SHA:

        raise RuntimeError(
            "Authorization canonical SHA mismatch."
        )


    if authorization[
        "targets_config_sha256"
    ] != EXPECTED_TARGETS_SHA:

        raise RuntimeError(
            "Authorization target-config SHA mismatch."
        )


    if authorization[
        "decoding_policy_sha256"
    ] != sha256_file(
        POLICY_PATH
    ):

        raise RuntimeError(
            "Authorization decoding-policy SHA mismatch."
        )


    if authorization[
        "parser_sha256"
    ] != sha256_file(
        PARSER_PATH
    ):

        raise RuntimeError(
            "Authorization parser SHA mismatch."
        )


    if authorization[
        "evaluator_sha256"
    ] != sha256_file(
        Path(__file__)
    ):

        raise RuntimeError(
            "Authorization evaluator SHA mismatch."
        )


    current_ledger_sha = sha256_file(
        LEDGER_PATH
    )


    if authorization[
        "ledger_sha256"
    ] != current_ledger_sha:

        raise RuntimeError(
            "Authorization ledger SHA mismatch."
        )


    output_dir = Path(
        authorization[
            "output_dir"
        ]
    ).resolve()


    runs_root = RUNS_ROOT.resolve()


    if not output_dir.is_relative_to(
        runs_root
    ):

        raise RuntimeError(
            "Authorized output path outside runs root."
        )


    if output_dir.exists():

        raise RuntimeError(
            "Authorized output directory already exists."
        )


    return (
        authorization,
        authorization_path,
        output_dir,
    )


# =====================================================================
# SYNTHETIC-ONLY SELF TEST
# =====================================================================

def synthetic_self_test():
    """
    No canonical case parsing.
    No model load.
    No inference.
    """

    targets, _, tokenizer_dir = (
        verify_frozen_runtime_files()
    )

    policy = load_json(
        POLICY_PATH
    )


    assert policy[
        "batch_size"
    ] == 1

    assert policy[
        "do_sample"
    ] is False

    assert policy[
        "num_beams"
    ] == 1

    assert policy[
        "max_new_tokens"
    ] == 256

    assert policy[
        "pad_token_id"
    ] == 128009


    valid = parse_completion(
        '{"decision":"YES","conditions":[],"citations":[],"rationale":"Allowed."}'
    )

    assert valid[
        "json_valid"
    ]

    assert valid[
        "decision_valid"
    ]

    assert valid[
        "contract_valid"
    ]


    fenced = parse_completion(
        '```json\n'
        '{"decision":"YES"}'
        '\n```'
    )

    assert not fenced[
        "json_valid"
    ]


    prefixed = parse_completion(
        'Answer: {"decision":"YES"}'
    )

    assert not prefixed[
        "json_valid"
    ]


    partial = parse_completion(
        '{"decision":"NO"}'
    )

    assert partial[
        "decision_valid"
    ]

    assert not partial[
        "contract_valid"
    ]


    from transformers import (
        AutoTokenizer,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        str(
            tokenizer_dir
        ),
        local_files_only=True,
    )


    synthetic_case = {
        "case_id":
            "SYNTHETIC_ONLY",

        "instruction":
            "Use only supplied evidence and return valid JSON.",

        "input":
            "Evidence: Synthetic policy permits action A.\n"
            "Question: Is action A permitted?",
    }


    rendered = render_case(
        synthetic_case,
        tokenizer,
    )


    ids = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=False,
    )[
        "input_ids"
    ]


    if len(
        ids
    ) <= 2:

        raise RuntimeError(
            "Synthetic rendered input has implausible token count."
        )


    if targets[
        "consumption_state"
    ] != "UNCONSUMED":

        raise RuntimeError(
            "Frozen target config consumption state changed."
        )


    print(
        "Parser valid-contract test: PASS"
    )

    print(
        "Parser fenced-output rejection: PASS"
    )

    print(
        "Parser prefixed-output rejection: PASS"
    )

    print(
        "Parser decision/contract separation: PASS"
    )

    print(
        "Shared tokenizer synthetic render: PASS"
    )

    print(
        "Synthetic rendered token count:",
        len(
            ids
        ),
    )

    print(
        "Canonical case content read: 0"
    )

    print(
        "Models loaded: 0"
    )

    print(
        "Inference performed: 0"
    )


# =====================================================================
# SCORING
# =====================================================================

def score_outputs(
    canonical_path,
    raw_rows,
):
    """
    This function runs only AFTER model inference completes and
    the model has been unloaded.
    """

    canonical = load_json(
        canonical_path
    )["cases"]


    output_by_id = {
        row[
            "case_id"
        ]:
            row
        for row in raw_rows
    }


    if len(
        output_by_id
    ) != 320:

        raise RuntimeError(
            "Expected 240 unique outputs."
        )


    total = 0
    correct = 0

    json_valid_count = 0
    decision_valid_count = 0
    contract_valid_count = 0

    predicted_labels = Counter()

    by_question_type = defaultdict(
        lambda: [
            0,
            0,
        ]
    )

    by_composition = defaultdict(
        lambda: [
            0,
            0,
        ]
    )

    scored_rows = []


    for case in canonical:

        case_id = case[
            "case_id"
        ]


        result = output_by_id[
            case_id
        ]


        gold_decision = case[
            "gold"
        ]


        predicted = (
            result[
                "decision"
            ]
            if result[
                "decision_valid"
            ]
            else None
        )


        is_correct = (
            predicted
            == gold_decision
        )


        total += 1

        correct += int(
            is_correct
        )

        json_valid_count += int(
            result[
                "json_valid"
            ]
        )

        decision_valid_count += int(
            result[
                "decision_valid"
            ]
        )

        contract_valid_count += int(
            result[
                "contract_valid"
            ]
        )


        if predicted is not None:

            predicted_labels[
                predicted
            ] += 1


        question_type = case[
            "question_type"
        ]

        composition = case[
            "composition"
        ]


        by_question_type[
            question_type
        ][0] += 1

        by_question_type[
            question_type
        ][1] += int(
            is_correct
        )


        by_composition[
            composition
        ][0] += 1

        by_composition[
            composition
        ][1] += int(
            is_correct
        )


        scored_rows.append({
            "case_id":
                case_id,

            "gold_decision":
                gold_decision,

            "predicted_decision":
                predicted,

            "correct":
                is_correct,

            "json_valid":
                result[
                    "json_valid"
                ],

            "decision_valid":
                result[
                    "decision_valid"
                ],

            "contract_valid":
                result[
                    "contract_valid"
                ],
        })


    if total != 320:

        raise RuntimeError(
            "Scoring row count mismatch."
        )


    summary = {
        "cases":
            total,

        "correct":
            correct,

        "decision_accuracy":
            correct
            / total,

        "json_valid_count":
            json_valid_count,

        "json_valid_rate":
            json_valid_count
            / total,

        "decision_valid_count":
            decision_valid_count,

        "decision_valid_rate":
            decision_valid_count
            / total,

        "contract_valid_count":
            contract_valid_count,

        "contract_valid_rate":
            contract_valid_count
            / total,

        "predicted_label_counts":
            dict(
                sorted(
                    predicted_labels.items()
                )
            ),

        "by_question_type": {
            name: {
                "cases":
                    values[
                        0
                    ],

                "correct":
                    values[
                        1
                    ],

                "accuracy":
                    values[
                        1
                    ]
                    / values[
                        0
                    ],
            }

            for name, values in sorted(
                by_question_type.items()
            )
        },

        "by_composition": {
            name: {
                "cases":
                    values[
                        0
                    ],

                "correct":
                    values[
                        1
                    ],

                "accuracy":
                    values[
                        1
                    ]
                    / values[
                        0
                    ],
            }

            for name, values in sorted(
                by_composition.items()
            )
        },
    }


    return (
        summary,
        scored_rows,
    )


# =====================================================================
# REAL EVALUATION
# =====================================================================

def run_real_evaluation(
    target_id,
    authorization_path,
):

    os.environ[
        "HF_HUB_OFFLINE"
    ] = "1"

    os.environ[
        "TRANSFORMERS_OFFLINE"
    ] = "1"


    targets_config, canonical, tokenizer_dir = (
        verify_frozen_runtime_files()
    )


    policy = load_json(
        POLICY_PATH
    )


    matches = [
        target
        for target in targets_config[
            "targets"
        ]
        if target[
            "target_id"
        ]
        == target_id
    ]


    if len(
        matches
    ) != 1:

        raise RuntimeError(
            "Unknown or duplicate target."
        )


    target = matches[
        0
    ]


    (
        authorization,
        authorization_file,
        output_dir,
    ) = verify_authorization(
        authorization_path,
        target_id,
    )


    verify_model_identity(
        target
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


    metadata_path = (
        output_dir
        / "RUN_METADATA.json"
    )

    raw_path = (
        output_dir
        / "RAW_OUTPUTS.jsonl"
    )

    score_path = (
        output_dir
        / "SCORE_SUMMARY.json"
    )

    scored_path = (
        output_dir
        / "SCORED_CASES.jsonl"
    )


    metadata = {
        "run_id":
            authorization[
                "run_id"
            ],

        "target_id":
            target_id,

        "status":
            "PREFLIGHT_COMPLETE_NO_INFERENCE_YET",

        "started_utc":
            utc_now(),

        "canonical_sha256":
            EXPECTED_CANONICAL_SHA,

        "targets_config_sha256":
            EXPECTED_TARGETS_SHA,

        "authorization_sha256":
            sha256_file(
                authorization_file
            ),

        "ledger_sha256_at_start":
            sha256_file(
                LEDGER_PATH
            ),

        "evaluator_sha256":
            sha256_file(
                Path(__file__)
            ),

        "parser_sha256":
            sha256_file(
                PARSER_PATH
            ),

        "decoding_policy_sha256":
            sha256_file(
                POLICY_PATH
            ),
    }


    atomic_json_write(
        metadata_path,
        metadata,
    )


    # ------------------------------------------------------------
    # Canonical data reduced immediately to blind inference fields.
    # ------------------------------------------------------------

    blind_cases = load_blind_cases(
        canonical
    )


    # ------------------------------------------------------------
    # Heavy imports only after authorization.
    # ------------------------------------------------------------

    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from peft import (
        PeftModel,
    )


    random.seed(
        policy[
            "seed"
        ]
    )

    torch.manual_seed(
        policy[
            "seed"
        ]
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            policy[
                "seed"
            ]
        )


    tokenizer = AutoTokenizer.from_pretrained(
        str(
            tokenizer_dir
        ),
        local_files_only=True,
    )


    if tokenizer.pad_token_id != policy[
        "pad_token_id"
    ]:

        raise RuntimeError(
            "Tokenizer pad token disagrees with policy."
        )


    model = AutoModelForCausalLM.from_pretrained(
        target[
            "base_snapshot"
        ],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map=policy[
            "device_map"
        ],
        low_cpu_mem_usage=True,
    )


    adapter_path = target.get(
        "adapter"
    )


    if adapter_path is not None:

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )


    model.eval()

    torch.set_grad_enabled(
        False
    )


    metadata[
        "status"
    ] = "MODEL_LOADED_INFERENCE_STARTED"

    metadata[
        "model_loaded_utc"
    ] = utc_now()


    atomic_json_write(
        metadata_path,
        metadata,
    )


    generation_kwargs = {
        "do_sample":
            policy[
                "do_sample"
            ],

        "num_beams":
            policy[
                "num_beams"
            ],

        "max_new_tokens":
            policy[
                "max_new_tokens"
            ],

        "use_cache":
            policy[
                "use_cache"
            ],

        "pad_token_id":
            policy[
                "pad_token_id"
            ],

        "eos_token_id":
            policy[
                "eos_token_id"
            ],
    }


    raw_rows = []


    with raw_path.open(
        "x",
        encoding="utf-8",
    ) as out:


        for index, case in enumerate(
            blind_cases,
            start=1,
        ):

            rendered = render_case(
                case,
                tokenizer,
            )


            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                truncation=False,
                return_tensors="pt",
            )


            device = next(
                model.parameters()
            ).device


            encoded = {
                key:
                    value.to(
                        device
                    )

                for key, value in (
                    encoded.items()
                )
            }


            input_length = int(
                encoded[
                    "input_ids"
                ].shape[
                    1
                ]
            )


            started = time.perf_counter()


            with torch.inference_mode():

                generated = model.generate(
                    **encoded,
                    **generation_kwargs,
                )


            if torch.cuda.is_available():

                torch.cuda.synchronize()


            latency = (
                time.perf_counter()
                - started
            )


            completion_ids = (
                generated[
                    0,
                    input_length:
                ]
                .detach()
                .cpu()
                .tolist()
            )


            raw_completion = tokenizer.decode(
                completion_ids,

                skip_special_tokens=
                    policy[
                        "decode"
                    ][
                        "skip_special_tokens"
                    ],

                clean_up_tokenization_spaces=
                    policy[
                        "decode"
                    ][
                        "clean_up_tokenization_spaces"
                    ],
            )


            parsed = parse_completion(
                raw_completion
            )


            row = {
                "case_id":
                    case[
                        "case_id"
                    ],

                "target_id":
                    target_id,

                "prompt_tokens":
                    input_length,

                "completion_tokens":
                    len(
                        completion_ids
                    ),

                "hit_max_new_tokens":
                    len(
                        completion_ids
                    )
                    >= policy[
                        "max_new_tokens"
                    ],

                "latency_seconds":
                    latency,

                "completion_token_ids":
                    completion_ids,

                "raw_completion":
                    raw_completion,

                "json_valid":
                    parsed[
                        "json_valid"
                    ],

                "decision_valid":
                    parsed[
                        "decision_valid"
                    ],

                "contract_valid":
                    parsed[
                        "contract_valid"
                    ],

                "decision":
                    parsed[
                        "decision"
                    ],

                "parser_error":
                    parsed[
                        "error"
                    ],
            }


            raw_rows.append(
                row
            )


            out.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )

            out.flush()

            os.fsync(
                out.fileno()
            )


            if (
                index % 20
                == 0
                or index == 240
            ):

                print(
                    f"{target_id}: "
                    f"{index}/240 cases generated"
                )


    # ------------------------------------------------------------
    # Unload model BEFORE scoring against gold.
    # ------------------------------------------------------------

    del model

    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.synchronize()


    metadata[
        "status"
    ] = "INFERENCE_COMPLETE_MODEL_UNLOADED"

    metadata[
        "inference_completed_utc"
    ] = utc_now()

    metadata[
        "outputs_generated"
    ] = len(
        raw_rows
    )


    atomic_json_write(
        metadata_path,
        metadata,
    )


    # ------------------------------------------------------------
    # Gold scoring only after model unload.
    # ------------------------------------------------------------

    summary, scored_rows = score_outputs(
        canonical,
        raw_rows,
    )


    summary[
        "run_id"
    ] = authorization[
        "run_id"
    ]

    summary[
        "target_id"
    ] = target_id

    summary[
        "canonical_sha256"
    ] = EXPECTED_CANONICAL_SHA

    summary[
        "raw_outputs_sha256"
    ] = sha256_file(
        raw_path
    )


    atomic_json_write(
        score_path,
        summary,
    )


    with scored_path.open(
        "x",
        encoding="utf-8",
    ) as f:

        for row in scored_rows:

            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
                + "\n"
            )


    metadata[
        "status"
    ] = "COMPLETE"

    metadata[
        "completed_utc"
    ] = utc_now()

    metadata[
        "raw_outputs_sha256"
    ] = sha256_file(
        raw_path
    )

    metadata[
        "score_summary_sha256"
    ] = sha256_file(
        score_path
    )

    metadata[
        "scored_cases_sha256"
    ] = sha256_file(
        scored_path
    )


    atomic_json_write(
        metadata_path,
        metadata,
    )


    print()
    print("Evaluation complete.")
    print("Target:", target_id)

    print(
        "Decision accuracy:",
        f"{summary['decision_accuracy']:.4f}",
    )

    print(
        "JSON-valid rate:",
        f"{summary['json_valid_rate']:.4f}",
    )

    print(
        "Contract-valid rate:",
        f"{summary['contract_valid_rate']:.4f}",
    )

    print(
        "Output directory:",
        output_dir,
    )


# =====================================================================
# CLI
# =====================================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--self-test",
        action="store_true",
    )


    parser.add_argument(
        "--target",
        choices=[
            "base_llama31_8b",
            "sentinel_v5_epoch2",
            "sentinel_v6_epoch1",
        ],
    )


    parser.add_argument(
        "--authorization-file",
    )


    args = parser.parse_args()


    if args.self_test:

        if (
            args.target is not None
            or args.authorization_file
            is not None
        ):

            raise RuntimeError(
                "--self-test cannot be combined with real-run arguments."
            )


        synthetic_self_test()

        return


    if args.target is None:

        raise RuntimeError(
            "--target required for real evaluation."
        )


    if args.authorization_file is None:

        raise RuntimeError(
            "--authorization-file mandatory for real evaluation."
        )


    run_real_evaluation(
        args.target,
        args.authorization_file,
    )


if __name__ == "__main__":
    main()
