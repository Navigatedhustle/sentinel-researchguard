from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json


def sha256_file(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def load_json(path):
    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def load_jsonl(path):
    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8"
    ) as f:

        for lineno, line in enumerate(
            f,
            start=1
        ):

            if not line.strip():
                raise RuntimeError(
                    f"Blank JSONL row at line {lineno}."
                )

            value = json.loads(line)

            if not isinstance(value, dict):
                raise RuntimeError(
                    f"Non-object JSONL row at line {lineno}."
                )

            rows.append(value)

    return rows


def write_json(path, value):
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True
        )
        + "\n",
        encoding="utf-8"
    )


def write_jsonl(path, rows):
    with Path(path).open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True
                )
                + "\n"
            )


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



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--canonical",
        required=True,
    )

    parser.add_argument(
        "--raw",
        required=True,
    )

    parser.add_argument(
        "--ledger",
        required=True,
    )

    parser.add_argument(
        "--authorization",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()


    canonical = Path(
        args.canonical
    ).resolve()

    raw = Path(
        args.raw
    ).resolve()

    ledger = Path(
        args.ledger
    ).resolve()

    authorization = Path(
        args.authorization
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()


    auth = load_json(
        authorization
    )


    required = {
        "version",
        "status",
        "run_id",
        "target_id",
        "canonical_sha256",
        "raw_outputs_sha256",
        "scorer_sha256",
        "ledger_sha256",
        "output_dir",
    }


    missing = required - set(auth)


    if missing:
        raise RuntimeError(
            "Authorization missing: "
            + repr(sorted(missing))
        )


    if auth["version"] != 1:
        raise RuntimeError(
            "Authorization version mismatch."
        )


    if (
        auth["status"]
        !=
        "AUTHORIZED_FOR_OFFLINE_SCORING"
    ):
        raise RuntimeError(
            "Offline scoring not authorized."
        )


    if (
        auth["run_id"]
        !=
        "challenge_v4_base_run01_recovery02_offline_scoring01"
    ):
        raise RuntimeError(
            "Run ID mismatch."
        )


    if (
        auth["target_id"]
        !=
        "base_llama31_8b"
    ):
        raise RuntimeError(
            "Target mismatch."
        )


    if (
        auth["canonical_sha256"]
        !=
        sha256_file(canonical)
    ):
        raise RuntimeError(
            "Canonical SHA mismatch."
        )


    if (
        auth["raw_outputs_sha256"]
        !=
        sha256_file(raw)
    ):
        raise RuntimeError(
            "RAW SHA mismatch."
        )


    if (
        auth["scorer_sha256"]
        !=
        sha256_file(
            Path(__file__).resolve()
        )
    ):
        raise RuntimeError(
            "Scorer SHA mismatch."
        )


    if (
        auth["ledger_sha256"]
        !=
        sha256_file(ledger)
    ):
        raise RuntimeError(
            "Ledger SHA mismatch."
        )


    if (
        auth["output_dir"]
        !=
        str(output_dir)
    ):
        raise RuntimeError(
            "Output-directory mismatch."
        )


    if output_dir.exists():
        raise RuntimeError(
            "Output directory already exists."
        )


    raw_rows = load_jsonl(raw)


    if len(raw_rows) != 320:
        raise RuntimeError(
            "RAW row count != 320."
        )


    summary, scored_rows = score_outputs(
        canonical,
        raw_rows,
    )


    if len(scored_rows) != 320:
        raise RuntimeError(
            "Scored rows != 320."
        )


    if summary.get("cases") != 320:
        raise RuntimeError(
            "Summary cases != 320."
        )


    output_dir.mkdir(
        parents=True
    )


    scored_path = (
        output_dir
        / "SCORED_CASES.jsonl"
    )

    summary_path = (
        output_dir
        / "SCORE_SUMMARY.json"
    )

    metadata_path = (
        output_dir
        / "SCORING_METADATA.json"
    )


    write_jsonl(
        scored_path,
        scored_rows
    )

    write_json(
        summary_path,
        summary
    )


    metadata = {
        "status":
            "OFFLINE_SCORING_COMPLETE",

        "run_id":
            auth["run_id"],

        "target_id":
            auth["target_id"],

        "canonical_sha256":
            sha256_file(canonical),

        "raw_outputs_sha256":
            sha256_file(raw),

        "scorer_sha256":
            sha256_file(
                Path(__file__).resolve()
            ),

        "authorization_sha256":
            sha256_file(
                authorization
            ),

        "authorized_ledger_sha256":
            auth["ledger_sha256"],

        "scored_rows":
            320,

        "model_loads":
            0,

        "generation_calls":
            0,

        "hidden_semantics_read":
            False,
    }


    write_json(
        metadata_path,
        metadata
    )


    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
