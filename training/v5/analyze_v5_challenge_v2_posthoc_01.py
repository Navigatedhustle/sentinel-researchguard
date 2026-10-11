"""
Sentinel V5 Challenge V2 post-hoc analysis 01.

Purpose:
- analyze the already-frozen one-shot Challenge V2 predictions
- find aggregate error concentrations
- characterize citation-selection errors
- characterize condition-ledger errors
- compare frozen DEVELOPMENT vs Challenge V2

Important:
- does NOT load a model
- does NOT generate text
- does NOT train
- does NOT modify predictions
- does NOT display individual benchmark cases
"""

from collections import Counter, defaultdict
from pathlib import Path

import argparse
import hashlib
import json
import re


BASE = Path("/workspace")

CHALLENGE = (
    BASE
    / "experiments/challenge_v2/"
      "sentinel_challenge_v2.json"
)

PREDICTIONS = (
    BASE
    / "experiments/v5/"
      "challenge_v2_one_shot_run01/"
      "predictions.jsonl"
)

METRICS = (
    BASE
    / "experiments/v5/"
      "challenge_v2_one_shot_run01/"
      "METRICS.json"
)

DEV_SUMMARY = (
    BASE
    / "experiments/v5/"
      "development_eval_run01/"
      "SUMMARY.json"
)


EXPECTED_CHALLENGE_SHA = (
    "30eb37ff39382f0d205adb7c17383b240"
    "25601ea3db1dd61f22b715af6c1e482"
)


LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
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


def normalize_text(value):

    return " ".join(
        str(value)
        .strip()
        .split()
    )


def evidence_refs(text):

    return set(
        re.findall(
            r"\[(E\d+)\]",
            text or "",
        )
    )


def safe_json(text):

    try:

        value = json.loads(
            text
        )

    except Exception:

        return None

    return value


def accuracy_table(
    rows,
    key,
    *,
    multi=False,
):

    totals = Counter()
    correct = Counter()
    errors = Counter()


    for row in rows:

        raw = row[
            "case"
        ].get(
            key
        )


        if multi:

            values = raw

            if not isinstance(
                values,
                list,
            ):

                values = []


            if not values:

                values = [
                    "<NONE>"
                ]


        else:

            if raw is None:

                values = [
                    "<NULL>"
                ]

            else:

                values = [
                    str(
                        raw
                    )
                ]


        for value in values:

            value = str(
                value
            )

            totals[
                value
            ] += 1


            if row[
                "prediction"
            ][
                "decision_correct"
            ]:

                correct[
                    value
                ] += 1

            else:

                errors[
                    value
                ] += 1


    result = []


    for value in totals:

        total = totals[
            value
        ]

        right = correct[
            value
        ]

        wrong = errors[
            value
        ]


        result.append({

            "value":
                value,

            "correct":
                right,

            "errors":
                wrong,

            "total":
                total,

            "accuracy":
                right
                / total,

            "error_rate":
                wrong
                / total,

            "share_of_all_errors":
                wrong
                / 24,
        })


    result.sort(
        key=lambda x: (
            x[
                "accuracy"
            ],
            -x[
                "total"
            ],
            x[
                "value"
            ],
        )
    )


    return result


def citation_taxonomy(
    gold,
    predicted,
    valid_refs,
):

    gold = list(
        gold
    )

    predicted = list(
        predicted
    )


    gold_set = set(
        gold
    )

    pred_set = set(
        predicted
    )


    invalid = sorted(
        pred_set
        - valid_refs
    )


    missing = sorted(
        gold_set
        - pred_set
    )


    extra = sorted(
        pred_set
        - gold_set
    )


    if invalid:

        primary = (
            "invalid_reference"
        )


    elif predicted == gold:

        primary = (
            "exact_list"
        )


    elif (
        len(
            predicted
        )
        == len(
            gold
        )
        and pred_set
        == gold_set
    ):

        primary = (
            "same_set_different_order"
        )


    elif (
        missing
        and not extra
    ):

        primary = (
            "missing_gold_citation_only"
        )


    elif (
        extra
        and not missing
    ):

        primary = (
            "extra_citation_only"
        )


    elif (
        missing
        and extra
    ):

        primary = (
            "missing_and_extra"
        )


    else:

        primary = (
            "other"
        )


    return {

        "primary":
            primary,

        "missing_count":
            len(
                missing
            ),

        "extra_count":
            len(
                extra
            ),

        "invalid_count":
            len(
                invalid
            ),

        "exact_set":
            (
                len(
                    predicted
                )
                == len(
                    gold
                )
                and pred_set
                == gold_set
            ),
    }


def condition_taxonomy(
    gold,
    predicted,
):

    gold = list(
        gold
    )

    predicted = list(
        predicted
    )


    if predicted == gold:

        return {
            "primary":
                "exact_list",

            "gold_count":
                len(
                    gold
                ),

            "predicted_count":
                len(
                    predicted
                ),

            "overlap_count":
                len(
                    gold
                ),
        }


    gold_norm = [
        normalize_text(
            value
        )
        for value in gold
    ]

    pred_norm = [
        normalize_text(
            value
        )
        for value in predicted
    ]


    if pred_norm == gold_norm:

        return {
            "primary":
                "whitespace_only_difference",

            "gold_count":
                len(
                    gold
                ),

            "predicted_count":
                len(
                    predicted
                ),

            "overlap_count":
                len(
                    gold
                ),
        }


    gold_set = set(
        gold_norm
    )

    pred_set = set(
        pred_norm
    )


    missing = (
        gold_set
        - pred_set
    )

    extra = (
        pred_set
        - gold_set
    )

    overlap = (
        gold_set
        & pred_set
    )


    if (
        len(
            gold_norm
        )
        == len(
            pred_norm
        )
        and gold_set
        == pred_set
    ):

        primary = (
            "same_set_different_order"
        )


    elif (
        missing
        and not extra
    ):

        primary = (
            "missing_gold_condition_only"
        )


    elif (
        extra
        and not missing
    ):

        primary = (
            "extra_condition_only"
        )


    elif (
        missing
        and extra
    ):

        primary = (
            "missing_and_extra_or_reworded"
        )


    else:

        primary = (
            "other"
        )


    return {

        "primary":
            primary,

        "gold_count":
            len(
                gold
            ),

        "predicted_count":
            len(
                predicted
            ),

        "overlap_count":
            len(
                overlap
            ),
    }


def print_group(
    title,
    rows,
    *,
    limit=30,
):

    print()
    print(
        "=" * 88
    )

    print(
        title
    )

    print(
        "=" * 88
    )


    print(
        f"{'group':42s} "
        f"{'correct':>8s} "
        f"{'total':>7s} "
        f"{'accuracy':>10s} "
        f"{'errors':>7s}"
    )


    for row in rows[
        :limit
    ]:

        print(
            f"{row['value'][:42]:42s} "
            f"{row['correct']:8d} "
            f"{row['total']:7d} "
            f"{row['accuracy']:10.4f} "
            f"{row['errors']:7d}"
        )


def main():

    args = parse_args()

    output_dir = Path(
        args.output_dir
    )


    if output_dir.exists():

        raise RuntimeError(
            "Output directory already exists."
        )


    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )


    # ================================================================
    # LOAD / VERIFY FROZEN INPUTS
    # ================================================================

    challenge_sha = sha256(
        CHALLENGE
    )


    if (
        challenge_sha
        != EXPECTED_CHALLENGE_SHA
    ):

        raise RuntimeError(
            "Challenge V2 hash mismatch."
        )


    challenge = json.loads(
        CHALLENGE.read_text()
    )


    if not isinstance(
        challenge,
        list,
    ):

        raise RuntimeError(
            "Challenge V2 root is not a list."
        )


    predictions = load_jsonl(
        PREDICTIONS
    )


    metrics = json.loads(
        METRICS.read_text()
    )


    dev = json.loads(
        DEV_SUMMARY.read_text()
    )


    if len(
        challenge
    ) != 240:

        raise RuntimeError(
            "Challenge case count mismatch."
        )


    if len(
        predictions
    ) != 240:

        raise RuntimeError(
            "Prediction count mismatch."
        )


    if metrics[
        "rows"
    ] != 240:

        raise RuntimeError(
            "Frozen METRICS row count mismatch."
        )


    if metrics[
        "decision_correct"
    ] != 216:

        raise RuntimeError(
            "Expected frozen headline result 216/240."
        )


    if metrics[
        "strict_json_correct"
    ] != 240:

        raise RuntimeError(
            "Expected frozen strict JSON result 240/240."
        )


    # ================================================================
    # JOIN BY CASE ID
    # ================================================================

    case_map = {
        str(
            case[
                "case_id"
            ]
        ):
            case

        for case in challenge
    }


    if len(
        case_map
    ) != 240:

        raise RuntimeError(
            "Challenge case IDs are not unique."
        )


    joined = []


    for prediction in predictions:

        case_id = str(
            prediction[
                "case_id"
            ]
        )


        if case_id not in case_map:

            raise RuntimeError(
                "Prediction case ID missing from benchmark."
            )


        case = case_map[
            case_id
        ]


        gold = case[
            "gold"
        ]


        if (
            prediction[
                "gold_label"
            ]
            != gold[
                "decision"
            ]
        ):

            raise RuntimeError(
                "Gold decision mismatch during join."
            )


        generated = safe_json(
            prediction[
                "generated_text"
            ]
        )


        if not isinstance(
            generated,
            dict,
        ):

            raise RuntimeError(
                "Frozen strict-JSON result is inconsistent."
            )


        joined.append({

            "case":
                case,

            "prediction":
                prediction,

            "generated":
                generated,

            "gold":
                gold,
        })


    if len(
        joined
    ) != 240:

        raise RuntimeError(
            "Joined row count mismatch."
        )


    decision_errors = [
        row
        for row in joined
        if not row[
            "prediction"
        ][
            "decision_correct"
        ]
    ]


    if len(
        decision_errors
    ) != 24:

        raise RuntimeError(
            "Expected exactly 24 decision errors."
        )


    # ================================================================
    # DEVELOPMENT VS EXTERNAL
    # ================================================================

    dev_epoch2 = (
        dev[
            "models"
        ][
            "epoch2"
        ]
    )


    comparison = {

        "development": {

            "decision_accuracy":
                dev_epoch2[
                    "decision_accuracy"
                ],

            "strict_json_validity":
                dev_epoch2[
                    "strict_json_validity"
                ],

            "citation_reference_validity":
                dev_epoch2[
                    "citation_reference_validity"
                ],

            "exact_citation_set_accuracy":
                dev_epoch2[
                    "exact_citation_set_accuracy"
                ],

            "exact_conditions_accuracy":
                dev_epoch2[
                    "exact_conditions_accuracy"
                ],

            "worst_label_accuracy":
                dev_epoch2[
                    "worst_label_accuracy"
                ],
        },

        "challenge_v2": {

            "decision_accuracy":
                metrics[
                    "decision_accuracy"
                ],

            "strict_json_validity":
                metrics[
                    "strict_json_validity"
                ],

            "citation_reference_validity":
                metrics[
                    "citation_reference_validity"
                ],

            "exact_citation_set_accuracy":
                metrics[
                    "exact_gold_citation_set_accuracy"
                ],

            "exact_conditions_accuracy":
                metrics[
                    "exact_gold_conditions_accuracy"
                ],

            "worst_label_accuracy":
                metrics[
                    "worst_label_accuracy"
                ],
        },
    }


    gaps = {}


    for key in comparison[
        "development"
    ]:

        gaps[
            key
        ] = (
            comparison[
                "challenge_v2"
            ][
                key
            ]
            - comparison[
                "development"
            ][
                key
            ]
        )


    comparison[
        "challenge_minus_development"
    ] = gaps


    # ================================================================
    # CONFUSION MATRIX
    # ================================================================

    confusion = {

        gold: {
            pred:
                0

            for pred in LABELS
        }

        for gold in LABELS
    }


    for row in joined:

        gold_label = (
            row[
                "gold"
            ][
                "decision"
            ]
        )

        predicted = (
            row[
                "generated"
            ][
                "decision"
            ]
        )


        confusion[
            gold_label
        ][
            predicted
        ] += 1


    # ================================================================
    # GROUP ANALYSES
    # ================================================================

    group_tables = {

        "composition":
            accuracy_table(
                joined,
                "composition",
            ),

        "question_type":
            accuracy_table(
                joined,
                "question_type",
            ),

        "input_order":
            accuracy_table(
                joined,
                "input_order",
            ),

        "evidence_count":
            accuracy_table(
                joined,
                "evidence_count",
            ),

        "relational_group":
            accuracy_table(
                joined,
                "relational_group",
            ),

        "reasoning_tags":
            accuracy_table(
                joined,
                "reasoning_tags",
                multi=True,
            ),

        "policy_pack":
            accuracy_table(
                joined,
                "policy_pack",
            ),
    }


    # ================================================================
    # CITATION TAXONOMY
    # ================================================================

    citation_primary = Counter()

    citation_primary_by_gold = defaultdict(
        Counter
    )

    citation_missing_counts = Counter()

    citation_extra_counts = Counter()


    for row in joined:

        valid_refs = evidence_refs(
            row[
                "case"
            ][
                "evidence_text"
            ]
        )


        taxon = citation_taxonomy(

            row[
                "gold"
            ][
                "citations"
            ],

            row[
                "generated"
            ][
                "citations"
            ],

            valid_refs,
        )


        citation_primary[
            taxon[
                "primary"
            ]
        ] += 1


        citation_primary_by_gold[
            row[
                "gold"
            ][
                "decision"
            ]
        ][
            taxon[
                "primary"
            ]
        ] += 1


        citation_missing_counts[
            taxon[
                "missing_count"
            ]
        ] += 1


        citation_extra_counts[
            taxon[
                "extra_count"
            ]
        ] += 1


    # ================================================================
    # CONDITION TAXONOMY
    # ================================================================

    condition_primary = Counter()

    condition_primary_by_gold = defaultdict(
        Counter
    )

    condition_gold_count = Counter()

    condition_pred_count = Counter()


    for row in joined:

        taxon = condition_taxonomy(

            row[
                "gold"
            ][
                "conditions"
            ],

            row[
                "generated"
            ][
                "conditions"
            ],
        )


        condition_primary[
            taxon[
                "primary"
            ]
        ] += 1


        condition_primary_by_gold[
            row[
                "gold"
            ][
                "decision"
            ]
        ][
            taxon[
                "primary"
            ]
        ] += 1


        condition_gold_count[
            taxon[
                "gold_count"
            ]
        ] += 1


        condition_pred_count[
            taxon[
                "predicted_count"
            ]
        ] += 1


    # ================================================================
    # DECISION ERROR RELATION TO EVIDENCE ACCOUNTING
    # ================================================================

    accounting_cross = Counter()


    for row in joined:

        valid_refs = evidence_refs(
            row[
                "case"
            ][
                "evidence_text"
            ]
        )


        citation = citation_taxonomy(

            row[
                "gold"
            ][
                "citations"
            ],

            row[
                "generated"
            ][
                "citations"
            ],

            valid_refs,
        )


        conditions = condition_taxonomy(

            row[
                "gold"
            ][
                "conditions"
            ],

            row[
                "generated"
            ][
                "conditions"
            ],
        )


        key = (

            (
                "decision_correct"
                if row[
                    "prediction"
                ][
                    "decision_correct"
                ]
                else "decision_wrong"
            ),

            (
                "citation_exact"
                if citation[
                    "exact_set"
                ]
                else "citation_not_exact"
            ),

            (
                "conditions_exact"
                if conditions[
                    "primary"
                ]
                == "exact_list"
                else "conditions_not_exact"
            ),
        )


        accounting_cross[
            key
        ] += 1


    # ================================================================
    # RESULT
    # ================================================================

    report = {

        "headline": {

            "cases":
                240,

            "correct":
                216,

            "errors":
                24,

            "decision_accuracy":
                0.90,

            "strict_json":
                1.0,
        },

        "development_comparison":
            comparison,

        "confusion_matrix":
            confusion,

        "group_tables":
            group_tables,

        "citation_taxonomy": {

            "primary":
                dict(
                    sorted(
                        citation_primary.items()
                    )
                ),

            "by_gold_label": {
                label:
                    dict(
                        sorted(
                            citation_primary_by_gold[
                                label
                            ].items()
                        )
                    )

                for label
                in LABELS
            },

            "missing_count_distribution":
                dict(
                    sorted(
                        citation_missing_counts.items()
                    )
                ),

            "extra_count_distribution":
                dict(
                    sorted(
                        citation_extra_counts.items()
                    )
                ),
        },

        "condition_taxonomy": {

            "primary":
                dict(
                    sorted(
                        condition_primary.items()
                    )
                ),

            "by_gold_label": {
                label:
                    dict(
                        sorted(
                            condition_primary_by_gold[
                                label
                            ].items()
                        )
                    )

                for label
                in LABELS
            },

            "gold_condition_count_distribution":
                dict(
                    sorted(
                        condition_gold_count.items()
                    )
                ),

            "predicted_condition_count_distribution":
                dict(
                    sorted(
                        condition_pred_count.items()
                    )
                ),
        },

        "decision_vs_accounting":
            {
                "|".join(
                    key
                ):
                    value

                for key, value
                in sorted(
                    accounting_cross.items()
                )
            },
    }


    write_json(
        output_dir
        / "AGGREGATE_ANALYSIS.json",
        report,
    )


    # ================================================================
    # HUMAN-READABLE CONSOLE REPORT
    # ================================================================

    print()
    print(
        "=" * 88
    )

    print(
        "HEADLINE"
    )

    print(
        "=" * 88
    )

    print(
        "Challenge V2 decision:",
        "216/240 = 90.00%"
    )

    print(
        "Decision errors:",
        len(
            decision_errors
        )
    )

    print(
        "Strict JSON:",
        "240/240 = 100.00%"
    )


    print()
    print(
        "=" * 88
    )

    print(
        "DEVELOPMENT → CHALLENGE V2"
    )

    print(
        "=" * 88
    )


    for key in (
        "decision_accuracy",
        "strict_json_validity",
        "citation_reference_validity",
        "exact_citation_set_accuracy",
        "exact_conditions_accuracy",
        "worst_label_accuracy",
    ):

        d = comparison[
            "development"
        ][
            key
        ]

        c = comparison[
            "challenge_v2"
        ][
            key
        ]

        gap = (
            c - d
        )


        print(
            f"{key:36s} "
            f"{d:8.4f} "
            f"→ "
            f"{c:8.4f} "
            f"gap={gap:+.4f}"
        )


    print()
    print(
        "=" * 88
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "=" * 88
    )


    print(
        f"{'gold':14s} "
        f"{'YES':>8s} "
        f"{'NO':>8s} "
        f"{'COND':>8s} "
        f"{'INSUFF':>8s}"
    )


    for gold in LABELS:

        row = confusion[
            gold
        ]


        print(
            f"{gold:14s} "
            f"{row['YES']:8d} "
            f"{row['NO']:8d} "
            f"{row['CONDITIONAL']:8d} "
            f"{row['INSUFFICIENT']:8d}"
        )


    print_group(
        "DECISION ACCURACY BY COMPOSITION",
        group_tables[
            "composition"
        ],
    )


    print_group(
        "DECISION ACCURACY BY QUESTION TYPE",
        group_tables[
            "question_type"
        ],
    )


    print_group(
        "DECISION ACCURACY BY INPUT ORDER",
        group_tables[
            "input_order"
        ],
    )


    print_group(
        "DECISION ACCURACY BY EVIDENCE COUNT",
        group_tables[
            "evidence_count"
        ],
    )


    print_group(
        "DECISION ACCURACY BY RELATIONAL GROUP",
        group_tables[
            "relational_group"
        ],
    )


    print_group(
        "DECISION ACCURACY BY REASONING TAG",
        group_tables[
            "reasoning_tags"
        ],
    )


    print()
    print(
        "=" * 88
    )

    print(
        "CITATION ERROR TAXONOMY"
    )

    print(
        "=" * 88
    )


    for key, value in sorted(
        citation_primary.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    ):

        print(
            f"{key:36s} "
            f"{value:3d}/240 "
            f"= {value / 240:.4f}"
        )


    print()
    print(
        "Citation taxonomy by gold label:"
    )


    for label in LABELS:

        print(
            f"  {label:12s}",
            dict(
                sorted(
                    citation_primary_by_gold[
                        label
                    ].items()
                )
            ),
        )


    print()
    print(
        "=" * 88
    )

    print(
        "CONDITION ERROR TAXONOMY"
    )

    print(
        "=" * 88
    )


    for key, value in sorted(
        condition_primary.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    ):

        print(
            f"{key:36s} "
            f"{value:3d}/240 "
            f"= {value / 240:.4f}"
        )


    print()
    print(
        "Condition taxonomy by gold label:"
    )


    for label in LABELS:

        print(
            f"  {label:12s}",
            dict(
                sorted(
                    condition_primary_by_gold[
                        label
                    ].items()
                )
            ),
        )


    print()
    print(
        "=" * 88
    )

    print(
        "DECISION × EVIDENCE ACCOUNTING"
    )

    print(
        "=" * 88
    )


    for key, value in sorted(
        accounting_cross.items()
    ):

        print(
            f"{' | '.join(key):75s} "
            f"{value:3d}"
        )


    print()
    print(
        "=" * 88
    )

    print(
        "POST-HOC ANALYSIS 01: PASS"
    )

    print(
        "=" * 88
    )

    print(
        "No model was loaded."
    )

    print(
        "No model generation occurred."
    )

    print(
        "No individual Challenge V2 case was displayed."
    )

    print(
        "Next step: inspect the 24 frozen decision errors."
    )


if __name__ == "__main__":
    main()
