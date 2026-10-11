"""
Sentinel V5 Challenge V2 post-hoc analysis 02.

Build a dossier for the 24 already-recorded decision failures.

NO:
- model loading
- generation
- training
- rescoring
- prompt modification

Challenge V2 has already been consumed.
"""

from collections import Counter, defaultdict
from pathlib import Path

import argparse
import hashlib
import json


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

EXPECTED_CHALLENGE_SHA = (
    "30eb37ff39382f0d205adb7c17383b240"
    "25601ea3db1dd61f22b715af6c1e482"
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

    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_jsonl(path, values):

    with path.open("w") as f:

        for value in values:

            f.write(
                json.dumps(
                    value,
                    sort_keys=True,
                )
            )

            f.write("\n")


def exact_citation_set(
    gold,
    predicted,
):

    return (
        len(gold)
        == len(predicted)
        and set(gold)
        == set(predicted)
    )


def condition_relation(
    gold,
    predicted,
):

    if predicted == gold:
        return "exact"

    if not gold and predicted:
        return "extra_when_gold_empty"

    if gold and not predicted:
        return "missing_all_gold_conditions"

    gold_set = {
        " ".join(x.split())
        for x in gold
    }

    pred_set = {
        " ".join(x.split())
        for x in predicted
    }


    missing = (
        gold_set
        - pred_set
    )

    extra = (
        pred_set
        - gold_set
    )


    if missing and not extra:
        return "missing_only"

    if extra and not missing:
        return "extra_only"

    if missing and extra:
        return "different_or_reworded"

    return "ordering_or_format"


def main():

    args = parse_args()

    outdir = Path(
        args.output_dir
    )


    if outdir.exists():

        raise RuntimeError(
            "Output directory already exists."
        )


    outdir.mkdir(
        parents=True,
        exist_ok=False,
    )


    # ================================================================
    # LOAD FROZEN DATA
    # ================================================================

    if (
        sha256(
            CHALLENGE
        )
        != EXPECTED_CHALLENGE_SHA
    ):

        raise RuntimeError(
            "Challenge V2 hash mismatch."
        )


    challenge = json.loads(
        CHALLENGE.read_text()
    )

    predictions = load_jsonl(
        PREDICTIONS
    )


    assert len(challenge) == 240
    assert len(predictions) == 240


    cases = {
        str(
            case[
                "case_id"
            ]
        ):
            case

        for case in challenge
    }


    predictions_by_id = {
        str(
            pred[
                "case_id"
            ]
        ):
            pred

        for pred in predictions
    }


    assert len(cases) == 240
    assert len(predictions_by_id) == 240

    assert (
        set(cases)
        == set(
            predictions_by_id
        )
    )


    failures = []


    for case_id in cases:

        case = cases[
            case_id
        ]

        pred_record = predictions_by_id[
            case_id
        ]


        if pred_record[
            "decision_correct"
        ]:

            continue


        gold = case[
            "gold"
        ]


        generated = json.loads(
            pred_record[
                "generated_text"
            ]
        )


        assert (
            gold[
                "decision"
            ]
            == pred_record[
                "gold_label"
            ]
        )


        assert (
            generated[
                "decision"
            ]
            == pred_record[
                "strict_decision"
            ]
        )


        failure = {

            "case_id":
                case_id,

            "composition":
                case.get(
                    "composition"
                ),

            "question_type":
                case.get(
                    "question_type"
                ),

            "reasoning_tags":
                case.get(
                    "reasoning_tags"
                ),

            "relational_group":
                case.get(
                    "relational_group"
                ),

            "policy_pack":
                case.get(
                    "policy_pack"
                ),

            "input_order":
                case.get(
                    "input_order"
                ),

            "evidence_count":
                case.get(
                    "evidence_count"
                ),

            "gold_decision":
                gold[
                    "decision"
                ],

            "predicted_decision":
                generated[
                    "decision"
                ],

            "transition":
                (
                    f"{gold['decision']}"
                    f"->{generated['decision']}"
                ),

            "question":
                case[
                    "question"
                ],

            "evidence_text":
                case[
                    "evidence_text"
                ],

            "benchmark_input":
                case[
                    "input"
                ],

            "gold_answer":
                gold.get(
                    "answer"
                ),

            "gold_evidence_sufficient":
                gold.get(
                    "evidence_sufficient"
                ),

            "gold_citations":
                gold[
                    "citations"
                ],

            "predicted_citations":
                generated[
                    "citations"
                ],

            "citation_set_exact":
                exact_citation_set(
                    gold[
                        "citations"
                    ],
                    generated[
                        "citations"
                    ],
                ),

            "gold_conditions":
                gold[
                    "conditions"
                ],

            "predicted_conditions":
                generated[
                    "conditions"
                ],

            "condition_relation":
                condition_relation(
                    gold[
                        "conditions"
                    ],
                    generated[
                        "conditions"
                    ],
                ),

            "model_rationale":
                generated[
                    "rationale"
                ],
        }


        failures.append(
            failure
        )


    if len(
        failures
    ) != 24:

        raise RuntimeError(
            f"Expected 24 failures; "
            f"found {len(failures)}."
        )


    # ================================================================
    # SORT
    # ================================================================

    failures.sort(
        key=lambda row: (
            row[
                "transition"
            ],
            row[
                "composition"
            ],
            row[
                "case_id"
            ],
        )
    )


    # ================================================================
    # AGGREGATES
    # ================================================================

    transitions = Counter(
        row[
            "transition"
        ]
        for row in failures
    )


    compositions = Counter(
        row[
            "composition"
        ]
        for row in failures
    )


    question_types = Counter(
        row[
            "question_type"
        ]
        for row in failures
    )


    evidence_counts = Counter(
        str(
            row[
                "evidence_count"
            ]
        )
        for row in failures
    )


    input_orders = Counter(
        row[
            "input_order"
        ]
        for row in failures
    )


    reasoning_tags = Counter()


    for row in failures:

        for tag in (
            row[
                "reasoning_tags"
            ]
            or []
        ):

            reasoning_tags[
                tag
            ] += 1


    citation_exact = sum(
        row[
            "citation_set_exact"
        ]
        for row in failures
    )


    condition_relations = Counter(
        row[
            "condition_relation"
        ]
        for row in failures
    )


    summary = {

        "failure_count":
            24,

        "transitions":
            dict(
                transitions.most_common()
            ),

        "compositions":
            dict(
                compositions.most_common()
            ),

        "question_types":
            dict(
                question_types.most_common()
            ),

        "evidence_counts":
            dict(
                sorted(
                    evidence_counts.items()
                )
            ),

        "input_orders":
            dict(
                input_orders.most_common()
            ),

        "reasoning_tags":
            dict(
                reasoning_tags.most_common()
            ),

        "failure_citation_set_exact":
            citation_exact,

        "failure_condition_relations":
            dict(
                condition_relations.most_common()
            ),
    }


    write_json(
        outdir
        / "FAILURE_SUMMARY.json",
        summary,
    )


    write_jsonl(
        outdir
        / "FAILURES.jsonl",
        failures,
    )


    # ================================================================
    # FAILURE INDEX TSV
    # ================================================================

    with (
        outdir
        / "FAILURE_INDEX.tsv"
    ).open("w") as f:

        f.write(
            "\t".join([
                "case_id",
                "transition",
                "composition",
                "question_type",
                "evidence_count",
                "input_order",
                "citation_exact",
                "condition_relation",
            ])
            + "\n"
        )


        for row in failures:

            f.write(
                "\t".join([
                    str(
                        row[
                            "case_id"
                        ]
                    ),

                    str(
                        row[
                            "transition"
                        ]
                    ),

                    str(
                        row[
                            "composition"
                        ]
                    ),

                    str(
                        row[
                            "question_type"
                        ]
                    ),

                    str(
                        row[
                            "evidence_count"
                        ]
                    ),

                    str(
                        row[
                            "input_order"
                        ]
                    ),

                    str(
                        row[
                            "citation_set_exact"
                        ]
                    ),

                    str(
                        row[
                            "condition_relation"
                        ]
                    ),
                ])
                + "\n"
            )


    # ================================================================
    # HUMAN-READABLE DOSSIER
    # ================================================================

    md = []


    md.append(
        "# Sentinel V5 — Challenge V2 Decision Failure Dossier"
    )

    md.append("")

    md.append(
        "Frozen one-shot result: "
        "**216/240 = 90.00% decision accuracy.**"
    )

    md.append("")

    md.append(
        "This dossier contains only the 24 already-recorded "
        "decision failures. No model was rerun."
    )

    md.append("")


    md.append(
        "## Failure transitions"
    )

    md.append("")


    for key, value in (
        transitions.most_common()
    ):

        md.append(
            f"- **{key}: {value}**"
        )


    md.append("")

    md.append(
        "## Failure compositions"
    )

    md.append("")


    for key, value in (
        compositions.most_common()
    ):

        md.append(
            f"- **{key}: {value}**"
        )


    md.append("")

    md.append("---")

    md.append("")


    current_transition = None


    for index, row in enumerate(
        failures,
        start=1,
    ):

        if (
            row[
                "transition"
            ]
            != current_transition
        ):

            current_transition = (
                row[
                    "transition"
                ]
            )

            md.append(
                f"# {current_transition}"
            )

            md.append("")


        md.append(
            f"## Failure {index:02d} — "
            f"{row['case_id']}"
        )

        md.append("")


        md.append(
            f"- **Composition:** "
            f"`{row['composition']}`"
        )

        md.append(
            f"- **Question type:** "
            f"`{row['question_type']}`"
        )

        md.append(
            f"- **Reasoning tags:** "
            f"`{row['reasoning_tags']}`"
        )

        md.append(
            f"- **Relational group:** "
            f"`{row['relational_group']}`"
        )

        md.append(
            f"- **Evidence count:** "
            f"{row['evidence_count']}"
        )

        md.append(
            f"- **Input order:** "
            f"`{row['input_order']}`"
        )

        md.append(
            f"- **Gold decision:** "
            f"**{row['gold_decision']}**"
        )

        md.append(
            f"- **Predicted decision:** "
            f"**{row['predicted_decision']}**"
        )

        md.append(
            f"- **Citation set exact:** "
            f"{row['citation_set_exact']}"
        )

        md.append(
            f"- **Condition relation:** "
            f"`{row['condition_relation']}`"
        )

        md.append("")


        md.append(
            "### Question"
        )

        md.append("")

        md.append(
            row[
                "question"
            ]
        )

        md.append("")


        md.append(
            "### Evidence"
        )

        md.append("")

        md.append("```text")

        md.append(
            row[
                "evidence_text"
            ]
        )

        md.append("```")

        md.append("")


        md.append(
            "### Gold"
        )

        md.append("")

        md.append(
            f"**Decision:** "
            f"{row['gold_decision']}"
        )

        md.append("")

        md.append(
            f"**Answer:** "
            f"{row['gold_answer']}"
        )

        md.append("")

        md.append(
            f"**Evidence sufficient:** "
            f"{row['gold_evidence_sufficient']}"
        )

        md.append("")

        md.append(
            "**Citations:**"
        )

        md.append("")

        md.append(
            "```json\n"
            + json.dumps(
                row[
                    "gold_citations"
                ],
                indent=2,
            )
            + "\n```"
        )

        md.append("")

        md.append(
            "**Conditions:**"
        )

        md.append("")

        md.append(
            "```json\n"
            + json.dumps(
                row[
                    "gold_conditions"
                ],
                indent=2,
            )
            + "\n```"
        )

        md.append("")


        md.append(
            "### Sentinel V5"
        )

        md.append("")

        md.append(
            f"**Decision:** "
            f"{row['predicted_decision']}"
        )

        md.append("")

        md.append(
            f"**Rationale:** "
            f"{row['model_rationale']}"
        )

        md.append("")

        md.append(
            "**Citations:**"
        )

        md.append("")

        md.append(
            "```json\n"
            + json.dumps(
                row[
                    "predicted_citations"
                ],
                indent=2,
            )
            + "\n```"
        )

        md.append("")

        md.append(
            "**Conditions:**"
        )

        md.append("")

        md.append(
            "```json\n"
            + json.dumps(
                row[
                    "predicted_conditions"
                ],
                indent=2,
            )
            + "\n```"
        )

        md.append("")

        md.append("---")

        md.append("")


    (
        outdir
        / "FAILURE_DOSSIER.md"
    ).write_text(
        "\n".join(
            md
        )
        + "\n"
    )


    # ================================================================
    # TERMINAL SUMMARY
    # ================================================================

    print("=" * 88)
    print("24-CASE FAILURE SUMMARY")
    print("=" * 88)


    print()
    print("DECISION TRANSITIONS")

    for key, value in (
        transitions.most_common()
    ):

        print(
            f"  {key:28s} {value:2d}"
        )


    print()
    print("COMPOSITIONS")

    for key, value in (
        compositions.most_common()
    ):

        print(
            f"  {key:42s} {value:2d}"
        )


    print()
    print("QUESTION TYPES")

    for key, value in (
        question_types.most_common()
    ):

        print(
            f"  {key:42s} {value:2d}"
        )


    print()
    print("INPUT ORDER")

    for key, value in (
        input_orders.most_common()
    ):

        print(
            f"  {key:42s} {value:2d}"
        )


    print()
    print("EVIDENCE COUNTS")

    for key, value in sorted(
        evidence_counts.items(),
        key=lambda x: int(
            x[0]
        ),
    ):

        print(
            f"  {key:42s} {value:2d}"
        )


    print()
    print("REASONING TAGS")

    for key, value in (
        reasoning_tags.most_common()
    ):

        print(
            f"  {key:42s} {value:2d}"
        )


    print()
    print(
        "Failures with exact gold citation set:",
        f"{citation_exact}/24"
    )


    print()
    print(
        "Condition relation among failures:"
    )

    for key, value in (
        condition_relations.most_common()
    ):

        print(
            f"  {key:42s} {value:2d}"
        )


    print()
    print("=" * 88)
    print("FAILURE INDEX")
    print("=" * 88)


    for index, row in enumerate(
        failures,
        start=1,
    ):

        print(
            f"{index:02d} "
            f"{row['transition']:26s} "
            f"{row['composition']:38s} "
            f"evidence={row['evidence_count']} "
            f"cite_exact={row['citation_set_exact']} "
            f"conditions={row['condition_relation']}"
        )


    print()
    print("=" * 88)
    print("POST-HOC ANALYSIS 02: PASS")
    print("=" * 88)

    print(
        "Failure dossier:",
        outdir
        / "FAILURE_DOSSIER.md"
    )

    print(
        "Structured failures:",
        outdir
        / "FAILURES.jsonl"
    )

    print(
        "No model was loaded."
    )

    print(
        "No generation occurred."
    )


if __name__ == "__main__":
    main()
