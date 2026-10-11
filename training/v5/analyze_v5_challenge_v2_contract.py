from collections import Counter, defaultdict
from pathlib import Path
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

OUTDIR = (
    BASE
    / "experiments/v5/"
      "challenge_v2_posthoc_contract_audit"
)

LABELS = (
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
)


def load_jsonl(path):

    rows = []

    with path.open() as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


challenge = json.loads(
    CHALLENGE.read_text()
)

predictions = load_jsonl(
    PREDICTIONS
)

assert len(challenge) == 240
assert len(predictions) == 240


case_map = {
    str(case["case_id"]): case
    for case in challenge
}

pred_map = {
    str(pred["case_id"]): pred
    for pred in predictions
}

assert set(case_map) == set(pred_map)


report = {}


print("=" * 88)
print("CHALLENGE V2 ANNOTATION CONTRACT")
print("=" * 88)


for label in LABELS:

    rows = [
        (
            case_map[cid],
            pred_map[cid],
        )

        for cid in case_map

        if case_map[cid][
            "gold"
        ][
            "decision"
        ] == label
    ]


    gold_citation_counts = Counter()
    pred_citation_counts = Counter()

    gold_condition_counts = Counter()
    pred_condition_counts = Counter()

    citation_exact = 0
    conditions_exact = 0

    decision_correct = 0


    for case, pred in rows:

        generated = json.loads(
            pred[
                "generated_text"
            ]
        )

        gold = case[
            "gold"
        ]


        gold_citations = gold[
            "citations"
        ]

        pred_citations = generated[
            "citations"
        ]


        gold_conditions = gold[
            "conditions"
        ]

        pred_conditions = generated[
            "conditions"
        ]


        gold_citation_counts[
            len(
                gold_citations
            )
        ] += 1

        pred_citation_counts[
            len(
                pred_citations
            )
        ] += 1


        gold_condition_counts[
            len(
                gold_conditions
            )
        ] += 1

        pred_condition_counts[
            len(
                pred_conditions
            )
        ] += 1


        if (
            len(
                gold_citations
            )
            == len(
                pred_citations
            )
            and set(
                gold_citations
            )
            == set(
                pred_citations
            )
        ):
            citation_exact += 1


        if (
            gold_conditions
            == pred_conditions
        ):
            conditions_exact += 1


        if pred[
            "decision_correct"
        ]:
            decision_correct += 1


    record = {

        "rows":
            len(
                rows
            ),

        "decision_correct":
            decision_correct,

        "gold_citation_count_distribution":
            dict(
                sorted(
                    gold_citation_counts.items()
                )
            ),

        "predicted_citation_count_distribution":
            dict(
                sorted(
                    pred_citation_counts.items()
                )
            ),

        "exact_citation_set":
            citation_exact,

        "gold_condition_count_distribution":
            dict(
                sorted(
                    gold_condition_counts.items()
                )
            ),

        "predicted_condition_count_distribution":
            dict(
                sorted(
                    pred_condition_counts.items()
                )
            ),

        "exact_conditions":
            conditions_exact,
    }


    report[
        label
    ] = record


    print()
    print(
        label
    )

    print(
        "-" * 88
    )

    print(
        "decision:",
        f"{decision_correct}/{len(rows)}"
    )

    print(
        "gold citation counts:",
        dict(
            sorted(
                gold_citation_counts.items()
            )
        )
    )

    print(
        "pred citation counts:",
        dict(
            sorted(
                pred_citation_counts.items()
            )
        )
    )

    print(
        "exact citation set:",
        f"{citation_exact}/{len(rows)}"
    )

    print(
        "gold condition counts:",
        dict(
            sorted(
                gold_condition_counts.items()
            )
        )
    )

    print(
        "pred condition counts:",
        dict(
            sorted(
                pred_condition_counts.items()
            )
        )
    )

    print(
        "exact conditions:",
        f"{conditions_exact}/{len(rows)}"
    )


OUTDIR.mkdir(
    parents=True,
    exist_ok=False,
)

(
    OUTDIR
    / "CONTRACT_AUDIT.json"
).write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)


print()
print("=" * 88)
print("POST-HOC CONTRACT AUDIT: PASS")
print("=" * 88)

print(
    "No model was loaded."
)

print(
    "No generation occurred."
)
