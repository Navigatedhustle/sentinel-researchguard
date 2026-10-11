from collections import Counter, defaultdict
from pathlib import Path
import json


BASE = Path("/workspace")

DATA_PATH = (
    BASE
    / "training/datasets/"
      "sentinel_validation_v43.jsonl"
)

RUN01_PATH = (
    BASE
    / "results/"
      "sentinel_v43_validation_lora_run01.json"
)

RUN02_PATH = (
    BASE
    / "results/"
      "sentinel_v44_run02_v43_validation.json"
)


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def extract_prediction(p):
    """
    Robustly recover the decision from the evaluator prediction record.
    """

    for key in [
        "predicted",
        "predicted_decision",
        "prediction",
        "predicted_label",
        "decision",
    ]:
        value = p.get(key)
        if value in {
            "YES",
            "NO",
            "CONDITIONAL",
            "INSUFFICIENT",
        }:
            return value

    for key in [
        "parsed",
        "parsed_output",
        "json",
        "output_json",
    ]:
        value = p.get(key)

        if (
            isinstance(value, dict)
            and value.get("decision") in {
                "YES",
                "NO",
                "CONDITIONAL",
                "INSUFFICIENT",
            }
        ):
            return value["decision"]

    for key in [
        "raw_model_output",
        "raw_output",
        "model_output",
        "generated_text",
        "text",
        "output",
    ]:
        value = p.get(key)

        if isinstance(value, str):
            try:
                obj = json.loads(value)
            except Exception:
                continue

            if obj.get("decision") in {
                "YES",
                "NO",
                "CONDITIONAL",
                "INSUFFICIENT",
            }:
                return obj["decision"]

    raise KeyError(
        "Could not recover prediction. Keys: "
        + repr(sorted(p.keys()))
    )


def gold_decision(row):
    return json.loads(
        row["output"]
    )["decision"]


def main():

    data = load_jsonl(DATA_PATH)

    run01_obj = json.loads(
        RUN01_PATH.read_text()
    )

    run02_obj = json.loads(
        RUN02_PATH.read_text()
    )

    run01 = run01_obj["predictions"]
    run02 = run02_obj["predictions"]

    assert len(data) == 960
    assert len(run01) == 960
    assert len(run02) == 960

    p1 = [
        extract_prediction(x)
        for x in run01
    ]

    p2 = [
        extract_prediction(x)
        for x in run02
    ]

    print("=" * 96)
    print("V4.4 RUN 02 — DECISION LOCALIZATION")
    print("=" * 96)

    # ------------------------------------------------------------------
    # Composition x gold label
    # ------------------------------------------------------------------

    stats = defaultdict(
        lambda: Counter(
            total=0,
            correct=0,
        )
    )

    confusion = defaultdict(Counter)

    for row, pred in zip(data, p2):

        gold = gold_decision(row)

        key = (
            row["composition"],
            gold,
        )

        stats[key]["total"] += 1

        if pred == gold:
            stats[key]["correct"] += 1

        confusion[key][pred] += 1

    print()
    print("COMPOSITION × GOLD LABEL")
    print("-" * 96)

    for key in sorted(stats):

        composition, label = key

        total = stats[key]["total"]
        correct = stats[key]["correct"]

        print(
            f"{composition:38s} "
            f"{label:13s} "
            f"{correct:3d}/{total:3d} "
            f"{correct/total:7.4f} "
            f"{dict(confusion[key])}"
        )

    # ------------------------------------------------------------------
    # Primary endpoint
    # ------------------------------------------------------------------

    temporal_no = [
        (row, pred)
        for row, pred in zip(
            data,
            p2,
        )
        if (
            row["composition"]
            == "temporal_override"
            and gold_decision(row) == "NO"
        )
    ]

    temporal_no_correct = sum(
        pred == "NO"
        for row, pred in temporal_no
    )

    print()
    print("=" * 96)
    print("PRIMARY ENDPOINT")
    print("=" * 96)

    print(
        "Run 01 temporal NO:   0/120 = 0.0000"
    )

    print(
        "Run 02 temporal NO:  "
        f"{temporal_no_correct}/"
        f"{len(temporal_no)} = "
        f"{temporal_no_correct/len(temporal_no):.4f}"
    )

    print(
        "Run 02 temporal NO errors:",
        Counter(
            pred
            for row, pred in temporal_no
            if pred != "NO"
        ),
    )

    # ------------------------------------------------------------------
    # All Run01 -> Run02 decision transitions
    # ------------------------------------------------------------------

    transitions = Counter(
        (a, b)
        for a, b in zip(
            p1,
            p2,
        )
    )

    changed = sum(
        a != b
        for a, b in zip(
            p1,
            p2,
        )
    )

    print()
    print("=" * 96)
    print("RUN 01 → RUN 02 DECISION CHANGES")
    print("=" * 96)

    print(
        f"changed rows: {changed}/960 "
        f"({changed/960:.4f})"
    )

    for transition, count in sorted(
        transitions.items()
    ):
        if transition[0] != transition[1]:
            print(
                f"{transition[0]:13s} -> "
                f"{transition[1]:13s}: "
                f"{count}"
            )

    # ------------------------------------------------------------------
    # Run02 failures
    # ------------------------------------------------------------------

    failures = []

    for index, (
        row,
        old_pred,
        new_pred,
    ) in enumerate(
        zip(
            data,
            p1,
            p2,
        )
    ):

        gold = gold_decision(row)

        if new_pred != gold:

            failures.append(
                (
                    index,
                    row,
                    gold,
                    old_pred,
                    new_pred,
                )
            )

    print()
    print("=" * 96)
    print("RUN 02 FAILURES")
    print("=" * 96)

    print(
        "total:",
        len(failures),
    )

    by_family = Counter(
        (
            row["composition"],
            row["family"],
            gold,
            new_pred,
        )
        for (
            index,
            row,
            gold,
            old_pred,
            new_pred,
        ) in failures
    )

    for key, count in sorted(
        by_family.items()
    ):

        composition, family, gold, pred = key

        print(
            f"{composition:30s} "
            f"{family:30s} "
            f"{gold:13s} -> "
            f"{pred:13s} "
            f"{count}"
        )

    # ------------------------------------------------------------------
    # YES regression localization
    # ------------------------------------------------------------------

    yes_failures = [
        item
        for item in failures
        if item[2] == "YES"
    ]

    print()
    print("=" * 96)
    print("YES REGRESSIONS")
    print("=" * 96)

    print(
        "YES failures:",
        len(yes_failures),
    )

    print(
        "by composition:",
        Counter(
            row["composition"]
            for (
                index,
                row,
                gold,
                old_pred,
                new_pred,
            ) in yes_failures
        ),
    )

    print(
        "predicted as:",
        Counter(
            new_pred
            for (
                index,
                row,
                gold,
                old_pred,
                new_pred,
            ) in yes_failures
        ),
    )

    print()
    print("First 15 YES failures:")

    for (
        index,
        row,
        gold,
        old_pred,
        new_pred,
    ) in yes_failures[:15]:

        print()
        print(
            f"row={index}"
        )
        print(
            "scenario:",
            row["scenario_id"],
        )
        print(
            "family:",
            row["family"],
        )
        print(
            "composition:",
            row["composition"],
        )
        print(
            "Run01:",
            old_pred,
        )
        print(
            "Run02:",
            new_pred,
        )

    print()
    print("=" * 96)
    print("CHALLENGE V1 WAS NOT READ BY THIS SCRIPT")
    print("=" * 96)


if __name__ == "__main__":
    main()
