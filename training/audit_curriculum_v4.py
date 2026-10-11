from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import re


BASE = Path("/workspace")
DATA = BASE / "training/datasets"

FILES = {
    "train": DATA / "sentinel_train_v4.jsonl",
    "validation": DATA / "sentinel_validation_v4.jsonl",
    "test": DATA / "sentinel_test_v4.jsonl",
}

EXPECTED_ROWS = {
    "train": 1800,
    "validation": 600,
    "test": 600,
}

EXPECTED_SCENARIOS = {
    "train": 600,
    "validation": 200,
    "test": 200,
}

EXPECTED_FAMILIES = {
    "train": 30,
    "validation": 10,
    "test": 10,
}

EXPECTED_PER_DECISION = {
    "train": 450,
    "validation": 150,
    "test": 150,
}

ASSETS = [
    "human-subject interview transcripts",
    "student advising records",
    "genomic sequencing files",
    "unpublished research manuscripts",
    "sponsor-confidential prototype designs",
]

DECISIONS = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
}


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                )

            rows.append(row)

    return rows


def parse_output(row):
    try:
        obj = json.loads(row["output"])
    except Exception as exc:
        raise AssertionError(
            f"{row.get('scenario_id')}: invalid output JSON: {exc}"
        )

    return obj


def evidence_part(input_text):
    marker = "\n\nQUESTION\n========\n"

    if marker not in input_text:
        raise AssertionError(
            "input missing QUESTION delimiter"
        )

    return input_text.split(marker, 1)[0]


def question_part(input_text):
    marker = "\n\nQUESTION\n========\n"

    if marker not in input_text:
        raise AssertionError(
            "input missing QUESTION delimiter"
        )

    q = input_text.split(marker, 1)[1].strip()

    prefixes = [
        "Decide the exact policy question below:\n",
        "Using only the supplied evidence, classify this question:\n",
    ]

    for prefix in prefixes:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()

    return q


def evidence_ids(input_text):
    part = evidence_part(input_text)

    return re.findall(
        r"^\[([^\]\n]+)\]$",
        part,
        flags=re.MULTILINE,
    )


def evidence_order(input_text):
    return tuple(evidence_ids(input_text))


def normalize_text(text):
    text = text.lower()

    for asset in ASSETS:
        text = text.replace(
            asset.lower(),
            "<asset>",
        )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def unresolved_placeholders(value):
    found = []

    def walk(x, path="root"):
        if isinstance(x, str):
            matches = re.findall(
                r"\{[a-zA-Z_][a-zA-Z0-9_]*\}",
                x,
            )

            for match in matches:
                found.append(
                    (path, match)
                )

        elif isinstance(x, list):
            for i, item in enumerate(x):
                walk(
                    item,
                    f"{path}[{i}]",
                )

        elif isinstance(x, dict):
            for key, item in x.items():
                walk(
                    item,
                    f"{path}.{key}",
                )

    walk(value)

    return found


def audit_split(split, rows):
    errors = []

    print()
    print(split.upper())
    print("-" * 72)

    print("rows:", len(rows))

    if len(rows) != EXPECTED_ROWS[split]:
        errors.append(
            f"{split}: expected {EXPECTED_ROWS[split]} rows, "
            f"got {len(rows)}"
        )

    scenario_groups = defaultdict(list)

    for row in rows:
        scenario_groups[
            row["scenario_id"]
        ].append(row)

    print(
        "scenarios:",
        len(scenario_groups),
    )

    if len(scenario_groups) != EXPECTED_SCENARIOS[split]:
        errors.append(
            f"{split}: expected "
            f"{EXPECTED_SCENARIOS[split]} scenarios, "
            f"got {len(scenario_groups)}"
        )

    families = {
        row["family"]
        for row in rows
    }

    print(
        "families:",
        len(families),
    )

    if len(families) != EXPECTED_FAMILIES[split]:
        errors.append(
            f"{split}: expected "
            f"{EXPECTED_FAMILIES[split]} families, "
            f"got {len(families)}"
        )

    decisions = Counter()

    exact_inputs = Counter(
        row["input"]
        for row in rows
    )

    duplicate_inputs = [
        text
        for text, count in exact_inputs.items()
        if count > 1
    ]

    print(
        "duplicate exact inputs within split:",
        len(duplicate_inputs),
    )

    if duplicate_inputs:
        errors.append(
            f"{split}: duplicate exact inputs exist"
        )

    placeholder_errors = 0
    citation_errors = 0
    semantic_errors = 0

    for row in rows:
        obj = parse_output(row)

        decision = obj.get("decision")
        decisions[decision] += 1

        if decision not in DECISIONS:
            semantic_errors += 1
            errors.append(
                f"{row['scenario_id']}: invalid decision {decision!r}"
            )

        ids = set(
            evidence_ids(row["input"])
        )

        citations = obj.get(
            "citations",
            [],
        )

        invalid = (
            set(citations) - ids
        )

        if invalid:
            citation_errors += 1
            errors.append(
                f"{row['scenario_id']}: "
                f"citations not in evidence: {sorted(invalid)}"
            )

        if decision == "INSUFFICIENT":
            if citations:
                semantic_errors += 1
                errors.append(
                    f"{row['scenario_id']}: "
                    "INSUFFICIENT has citations"
                )

            if obj.get("evidence_sufficient") is not False:
                semantic_errors += 1
                errors.append(
                    f"{row['scenario_id']}: "
                    "INSUFFICIENT marked sufficient"
                )

        else:
            if obj.get("evidence_sufficient") is not True:
                semantic_errors += 1
                errors.append(
                    f"{row['scenario_id']}: "
                    f"{decision} marked insufficient"
                )

        conditions = obj.get(
            "conditions",
            [],
        )

        if (
            decision == "CONDITIONAL"
            and not conditions
        ):
            semantic_errors += 1
            errors.append(
                f"{row['scenario_id']}: "
                "CONDITIONAL lacks conditions"
            )

        if (
            decision != "CONDITIONAL"
            and conditions
        ):
            # Not automatically wrong in every ontology,
            # so report rather than immediately fail.
            semantic_errors += 1
            errors.append(
                f"{row['scenario_id']}: "
                f"{decision} unexpectedly has conditions "
                f"{conditions!r}"
            )

        unresolved = unresolved_placeholders(
            {
                "input": row["input"],
                "answer": obj.get("answer"),
                "conditions": conditions,
            }
        )

        if unresolved:
            placeholder_errors += 1
            errors.append(
                f"{row['scenario_id']}: "
                f"unresolved placeholders {unresolved}"
            )

    print(
        "decisions:",
        dict(sorted(decisions.items())),
    )

    for decision in DECISIONS:
        actual = decisions[decision]
        expected = EXPECTED_PER_DECISION[split]

        if actual != expected:
            errors.append(
                f"{split}: {decision} expected "
                f"{expected}, got {actual}"
            )

    print(
        "citation errors:",
        citation_errors,
    )

    print(
        "placeholder errors:",
        placeholder_errors,
    )

    # Scenario-level invariant checks.
    bad_surface_sets = []
    bad_outputs = []
    bad_metadata = []

    varied_evidence_order = 0
    total_multi_evidence = 0

    for scenario_id, group in scenario_groups.items():

        surfaces = {
            row["surface_variant"]
            for row in group
        }

        if surfaces != {0, 1, 2}:
            bad_surface_sets.append(
                (scenario_id, surfaces)
            )

        if len(group) != 3:
            bad_surface_sets.append(
                (scenario_id, f"count={len(group)}")
            )

        outputs = {
            row["output"]
            for row in group
        }

        if len(outputs) != 1:
            bad_outputs.append(
                scenario_id
            )

        metadata_signatures = {
            (
                row["family"],
                row["composition"],
                row["question_type"],
                row["asset"],
            )
            for row in group
        }

        if len(metadata_signatures) != 1:
            bad_metadata.append(
                scenario_id
            )

        orders = {
            evidence_order(row["input"])
            for row in group
        }

        first_ids = evidence_order(
            group[0]["input"]
        )

        if len(first_ids) > 1:
            total_multi_evidence += 1

            if len(orders) > 1:
                varied_evidence_order += 1

        questions = {
            normalize_text(
                question_part(row["input"])
            )
            for row in group
        }

        if len(questions) != 1:
            errors.append(
                f"{scenario_id}: "
                "surface variants changed underlying question"
            )

    print(
        "bad surface sets:",
        len(bad_surface_sets),
    )
    print(
        "inconsistent outputs within scenario:",
        len(bad_outputs),
    )
    print(
        "inconsistent metadata within scenario:",
        len(bad_metadata),
    )

    if total_multi_evidence:
        pct = (
            100.0
            * varied_evidence_order
            / total_multi_evidence
        )

        print(
            "multi-evidence scenarios with varied ordering:",
            f"{varied_evidence_order}/"
            f"{total_multi_evidence} "
            f"({pct:.1f}%)",
        )

        # This is intentionally a weak floor.
        # We only want to catch a broken shuffler.
        if pct < 50.0:
            errors.append(
                f"{split}: evidence ordering variation "
                f"too low ({pct:.1f}%)"
            )

    if bad_surface_sets:
        errors.append(
            f"{split}: malformed surface-variant groups"
        )

    if bad_outputs:
        errors.append(
            f"{split}: output changes across surface variants"
        )

    if bad_metadata:
        errors.append(
            f"{split}: metadata changes across surface variants"
        )

    return errors, scenario_groups


def scenario_representatives(groups):
    reps = {}

    for scenario_id, rows in groups.items():
        row = sorted(
            rows,
            key=lambda x: x["surface_variant"],
        )[0]

        reps[scenario_id] = {
            "family": row["family"],
            "composition": row["composition"],
            "question": normalize_text(
                question_part(row["input"])
            ),
            "evidence": normalize_text(
                evidence_part(row["input"])
            ),
        }

    return reps


def lexical_near_duplicate_audit(all_groups):
    print()
    print("CROSS-SPLIT NEAR-DUPLICATE AUDIT")
    print("-" * 72)

    reps = {
        split: scenario_representatives(groups)
        for split, groups in all_groups.items()
    }

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    suspicious = []

    for a, b in pairs:
        best = []

        for aid, ar in reps[a].items():
            for bid, br in reps[b].items():

                # Questions are short enough that SequenceMatcher is
                # useful as a conservative lexical contamination alarm.
                score = SequenceMatcher(
                    None,
                    ar["question"],
                    br["question"],
                ).ratio()

                if score >= 0.92:
                    best.append(
                        (
                            score,
                            aid,
                            bid,
                            ar["family"],
                            br["family"],
                            ar["question"],
                            br["question"],
                        )
                    )

        best.sort(
            reverse=True,
            key=lambda x: x[0],
        )

        print(
            f"{a} vs {b}: "
            f"{len(best)} question pairs >= 0.92 similarity"
        )

        suspicious.extend(
            (a, b, item)
            for item in best
        )

        for item in best[:5]:
            (
                score,
                aid,
                bid,
                af,
                bf,
                aq,
                bq,
            ) = item

            print()
            print(
                f"  similarity={score:.3f}"
            )
            print(
                f"  {a}: {aid} [{af}]"
            )
            print(
                f"    {aq}"
            )
            print(
                f"  {b}: {bid} [{bf}]"
            )
            print(
                f"    {bq}"
            )

    return suspicious


def print_content_samples(rows_by_split):
    print()
    print("QUALITATIVE SAMPLE")
    print("-" * 72)

    seen = set()

    for split in (
        "train",
        "validation",
        "test",
    ):
        for row in rows_by_split[split]:
            composition = row["composition"]

            key = (
                split,
                composition,
            )

            if key in seen:
                continue

            seen.add(key)

            obj = parse_output(row)

            print()
            print("=" * 72)
            print(
                split.upper(),
                "|",
                composition,
                "|",
                obj["decision"],
            )
            print(
                "family:",
                row["family"],
            )
            print(
                "scenario:",
                row["scenario_id"],
            )
            print()
            print("QUESTION:")
            print(
                question_part(row["input"])
            )
            print()
            print("ANSWER:")
            print(
                obj.get("answer")
            )
            print(
                "citations:",
                obj.get("citations")
            )
            print(
                "conditions:",
                obj.get("conditions")
            )


def main():
    print("=" * 72)
    print("SENTINEL V4 DEEP DATASET AUDIT")
    print("=" * 72)

    rows_by_split = {
        split: load_jsonl(path)
        for split, path in FILES.items()
    }

    all_errors = []
    all_groups = {}

    for split, rows in rows_by_split.items():
        errors, groups = audit_split(
            split,
            rows,
        )

        all_errors.extend(errors)
        all_groups[split] = groups

    suspicious = lexical_near_duplicate_audit(
        all_groups
    )

    print_content_samples(
        rows_by_split
    )

    print()
    print("=" * 72)

    if all_errors:
        print(
            f"AUDIT FAILED: {len(all_errors)} issue(s)"
        )
        print("=" * 72)

        for error in all_errors[:50]:
            print(
                "ERROR:",
                error,
            )

        if len(all_errors) > 50:
            print(
                f"... plus {len(all_errors) - 50} more"
            )

        raise SystemExit(1)

    print("MECHANICAL AUDIT PASSED")

    if suspicious:
        print(
            f"REVIEW REQUIRED: "
            f"{len(suspicious)} high-similarity "
            "cross-split question pair(s)"
        )
    else:
        print(
            "No >=0.92 cross-split question similarity alarms"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
