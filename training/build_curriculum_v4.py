from pathlib import Path
from collections import Counter
import hashlib
import json
import random
import re

from training.v4_family_specs_v41 import FAMILIES


BASE = Path("/workspace")
OUT = BASE / "training/datasets"

SEED = 20260814
PARAPHRASES = 3

ASSETS = [
    "human-subject interview transcripts",
    "student advising records",
    "genomic sequencing files",
    "unpublished research manuscripts",
    "sponsor-confidential prototype designs",
]


INSTRUCTION = """You are Sentinel, a policy-governance decision model.

Use ONLY the supplied evidence.

Classify the exact question as one of:

YES
NO
CONDITIONAL
INSUFFICIENT

Apply these rules in order:

INSUFFICIENT:
Choose INSUFFICIENT when the supplied evidence does not directly govern
the exact issue asked about. Related evidence is not enough. Do not invent
missing approvers, technologies, durations, thresholds, locations, or rules.

NO:
Choose NO when the evidence directly prohibits the exact action or directly
establishes that the proposition in the question is false.

CONDITIONAL:
Choose CONDITIONAL when the action may be permitted, but one or more
material requirements remain unresolved.

YES:
Choose YES only when the evidence directly permits the exact action and
every material condition is already established as satisfied by the question.

Silence is not prohibition.
Semantic similarity is not evidence.

Return valid JSON only.
""".strip()


QUESTION_FRAMES = [
    "{question}",
    "Decide the exact policy question below:\n{question}",
    "Using only the supplied evidence, classify this question:\n{question}",
]


def slug(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def fill_asset(value, asset):
    """
    Replace the abstract {asset} placeholder while leaving all other text
    untouched.
    """
    if isinstance(value, str):
        return value.replace("{asset}", asset)

    if isinstance(value, list):
        return [
            fill_asset(x, asset)
            for x in value
        ]

    if isinstance(value, dict):
        return {
            k: fill_asset(v, asset)
            for k, v in value.items()
        }

    return value


def stable_seed(*parts):
    payload = "||".join(str(x) for x in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def render_evidence(evidence, scenario_id, surface_idx):
    """
    Shuffle evidence order deterministically so the model cannot rely on
    passage position while citation IDs remain stable.
    """
    ordered = list(evidence)

    rng = random.Random(
        stable_seed(SEED, scenario_id, surface_idx, "evidence_order")
    )
    rng.shuffle(ordered)

    blocks = []

    for item in ordered:
        blocks.append(
            "\n".join(
                [
                    f"[{item['id']}]",
                    f"Source: {item['source']}",
                    f"Section: {item['section']}",
                    f"Text: {item['text']}",
                ]
            )
        )

    return "\n\n".join(blocks)


def render_input(evidence, question, scenario_id, surface_idx):
    evidence_text = render_evidence(
        evidence,
        scenario_id,
        surface_idx,
    )

    framed_question = QUESTION_FRAMES[surface_idx].format(
        question=question
    )

    return (
        "SUPPLIED EVIDENCE\n"
        "=================\n"
        f"{evidence_text}\n\n"
        "QUESTION\n"
        "========\n"
        f"{framed_question}"
    )


def normalize_cases(family):
    """
    Normalize the family-spec case representation.

    V4 family specs may store cases as:

        {
            "YES": {...},
            "NO": {...},
            "CONDITIONAL": {...},
            "INSUFFICIENT": {...},
        }

    while the curriculum engine operates on a list of case dictionaries.
    Preserve the decision label explicitly without mutating FAMILIES.
    """
    cases = family["cases"]

    if isinstance(cases, dict):
        normalized = []

        for decision, case in cases.items():
            if not isinstance(case, dict):
                raise TypeError(
                    f"{family.get('name', '<unnamed>')}/{decision}: "
                    f"case must be dict, got {type(case).__name__}"
                )

            item = dict(case)

            embedded = item.get("decision")

            if embedded is not None and embedded != decision:
                raise ValueError(
                    f"{family.get('name', '<unnamed>')}/{decision}: "
                    f"embedded decision {embedded!r} conflicts "
                    f"with mapping key {decision!r}"
                )

            item["decision"] = decision
            normalized.append(item)

        return normalized

    if isinstance(cases, list):
        return [
            dict(case)
            for case in cases
        ]

    raise TypeError(
        f"{family.get('name', '<unnamed>')}: "
        f"cases must be dict or list, got {type(cases).__name__}"
    )


def validate_family(split, family):
    required_family_fields = {
        "name",
        "composition",
        "evidence",
        "cases",
    }

    missing = required_family_fields - set(family)

    if missing:
        raise ValueError(
            f"{split}/{family.get('name', '<unnamed>')}: "
            f"missing family fields {sorted(missing)}"
        )

    evidence = family["evidence"]
    cases = normalize_cases(family)

    evidence_ids = [
        item["id"]
        for item in evidence
    ]

    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError(
            f"{split}/{family['name']}: duplicate evidence IDs"
        )

    decisions = [
        case["decision"]
        for case in cases
    ]

    expected = {
        "YES",
        "NO",
        "CONDITIONAL",
        "INSUFFICIENT",
    }

    if set(decisions) != expected:
        raise ValueError(
            f"{split}/{family['name']}: decisions={decisions}; "
            f"expected exactly {sorted(expected)}"
        )

    if len(decisions) != 4:
        raise ValueError(
            f"{split}/{family['name']}: expected 4 cases"
        )

    valid_ids = set(evidence_ids)

    for case in cases:
        required_case_fields = {
            "decision",
            "question",
            "question_type",
            "citations",
            "rationale",
            "reasoning_tags",
            "conditions",
        }

        missing_case = required_case_fields - set(case)

        if missing_case:
            raise ValueError(
                f"{split}/{family['name']}/{case.get('decision')}: "
                f"missing case fields {sorted(missing_case)}"
            )

        invalid_citations = (
            set(case["citations"]) - valid_ids
        )

        if invalid_citations:
            raise ValueError(
                f"{split}/{family['name']}/{case['decision']}: "
                f"invalid citations {sorted(invalid_citations)}"
            )

        if (
            case["decision"] == "INSUFFICIENT"
            and case["citations"]
        ):
            raise ValueError(
                f"{split}/{family['name']}: "
                "INSUFFICIENT case must have zero citations"
            )

        if (
            case["decision"] == "CONDITIONAL"
            and not case["conditions"]
        ):
            raise ValueError(
                f"{split}/{family['name']}: "
                "CONDITIONAL case needs unresolved conditions"
            )


def build_split(split, families):
    rows = []

    for family in families:
        validate_family(split, family)

        for asset_idx, asset in enumerate(ASSETS):
            evidence = fill_asset(
                family["evidence"],
                asset,
            )

            for case in normalize_cases(family):
                populated_case = fill_asset(
                    case,
                    asset,
                )

                decision = populated_case["decision"]

                scenario_id = (
                    f"{split}__"
                    f"{slug(family['name'])}__"
                    f"asset_{asset_idx}__"
                    f"{decision.lower()}"
                )

                citations = list(
                    populated_case["citations"]
                )

                if decision == "INSUFFICIENT":
                    citations = []

                output_obj = {
                    "evidence_sufficient": (
                        decision != "INSUFFICIENT"
                    ),
                    "decision": decision,
                    "conditions": list(
                        populated_case["conditions"]
                    ),
                    "answer": populated_case["rationale"],
                    "citations": citations,
                }

                for surface_idx in range(PARAPHRASES):
                    input_text = render_input(
                        evidence=evidence,
                        question=populated_case["question"],
                        scenario_id=scenario_id,
                        surface_idx=surface_idx,
                    )

                    row = {
                        "scenario_id": scenario_id,
                        "family": family["name"],
                        "archetype": family["name"],
                        "composition": family["composition"],
                        "question_type": populated_case[
                            "question_type"
                        ],
                        "asset": asset,
                        "surface_variant": surface_idx,
                        "tags": list(
                            populated_case["reasoning_tags"]
                        ),
                        "instruction": INSTRUCTION,
                        "input": input_text,
                        "output": json.dumps(
                            output_obj,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }

                    rows.append(row)

    rng = random.Random(
        stable_seed(SEED, split, "row_shuffle")
    )
    rng.shuffle(rows)

    return rows


def audit_split(split, rows, families):
    print()
    print(split.upper())
    print("-" * 72)

    print("rows:", len(rows))
    print(
        "scenarios:",
        len({r["scenario_id"] for r in rows}),
    )
    print(
        "families:",
        len({r["family"] for r in rows}),
    )

    decisions = Counter(
        json.loads(r["output"])["decision"]
        for r in rows
    )

    print(
        "decisions:",
        dict(sorted(decisions.items())),
    )

    compositions = Counter(
        r["composition"]
        for r in rows
    )

    print(
        "compositions:",
        dict(sorted(compositions.items())),
    )

    family_counts = Counter(
        r["family"]
        for r in rows
    )

    expected_per_family = (
        len(ASSETS)
        * 4
        * PARAPHRASES
    )

    bad_family_counts = {
        name: count
        for name, count in family_counts.items()
        if count != expected_per_family
    }

    print(
        "bad family row counts:",
        bad_family_counts,
    )

    expected_rows = (
        len(families)
        * len(ASSETS)
        * 4
        * PARAPHRASES
    )

    if len(rows) != expected_rows:
        raise AssertionError(
            f"{split}: expected {expected_rows} rows, "
            f"got {len(rows)}"
        )

    expected_per_decision = expected_rows // 4

    if any(
        decisions[d] != expected_per_decision
        for d in (
            "YES",
            "NO",
            "CONDITIONAL",
            "INSUFFICIENT",
        )
    ):
        raise AssertionError(
            f"{split}: decision imbalance {decisions}"
        )

    if bad_family_counts:
        raise AssertionError(
            f"{split}: family row-count error"
        )

    # Strict JSON/output semantic checks.
    for row in rows:
        obj = json.loads(row["output"])

        if obj["decision"] == "INSUFFICIENT":
            if obj["citations"]:
                raise AssertionError(
                    f"{row['scenario_id']}: "
                    "INSUFFICIENT has citations"
                )

            if obj["evidence_sufficient"]:
                raise AssertionError(
                    f"{row['scenario_id']}: "
                    "INSUFFICIENT marked sufficient"
                )

        else:
            if not obj["evidence_sufficient"]:
                raise AssertionError(
                    f"{row['scenario_id']}: "
                    "governed decision marked insufficient"
                )

        if (
            obj["decision"] == "CONDITIONAL"
            and not obj["conditions"]
        ):
            raise AssertionError(
                f"{row['scenario_id']}: "
                "CONDITIONAL lacks conditions"
            )


def exact_input_hashes(rows):
    return {
        hashlib.sha256(
            r["input"].encode("utf-8")
        ).hexdigest()
        for r in rows
    }


def scenario_ids(rows):
    return {
        r["scenario_id"]
        for r in rows
    }


def family_names(rows):
    return {
        r["family"]
        for r in rows
    }


def write_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_splits = {
        "train",
        "validation",
        "test",
    }

    if set(FAMILIES) != expected_splits:
        raise ValueError(
            f"FAMILIES splits={sorted(FAMILIES)}; "
            f"expected {sorted(expected_splits)}"
        )

    expected_family_counts = {
        "train": 30,
        "validation": 10,
        "test": 10,
    }

    for split, expected in expected_family_counts.items():
        actual = len(FAMILIES[split])

        if actual != expected:
            raise AssertionError(
                f"{split}: expected {expected} families, "
                f"got {actual}"
            )

    rows = {
        split: build_split(
            split,
            FAMILIES[split],
        )
        for split in (
            "train",
            "validation",
            "test",
        )
    }

    print("=" * 72)
    print("SENTINEL V4 COMPOSITIONAL CURRICULUM")
    print("=" * 72)

    for split in (
        "train",
        "validation",
        "test",
    ):
        audit_split(
            split,
            rows[split],
            FAMILIES[split],
        )

    print()
    print("LEAKAGE AUDIT")
    print("-" * 72)

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for a, b in pairs:
        family_overlap = (
            family_names(rows[a])
            & family_names(rows[b])
        )

        scenario_overlap = (
            scenario_ids(rows[a])
            & scenario_ids(rows[b])
        )

        input_overlap = (
            exact_input_hashes(rows[a])
            & exact_input_hashes(rows[b])
        )

        print(
            f"{a} vs {b}: "
            f"family={len(family_overlap)} "
            f"scenario={len(scenario_overlap)} "
            f"exact_input={len(input_overlap)}"
        )

        if family_overlap:
            raise AssertionError(
                f"{a}/{b} family leakage: "
                f"{sorted(family_overlap)}"
            )

        if scenario_overlap:
            raise AssertionError(
                f"{a}/{b} scenario leakage"
            )

        if input_overlap:
            raise AssertionError(
                f"{a}/{b} exact-input leakage"
            )

    paths = {
        "train": (
            OUT / "sentinel_train_v4.jsonl"
        ),
        "validation": (
            OUT / "sentinel_validation_v4.jsonl"
        ),
        "test": (
            OUT / "sentinel_test_v4.jsonl"
        ),
    }

    for split, path in paths.items():
        write_jsonl(
            path,
            rows[split],
        )

        print(
            f"WROTE {split}: "
            f"{path} "
            f"({len(rows[split])} rows)"
        )

    print()
    print("=" * 72)
    print("V4 DATASET BUILD PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
