from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re


BASE = Path("/workspace")

TRAIN = (
    BASE
    / "training/v6/full_candidate_v2/"
      "sentinel_v6_train_candidate_v2.jsonl"
)

DEV = (
    BASE
    / "training/v6/full_candidate_v2/"
      "sentinel_v6_development_candidate_v2.jsonl"
)

PILOT = (
    BASE
    / "training/v6/pilot/"
      "sentinel_v6_relational_pilot.jsonl"
)

OUT_JSON = (
    BASE
    / "experiments/v6/full_candidate_v2/"
      "V6_FULL_CURRICULUM_V2_SEMANTIC_CAUSAL_AUDIT.json"
)

OUT_TXT = (
    BASE
    / "experiments/v6/full_candidate_v2/"
      "V6_FULL_CURRICULUM_V2_SEMANTIC_CAUSAL_AUDIT.txt"
)


LABELS = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
}


EXPECTED_KEYS = {
    "evidence",
    "evidence_hash",
    "family",
    "group_variant",
    "input",
    "input_order",
    "instruction",
    "operator",
    "output",
    "question",
    "relational_group",
    "row_id",
    "scenario_id",
    "split",
    "surface_variant",
    "target",
    "target_evidence_ref",
    "target_id",
    "target_index",
    "target_name",
    "world",
    "world_index",
}


def sha(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def norm(text):
    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
    )


def fail(message):
    raise AssertionError(
        message
    )


def split_evidence(evidence):
    matches = list(
        re.finditer(
            r"(?m)^\[(E[1-4])\]\s+",
            evidence,
        )
    )

    if len(matches) != 4:
        fail(
            "evidence does not contain "
            "exactly E1..E4 blocks"
        )

    blocks = {}

    for index, match in enumerate(
        matches
    ):
        ref = match.group(1)

        start = match.end()

        end = (
            matches[
                index + 1
            ].start()
            if index + 1
            < len(matches)
            else len(evidence)
        )

        blocks[ref] = (
            evidence[
                start:end
            ].strip()
        )

    if set(blocks) != {
        "E1",
        "E2",
        "E3",
        "E4",
    }:
        fail(
            "evidence refs malformed"
        )

    return blocks


def derive_decision(block):
    low = block.lower()

    insufficient_markers = (
        "does not establish a governing rule for this exact request",
        "governance for this exact request is not established",
    )

    if any(
        marker in low
        for marker in insufficient_markers
    ):
        return "INSUFFICIENT"

    direct_no_markers = (
        "the controlling rule prohibits the requested action",
        "under the operative rule, the requested action is barred",
    )

    if any(
        marker in low
        for marker in direct_no_markers
    ):
        return "NO"

    failed_markers = (
        "is documented as failed",
        "required control '",
    )

    if "is documented as failed" in low:
        return "NO"

    if (
        "the record confirms that the required control '"
        in low
        and
        "' failed" in low
    ):
        return "NO"

    unresolved_markers = (
        "does not establish whether the material requirement",
        "leaves the required control",
    )

    if any(
        marker in low
        for marker in unresolved_markers
    ):
        return "CONDITIONAL"

    permission_markers = (
        "the controlling rule permits the requested action",
        "under the operative rule, the requested action is allowed",
    )

    if any(
        marker in low
        for marker in permission_markers
    ):
        return "YES"

    fail(
        "independent semantic adjudicator "
        "could not derive decision"
    )


def audit_split(
    name,
    rows,
    expected_rows,
    expected_families,
):
    if len(rows) != expected_rows:
        fail(
            f"{name}: row count "
            f"{len(rows)} != {expected_rows}"
        )

    if any(
        set(row) != EXPECTED_KEYS
        for row in rows
    ):
        fail(
            f"{name}: row-schema mismatch"
        )

    row_ids = [
        row["row_id"]
        for row in rows
    ]

    if len(row_ids) != len(
        set(row_ids)
    ):
        fail(
            f"{name}: duplicate row IDs"
        )

    families = {
        row["family"]
        for row in rows
    }

    if len(families) != expected_families:
        fail(
            f"{name}: family count "
            f"{len(families)} "
            f"!= {expected_families}"
        )

    groups = defaultdict(
        list
    )

    family_rows = defaultdict(
        list
    )

    semantic_errors = []

    annotation_errors = []

    for row in rows:
        groups[
            row[
                "relational_group"
            ]
        ].append(row)

        family_rows[
            row["family"]
        ].append(row)

        if row["target"] not in LABELS:
            fail(
                f"{name}: invalid target label"
            )

        if hashlib.sha256(
            row[
                "evidence"
            ].encode(
                "utf-8"
            )
        ).hexdigest() != row[
            "evidence_hash"
        ]:
            fail(
                f"{name}: evidence hash mismatch"
            )

        blocks = split_evidence(
            row["evidence"]
        )

        ref = row[
            "target_evidence_ref"
        ]

        if ref not in blocks:
            fail(
                f"{name}: target evidence ref missing"
            )

        block = blocks[
            ref
        ]

        # Exact target must actually appear in the cited evidence block.
        if (
            row[
                "target_name"
            ].lower()
            not in block.lower()
        ):
            fail(
                f"{name}: cited target block "
                "does not name exact target"
            )

        derived = derive_decision(
            block
        )

        if derived != row[
            "target"
        ]:
            semantic_errors.append(
                (
                    row["row_id"],
                    row["target"],
                    derived,
                )
            )

        try:
            output = json.loads(
                row["output"]
            )
        except Exception:
            fail(
                f"{name}: invalid output JSON"
            )

        if set(output) != {
            "decision",
            "citations",
            "conditions",
            "rationale",
        }:
            fail(
                f"{name}: output keys mismatch"
            )

        if output[
            "decision"
        ] != row[
            "target"
        ]:
            annotation_errors.append(
                (
                    row["row_id"],
                    "decision",
                )
            )

        if output[
            "citations"
        ] != [ref]:
            annotation_errors.append(
                (
                    row["row_id"],
                    "citation",
                )
            )

        conditions = output[
            "conditions"
        ]

        if not isinstance(
            conditions,
            list,
        ):
            annotation_errors.append(
                (
                    row["row_id"],
                    "conditions_type",
                )
            )

            conditions = []

        for condition in conditions:
            if not re.match(
                r"^(SATISFIED|FAILED|UNRESOLVED) \| .+$",
                condition,
            ):
                annotation_errors.append(
                    (
                        row["row_id"],
                        "condition_format",
                    )
                )

        decision = output[
            "decision"
        ]

        if decision == "YES":
            if (
                not conditions
                or any(
                    not c.startswith(
                        "SATISFIED | "
                    )
                    for c in conditions
                )
            ):
                annotation_errors.append(
                    (
                        row["row_id"],
                        "yes_ledger",
                    )
                )

        elif decision == "CONDITIONAL":
            if (
                not any(
                    c.startswith(
                        "UNRESOLVED | "
                    )
                    for c in conditions
                )
                or any(
                    c.startswith(
                        "FAILED | "
                    )
                    for c in conditions
                )
            ):
                annotation_errors.append(
                    (
                        row["row_id"],
                        "conditional_ledger",
                    )
                )

        elif decision == "NO":
            if conditions:
                if not any(
                    c.startswith(
                        "FAILED | "
                    )
                    for c in conditions
                ):
                    annotation_errors.append(
                        (
                            row["row_id"],
                            "no_ledger",
                        )
                    )

        elif decision == "INSUFFICIENT":
            if conditions:
                annotation_errors.append(
                    (
                        row["row_id"],
                        "insufficient_ledger",
                    )
                )

        if (
            not isinstance(
                output[
                    "rationale"
                ],
                str,
            )
            or
            not output[
                "rationale"
            ].strip()
        ):
            annotation_errors.append(
                (
                    row["row_id"],
                    "rationale",
                )
            )


    if semantic_errors:
        fail(
            f"{name}: independent semantic "
            "errors: "
            f"{semantic_errors[:10]}"
        )

    if annotation_errors:
        fail(
            f"{name}: annotation-contract "
            "errors: "
            f"{annotation_errors[:10]}"
        )


    expected_groups = (
        expected_families
        * 8
    )

    if len(groups) != expected_groups:
        fail(
            f"{name}: relational groups "
            f"{len(groups)} != "
            f"{expected_groups}"
        )


    causal_tracks = 0
    causal_failures = []

    world_failures = []

    for group_id, group in groups.items():
        if len(group) != 16:
            fail(
                f"{name}: {group_id} "
                f"rows != 16"
            )

        worlds = defaultdict(
            list
        )

        targets = defaultdict(
            list
        )

        for row in group:
            worlds[
                row["world_index"]
            ].append(row)

            targets[
                row["target_index"]
            ].append(row)

        if set(worlds) != {
            0,
            1,
            2,
            3,
        }:
            fail(
                f"{name}: {group_id} "
                "world coverage failure"
            )

        if set(targets) != {
            0,
            1,
            2,
            3,
        }:
            fail(
                f"{name}: {group_id} "
                "target coverage failure"
            )


        for world_index, wrows in worlds.items():
            labels = Counter(
                row["target"]
                for row in wrows
            )

            if labels != Counter({
                "YES": 1,
                "NO": 1,
                "CONDITIONAL": 1,
                "INSUFFICIENT": 1,
            }):
                world_failures.append(
                    (
                        group_id,
                        world_index,
                        dict(labels),
                    )
                )

            hashes = {
                row[
                    "evidence_hash"
                ]
                for row in wrows
            }

            if len(hashes) != 1:
                world_failures.append(
                    (
                        group_id,
                        world_index,
                        "world evidence mismatch",
                    )
                )


        for target_index, trows in targets.items():
            causal_tracks += 1

            labels = Counter(
                row["target"]
                for row in trows
            )

            if labels != Counter({
                "YES": 1,
                "NO": 1,
                "CONDITIONAL": 1,
                "INSUFFICIENT": 1,
            }):
                causal_failures.append(
                    (
                        group_id,
                        target_index,
                        dict(labels),
                    )
                )

            questions = {
                row[
                    "question"
                ]
                for row in trows
            }

            if len(questions) != 1:
                causal_failures.append(
                    (
                        group_id,
                        target_index,
                        "question changed across worlds",
                    )
                )

            target_blocks = []

            for row in trows:
                blocks = split_evidence(
                    row["evidence"]
                )

                target_blocks.append(
                    blocks[
                        row[
                            "target_evidence_ref"
                        ]
                    ]
                )

            if len(
                {
                    text_hash
                    for text_hash in (
                        hashlib.sha256(
                            x.encode(
                                "utf-8"
                            )
                        ).hexdigest()
                        for x in target_blocks
                    )
                }
            ) != 4:
                causal_failures.append(
                    (
                        group_id,
                        target_index,
                        "target evidence not four-state unique",
                    )
                )


    if world_failures:
        fail(
            f"{name}: world Latin failures: "
            f"{world_failures[:10]}"
        )

    if causal_failures:
        fail(
            f"{name}: fixed-target causal "
            "failures: "
            f"{causal_failures[:10]}"
        )


    for family, frows in family_rows.items():
        if len(frows) != 128:
            fail(
                f"{name}: family {family} "
                f"rows != 128"
            )

        labels = Counter(
            row["target"]
            for row in frows
        )

        if labels != Counter({
            "YES": 32,
            "NO": 32,
            "CONDITIONAL": 32,
            "INSUFFICIENT": 32,
        }):
            fail(
                f"{name}: family {family} "
                "label imbalance"
            )

        input_orders = Counter(
            row[
                "input_order"
            ]
            for row in frows
        )

        if input_orders != Counter({
            "question_first": 64,
            "evidence_first": 64,
        }):
            fail(
                f"{name}: family {family} "
                "input-order imbalance"
            )

        surfaces = Counter(
            row[
                "surface_variant"
            ]
            for row in frows
        )

        if surfaces != Counter({
            0: 64,
            1: 64,
        }):
            fail(
                f"{name}: family {family} "
                "surface imbalance"
            )

        variants = Counter(
            row[
                "group_variant"
            ]
            for row in frows
        )

        if variants != Counter({
            0: 64,
            1: 64,
        }):
            fail(
                f"{name}: family {family} "
                "group-variant imbalance"
            )


    return {
        "rows":
            len(rows),

        "families":
            len(families),

        "relational_groups":
            len(groups),

        "semantic_errors":
            len(semantic_errors),

        "annotation_errors":
            len(annotation_errors),

        "fixed_target_causal_tracks":
            causal_tracks,

        "causal_failures":
            len(causal_failures),

        "world_latin_failures":
            len(world_failures),

        "labels":
            dict(
                sorted(
                    Counter(
                        row["target"]
                        for row in rows
                    ).items()
                )
            ),

        "operators":
            dict(
                sorted(
                    Counter(
                        row["operator"]
                        for row in rows
                    ).items()
                )
            ),
    }


def main():
    train = load_jsonl(
        TRAIN
    )

    dev = load_jsonl(
        DEV
    )

    pilot = load_jsonl(
        PILOT
    )


    # ================================================================
    # SPLIT LEAKAGE
    # ================================================================

    train_families = {
        row["family"]
        for row in train
    }

    dev_families = {
        row["family"]
        for row in dev
    }

    if train_families & dev_families:
        fail(
            "TRAIN/DEVELOPMENT family leakage"
        )


    train_inputs = {
        norm(
            row["input"]
        )
        for row in train
    }

    dev_inputs = {
        norm(
            row["input"]
        )
        for row in dev
    }

    exact_split_input_overlap = len(
        train_inputs
        & dev_inputs
    )

    if exact_split_input_overlap:
        fail(
            "exact TRAIN/DEVELOPMENT input overlap"
        )


    pilot_inputs = {
        norm(
            row["input"]
        )
        for row in pilot
        if isinstance(
            row.get(
                "input"
            ),
            str,
        )
    }

    production_inputs = (
        train_inputs
        | dev_inputs
    )

    pilot_exact_overlap = len(
        pilot_inputs
        & production_inputs
    )

    if pilot_exact_overlap:
        fail(
            "exact frozen-pilot input reuse detected"
        )


    train_result = audit_split(
        "TRAIN",
        train,
        4096,
        32,
    )

    dev_result = audit_split(
        "DEVELOPMENT",
        dev,
        1024,
        8,
    )


    # ================================================================
    # QUESTION / EVIDENCE EXACT-MARGINAL SANITY
    # ================================================================

    combined = (
        train
        + dev
    )

    question_buckets = defaultdict(
        Counter
    )

    evidence_buckets = defaultdict(
        Counter
    )

    for row in combined:
        question_buckets[
            row["question"]
        ][
            row["target"]
        ] += 1

        evidence_buckets[
            row["evidence_hash"]
        ][
            row["target"]
        ] += 1


    bad_questions = [
        q
        for q, counts
        in question_buckets.items()
        if counts != Counter({
            "YES": 1,
            "NO": 1,
            "CONDITIONAL": 1,
            "INSUFFICIENT": 1,
        })
    ]

    bad_evidence = [
        h
        for h, counts
        in evidence_buckets.items()
        if counts != Counter({
            "YES": 1,
            "NO": 1,
            "CONDITIONAL": 1,
            "INSUFFICIENT": 1,
        })
    ]


    if bad_questions:
        fail(
            "exact question marginal "
            "is label-predictive"
        )

    if bad_evidence:
        fail(
            "exact evidence-pack marginal "
            "is label-predictive"
        )


    report = {
        "status":
            "PASS",

        "train":
            train_result,

        "development":
            dev_result,

        "combined": {
            "rows":
                len(combined),

            "families":
                len(
                    {
                        row["family"]
                        for row in combined
                    }
                ),

            "relational_groups":
                len(
                    {
                        row[
                            "relational_group"
                        ]
                        for row in combined
                    }
                ),

            "fixed_target_causal_tracks":
                train_result[
                    "fixed_target_causal_tracks"
                ]
                + dev_result[
                    "fixed_target_causal_tracks"
                ],

            "semantic_errors":
                0,

            "causal_failures":
                0,
        },

        "leakage": {
            "train_development_family_overlap":
                0,

            "train_development_exact_input_overlap":
                exact_split_input_overlap,

            "pilot_exact_input_overlap":
                pilot_exact_overlap,

            "bad_exact_question_marginals":
                len(
                    bad_questions
                ),

            "bad_exact_evidence_marginals":
                len(
                    bad_evidence
                ),
        },

        "model_loads":
            0,

        "inference_calls":
            0,

        "challenge_v4_files_read":
            0,
    }


    OUT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


    lines = [
        "SENTINEL V6 FULL CURRICULUM — SEMANTIC / CAUSAL AUDIT",
        "=" * 78,
        "",
        "STATUS: PASS",
        "",
        f"TRAIN rows: {train_result['rows']}",
        f"TRAIN families: {train_result['families']}",
        f"TRAIN relational groups: {train_result['relational_groups']}",
        f"TRAIN causal tracks: {train_result['fixed_target_causal_tracks']}",
        "",
        f"DEVELOPMENT rows: {dev_result['rows']}",
        f"DEVELOPMENT families: {dev_result['families']}",
        f"DEVELOPMENT relational groups: {dev_result['relational_groups']}",
        f"DEVELOPMENT causal tracks: {dev_result['fixed_target_causal_tracks']}",
        "",
        "Semantic disagreements: 0",
        "Annotation errors: 0",
        "Fixed-target causal failures: 0",
        "World Latin failures: 0",
        "TRAIN/DEV family overlap: 0",
        f"TRAIN/DEV exact input overlap: {exact_split_input_overlap}",
        f"Pilot exact input overlap: {pilot_exact_overlap}",
        "Exact-question shortcut failures: 0",
        "Exact-evidence shortcut failures: 0",
        "",
        "Model loads: 0",
        "Inference: 0",
        "Challenge V4 files read: 0",
    ]


    OUT_TXT.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


    print()
    print("=" * 72)
    print("V6 FULL CURRICULUM SEMANTIC / CAUSAL AUDIT")
    print("=" * 72)
    print("PASS")
    print()
    print(
        "TRAIN:",
        train_result,
    )
    print()
    print(
        "DEVELOPMENT:",
        dev_result,
    )
    print()
    print(
        "Combined fixed-target causal tracks:",
        report[
            "combined"
        ][
            "fixed_target_causal_tracks"
        ],
    )
    print(
        "Semantic disagreements: 0"
    )
    print(
        "Causal failures: 0"
    )
    print(
        "TRAIN/DEV family overlap: 0"
    )
    print(
        "TRAIN/DEV exact input overlap:",
        exact_split_input_overlap,
    )
    print(
        "Frozen-pilot exact input overlap:",
        pilot_exact_overlap,
    )
    print(
        "Question marginal failures:",
        len(
            bad_questions
        ),
    )
    print(
        "Evidence marginal failures:",
        len(
            bad_evidence
        ),
    )
    print()
    print(
        "Audit JSON SHA256:",
        sha(
            OUT_JSON
        ),
    )
    print(
        "Audit TXT SHA256:",
        sha(
            OUT_TXT
        ),
    )
    print()
    print("Model loads: 0")
    print("Inference: 0")
    print("Challenge V4 files read: 0")


if __name__ == "__main__":
    main()
