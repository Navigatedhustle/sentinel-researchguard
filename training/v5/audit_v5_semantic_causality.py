"""
Sentinel V5 semantic / causal consistency audit.

Purpose
-------

Verify that every generated row faithfully corresponds to the
executable reasoning engine and that the intended Latin-square
counterfactual structure is actually present.

Checks include:

- dataset hashes
- row schema
- family/split membership
- Latin-square gold label
- exact question realization
- exact input rendering
- evidence hash
- expected governing facts
- expected local distractors
- exact minimal citation set
- no distractor citations
- exact conditions
- exact rationale
- same-question / different-evidence counterfactuals
- same-evidence / different-question counterfactuals
- four-label causal rotations

This script does NOT read Challenge V1 or Challenge V2.
It does NOT train or evaluate Sentinel.
"""

from collections import Counter, defaultdict
from pathlib import Path

import hashlib
import json
import re
import sys

from training.v5.build_v5_curriculum import (
    TRAIN_QUESTION_TEMPLATES,
    DEVELOPMENT_QUESTION_TEMPLATES,
    render_input,
)

from training.v5.v5_family_realizers import (
    FAMILY_INDEX,
)

from training.v5.v5_relational_engine import (
    LABELS,
    validate_state,
)


BASE = Path(
    "/workspace"
)

TRAIN = (
    BASE
    / "training/v5/datasets/"
      "sentinel_v5_train.jsonl"
)

DEV = (
    BASE
    / "training/v5/datasets/"
      "sentinel_v5_development.jsonl"
)


EXPECTED_TRAIN_SHA = (
    "945eef0841648911aaee5b88a5f7efd6"
    "c84bcad7399697f7f786f4774645d972"
)

EXPECTED_DEV_SHA = (
    "4e9fe2effa908d8994fdfb3be486661f"
    "9ef178860725fbd887d60b5181900124"
)


EXPECTED_ROW_KEYS = {
    "asset",
    "evidence",
    "evidence_hash",
    "family",
    "input",
    "input_order",
    "instruction",
    "operator",
    "output",
    "question",
    "relational_group",
    "scenario_id",
    "split",
    "surface_variant",
    "target",
    "target_index",
    "world",
}


EXPECTED_OUTPUT_KEYS = {
    "decision",
    "citations",
    "conditions",
    "rationale",
}


FAILURES = []


def fail(
    message,
):

    FAILURES.append(
        message
    )


def sha256(
    path,
):

    h = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(
                block
            )

    return h.hexdigest()


def hash_text(
    text,
):

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def load_jsonl(
    path,
):

    rows = []

    with path.open() as f:

        for line in f:

            line = line.strip()

            if line:

                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def parse_evidence(
    evidence,
):
    """
    Parse:

        [E1] text

        [E2] text

    into:

        {
            "E1": "text",
            "E2": "text",
        }
    """

    matches = list(
        re.finditer(
            r"(?ms)^\[(E\d+)\]\s+"
            r"(.*?)"
            r"(?=^\[E\d+\]\s+|\Z)",
            evidence,
        )
    )

    chunks = {}

    for match in matches:

        eid = match.group(
            1
        )

        text = match.group(
            2
        ).strip()

        if eid in chunks:

            raise AssertionError(
                f"Duplicate evidence ID: {eid}"
            )

        chunks[
            eid
        ] = text

    return chunks


def question_templates_for(
    split,
):

    if split == "train":

        return TRAIN_QUESTION_TEMPLATES

    if split == "development":

        return DEVELOPMENT_QUESTION_TEMPLATES

    raise ValueError(
        split
    )


def expected_scenario_id(
    row,
):

    return (
        f"v5__{row['split']}__"
        f"{row['operator']}__"
        f"{row['family']}__"
        f"world{row['world']}__"
        f"target{row['target_index']}__"
        f"surface{row['surface_variant']}"
    )


def audit_row(
    row,
):

    # ================================================================
    # BASIC SCHEMA
    # ================================================================

    if set(
        row
    ) != EXPECTED_ROW_KEYS:

        fail(
            f"Row key mismatch: "
            f"{row.get('scenario_id')}"
        )

        return


    split = row[
        "split"
    ]

    family = row[
        "family"
    ]

    operator = row[
        "operator"
    ]

    asset = row[
        "asset"
    ]

    world = row[
        "world"
    ]

    target_index = row[
        "target_index"
    ]

    surface = row[
        "surface_variant"
    ]

    label = row[
        "target"
    ]


    # ================================================================
    # ONTOLOGY / SPLIT
    # ================================================================

    if family not in FAMILY_INDEX:

        fail(
            f"Unknown family: {family}"
        )

        return


    spec = FAMILY_INDEX[
        family
    ]


    if spec[
        "operator"
    ] != operator:

        fail(
            f"Operator mismatch: "
            f"{row['scenario_id']}"
        )


    if spec[
        "split"
    ] != split:

        fail(
            f"Split mismatch: "
            f"{row['scenario_id']}"
        )


    # ================================================================
    # CONSTRUCTION COORDINATES
    # ================================================================

    if world not in {
        0,
        1,
        2,
        3,
    }:

        fail(
            f"Bad world index: "
            f"{row['scenario_id']}"
        )


    if target_index not in {
        0,
        1,
        2,
        3,
    }:

        fail(
            f"Bad target index: "
            f"{row['scenario_id']}"
        )


    expected_label = LABELS[
        (
            target_index
            + world
        )
        % 4
    ]


    if label != expected_label:

        fail(
            f"Latin-square label mismatch: "
            f"{row['scenario_id']} "
            f"{label} != {expected_label}"
        )


    # ================================================================
    # SCENARIO ID / GROUP
    # ================================================================

    if row[
        "scenario_id"
    ] != expected_scenario_id(
        row
    ):

        fail(
            f"Scenario ID mismatch: "
            f"{row['scenario_id']}"
        )


    expected_group = (
        f"{operator}__{family}"
    )


    if row[
        "relational_group"
    ] != expected_group:

        fail(
            f"Relational group mismatch: "
            f"{row['scenario_id']}"
        )


    # ================================================================
    # REGENERATE INTENDED SEMANTIC STATE
    # ================================================================

    semantic_variant = (
        surface
        % 3
    )


    try:

        state = validate_state(
            operator=operator,
            family=family,
            asset=asset,
            label=label,
            variant=semantic_variant,
        )

    except Exception as exc:

        fail(
            f"State regeneration failed: "
            f"{row['scenario_id']} "
            f"{type(exc).__name__}: {exc}"
        )

        return


    # ================================================================
    # QUESTION
    # ================================================================

    templates = question_templates_for(
        split
    )


    if not (
        0
        <= surface
        < len(
            templates
        )
    ):

        fail(
            f"Surface index outside split template range: "
            f"{row['scenario_id']}"
        )

        return


    expected_question = templates[
        surface
    ].format(
        action=state[
            "action"
        ]
    )


    if row[
        "question"
    ] != expected_question:

        fail(
            f"Question realization mismatch: "
            f"{row['scenario_id']}"
        )


    # ================================================================
    # EVIDENCE HASH / INPUT RENDERING
    # ================================================================

    if row[
        "evidence_hash"
    ] != hash_text(
        row[
            "evidence"
        ]
    ):

        fail(
            f"Evidence hash mismatch: "
            f"{row['scenario_id']}"
        )


    expected_input = render_input(
        row[
            "question"
        ],
        row[
            "evidence"
        ],
        row[
            "input_order"
        ],
    )


    if row[
        "input"
    ] != expected_input:

        fail(
            f"Rendered input mismatch: "
            f"{row['scenario_id']}"
        )


    # ================================================================
    # PARSE OUTPUT
    # ================================================================

    try:

        output = json.loads(
            row[
                "output"
            ]
        )

    except Exception as exc:

        fail(
            f"Invalid output JSON: "
            f"{row['scenario_id']} "
            f"{exc}"
        )

        return


    if set(
        output
    ) != EXPECTED_OUTPUT_KEYS:

        fail(
            f"Output schema mismatch: "
            f"{row['scenario_id']}"
        )


    if output[
        "decision"
    ] != label:

        fail(
            f"Output decision mismatch: "
            f"{row['scenario_id']}"
        )


    if output[
        "conditions"
    ] != state[
        "conditions"
    ]:

        fail(
            f"Condition mismatch: "
            f"{row['scenario_id']}"
        )


    if output[
        "rationale"
    ] != state[
        "rationale"
    ]:

        fail(
            f"Rationale mismatch: "
            f"{row['scenario_id']}"
        )


    # ================================================================
    # EVIDENCE CHUNKS
    # ================================================================

    try:

        chunks = parse_evidence(
            row[
                "evidence"
            ]
        )

    except Exception as exc:

        fail(
            f"Evidence parse failure: "
            f"{row['scenario_id']} "
            f"{exc}"
        )

        return


    if not chunks:

        fail(
            f"No evidence chunks parsed: "
            f"{row['scenario_id']}"
        )

        return


    # Build text -> IDs lookup.

    text_to_ids = defaultdict(
        list
    )


    for eid, text in chunks.items():

        text_to_ids[
            text
        ].append(
            eid
        )


    # ================================================================
    # GOVERNING FACT PRESENCE + MINIMAL CITATIONS
    # ================================================================

    expected_citations = []


    for fact in state[
        "governing_facts"
    ]:

        ids = text_to_ids.get(
            fact,
            [],
        )


        if len(
            ids
        ) != 1:

            fail(
                f"Governing fact count != 1: "
                f"{row['scenario_id']}"
            )

            continue


        expected_citations.append(
            ids[
                0
            ]
        )


    actual_citations = output[
        "citations"
    ]


    if actual_citations != expected_citations:

        fail(
            f"Exact minimal citation mismatch: "
            f"{row['scenario_id']}"
        )


    if len(
        actual_citations
    ) != len(
        set(
            actual_citations
        )
    ):

        fail(
            f"Duplicate citations: "
            f"{row['scenario_id']}"
        )


    # ================================================================
    # DISTRACTOR PRESENCE / NON-CITATION
    # ================================================================

    cited_texts = {
        chunks[
            eid
        ]
        for eid
        in actual_citations
        if eid in chunks
    }


    for distractor in state[
        "distractors"
    ]:

        ids = text_to_ids.get(
            distractor,
            [],
        )


        if len(
            ids
        ) != 1:

            fail(
                f"Local distractor count != 1: "
                f"{row['scenario_id']}"
            )

            continue


        if distractor in cited_texts:

            fail(
                f"Distractor incorrectly cited: "
                f"{row['scenario_id']}"
            )


    # ================================================================
    # LABEL-SPECIFIC CONTRACT
    # ================================================================

    if label == "INSUFFICIENT":

        if state[
            "governing_facts"
        ]:

            fail(
                f"INSUFFICIENT has governing facts: "
                f"{row['scenario_id']}"
            )


        if actual_citations:

            fail(
                f"INSUFFICIENT has citations: "
                f"{row['scenario_id']}"
            )


        if output[
            "conditions"
        ]:

            fail(
                f"INSUFFICIENT has conditions: "
                f"{row['scenario_id']}"
            )


    elif label == "CONDITIONAL":

        if not actual_citations:

            fail(
                f"CONDITIONAL lacks citations: "
                f"{row['scenario_id']}"
            )


        if not output[
            "conditions"
        ]:

            fail(
                f"CONDITIONAL lacks condition: "
                f"{row['scenario_id']}"
            )


    else:

        if not actual_citations:

            fail(
                f"{label} lacks governing citations: "
                f"{row['scenario_id']}"
            )


        if output[
            "conditions"
        ]:

            fail(
                f"{label} unexpectedly has conditions: "
                f"{row['scenario_id']}"
            )


def audit_counterfactual_structure(
    rows,
    split,
):

    print()
    print(
        "=" * 88
    )

    print(
        f"COUNTERFACTUAL STRUCTURE — {split.upper()}"
    )

    print(
        "=" * 88
    )


    # ================================================================
    # SAME QUESTION + DIFFERENT EVIDENCE => ALL FOUR LABELS
    # ================================================================

    by_question_coordinate = defaultdict(
        list
    )


    for row in rows:

        key = (
            row[
                "family"
            ],
            row[
                "surface_variant"
            ],
            row[
                "target_index"
            ],
        )

        by_question_coordinate[
            key
        ].append(
            row
        )


    bad_question_groups = 0


    for key, group in by_question_coordinate.items():

        if len(
            group
        ) != 4:

            bad_question_groups += 1
            continue


        questions = {
            row[
                "question"
            ]
            for row in group
        }

        worlds = {
            row[
                "world"
            ]
            for row in group
        }

        evidence = {
            row[
                "evidence_hash"
            ]
            for row in group
        }

        labels = {
            row[
                "target"
            ]
            for row in group
        }


        if len(
            questions
        ) != 1:

            bad_question_groups += 1


        if worlds != {
            0,
            1,
            2,
            3,
        }:

            bad_question_groups += 1


        if len(
            evidence
        ) != 4:

            bad_question_groups += 1


        if labels != set(
            LABELS
        ):

            bad_question_groups += 1


    if bad_question_groups:

        fail(
            f"{split}: bad same-question "
            f"counterfactual groups={bad_question_groups}"
        )


    # ================================================================
    # SAME EVIDENCE + DIFFERENT QUESTIONS => ALL FOUR LABELS
    # ================================================================

    by_evidence_coordinate = defaultdict(
        list
    )


    for row in rows:

        key = (
            row[
                "family"
            ],
            row[
                "surface_variant"
            ],
            row[
                "world"
            ],
        )

        by_evidence_coordinate[
            key
        ].append(
            row
        )


    bad_evidence_groups = 0


    for key, group in by_evidence_coordinate.items():

        if len(
            group
        ) != 4:

            bad_evidence_groups += 1
            continue


        evidence = {
            row[
                "evidence_hash"
            ]
            for row in group
        }

        targets = {
            row[
                "target_index"
            ]
            for row in group
        }

        questions = {
            row[
                "question"
            ]
            for row in group
        }

        labels = {
            row[
                "target"
            ]
            for row in group
        }


        if len(
            evidence
        ) != 1:

            bad_evidence_groups += 1


        if targets != {
            0,
            1,
            2,
            3,
        }:

            bad_evidence_groups += 1


        if len(
            questions
        ) != 4:

            bad_evidence_groups += 1


        if labels != set(
            LABELS
        ):

            bad_evidence_groups += 1


    if bad_evidence_groups:

        fail(
            f"{split}: bad same-evidence "
            f"counterfactual groups={bad_evidence_groups}"
        )


    print(
        "same-question counterfactual groups:",
        len(
            by_question_coordinate
        ),
    )

    print(
        "bad same-question groups:",
        bad_question_groups,
    )

    print(
        "same-evidence counterfactual groups:",
        len(
            by_evidence_coordinate
        ),
    )

    print(
        "bad same-evidence groups:",
        bad_evidence_groups,
    )


def main():

    print(
        "=" * 88
    )

    print(
        "SENTINEL V5 SEMANTIC / CAUSAL CONSISTENCY AUDIT"
    )

    print(
        "=" * 88
    )


    # ================================================================
    # HASH INTEGRITY
    # ================================================================

    train_sha = sha256(
        TRAIN
    )

    dev_sha = sha256(
        DEV
    )


    print()
    print(
        "TRAIN SHA:",
        train_sha,
    )

    print(
        "DEV SHA:  ",
        dev_sha,
    )


    if train_sha != EXPECTED_TRAIN_SHA:

        fail(
            "TRAIN hash mismatch."
        )


    if dev_sha != EXPECTED_DEV_SHA:

        fail(
            "DEVELOPMENT hash mismatch."
        )


    # ================================================================
    # LOAD
    # ================================================================

    train_rows = load_jsonl(
        TRAIN
    )

    dev_rows = load_jsonl(
        DEV
    )


    if len(
        train_rows
    ) != 6144:

        fail(
            "TRAIN row count mismatch."
        )


    if len(
        dev_rows
    ) != 1024:

        fail(
            "DEVELOPMENT row count mismatch."
        )


    print()
    print(
        "TRAIN rows:",
        len(
            train_rows
        ),
    )

    print(
        "DEV rows:",
        len(
            dev_rows
        ),
    )


    # ================================================================
    # ROW-BY-ROW SEMANTIC AUDIT
    # ================================================================

    print()
    print(
        "=" * 88
    )

    print(
        "ROW-BY-ROW SEMANTIC REGENERATION"
    )

    print(
        "=" * 88
    )


    before = len(
        FAILURES
    )


    for row in train_rows:

        audit_row(
            row
        )


    train_failures = (
        len(
            FAILURES
        )
        - before
    )


    before = len(
        FAILURES
    )


    for row in dev_rows:

        audit_row(
            row
        )


    dev_failures = (
        len(
            FAILURES
        )
        - before
    )


    print(
        "TRAIN rows checked:",
        len(
            train_rows
        )
    )

    print(
        "TRAIN semantic failures:",
        train_failures,
    )

    print(
        "DEV rows checked:",
        len(
            dev_rows
        )
    )

    print(
        "DEV semantic failures:",
        dev_failures,
    )


    # ================================================================
    # CAUSAL STRUCTURE
    # ================================================================

    audit_counterfactual_structure(
        train_rows,
        "train",
    )

    audit_counterfactual_structure(
        dev_rows,
        "development",
    )


    # ================================================================
    # AGGREGATE SANITY
    # ================================================================

    print()
    print(
        "=" * 88
    )

    print(
        "AGGREGATE SEMANTIC SANITY"
    )

    print(
        "=" * 88
    )


    all_rows = (
        train_rows
        + dev_rows
    )


    label_counts = Counter(
        row[
            "target"
        ]
        for row in all_rows
    )


    citation_cardinality = Counter()


    for row in all_rows:

        output = json.loads(
            row[
                "output"
            ]
        )

        citation_cardinality[
            (
                row[
                    "target"
                ],
                len(
                    output[
                        "citations"
                    ]
                ),
            )
        ] += 1


    no_rationales = {
        json.loads(
            row[
                "output"
            ]
        )[
            "rationale"
        ]
        for row in all_rows
        if row[
            "target"
        ] == "NO"
    }


    insufficient_rationales = {
        json.loads(
            row[
                "output"
            ]
        )[
            "rationale"
        ]
        for row in all_rows
        if row[
            "target"
        ] == "INSUFFICIENT"
    }


    print(
        "labels:",
        dict(
            sorted(
                label_counts.items()
            )
        ),
    )

    print(
        "citation cardinality by label:"
    )

    for key, count in sorted(
        citation_cardinality.items()
    ):

        print(
            " ",
            key,
            count,
        )


    print(
        "distinct NO rationales:",
        len(
            no_rationales
        ),
    )

    print(
        "distinct INSUFFICIENT rationales:",
        len(
            insufficient_rationales
        ),
    )


    if len(
        no_rationales
    ) < 6:

        fail(
            "NO rationale diversity unexpectedly low."
        )


    if len(
        insufficient_rationales
    ) < 6:

        fail(
            "INSUFFICIENT rationale diversity unexpectedly low."
        )


    # ================================================================
    # FINAL VERDICT
    # ================================================================

    print()
    print(
        "=" * 88
    )

    print(
        "SEMANTIC / CAUSAL AUDIT VERDICT"
    )

    print(
        "=" * 88
    )


    if FAILURES:

        print()
        print(
            "FAILURES:",
            len(
                FAILURES
            ),
        )

        # Do not dump thousands of failures.
        # First 25 are enough to diagnose safely.

        for item in FAILURES[
            :25
        ]:

            print(
                "  -",
                item,
            )


        if len(
            FAILURES
        ) > 25:

            print(
                "  ... additional failures suppressed:",
                len(
                    FAILURES
                )
                - 25,
            )


        print()
        print(
            "V5 SEMANTIC / CAUSAL AUDIT: FAIL"
        )

        print(
            "DO NOT FREEZE."
        )

        print(
            "DO NOT TRAIN."
        )

        sys.exit(
            1
        )


    print()
    print(
        "FAILURES: 0"
    )

    print(
        "V5 SEMANTIC / CAUSAL AUDIT: PASS"
    )

    print(
        "All 7168 rows exactly match the executable reasoning engine."
    )

    print(
        "Counterfactual causal structure is intact."
    )

    print(
        "Dataset remains unfrozen."
    )

    print(
        "No model training performed."
    )


if __name__ == "__main__":
    main()
