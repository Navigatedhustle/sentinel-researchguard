from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import sys


BASE = Path("/workspace")

PILOT = (
    BASE
    / "training/v5/pilot/"
      "sentinel_v5_relational_pilot.jsonl"
)

EXPECTED_PILOT_SHA = (
    "a9d6c5714e8e4e828a4cfbf499f75e21"
    "5dae10e78dc32560a87473eb198930ca"
)

CHALLENGE_V2 = (
    BASE
    / "experiments/challenge_v2/"
      "sentinel_challenge_v2.json"
)

EXPECTED_CHALLENGE_SHA = (
    "30eb37ff39382f0d205adb7c17383b240"
    "25601ea3db1dd61f22b715af6c1e482"
)

LABELS = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
}

EXPECTED_OPERATORS = {
    "operative_rule_selection",
    "prohibition_short_circuit",
    "permission_condition_ledger",
    "exception_activation_chain",
    "authority_state_resolution",
    "category_rule_composition",
    "scope_destination_resolution",
    "evidence_relevance_boundary",
}

ERRORS = []


def fail(message):
    ERRORS.append(message)


def norm(text):
    return re.sub(
        r"\s+",
        " ",
        str(text).strip().lower(),
    )


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def parse_evidence(text):
    pattern = re.compile(
        r"\[(E\d+)\]\s*(.*?)"
        r"(?=\n\n\[E\d+\]|\Z)",
        flags=re.S,
    )

    return {
        eid: body.strip()
        for eid, body
        in pattern.findall(text)
    }


# ============================================================
# INDEPENDENT OPERATOR SEMANTIC SIGNATURES
# ============================================================
#
# infer_label() receives only:
#
#     operator
#     target-specific evidence text
#
# It does NOT receive:
#
#     row["target"]
#     output["decision"]
#     scenario_id
#
# ============================================================


SIGNATURES = {

    "operative_rule_selection": {

        "YES": [
            "active project charter is the operative rule",
            "active project charter permits staging",
            "technical review is complete",
        ],

        "NO": [
            "active project charter is the operative rule",
            "active project charter expressly prohibits staging",
        ],

        "CONDITIONAL": [
            "active project charter is the operative rule",
            "technical-review status is not established",
        ],

        "INSUFFICIENT": [
            "no supplied rule governs whether it may be staged",
        ],
    },


    "prohibition_short_circuit": {

        "YES": [
            "controlling interface rule permits",
            "transport encryption is active",
        ],

        "NO": [
            "controlling interface rule expressly prohibits transmission",
        ],

        "CONDITIONAL": [
            "release-authorization status is not established",
        ],

        "INSUFFICIENT": [
            "provides no rule for transmission through interface r",
        ],
    },


    "permission_condition_ledger": {

        "YES": [
            "release",
            "is permitted only after technical review and advance notice",
            "technical review is complete",
            "advance notice has been recorded",
        ],

        "NO": [
            "governing release rule expressly prohibits release",
        ],

        "CONDITIONAL": [
            "is permitted only after technical review and advance notice",
            "advance-notice status is not established",
        ],

        "INSUFFICIENT": [
            "no supplied provision governs release to the registered research workspace",
        ],
    },


    "exception_activation_chain": {

        "YES": [
            "standing rule prohibits moving",
            "activates an exception to the standing prohibition",
            "the maintenance window is active and operations clearance is recorded",
        ],

        "NO": [
            "standing rule prohibits moving",
            "no maintenance window is active",
        ],

        "CONDITIONAL": [
            "standing rule prohibits moving",
            "maintenance window is active",
            "operations-clearance status is not established",
        ],

        "INSUFFICIENT": [
            "no supplied rule governs whether it may leave the normal processing enclave",
        ],
    },


    "authority_state_resolution": {

        "YES": [
            "may be approved by the principal reviewer or a temporary reviewer",
            "recorded as currently effective",
        ],

        "NO": [
            "may be approved by the principal reviewer or a temporary reviewer",
            "was revoked before this publication request",
        ],

        "CONDITIONAL": [
            "may be approved by the principal reviewer or a temporary reviewer",
            "does not establish whether that assignment is currently effective",
        ],

        "INSUFFICIENT": [
            "does not state whether a temporary reviewer may approve publication",
        ],
    },


    "category_rule_composition": {

        "YES": [
            "satisfies the definition of a managed-compute workload",
            "managed-compute workloads",
            "may run on the shared analysis cluster",
            "the cluster is certified",
        ],

        "NO": [
            "satisfies the definition of a restricted-compute workload",
            "restricted-compute workloads",
            "expressly prohibited from running on the shared analysis cluster",
        ],

        "CONDITIONAL": [
            "satisfies the definition of a managed-compute workload",
            "current certification status is not established",
        ],

        "INSUFFICIENT": [
            "satisfies the definition of a managed-compute workload",
            "no supplied provision states whether managed-compute workloads may run",
        ],
    },


    "scope_destination_resolution": {

        "YES": [
            "classified as an approved recipient class",
            "required recipient controls are active",
        ],

        "NO": [
            "explicitly classified as an excluded recipient class",
            "may not be sent to excluded recipient classes",
        ],

        "CONDITIONAL": [
            "classified as an approved recipient class",
            "agreement status is not established",
        ],

        "INSUFFICIENT": [
            "does not classify recipient group delta",
        ],
    },


    "evidence_relevance_boundary": {

        "YES": [
            "channel k rule permits submission",
            "transmission validation is complete",
        ],

        "NO": [
            "channel k rule expressly prohibits submission",
        ],

        "CONDITIONAL": [
            "channel k rule permits submission",
            "validation status is not established",
        ],

        "INSUFFICIENT": [
            "contains no provision governing channel k",
        ],
    },
}


def infer_label(
    operator,
    target_evidence,
):
    text = norm(
        target_evidence
    )

    matches = []

    for label, signatures in (
        SIGNATURES[
            operator
        ].items()
    ):

        if all(
            norm(signature)
            in text
            for signature
            in signatures
        ):
            matches.append(
                label
            )

    if len(matches) == 1:
        return matches[0], matches

    return None, matches


# ============================================================
# EXPECTED CITATION DEPTH
# ============================================================
#
# This follows the pilot architecture—not an arbitrary demand that
# every operator be multi-citation.
# ============================================================


def expected_citation_count(
    operator,
    label,
):

    if operator in {
        "prohibition_short_circuit",
        "evidence_relevance_boundary",
    }:
        return 1

    if operator == "operative_rule_selection":
        return (
            1
            if label == "INSUFFICIENT"
            else 2
        )

    if operator == "permission_condition_ledger":
        return (
            2
            if label in {
                "YES",
                "CONDITIONAL",
            }
            else 1
        )

    if operator in {
        "exception_activation_chain",
        "authority_state_resolution",
        "category_rule_composition",
        "scope_destination_resolution",
    }:
        return (
            1
            if label == "INSUFFICIENT"
            else 2
        )

    raise ValueError(
        operator
    )


def main():

    print("=" * 88)
    print(
        "SENTINEL V5 RELATIONAL PILOT — "
        "SEMANTIC / CITATION AUDIT"
    )
    print("=" * 88)


    # ========================================================
    # IMMUTABILITY
    # ========================================================

    pilot_sha = sha256(
        PILOT
    )

    challenge_sha = sha256(
        CHALLENGE_V2
    )

    print(
        "pilot SHA:",
        pilot_sha,
    )

    print(
        "Challenge V2 SHA:",
        challenge_sha,
    )

    if pilot_sha != EXPECTED_PILOT_SHA:
        fail(
            "Pilot dataset changed before semantic audit."
        )

    if challenge_sha != EXPECTED_CHALLENGE_SHA:
        fail(
            "Challenge V2 hash mismatch."
        )


    # ========================================================
    # LOAD PILOT ONLY
    # ========================================================

    rows = []

    with PILOT.open() as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(
                        line
                    )
                )

    print()
    print(
        "rows:",
        len(rows),
    )

    if len(rows) != 256:
        fail(
            f"Expected 256 pilot rows, got {len(rows)}."
        )


    # ========================================================
    # SEMANTIC / CITATION AUDIT
    # ========================================================

    target_question_errors = 0

    target_location_errors = 0
    citation_coverage_errors = 0
    citation_depth_errors = 0

    oracle_ambiguous = 0
    oracle_mismatches = 0

    surface_consistency_errors = 0

    oracle_counts = Counter()
    citation_counts = Counter()

    per_operator = defaultdict(
        lambda: {
            "rows": 0,
            "oracle_correct": 0,
            "citation_exact": 0,
        }
    )

    logical_groups = defaultdict(
        list
    )


    for row in rows:

        operator = row[
            "operator"
        ]

        if operator not in EXPECTED_OPERATORS:
            fail(
                f"Unexpected operator: {operator}"
            )
            continue

        asset = row[
            "asset"
        ]

        question = row[
            "question"
        ]

        per_operator[
            operator
        ][
            "rows"
        ] += 1


        # ----------------------------------------------------
        # The target named in metadata must actually be the
        # proposition queried.
        # ----------------------------------------------------

        if (
            asset.lower()
            not in question.lower()
        ):
            target_question_errors += 1


        # ----------------------------------------------------
        # Find target-specific evidence independently of gold
        # citations.
        # ----------------------------------------------------

        evidence = parse_evidence(
            row[
                "evidence"
            ]
        )

        target_eids = {
            eid
            for eid, text
            in evidence.items()
            if asset.lower()
            in text.lower()
        }

        if not target_eids:
            target_location_errors += 1
            continue


        target_text = " ".join(
            evidence[
                eid
            ]
            for eid in sorted(
                target_eids
            )
        )


        # ----------------------------------------------------
        # Parse output only after target evidence has been
        # independently identified.
        # ----------------------------------------------------

        try:
            output = json.loads(
                row[
                    "output"
                ]
            )
        except Exception:
            fail(
                "Pilot output became non-JSON."
            )
            continue


        gold_label = output[
            "decision"
        ]

        citations = output[
            "citations"
        ]

        cited_set = set(
            citations
        )


        # ----------------------------------------------------
        # Citation set must exactly equal all target-specific
        # governing facts in this pilot architecture.
        #
        # This detects:
        # - missing one leg of a two-step proof
        # - citing a distractor
        # - citing another entity's rule
        # ----------------------------------------------------

        if cited_set != target_eids:

            citation_coverage_errors += 1

        else:

            per_operator[
                operator
            ][
                "citation_exact"
            ] += 1


        citation_counts[
            len(
                citations
            )
        ] += 1


        expected_depth = (
            expected_citation_count(
                operator,
                gold_label,
            )
        )

        if len(
            citations
        ) != expected_depth:

            citation_depth_errors += 1


        # ----------------------------------------------------
        # Independent semantic oracle.
        # ----------------------------------------------------

        predicted, matches = infer_label(
            operator,
            target_text,
        )

        if predicted is None:

            oracle_ambiguous += 1

        else:

            oracle_counts[
                predicted
            ] += 1

            if (
                predicted
                != gold_label
            ):

                oracle_mismatches += 1

            elif (
                predicted
                != row[
                    "target"
                ]
            ):

                oracle_mismatches += 1

            else:

                per_operator[
                    operator
                ][
                    "oracle_correct"
                ] += 1


        # ----------------------------------------------------
        # Logical pair consistency across two surface variants.
        # ----------------------------------------------------

        logical_key = (
            operator,
            row[
                "world"
            ],
            row[
                "target_index"
            ],
        )

        logical_groups[
            logical_key
        ].append(
            {
                "evidence_hash":
                    row[
                        "evidence_hash"
                    ],

                "target":
                    row[
                        "target"
                    ],

                "output":
                    output,

                "input_order":
                    row[
                        "input_order"
                    ],
            }
        )


    # ========================================================
    # SURFACE CONSISTENCY
    # ========================================================

    for key, group in (
        logical_groups.items()
    ):

        if len(group) != 2:

            surface_consistency_errors += 1
            continue

        a, b = group

        if (
            a[
                "evidence_hash"
            ]
            != b[
                "evidence_hash"
            ]
        ):
            surface_consistency_errors += 1

        if (
            a[
                "target"
            ]
            != b[
                "target"
            ]
        ):
            surface_consistency_errors += 1

        if (
            a[
                "output"
            ]
            != b[
                "output"
            ]
        ):
            surface_consistency_errors += 1

        if (
            a[
                "input_order"
            ]
            == b[
                "input_order"
            ]
        ):
            surface_consistency_errors += 1


    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 88)
    print(
        "TARGET / CITATION RESULTS"
    )
    print("=" * 88)

    print(
        "target/question errors:",
        target_question_errors,
    )

    print(
        "target evidence-location errors:",
        target_location_errors,
    )

    print(
        "citation exact-coverage errors:",
        citation_coverage_errors,
    )

    print(
        "citation depth errors:",
        citation_depth_errors,
    )

    print(
        "citation cardinality:",
        dict(
            sorted(
                citation_counts.items()
            )
        ),
    )


    print()
    print("=" * 88)
    print(
        "SEMANTIC ORACLE RESULTS"
    )
    print("=" * 88)

    print(
        "oracle inferred labels:",
        dict(
            sorted(
                oracle_counts.items()
            )
        ),
    )

    print(
        "oracle ambiguous/unclassified:",
        oracle_ambiguous,
    )

    print(
        "oracle/target mismatches:",
        oracle_mismatches,
    )

    print(
        "surface consistency errors:",
        surface_consistency_errors,
    )


    print()
    print("=" * 88)
    print(
        "PER OPERATOR"
    )
    print("=" * 88)

    for operator in sorted(
        per_operator
    ):

        stats = per_operator[
            operator
        ]

        print(
            f"{operator:36s} "
            f"rows={stats['rows']:3d} "
            f"oracle={stats['oracle_correct']:3d}/"
            f"{stats['rows']:3d} "
            f"citation={stats['citation_exact']:3d}/"
            f"{stats['rows']:3d}"
        )


    # ========================================================
    # HARD GATES
    # ========================================================

    if target_question_errors:
        fail(
            f"{target_question_errors} rows query a different "
            "target than their metadata identifies."
        )

    if target_location_errors:
        fail(
            f"{target_location_errors} rows lack target-specific evidence."
        )

    if citation_coverage_errors:
        fail(
            f"{citation_coverage_errors} citation sets do not "
            "exactly cover target-specific governing evidence."
        )

    if citation_depth_errors:
        fail(
            f"{citation_depth_errors} rows violate expected "
            "pilot citation depth."
        )

    if oracle_ambiguous:
        fail(
            f"{oracle_ambiguous} rows could not be assigned "
            "exactly one evidence-derived semantic label."
        )

    if oracle_mismatches:
        fail(
            f"{oracle_mismatches} evidence-derived labels "
            "disagree with the target."
        )

    if surface_consistency_errors:
        fail(
            f"{surface_consistency_errors} logical pairs differ "
            "across surface variants."
        )


    expected_oracle = Counter({
        label: 64
        for label in LABELS
    })

    if oracle_counts != expected_oracle:
        fail(
            "Semantic oracle did not independently reconstruct "
            "exactly 64 cases per label."
        )


    for operator, stats in (
        per_operator.items()
    ):

        if stats[
            "rows"
        ] != 32:

            fail(
                f"{operator}: expected 32 rows."
            )

        if (
            stats[
                "oracle_correct"
            ]
            != 32
        ):

            fail(
                f"{operator}: semantic oracle not 32/32."
            )

        if (
            stats[
                "citation_exact"
            ]
            != 32
        ):

            fail(
                f"{operator}: citation coverage not 32/32."
            )


    # ========================================================
    # POST-AUDIT IMMUTABILITY
    # ========================================================

    after_pilot_sha = sha256(
        PILOT
    )

    after_challenge_sha = sha256(
        CHALLENGE_V2
    )

    print()
    print(
        "post-audit pilot SHA:",
        after_pilot_sha,
    )

    print(
        "post-audit Challenge V2 SHA:",
        after_challenge_sha,
    )

    if after_pilot_sha != EXPECTED_PILOT_SHA:
        fail(
            "Pilot changed during semantic audit."
        )

    if after_challenge_sha != EXPECTED_CHALLENGE_SHA:
        fail(
            "Challenge V2 changed during semantic audit."
        )


    # ========================================================
    # VERDICT
    # ========================================================

    print()
    print("=" * 88)
    print(
        "V5 PILOT SEMANTIC AUDIT VERDICT"
    )
    print("=" * 88)

    if ERRORS:

        print(
            "FAILURES:",
            len(
                ERRORS
            ),
        )

        for error in ERRORS:
            print(
                "FAIL:",
                error,
            )

        print()
        print(
            "V5 RELATIONAL PILOT SEMANTICS: FAIL"
        )

        sys.exit(1)


    print(
        "FAILURES: 0"
    )

    print()
    print(
        "V5 RELATIONAL PILOT SEMANTICS: PASS"
    )

    print(
        "Evidence-derived oracle agrees with all 256 targets."
    )

    print(
        "Citation sets exactly cover each target's "
        "governing pilot evidence."
    )

    print(
        "Pilot is eligible to inform the full V5 generator architecture."
    )


if __name__ == "__main__":
    main()
