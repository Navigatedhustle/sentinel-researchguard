from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import sys


BASE = Path("/workspace")

CHALLENGE = (
    BASE
    / "experiments/challenge_v2/"
      "sentinel_challenge_v2.json"
)

EXPECTED_SHA = (
    "30eb37ff39382f0d205adb7c17383b240"
    "25601ea3db1dd61f22b715af6c1e482"
)

LABELS = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
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


def citation_map(text):
    """
    Parse:

    [E1] ...

    [E2] ...
    """

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
# QUESTION -> TARGET ENTITY
# ============================================================

TARGET_PATTERNS = {

    "specific_rule_precedence":
        r"share the (.+?) with an external collaborator",

    "superseding_rule_precedence":
        r"release the (.+?) through the sponsor portal",

    "exception_activation":
        r"export the (.+?) outside the approved environment",

    "nested_exception_requirements":
        r"transfer the (.+?) to an external research partner",

    "delegated_authority_validity":
        r"approve release of the (.+?)(?:\?|$)",

    "temporal_authorization":
        r"access the (.+?) remotely",

    "definition_then_authorization":
        r"process the (.+?) on a local workstation",

    "purpose_scope_authorization":
        r"reuse the (.+?) for secondary academic research",

    "jurisdiction_then_requirement":
        r"transfer the (.+?) to Territory Q",

    "prohibition_vs_unresolved_fact":
        r"send the (.+?) by institutional email",

    "permission_with_material_condition":
        r"publish the (.+?) in a public repository",

    "evidence_relevance_sufficiency":
        r"use the (.+?) with an external generative-AI service",
}


def extract_target(row):

    composition = row[
        "composition"
    ]

    pattern = TARGET_PATTERNS[
        composition
    ]

    match = re.search(
        pattern,
        row[
            "question"
        ],
        flags=re.I,
    )

    if not match:
        return None

    return match.group(
        1
    ).strip()


# ============================================================
# INDEPENDENT SEMANTIC SIGNATURES
# ============================================================
#
# These signatures do NOT inspect case_id or gold decision.
#
# For each composition we look only at the target-specific
# governing rule text and infer its semantic state.
#
# Each target segment must match exactly one label.
# ============================================================


SIGNATURES = {

    "specific_rule_precedence": {

        "YES": [
            "disclosure review is recorded as approved",
        ],

        "NO": [
            "specific controlled-data rule controls and expressly prohibits",
        ],

        "CONDITIONAL": [
            "agreement is active",
            "does not establish whether that agreement is active",
        ],

        "INSUFFICIENT": [
            "do not state a rule governing external collaborator sharing",
        ],
    },


    "superseding_rule_precedence": {

        "YES": [
            "sponsor review is complete",
        ],

        "NO": [
            "sponsor signoff was denied",
        ],

        "CONDITIONAL": [
            "current signoff status is not established",
        ],

        "INSUFFICIENT": [
            "no supplied provision states whether",
            "released through a sponsor portal",
        ],
    },


    "exception_activation": {

        "YES": [
            "a recovery event is declared",
            "written clearance is recorded",
        ],

        "NO": [
            "no recovery event has been declared",
        ],

        "CONDITIONAL": [
            "written-clearance status is not established",
        ],

        "INSUFFICIENT": [
            "does not state whether export outside an approved environment",
        ],
    },


    "nested_exception_requirements": {

        "YES": [
            "waiver is active",
            "sponsor consent is recorded",
        ],

        "NO": [
            "institutional waiver was denied",
        ],

        "CONDITIONAL": [
            "sponsor-consent status is not established",
        ],

        "INSUFFICIENT": [
            "does not establish any rule governing external research-partner transfer",
        ],
    },


    "delegated_authority_validity": {

        "YES": [
            "recorded as the current delegate",
            "required department approval exists",
        ],

        "NO": [
            "delegation was revoked",
        ],

        "CONDITIONAL": [
            "does not establish whether that delegation is currently recorded",
        ],

        "INSUFFICIENT": [
            "does not identify the acting data steward as a delegate",
            "state who may approve this release",
        ],
    },


    "temporal_authorization": {

        "YES": [
            "request occurs june 15",
            "annual review is complete",
        ],

        "NO": [
            "authorization expired may 31",
        ],

        "CONDITIONAL": [
            "annual-review status is not established",
        ],

        "INSUFFICIENT": [
            "does not state any rule governing whether remote access is allowed",
        ],
    },


    "definition_then_authorization": {

        "YES": [
            "meets that definition",
            "workstation is recorded as managed",
        ],

        "NO": [
            "restricted local workloads are expressly prohibited",
        ],

        "CONDITIONAL": [
            "endpoint-security approval status is not established",
        ],

        "INSUFFICIENT": [
            "no supplied provision states whether qualified local workloads may be processed",
        ],
    },


    "purpose_scope_authorization": {

        "YES": [
            "ethics review is complete",
        ],

        "NO": [
            "explicitly excludes secondary academic reuse",
        ],

        "CONDITIONAL": [
            "ethics-review status is not established",
        ],

        "INSUFFICIENT": [
            "does not define whether the proposed secondary academic activity falls",
        ],
    },


    "jurisdiction_then_requirement": {

        "YES": [
            "required clauses are active",
        ],

        "NO": [
            "classified as a prohibited destination",
        ],

        "CONDITIONAL": [
            "clause status is not established",
        ],

        "INSUFFICIENT": [
            "does not classify territory q in either category",
        ],
    },


    "prohibition_vs_unresolved_fact": {

        "YES": [
            "transport encryption is enabled",
        ],

        "NO": [
            "institutional email transmission",
            "is expressly prohibited",
            "separate disclosure review",
        ],

        "CONDITIONAL": [
            "approval status is not established",
        ],

        "INSUFFICIENT": [
            "does not state a rule for institutional email transmission",
        ],
    },


    "permission_with_material_condition": {

        "YES": [
            "disclosure review is approved",
            "sponsor notice was sent on time",
        ],

        "NO": [
            "expressly prohibits publication in public repositories",
        ],

        "CONDITIONAL": [
            "sponsor-notice status is not established",
        ],

        "INSUFFICIENT": [
            "does not state whether",
            "public repository",
        ],
    },


    "evidence_relevance_sufficiency": {

        "YES": [
            "provider is approved",
            "data minimization is complete",
        ],

        "NO": [
            "expressly prohibits sending",
            "external generative-ai services",
        ],

        "CONDITIONAL": [
            "provider-approval status is not established",
        ],

        "INSUFFICIENT": [
            "contains no provision addressing external generative-ai services",
        ],
    },
}


def infer_label(
    composition,
    segment,
):

    text = norm(
        segment
    )

    matches = []

    for label, signatures in SIGNATURES[
        composition
    ].items():

        if all(
            norm(signature)
            in text
            for signature
            in signatures
        ):
            matches.append(
                label
            )

    if len(
        matches
    ) == 1:

        return (
            matches[0],
            matches,
        )

    return (
        None,
        matches,
    )


# ============================================================
# TARGET-SEGMENT ISOLATION
# ============================================================
#
# Relational evidence packs can merge multiple target rules into
# one E passage.
#
# We therefore isolate the text belonging to the entity asked
# about:
#
#     sentence containing target entity
#       ↓
#     until sentence containing next entity in same group
#
# This prevents another target's label state from satisfying
# the semantic oracle.
# ============================================================


def sentence_start(
    text,
    position,
):

    before = text[
        :position
    ]

    candidates = [
        before.rfind(". "),
        before.rfind("! "),
        before.rfind("? "),
    ]

    boundary = max(
        candidates
    )

    if boundary == -1:
        return 0

    return boundary + 2


def isolate_target_segment(
    passage,
    target,
    group_targets,
):

    lowered = passage.lower()

    target_pos = lowered.find(
        target.lower()
    )

    if target_pos == -1:
        return None

    start = sentence_start(
        passage,
        target_pos,
    )

    end = len(
        passage
    )

    for other in group_targets:

        if norm(
            other
        ) == norm(
            target
        ):
            continue

        pos = lowered.find(
            other.lower(),
            target_pos
            + len(
                target
            ),
        )

        if pos == -1:
            continue

        candidate_end = sentence_start(
            passage,
            pos,
        )

        if (
            candidate_end > start
            and candidate_end < end
        ):

            end = candidate_end

    return passage[
        start:end
    ].strip()


# ============================================================
# GOLD ANSWER / CONDITION SEMANTICS
# ============================================================


def audit_gold_semantics(
    row,
):

    gold = row[
        "gold"
    ]

    label = gold[
        "decision"
    ]

    answer = norm(
        gold[
            "answer"
        ]
    )

    conditions = [
        norm(x)
        for x in gold[
            "conditions"
        ]
    ]

    errors = []

    if label == "CONDITIONAL":

        if not conditions:

            errors.append(
                "CONDITIONAL has no conditions"
            )

        if not all(
            "unresolved" in condition
            for condition in conditions
        ):

            errors.append(
                "CONDITIONAL condition lacks unresolved state"
            )

        if not (
            "unresolved" in answer
            or "remains" in answer
        ):

            errors.append(
                "CONDITIONAL answer lacks unresolved-state language"
            )


    elif label == "NO":

        if conditions:

            errors.append(
                "NO unexpectedly has conditions"
            )

        negative_markers = [
            "prohibit",
            "not permitted",
            "revocation",
            "revoked",
            "expired",
            "outside",
            "denied",
            "failed",
            "prevents",
            "inactive",
        ]

        if not any(
            marker in answer
            for marker in negative_markers
        ):

            errors.append(
                "NO answer lacks controlling-negative rationale"
            )


    elif label == "INSUFFICIENT":

        if conditions:

            errors.append(
                "INSUFFICIENT unexpectedly has conditions"
            )

        insufficient_markers = [
            "does not",
            "no supplied",
            "missing",
            "not establish",

            # Equivalent missing-governance formulations.
            "do not govern",
            "not a governing",
            "no governing",
            "different channel",
            "no provision",
            "no rule",
        ]

        if not any(
            marker in answer
            for marker in insufficient_markers
        ):

            errors.append(
                "INSUFFICIENT answer lacks missing-governance rationale"
            )


    elif label == "YES":

        bad_markers = [
            "unresolved",
            "not established",
            "prohibit",
            "denied",
            "expired",
            "revoked",
            "outside the authorized",
            "missing",
        ]

        if any(
            marker in answer
            for marker in bad_markers
        ):

            errors.append(
                "YES answer contains unresolved/negative state"
            )

        if any(
            "unresolved" in condition
            for condition in conditions
        ):

            errors.append(
                "YES contains unresolved condition"
            )

    return errors


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 88)
    print(
        "SENTINEL CHALLENGE V2 — SEMANTIC / CITATION AUDIT"
    )
    print("=" * 88)

    current_sha = sha256(
        CHALLENGE
    )

    print(
        "candidate SHA256:",
        current_sha,
    )

    if current_sha != EXPECTED_SHA:

        fail(
            "Challenge V2 changed before semantic audit."
        )

    rows = json.loads(
        CHALLENGE.read_text()
    )

    if (
        not isinstance(
            rows,
            list,
        )
        or len(
            rows
        ) != 240
    ):

        fail(
            "Expected exactly 240 Challenge V2 rows."
        )

        finish()
        return


    # --------------------------------------------------------
    # Extract every row's target entity before looking at gold.
    # --------------------------------------------------------

    targets = {}

    target_errors = 0

    for row in rows:

        target = extract_target(
            row
        )

        if not target:

            target_errors += 1
            continue

        targets[
            row[
                "case_id"
            ]
        ] = target

    print()
    print(
        "target extraction errors:",
        target_errors,
    )

    if target_errors:

        fail(
            f"{target_errors} target extraction errors."
        )


    # --------------------------------------------------------
    # Build group target sets.
    # --------------------------------------------------------

    group_targets = defaultdict(
        set
    )

    for row in rows:

        group = row.get(
            "relational_group"
        )

        if not group:
            continue

        target = targets.get(
            row[
                "case_id"
            ]
        )

        if target:

            group_targets[
                group
            ].add(
                target
            )


    bad_group_target_counts = 0

    for group, values in group_targets.items():

        if len(
            values
        ) != 4:

            bad_group_target_counts += 1

    print(
        "relational groups with != 4 targets:",
        bad_group_target_counts,
    )

    if bad_group_target_counts:

        fail(
            f"{bad_group_target_counts} relational groups "
            "do not contain exactly four target entities."
        )


    # --------------------------------------------------------
    # Independent target-specific semantic oracle.
    # --------------------------------------------------------

    oracle_counts = Counter()
    oracle_mismatches = 0
    oracle_ambiguous = 0

    citation_target_errors = 0
    citation_minimality_errors = 0

    gold_semantic_errors = 0

    per_composition = defaultdict(
        lambda: {
            "rows": 0,
            "oracle_correct": 0,
            "citation_correct": 0,
        }
    )


    for row in rows:

        case_id = row[
            "case_id"
        ]

        composition = row[
            "composition"
        ]

        target = targets.get(
            case_id
        )

        gold = row[
            "gold"
        ]

        expected = gold[
            "decision"
        ]

        per_composition[
            composition
        ][
            "rows"
        ] += 1


        # ----------------------------------------------------
        # Citation must point to the ONLY passage containing
        # the target entity.
        # ----------------------------------------------------

        cmap = citation_map(
            row[
                "evidence_text"
            ]
        )

        citations = gold[
            "citations"
        ]

        if len(
            citations
        ) != 1:

            citation_target_errors += 1
            continue

        cited_eid = citations[
            0
        ]

        cited_text = cmap.get(
            cited_eid,
            "",
        )

        if not target:

            citation_target_errors += 1
            continue

        containing = [
            eid
            for eid, text
            in cmap.items()
            if target.lower()
            in text.lower()
        ]

        if containing != [
            cited_eid
        ]:

            citation_target_errors += 1

        else:

            per_composition[
                composition
            ][
                "citation_correct"
            ] += 1


        # ----------------------------------------------------
        # No non-cited passage may independently mention the
        # target entity. This is our minimal-governing-location
        # criterion.
        # ----------------------------------------------------

        non_cited_target_mentions = [
            eid
            for eid, text
            in cmap.items()
            if (
                eid != cited_eid
                and target.lower()
                in text.lower()
            )
        ]

        if non_cited_target_mentions:

            citation_minimality_errors += 1


        # ----------------------------------------------------
        # Isolate the target's governing rule text.
        # ----------------------------------------------------

        group = row.get(
            "relational_group"
        )

        if group:

            peers = group_targets[
                group
            ]

        else:

            peers = {
                target
            }

        segment = isolate_target_segment(
            cited_text,
            target,
            peers,
        )

        if not segment:

            oracle_ambiguous += 1
            continue


        # ----------------------------------------------------
        # Infer label from evidence semantics WITHOUT passing
        # expected/gold to the inference function.
        # ----------------------------------------------------

        predicted, matches = infer_label(
            composition,
            segment,
        )

        if predicted is None:

            oracle_ambiguous += 1

            continue

        oracle_counts[
            predicted
        ] += 1

        if predicted != expected:

            oracle_mismatches += 1

        else:

            per_composition[
                composition
            ][
                "oracle_correct"
            ] += 1


        # ----------------------------------------------------
        # Gold answer / condition coherence.
        # ----------------------------------------------------

        semantic_errors = audit_gold_semantics(
            row
        )

        gold_semantic_errors += len(
            semantic_errors
        )


    print()
    print("=" * 88)
    print(
        "TARGET-SPECIFIC CITATION RESULTS"
    )
    print("=" * 88)

    print(
        "citation target-location errors:",
        citation_target_errors,
    )

    print(
        "citation minimality errors:",
        citation_minimality_errors,
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
        "oracle/gold mismatches:",
        oracle_mismatches,
    )

    print(
        "gold answer/condition semantic errors:",
        gold_semantic_errors,
    )


    print()
    print("=" * 88)
    print(
        "PER COMPOSITION"
    )
    print("=" * 88)

    for composition in sorted(
        per_composition
    ):

        stats = per_composition[
            composition
        ]

        print(
            f"{composition:40s} "
            f"rows={stats['rows']:2d} "
            f"oracle={stats['oracle_correct']:2d}/"
            f"{stats['rows']:2d} "
            f"citation={stats['citation_correct']:2d}/"
            f"{stats['rows']:2d}"
        )


    if citation_target_errors:

        fail(
            f"{citation_target_errors} gold citations "
            "do not uniquely locate the target governing passage."
        )

    if citation_minimality_errors:

        fail(
            f"{citation_minimality_errors} non-cited passages "
            "also mention the target entity."
        )

    if oracle_ambiguous:

        fail(
            f"{oracle_ambiguous} cases could not be assigned "
            "exactly one semantic oracle label."
        )

    if oracle_mismatches:

        fail(
            f"{oracle_mismatches} evidence-derived oracle labels "
            "disagree with gold."
        )

    if gold_semantic_errors:

        fail(
            f"{gold_semantic_errors} answer/condition semantic errors."
        )


    # --------------------------------------------------------
    # Ensure candidate was not changed during audit.
    # --------------------------------------------------------

    after_sha = sha256(
        CHALLENGE
    )

    print()
    print(
        "post-audit SHA256:",
        after_sha,
    )

    if after_sha != EXPECTED_SHA:

        fail(
            "Challenge V2 changed during semantic audit."
        )


    finish()


def finish():

    print()
    print("=" * 88)
    print(
        "SEMANTIC AUDIT VERDICT"
    )
    print("=" * 88)

    if ERRORS:

        print(
            "FAILURES:",
            len(
                ERRORS
            ),
        )

        for message in ERRORS:

            print(
                "FAIL:",
                message,
            )

        print()
        print(
            "CHALLENGE V2 STATUS: FAIL — DO NOT FREEZE"
        )

        sys.exit(1)

    print(
        "FAILURES: 0"
    )

    print()
    print(
        "CHALLENGE V2 SEMANTIC/CITATION STATUS: PASS"
    )

    print(
        "Candidate is eligible for freeze after "
        "structural audit re-verification."
    )


if __name__ == "__main__":
    main()
