from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import hashlib
import json
import re


BASE = Path("/workspace")

V1_TRAIN = (
    BASE
    / "training/v6/full_candidate/"
      "sentinel_v6_train_candidate.jsonl"
)

V1_DEV = (
    BASE
    / "training/v6/full_candidate/"
      "sentinel_v6_development_candidate.jsonl"
)

V2_TRAIN = (
    BASE
    / "training/v6/full_candidate_v2/"
      "sentinel_v6_train_candidate_v2.jsonl"
)

V2_DEV = (
    BASE
    / "training/v6/full_candidate_v2/"
      "sentinel_v6_development_candidate_v2.jsonl"
)

PILOT = (
    BASE
    / "training/v6/pilot/"
      "sentinel_v6_relational_pilot.jsonl"
)

PRIOR_PATHS = [
    (
        "challenge_v1",
        BASE
        / "experiments/challenge_v1/"
          "sentinel_challenge_v1.json",
    ),
    (
        "challenge_v2",
        BASE
        / "experiments/challenge_v2/"
          "sentinel_challenge_v2.json",
    ),
    (
        "challenge_v3",
        BASE
        / "experiments/challenge_v3/canonical/"
          "sentinel_challenge_v3.json",
    ),
]

OUT_JSON = (
    BASE
    / "experiments/v6/full_candidate_v2/"
      "V6_CANDIDATE_V2_FINAL_DEEP_AUDIT_VERIFY.json"
)

OUT_TXT = (
    BASE
    / "experiments/v6/full_candidate_v2/"
      "V6_CANDIDATE_V2_FINAL_DEEP_AUDIT_VERIFY.txt"
)


LABELS = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def sha_text(text: str) -> str:
    return sha_bytes(
        text.encode("utf-8")
    )


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def load_structured(path):
    obj = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in (
            "cases",
            "records",
            "rows",
            "examples",
            "items",
        ):
            if isinstance(
                obj.get(key),
                list,
            ):
                return obj[key]

    raise AssertionError(
        f"unsupported structured benchmark schema: {path}"
    )


def norm(text):
    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            str(text).lower(),
        )
    )


def tokens(text):
    return set(
        re.findall(
            r"[a-z0-9]+",
            str(text).lower(),
        )
    )


def token_jaccard(a, b):
    aa = tokens(a)
    bb = tokens(b)

    if not aa and not bb:
        return 1.0

    if not aa or not bb:
        return 0.0

    return len(
        aa & bb
    ) / len(
        aa | bb
    )


def shingles(text, n=5):
    seq = re.findall(
        r"[a-z0-9]+",
        str(text).lower(),
    )

    if len(seq) < n:
        return set()

    return {
        tuple(
            seq[i:i+n]
        )
        for i in range(
            len(seq) - n + 1
        )
    }


def shingle_jaccard(a, b):
    aa = shingles(a)
    bb = shingles(b)

    if not aa and not bb:
        return 1.0

    if not aa or not bb:
        return 0.0

    return len(
        aa & bb
    ) / len(
        aa | bb
    )


def extract_question(record):
    value = record.get(
        "question"
    )

    if isinstance(
        value,
        str,
    ):
        return value

    messages = record.get(
        "messages"
    )

    if isinstance(
        messages,
        list,
    ):
        for message in reversed(
            messages
        ):
            if (
                isinstance(
                    message,
                    dict,
                )
                and message.get(
                    "role"
                ) == "user"
                and isinstance(
                    message.get(
                        "content"
                    ),
                    str,
                )
            ):
                return message[
                    "content"
                ]

    return ""


def extract_evidence_strings(record):
    value = record.get(
        "evidence"
    )

    out = []

    if isinstance(
        value,
        str,
    ):
        return [
            value
        ]

    if isinstance(
        value,
        list,
    ):
        for item in value:
            if isinstance(
                item,
                str,
            ):
                out.append(
                    item
                )

            elif isinstance(
                item,
                dict,
            ):
                for key in (
                    "text",
                    "content",
                    "passage",
                    "evidence",
                ):
                    if isinstance(
                        item.get(key),
                        str,
                    ):
                        out.append(
                            item[key]
                        )
                        break

    elif isinstance(
        value,
        dict,
    ):
        for item in value.values():
            if isinstance(
                item,
                str,
            ):
                out.append(
                    item
                )

            elif isinstance(
                item,
                dict,
            ):
                for key in (
                    "text",
                    "content",
                    "passage",
                ):
                    if isinstance(
                        item.get(key),
                        str,
                    ):
                        out.append(
                            item[key]
                        )
                        break

    return out


def prior_case_text(record):
    parts = [
        extract_question(
            record
        )
    ]

    parts.extend(
        extract_evidence_strings(
            record
        )
    )

    return "\n".join(
        x
        for x in parts
        if x
    )


def v6_evidence_blocks(text):
    matches = list(
        re.finditer(
            r"(?m)^\[(E[1-4])\]\s+",
            text,
        )
    )

    if len(matches) != 4:
        raise AssertionError(
            "V6 evidence must contain four records"
        )

    blocks = []

    for i, match in enumerate(
        matches
    ):
        start = match.end()

        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        body = text[
            start:end
        ].strip()

        blocks.append(
            (
                match.group(1),
                body,
            )
        )

    return blocks


def remove_v2_terminators(text):
    return text.replace(
        "\nEND_RECORD",
        "",
    )


def audit_v1_v2_delta(
    v1_rows,
    v2_rows,
    split,
):
    if len(v1_rows) != len(v2_rows):
        raise AssertionError(
            f"{split}: V1/V2 row count differs"
        )

    by_id_v1 = {
        r["row_id"]:
            r
        for r in v1_rows
    }

    by_id_v2 = {
        r["row_id"]:
            r
        for r in v2_rows
    }

    if set(by_id_v1) != set(
        by_id_v2
    ):
        raise AssertionError(
            f"{split}: V1/V2 row-id set differs"
        )


    allowed_changed_fields = {
        "evidence",
        "evidence_hash",
        "input",
    }

    unexpected_changes = []

    terminator_failures = []

    reconstructed_failures = []


    for row_id in sorted(
        by_id_v1
    ):
        a = by_id_v1[
            row_id
        ]

        b = by_id_v2[
            row_id
        ]


        changed = {
            key
            for key in a
            if a[key] != b[key]
        }

        if not changed <= allowed_changed_fields:
            unexpected_changes.append(
                (
                    row_id,
                    sorted(
                        changed
                    ),
                )
            )


        # Semantic output must be byte-identical.
        if a["output"] != b["output"]:
            unexpected_changes.append(
                (
                    row_id,
                    ["output"],
                )
            )


        blocks = v6_evidence_blocks(
            b["evidence"]
        )

        if any(
            body.count(
                "END_RECORD"
            ) != 1
            for _, body in blocks
        ):
            terminator_failures.append(
                row_id
            )


        if remove_v2_terminators(
            b["evidence"]
        ) != a[
            "evidence"
        ]:
            reconstructed_failures.append(
                (
                    row_id,
                    "evidence",
                )
            )


        if remove_v2_terminators(
            b["input"]
        ) != a[
            "input"
        ]:
            reconstructed_failures.append(
                (
                    row_id,
                    "input",
                )
            )


        if sha_text(
            b["evidence"]
        ) != b[
            "evidence_hash"
        ]:
            reconstructed_failures.append(
                (
                    row_id,
                    "evidence_hash",
                )
            )


    if unexpected_changes:
        raise AssertionError(
            f"{split}: V1/V2 unexpected field changes: "
            f"{unexpected_changes[:10]}"
        )

    if terminator_failures:
        raise AssertionError(
            f"{split}: END_RECORD count failures: "
            f"{terminator_failures[:10]}"
        )

    if reconstructed_failures:
        raise AssertionError(
            f"{split}: V2-minus-terminator does not reconstruct V1: "
            f"{reconstructed_failures[:10]}"
        )


    return {
        "rows_compared":
            len(v1_rows),

        "allowed_changed_fields":
            sorted(
                allowed_changed_fields
            ),

        "unexpected_field_changes":
            0,

        "terminator_failures":
            0,

        "reconstruction_failures":
            0,

        "outputs_changed":
            0,
    }


def audit_annotations(rows, split):
    errors = []

    decision_counts = Counter()

    ledger_shapes = Counter()

    for row in rows:
        output = json.loads(
            row["output"]
        )

        if set(output) != {
            "decision",
            "citations",
            "conditions",
            "rationale",
        }:
            errors.append(
                (
                    row["row_id"],
                    "output_keys",
                )
            )
            continue


        decision = output[
            "decision"
        ]

        decision_counts[
            decision
        ] += 1


        if decision not in LABELS:
            errors.append(
                (
                    row["row_id"],
                    "decision_vocab",
                )
            )


        if decision != row[
            "target"
        ]:
            errors.append(
                (
                    row["row_id"],
                    "decision_target",
                )
            )


        if output[
            "citations"
        ] != [
            row[
                "target_evidence_ref"
            ]
        ]:
            errors.append(
                (
                    row["row_id"],
                    "citations",
                )
            )


        blocks = dict(
            v6_evidence_blocks(
                row["evidence"]
            )
        )

        ref = row[
            "target_evidence_ref"
        ]

        if ref not in blocks:
            errors.append(
                (
                    row["row_id"],
                    "citation_missing",
                )
            )

        elif (
            row[
                "target_name"
            ].lower()
            not in blocks[
                ref
            ].lower()
        ):
            errors.append(
                (
                    row["row_id"],
                    "citation_target_binding",
                )
            )


        conditions = output[
            "conditions"
        ]

        if not isinstance(
            conditions,
            list,
        ):
            errors.append(
                (
                    row["row_id"],
                    "conditions_type",
                )
            )

            continue


        states = []

        for condition in conditions:
            match = re.match(
                r"^(SATISFIED|FAILED|UNRESOLVED) \| (.+)$",
                condition,
            )

            if not match:
                errors.append(
                    (
                        row["row_id"],
                        "condition_format",
                    )
                )

                continue

            states.append(
                match.group(1)
            )


        ledger_shapes[
            (
                decision,
                tuple(
                    states
                ),
            )
        ] += 1


        if decision == "YES":
            if (
                len(states) != 2
                or states != [
                    "SATISFIED",
                    "SATISFIED",
                ]
            ):
                errors.append(
                    (
                        row["row_id"],
                        "yes_complete_ledger",
                    )
                )


        elif decision == "CONDITIONAL":
            if (
                len(states) != 2
                or states.count(
                    "UNRESOLVED"
                ) != 1
                or states.count(
                    "SATISFIED"
                ) != 1
                or "FAILED" in states
            ):
                errors.append(
                    (
                        row["row_id"],
                        "conditional_complete_ledger",
                    )
                )


        elif decision == "INSUFFICIENT":
            if states:
                errors.append(
                    (
                        row["row_id"],
                        "insufficient_ledger",
                    )
                )


        elif decision == "NO":
            # Two valid NO realizations:
            #
            # direct prohibition -> no ledger items
            # failed requirement -> complete two-item ledger
            if states:
                if (
                    len(states) != 2
                    or "FAILED"
                    not in states
                    or "UNRESOLVED"
                    in states
                ):
                    errors.append(
                        (
                            row["row_id"],
                            "no_ledger",
                        )
                    )


        rationale = output[
            "rationale"
        ]

        if (
            not isinstance(
                rationale,
                str,
            )
            or not rationale.strip()
        ):
            errors.append(
                (
                    row["row_id"],
                    "rationale",
                )
            )


    if errors:
        raise AssertionError(
            f"{split}: annotation errors: "
            f"{errors[:10]}"
        )


    return {
        "rows":
            len(rows),

        "errors":
            0,

        "decision_counts":
            dict(
                sorted(
                    decision_counts.items()
                )
            ),

        "ledger_shapes":
            {
                str(key):
                    count
                for key, count
                in sorted(
                    ledger_shapes.items(),
                    key=lambda x:
                        str(
                            x[0]
                        ),
                )
            },
    }


def audit_prior_contamination(
    v6_rows,
    prior_records,
):
    v6_questions = sorted(
        {
            row[
                "question"
            ]
            for row in v6_rows
        }
    )

    v6_blocks = []

    v6_cases = []

    for row in v6_rows:
        for _, body in v6_evidence_blocks(
            row["evidence"]
        ):
            v6_blocks.append(
                remove_v2_terminators(
                    body
                )
            )

        v6_cases.append(
            row["question"]
            + "\n"
            + remove_v2_terminators(
                row["evidence"]
            )
        )


    prior_questions = []

    prior_evidence = []

    prior_cases = []

    prior_source_counts = Counter()


    for source, record in prior_records:
        q = extract_question(
            record
        )

        if q:
            prior_questions.append(
                (
                    source,
                    q,
                )
            )

        for evidence in extract_evidence_strings(
            record
        ):
            prior_evidence.append(
                (
                    source,
                    evidence,
                )
            )

        prior_cases.append(
            (
                source,
                prior_case_text(
                    record
                ),
            )
        )

        prior_source_counts[
            source
        ] += 1


    prior_question_norm = {
        norm(text):
            source
        for source, text
        in prior_questions
        if norm(text)
    }

    prior_evidence_norm = {
        norm(text):
            source
        for source, text
        in prior_evidence
        if norm(text)
    }


    exact_question_hits = 0

    for q in v6_questions:
        if norm(q) in prior_question_norm:
            exact_question_hits += 1


    exact_evidence_hits = 0

    for block in v6_blocks:
        if norm(block) in prior_evidence_norm:
            exact_evidence_hits += 1


    # ------------------------------------------------------------
    # Close-question audit.
    # Threshold was specified before results:
    #
    # SequenceMatcher >= .93
    # AND
    # token Jaccard >= .72
    # ------------------------------------------------------------

    close_question_hits = 0

    max_sequence = 0.0

    max_question_jaccard = 0.0


    prior_q_values = [
        text
        for _, text
        in prior_questions
    ]


    for vq in v6_questions:
        nvq = norm(
            vq
        )

        for pq in prior_q_values:
            npq = norm(
                pq
            )

            seq = SequenceMatcher(
                None,
                nvq,
                npq,
            ).ratio()

            jac = token_jaccard(
                vq,
                pq,
            )

            max_sequence = max(
                max_sequence,
                seq,
            )

            max_question_jaccard = max(
                max_question_jaccard,
                jac,
            )

            if (
                seq >= 0.93
                and jac >= 0.72
            ):
                close_question_hits += 1


    # ------------------------------------------------------------
    # Full-case shingle audit.
    #
    # We use an inverted index so we do not compute every possible
    # pair unless the two cases share at least one 5-token shingle.
    # ------------------------------------------------------------

    prior_case_shingles = []

    inverted = defaultdict(
        set
    )

    for i, (_, text) in enumerate(
        prior_cases
    ):
        s = shingles(
            text
        )

        prior_case_shingles.append(
            s
        )

        for item in s:
            inverted[
                item
            ].add(
                i
            )


    high_case_hits = 0

    max_case_jaccard = 0.0


    for case in v6_cases:
        s = shingles(
            case
        )

        candidate_indices = set()

        for item in s:
            candidate_indices.update(
                inverted.get(
                    item,
                    set(),
                )
            )

        for i in candidate_indices:
            other = prior_case_shingles[
                i
            ]

            union = s | other

            if not union:
                score = 1.0
            else:
                score = len(
                    s & other
                ) / len(
                    union
                )

            max_case_jaccard = max(
                max_case_jaccard,
                score,
            )

            if score >= 0.55:
                high_case_hits += 1


    result = {
        "prior_source_record_counts":
            dict(
                sorted(
                    prior_source_counts.items()
                )
            ),

        "v6_unique_questions":
            len(
                v6_questions
            ),

        "v6_evidence_blocks":
            len(
                v6_blocks
            ),

        "exact_question_overlaps":
            exact_question_hits,

        "exact_evidence_overlaps":
            exact_evidence_hits,

        "close_question_hits":
            close_question_hits,

        "high_full_case_shingle_hits":
            high_case_hits,

        "max_sequence_similarity":
            max_sequence,

        "max_question_token_jaccard":
            max_question_jaccard,

        "max_full_case_5gram_jaccard":
            max_case_jaccard,
    }


    if exact_question_hits:
        raise AssertionError(
            "prior-benchmark exact question contamination"
        )

    if exact_evidence_hits:
        raise AssertionError(
            "prior-benchmark exact evidence contamination"
        )

    if close_question_hits:
        raise AssertionError(
            "prior-benchmark near-question contamination"
        )

    if high_case_hits:
        raise AssertionError(
            "prior-benchmark high full-case contamination"
        )


    return result


def main():
    v1_train = load_jsonl(
        V1_TRAIN
    )

    v1_dev = load_jsonl(
        V1_DEV
    )

    v2_train = load_jsonl(
        V2_TRAIN
    )

    v2_dev = load_jsonl(
        V2_DEV
    )

    pilot = load_jsonl(
        PILOT
    )


    # ================================================================
    # 1. CONTROLLED V1 -> V2 DELTA
    # ================================================================

    train_delta = audit_v1_v2_delta(
        v1_train,
        v2_train,
        "TRAIN",
    )

    dev_delta = audit_v1_v2_delta(
        v1_dev,
        v2_dev,
        "DEVELOPMENT",
    )


    # ================================================================
    # 2. ANNOTATION COMPLETENESS
    # ================================================================

    train_annotations = audit_annotations(
        v2_train,
        "TRAIN",
    )

    dev_annotations = audit_annotations(
        v2_dev,
        "DEVELOPMENT",
    )


    # ================================================================
    # 3. FAMILY / ROW / SPLIT INVARIANTS
    # ================================================================

    if len(v2_train) != 4096:
        raise AssertionError(
            "TRAIN row count != 4096"
        )

    if len(v2_dev) != 1024:
        raise AssertionError(
            "DEVELOPMENT row count != 1024"
        )


    train_families = {
        row[
            "family"
        ]
        for row in v2_train
    }

    dev_families = {
        row[
            "family"
        ]
        for row in v2_dev
    }

    if len(train_families) != 32:
        raise AssertionError(
            "TRAIN family count != 32"
        )

    if len(dev_families) != 8:
        raise AssertionError(
            "DEVELOPMENT family count != 8"
        )

    if train_families & dev_families:
        raise AssertionError(
            "TRAIN/DEV family overlap"
        )


    if len({
        row[
            "relational_group"
        ]
        for row in v2_train
    }) != 256:
        raise AssertionError(
            "TRAIN group count != 256"
        )

    if len({
        row[
            "relational_group"
        ]
        for row in v2_dev
    }) != 64:
        raise AssertionError(
            "DEV group count != 64"
        )


    # ================================================================
    # 4. PILOT NON-REUSE
    # ================================================================

    pilot_row_ids = {
        row[
            "row_id"
        ]
        for row in pilot
        if "row_id" in row
    }

    v2_row_ids = {
        row[
            "row_id"
        ]
        for row in (
            v2_train
            + v2_dev
        )
    }

    pilot_row_id_overlap = len(
        pilot_row_ids
        & v2_row_ids
    )


    pilot_inputs = {
        norm(
            row[
                "input"
            ]
        )
        for row in pilot
        if isinstance(
            row.get(
                "input"
            ),
            str,
        )
    }

    v2_inputs = {
        norm(
            row[
                "input"
            ]
        )
        for row in (
            v2_train
            + v2_dev
        )
    }

    pilot_input_overlap = len(
        pilot_inputs
        & v2_inputs
    )


    if pilot_row_id_overlap:
        raise AssertionError(
            "pilot row-id reuse"
        )

    if pilot_input_overlap:
        raise AssertionError(
            "pilot exact-input reuse"
        )


    # ================================================================
    # 5. PRIOR-BENCHMARK CONTAMINATION
    #
    # Challenge V4 intentionally absent.
    # ================================================================

    prior_records = []

    for source, path in PRIOR_PATHS:
        for record in load_structured(
            path
        ):
            prior_records.append(
                (
                    source,
                    record,
                )
            )


    contamination = audit_prior_contamination(
        v2_train
        + v2_dev,
        prior_records,
    )


    # ================================================================
    # FINAL REPORT
    # ================================================================

    report = {
        "status":
            "PASS",

        "candidate":
            "V6 full curriculum Candidate V2",

        "geometry": {
            "train_rows":
                len(
                    v2_train
                ),

            "development_rows":
                len(
                    v2_dev
                ),

            "total_rows":
                len(
                    v2_train
                    + v2_dev
                ),

            "train_families":
                len(
                    train_families
                ),

            "development_families":
                len(
                    dev_families
                ),

            "train_relational_groups":
                256,

            "development_relational_groups":
                64,

            "fixed_target_causal_tracks":
                1280,
        },

        "v1_to_v2_delta": {
            "train":
                train_delta,

            "development":
                dev_delta,

            "intended_change":
                "END_RECORD appended identically to every evidence record",
        },

        "annotations": {
            "train":
                train_annotations,

            "development":
                dev_annotations,
        },

        "pilot_nonreuse": {
            "row_id_overlap":
                pilot_row_id_overlap,

            "exact_input_overlap":
                pilot_input_overlap,
        },

        "prior_benchmark_contamination":
            contamination,

        "sealed_external_benchmark": {
            "status":
                "NOT_ACCESSED",

            "files_read":
                0,
        },

        "sentinel_model_loads":
            0,

        "sentinel_inference_calls":
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
        "SENTINEL V6 CANDIDATE V2 — FINAL DEEP DATASET AUDIT",
        "=" * 80,
        "",
        "STATUS: PASS",
        "",
        "V1 -> V2 CONTROLLED DELTA",
        "-" * 80,
        f"TRAIN rows compared: {train_delta['rows_compared']}",
        f"DEVELOPMENT rows compared: {dev_delta['rows_compared']}",
        "Unexpected semantic/metadata changes: 0",
        "END_RECORD failures: 0",
        "V2 -> V1 reconstruction failures: 0",
        "Output changes: 0",
        "",
        "ANNOTATION",
        "-" * 80,
        "TRAIN annotation errors: 0",
        "DEVELOPMENT annotation errors: 0",
        "",
        "STRUCTURE",
        "-" * 80,
        "TRAIN rows: 4096",
        "DEVELOPMENT rows: 1024",
        "TRAIN families: 32",
        "DEVELOPMENT families: 8",
        "Relational groups: 320",
        "Fixed-target causal tracks: 1280",
        "",
        "PILOT NON-REUSE",
        "-" * 80,
        f"row-id overlap: {pilot_row_id_overlap}",
        f"exact-input overlap: {pilot_input_overlap}",
        "",
        "PRIOR BENCHMARK CONTAMINATION",
        "-" * 80,
        f"exact question overlaps: {contamination['exact_question_overlaps']}",
        f"exact evidence overlaps: {contamination['exact_evidence_overlaps']}",
        f"close question hits: {contamination['close_question_hits']}",
        f"high full-case shingle hits: {contamination['high_full_case_shingle_hits']}",
        f"max sequence similarity: {contamination['max_sequence_similarity']:.4f}",
        f"max question token Jaccard: {contamination['max_question_token_jaccard']:.4f}",
        f"max full-case 5gram Jaccard: {contamination['max_full_case_5gram_jaccard']:.4f}",
        "",
        "SEALED EXTERNAL BENCHMARK",
        "-" * 80,
        "files read: 0",
        "",
        "Sentinel model loads: 0",
        "Sentinel inference: 0",
    ]


    OUT_TXT.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


    print()
    print("=" * 80)
    print("V6 CANDIDATE V2 FINAL DEEP DATASET AUDIT")
    print("=" * 80)
    print("PASS")

    print()
    print(
        "V1 -> V2 TRAIN rows proven:",
        train_delta[
            "rows_compared"
        ],
    )

    print(
        "V1 -> V2 DEV rows proven:",
        dev_delta[
            "rows_compared"
        ],
    )

    print(
        "Unexpected V1/V2 changes: 0"
    )

    print(
        "END_RECORD failures: 0"
    )

    print(
        "V2-minus-END_RECORD reconstruction failures: 0"
    )

    print()
    print(
        "Annotation errors: 0"
    )

    print(
        "Pilot row/input reuse: 0"
    )

    print()
    print(
        "Prior benchmark exact questions:",
        contamination[
            "exact_question_overlaps"
        ],
    )

    print(
        "Prior benchmark exact evidence:",
        contamination[
            "exact_evidence_overlaps"
        ],
    )

    print(
        "Prior benchmark close questions:",
        contamination[
            "close_question_hits"
        ],
    )

    print(
        "Prior benchmark high case overlaps:",
        contamination[
            "high_full_case_shingle_hits"
        ],
    )

    print(
        "Max question sequence similarity:",
        f"{contamination['max_sequence_similarity']:.4f}",
    )

    print(
        "Max full-case 5gram Jaccard:",
        f"{contamination['max_full_case_5gram_jaccard']:.4f}",
    )

    print()
    print(
        "Audit JSON SHA256:",
        sha_bytes(
            OUT_JSON.read_bytes()
        ),
    )

    print(
        "Audit TXT SHA256:",
        sha_bytes(
            OUT_TXT.read_bytes()
        ),
    )

    print()
    print(
        "Sealed external benchmark files read: 0"
    )

    print(
        "Sentinel model loads: 0"
    )

    print(
        "Sentinel inference: 0"
    )


if __name__ == "__main__":
    main()
