"""
Sentinel V5 <-> Challenge V2 mechanical contamination audit.

PURPOSE
-------

Determine whether V5 TRAIN or DEVELOPMENT contains exact or suspicious
near-copy relationships to the sealed Challenge V2 benchmark.

ALLOWED BENCHMARK INTERACTION
-----------------------------

This script mechanically reads Challenge V2 ONLY to calculate aggregate:

- normalized exact overlaps
- cryptographic normalized overlaps
- character n-gram similarity
- normalized sequence similarity

It DOES NOT:

- print benchmark cases
- print benchmark questions
- print benchmark evidence
- print benchmark labels
- print benchmark IDs
- print nearest-neighbor examples
- use benchmark content to generate training data
- evaluate a Sentinel model
- modify the benchmark

Only aggregate counts and similarity statistics are emitted.
"""

from pathlib import Path
from difflib import SequenceMatcher

import hashlib
import json
import re
import sys

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


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

CHALLENGE = (
    BASE
    / "experiments/challenge_v2/"
      "sentinel_challenge_v2.json"
)


EXPECTED_TRAIN_SHA = (
    "945eef0841648911aaee5b88a5f7efd6"
    "c84bcad7399697f7f786f4774645d972"
)

EXPECTED_DEV_SHA = (
    "4e9fe2effa908d8994fdfb3be486661f"
    "9ef178860725fbd887d60b5181900124"
)

EXPECTED_CHALLENGE_SHA = (
    "30eb37ff39382f0d205adb7c17383b240"
    "25601ea3db1dd61f22b715af6c1e482"
)


# Conservative near-copy gates.
#
# A case must be extremely similar under BOTH:
#
#   character TF-IDF cosine
#   normalized sequence similarity
#
# before we classify it as suspicious.
#
# Question-only similarity is reported separately because generic
# governance question scaffolding may legitimately resemble other
# governance questions.

PAIR_CHAR_THRESHOLD = 0.92
PAIR_SEQUENCE_THRESHOLD = 0.90

EVIDENCE_CHAR_THRESHOLD = 0.92
EVIDENCE_SEQUENCE_THRESHOLD = 0.90

QUESTION_WARNING_THRESHOLD = 0.92


FAILURES = []
WARNINGS = []


def fail(
    message,
):

    FAILURES.append(
        message
    )


def warn(
    message,
):

    WARNINGS.append(
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


def norm(
    text,
):
    """
    Normalize for mechanical contamination comparison.

    Evidence identifiers are removed because E1/E2 ordering should not
    make two otherwise identical passages appear different.
    """

    text = str(
        text
    )

    text = re.sub(
        r"\[E\d+\]",
        " ",
        text,
        flags=re.I,
    )

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalized_sha(
    text,
):

    return hashlib.sha256(
        norm(
            text
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def textify(
    value,
):
    """
    Convert nested benchmark evidence representations into text
    without printing any contents.
    """

    if value is None:

        return ""


    if isinstance(
        value,
        str,
    ):

        return value


    if isinstance(
        value,
        (
            int,
            float,
            bool,
        ),
    ):

        return str(
            value
        )


    if isinstance(
        value,
        list,
    ):

        parts = [
            textify(
                item
            )
            for item in value
        ]

        return "\n".join(
            part
            for part in parts
            if part.strip()
        )


    if isinstance(
        value,
        dict,
    ):

        preferred = []

        for key in (
            "text",
            "content",
            "passage",
            "body",
            "value",
        ):

            if key in value:

                candidate = textify(
                    value[
                        key
                    ]
                )

                if candidate.strip():

                    preferred.append(
                        candidate
                    )


        if preferred:

            return "\n".join(
                preferred
            )


        parts = []

        for key, item in value.items():

            # Exclude likely labels / outputs / metadata.
            if str(
                key
            ).lower() in {
                "label",
                "target",
                "answer",
                "decision",
                "output",
                "response",
                "rationale",
                "citations",
                "conditions",
                "id",
                "case_id",
                "scenario_id",
                "split",
                "family",
                "operator",
            }:

                continue


            candidate = textify(
                item
            )

            if candidate.strip():

                parts.append(
                    candidate
                )


        return "\n".join(
            parts
        )


    return ""


def case_list_from_json(
    root,
):
    """
    Locate the benchmark case array without exposing case contents.
    """

    if isinstance(
        root,
        list,
    ):

        if all(
            isinstance(
                item,
                dict,
            )
            for item in root
        ):

            return root


    if isinstance(
        root,
        dict,
    ):

        preferred_keys = (
            "cases",
            "examples",
            "items",
            "rows",
            "data",
            "benchmark",
        )


        for key in preferred_keys:

            value = root.get(
                key
            )

            if (
                isinstance(
                    value,
                    list,
                )
                and all(
                    isinstance(
                        item,
                        dict,
                    )
                    for item in value
                )
            ):

                return value


        candidates = [
            value
            for value in root.values()
            if (
                isinstance(
                    value,
                    list,
                )
                and value
                and all(
                    isinstance(
                        item,
                        dict,
                    )
                    for item in value
                )
            )
        ]


        if len(
            candidates
        ) == 1:

            return candidates[
                0
            ]


    raise RuntimeError(
        "Could not mechanically locate Challenge V2 case array."
    )


def direct_field(
    case,
    keys,
):

    lowered = {
        str(
            key
        ).lower():
            key
        for key in case
    }


    for wanted in keys:

        actual = lowered.get(
            wanted.lower()
        )

        if actual is not None:

            value = textify(
                case[
                    actual
                ]
            )

            if value.strip():

                return value


    return ""


def parse_rendered_input(
    text,
):
    """
    Mechanically split common rendered question/evidence formats.

    Does not print the input.
    """

    if not text:

        return (
            "",
            "",
        )


    # QUESTION first.

    match = re.search(
        r"(?is)"
        r"\bQUESTION\b\s*:?\s*"
        r"(.*?)"
        r"\s*\b(?:SUPPLIED\s+EVIDENCE|EVIDENCE)\b\s*:?\s*"
        r"(.*)",
        text,
    )


    if match:

        return (
            match.group(
                1
            ).strip(),
            match.group(
                2
            ).strip(),
        )


    # EVIDENCE first.

    match = re.search(
        r"(?is)"
        r"\b(?:SUPPLIED\s+EVIDENCE|EVIDENCE)\b\s*:?\s*"
        r"(.*?)"
        r"\s*\bQUESTION\b\s*:?\s*"
        r"(.*)",
        text,
    )


    if match:

        return (
            match.group(
                2
            ).strip(),
            match.group(
                1
            ).strip(),
        )


    return (
        "",
        "",
    )


def extract_challenge_case(
    case,
):
    """
    Extract question/evidence mechanically.

    No content is emitted.
    """

    question = direct_field(
        case,
        (
            "question",
            "query",
            "user_question",
        ),
    )


    evidence = direct_field(
        case,
        (
            "evidence",
            "supplied_evidence",
            "passages",
            "documents",
            "context",
            "policy_text",
            "policies",
        ),
    )


    rendered = direct_field(
        case,
        (
            "input",
            "prompt",
            "user_input",
        ),
    )


    # Some benchmark formats place rendered prompt inside messages.

    if not rendered:

        rendered = direct_field(
            case,
            (
                "messages",
                "conversation",
            ),
        )


    parsed_question = ""
    parsed_evidence = ""


    if rendered:

        (
            parsed_question,
            parsed_evidence,
        ) = parse_rendered_input(
            rendered
        )


    if not question:

        question = parsed_question


    if not evidence:

        evidence = parsed_evidence


    return {
        "question":
            question,

        "evidence":
            evidence,
    }


def exact_overlap_count(
    reference,
    probes,
):

    reference_hashes = {
        normalized_sha(
            item
        )
        for item in reference
    }

    return sum(
        normalized_sha(
            item
        )
        in reference_hashes
        for item in probes
    )


def paired_text(
    question,
    evidence,
):

    return (
        norm(
            question
        )
        + "\n<pair>\n"
        + norm(
            evidence
        )
    )


def similarity_probe(
    reference,
    probes,
):
    """
    Return only aggregate nearest-neighbor statistics.

    Individual nearest matches are NEVER returned or printed.
    """

    reference = [
        norm(
            item
        )
        for item in reference
    ]

    probes = [
        norm(
            item
        )
        for item in probes
    ]


    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(
            4,
            6,
        ),
        lowercase=False,
        min_df=1,
        sublinear_tf=True,
        max_features=120000,
    )


    all_text = (
        reference
        + probes
    )


    matrix = vectorizer.fit_transform(
        all_text
    )


    reference_matrix = matrix[
        :len(
            reference
        )
    ]

    probe_matrix = matrix[
        len(
            reference
        ):
    ]


    similarities = (
        probe_matrix
        @ reference_matrix.T
    ).tocsr()


    cosine_max = []

    sequence_at_cosine_max = []


    for probe_index in range(
        len(
            probes
        )
    ):

        row = similarities.getrow(
            probe_index
        )


        if row.nnz == 0:

            cosine_max.append(
                0.0
            )

            sequence_at_cosine_max.append(
                0.0
            )

            continue


        local_index = int(
            np.argmax(
                row.data
            )
        )

        reference_index = int(
            row.indices[
                local_index
            ]
        )

        cosine = float(
            row.data[
                local_index
            ]
        )


        sequence = SequenceMatcher(
            None,
            probes[
                probe_index
            ],
            reference[
                reference_index
            ],
            autojunk=False,
        ).ratio()


        cosine_max.append(
            cosine
        )

        sequence_at_cosine_max.append(
            float(
                sequence
            )
        )


    return (
        np.asarray(
            cosine_max,
            dtype=float,
        ),
        np.asarray(
            sequence_at_cosine_max,
            dtype=float,
        ),
    )


def metric_summary(
    name,
    cosine,
    sequence,
):

    quantiles = (
        0.50,
        0.90,
        0.95,
        0.99,
        1.00,
    )


    print()
    print(
        name
    )


    for q in quantiles:

        print(
            f"  cosine p{int(q * 100):02d}: "
            f"{np.quantile(cosine, q):.4f}"
        )


    print(
        f"  sequence-at-nearest max: "
        f"{sequence.max():.4f}"
    )


def suspicious_count(
    cosine,
    sequence,
    cosine_threshold,
    sequence_threshold,
):

    return int(
        np.sum(
            (
                cosine
                >= cosine_threshold
            )
            & (
                sequence
                >= sequence_threshold
            )
        )
    )


def audit_against_split(
    split_name,
    rows,
    challenge_cases,
):

    print()
    print(
        "=" * 88
    )

    print(
        f"CHALLENGE V2 vs {split_name}"
    )

    print(
        "=" * 88
    )


    reference_questions = [
        row[
            "question"
        ]
        for row in rows
    ]

    reference_evidence = [
        row[
            "evidence"
        ]
        for row in rows
    ]

    reference_pairs = [
        paired_text(
            row[
                "question"
            ],
            row[
                "evidence"
            ],
        )
        for row in rows
    ]


    challenge_questions = [
        case[
            "question"
        ]
        for case in challenge_cases
    ]

    challenge_evidence = [
        case[
            "evidence"
        ]
        for case in challenge_cases
    ]

    challenge_pairs = [
        paired_text(
            case[
                "question"
            ],
            case[
                "evidence"
            ],
        )
        for case in challenge_cases
    ]


    # ================================================================
    # EXACT NORMALIZED OVERLAP
    # ================================================================

    exact_question = exact_overlap_count(
        reference_questions,
        challenge_questions,
    )

    exact_evidence = exact_overlap_count(
        reference_evidence,
        challenge_evidence,
    )

    exact_pair = exact_overlap_count(
        reference_pairs,
        challenge_pairs,
    )


    print(
        "normalized exact question overlaps:",
        exact_question,
    )

    print(
        "normalized exact evidence overlaps:",
        exact_evidence,
    )

    print(
        "normalized exact question/evidence pair overlaps:",
        exact_pair,
    )


    if exact_question:

        fail(
            f"{split_name}: exact normalized question overlap "
            f"with Challenge V2 = {exact_question}"
        )


    if exact_evidence:

        fail(
            f"{split_name}: exact normalized evidence overlap "
            f"with Challenge V2 = {exact_evidence}"
        )


    if exact_pair:

        fail(
            f"{split_name}: exact normalized pair overlap "
            f"with Challenge V2 = {exact_pair}"
        )


    # ================================================================
    # QUESTION SIMILARITY
    # ================================================================

    (
        q_cosine,
        q_sequence,
    ) = similarity_probe(
        reference_questions,
        challenge_questions,
    )


    metric_summary(
        "QUESTION nearest-neighbor similarity",
        q_cosine,
        q_sequence,
    )


    question_high = int(
        np.sum(
            q_cosine
            >= QUESTION_WARNING_THRESHOLD
        )
    )


    print(
        "  question cosine >= "
        f"{QUESTION_WARNING_THRESHOLD:.2f}:",
        question_high,
    )


    if question_high:

        warn(
            f"{split_name}: {question_high} Challenge V2 "
            f"questions have character cosine >= "
            f"{QUESTION_WARNING_THRESHOLD:.2f}. "
            f"No individual cases were inspected."
        )


    # ================================================================
    # EVIDENCE SIMILARITY
    # ================================================================

    (
        e_cosine,
        e_sequence,
    ) = similarity_probe(
        reference_evidence,
        challenge_evidence,
    )


    metric_summary(
        "EVIDENCE nearest-neighbor similarity",
        e_cosine,
        e_sequence,
    )


    suspicious_evidence = suspicious_count(
        e_cosine,
        e_sequence,
        EVIDENCE_CHAR_THRESHOLD,
        EVIDENCE_SEQUENCE_THRESHOLD,
    )


    print(
        "  suspicious evidence near-copies:",
        suspicious_evidence,
    )


    if suspicious_evidence:

        fail(
            f"{split_name}: suspicious Challenge V2 "
            f"evidence near-copy count={suspicious_evidence}"
        )


    # ================================================================
    # FULL RELATIONAL PAIR SIMILARITY
    # ================================================================

    (
        p_cosine,
        p_sequence,
    ) = similarity_probe(
        reference_pairs,
        challenge_pairs,
    )


    metric_summary(
        "QUESTION+EVIDENCE nearest-neighbor similarity",
        p_cosine,
        p_sequence,
    )


    suspicious_pairs = suspicious_count(
        p_cosine,
        p_sequence,
        PAIR_CHAR_THRESHOLD,
        PAIR_SEQUENCE_THRESHOLD,
    )


    print(
        "  suspicious full-pair near-copies:",
        suspicious_pairs,
    )


    if suspicious_pairs:

        fail(
            f"{split_name}: suspicious Challenge V2 "
            f"question/evidence pair near-copy count="
            f"{suspicious_pairs}"
        )


def main():

    print(
        "=" * 88
    )

    print(
        "SENTINEL V5 / CHALLENGE V2 CONTAMINATION AUDIT"
    )

    print(
        "=" * 88
    )


    # ================================================================
    # HASH SAFETY
    # ================================================================

    train_sha = sha256(
        TRAIN
    )

    dev_sha = sha256(
        DEV
    )

    challenge_sha = sha256(
        CHALLENGE
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

    print(
        "Challenge V2 SHA:",
        challenge_sha,
    )


    if train_sha != EXPECTED_TRAIN_SHA:

        fail(
            "TRAIN hash mismatch."
        )


    if dev_sha != EXPECTED_DEV_SHA:

        fail(
            "DEVELOPMENT hash mismatch."
        )


    if challenge_sha != EXPECTED_CHALLENGE_SHA:

        fail(
            "Challenge V2 hash mismatch."
        )


    if FAILURES:

        raise RuntimeError(
            "Hash safety gate failed before contamination audit."
        )


    # ================================================================
    # LOAD V5
    # ================================================================

    train_rows = load_jsonl(
        TRAIN
    )

    dev_rows = load_jsonl(
        DEV
    )


    # ================================================================
    # MECHANICALLY LOAD SEALED BENCHMARK
    # ================================================================

    with CHALLENGE.open() as f:

        root = json.load(
            f
        )


    raw_cases = case_list_from_json(
        root
    )


    print()
    print(
        "Challenge V2 cases mechanically located:",
        len(
            raw_cases
        ),
    )


    challenge_cases = []

    missing_question = 0
    missing_evidence = 0


    for case in raw_cases:

        extracted = extract_challenge_case(
            case
        )


        if not extracted[
            "question"
        ].strip():

            missing_question += 1


        if not extracted[
            "evidence"
        ].strip():

            missing_evidence += 1


        challenge_cases.append(
            extracted
        )


    print(
        "cases missing mechanically extractable question:",
        missing_question,
    )

    print(
        "cases missing mechanically extractable evidence:",
        missing_evidence,
    )


    if missing_question:

        fail(
            f"Challenge extraction missing question for "
            f"{missing_question} cases."
        )


    if missing_evidence:

        fail(
            f"Challenge extraction missing evidence for "
            f"{missing_evidence} cases."
        )


    if FAILURES:

        print()
        print(
            "Extraction failed safely."
        )

        print(
            "No individual Challenge V2 content was printed."
        )

        sys.exit(
            1
        )


    # ================================================================
    # CONTAMINATION AUDITS
    # ================================================================

    audit_against_split(
        "V5 TRAIN",
        train_rows,
        challenge_cases,
    )


    audit_against_split(
        "V5 DEVELOPMENT",
        dev_rows,
        challenge_cases,
    )


    # ================================================================
    # VERIFY SEALED BENCHMARK STILL UNCHANGED
    # ================================================================

    challenge_sha_after = sha256(
        CHALLENGE
    )


    if (
        challenge_sha_after
        != EXPECTED_CHALLENGE_SHA
    ):

        fail(
            "Challenge V2 changed during contamination audit."
        )


    # ================================================================
    # VERDICT
    # ================================================================

    print()
    print(
        "=" * 88
    )

    print(
        "CONTAMINATION AUDIT VERDICT"
    )

    print(
        "=" * 88
    )


    if WARNINGS:

        print()
        print(
            "WARNINGS:"
        )

        for item in WARNINGS:

            print(
                "  -",
                item,
            )


    if FAILURES:

        print()
        print(
            "FAILURES:",
            len(
                FAILURES
            ),
        )

        for item in FAILURES:

            print(
                "  -",
                item,
            )


        print()
        print(
            "V5 / CHALLENGE V2 CONTAMINATION AUDIT: FAIL"
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
        "V5 / CHALLENGE V2 CONTAMINATION AUDIT: PASS"
    )

    print(
        "Exact normalized overlap: 0."
    )

    print(
        "No suspicious evidence or full-pair near-copies detected."
    )

    print(
        "Challenge V2 remained unchanged."
    )

    print(
        "No individual Challenge V2 cases were displayed."
    )

    print(
        "No Sentinel model evaluation occurred."
    )

    print(
        "No Sentinel model training occurred."
    )


if __name__ == "__main__":
    main()
