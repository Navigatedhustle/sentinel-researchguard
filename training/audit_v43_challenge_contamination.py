from difflib import SequenceMatcher
from pathlib import Path
from collections import Counter
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE = Path("/workspace")

CHALLENGE = (
    BASE
    / "experiments/challenge_v1"
    / "sentinel_challenge_v1.json"
)

DATASETS = {
    "train":
        BASE / "training/datasets/sentinel_train_v43.jsonl",

    "validation":
        BASE / "training/datasets/sentinel_validation_v43.jsonl",

    "test":
        BASE / "training/datasets/sentinel_test_v43.jsonl",
}


ASSETS = [
    "human-subject interview transcripts",
    "student advising records",
    "genomic sequencing files",
    "unpublished research manuscripts",
    "sponsor-confidential prototype designs",
]


# Conservative gates.
QUESTION_HARD_SEQ = 0.92
QUESTION_HARD_JACCARD = 0.75

QUESTION_REVIEW_SEQ = 0.88
QUESTION_REVIEW_JACCARD = 0.65

FULL_INPUT_HARD_COSINE = 0.85
FULL_INPUT_REVIEW_COSINE = 0.70


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def extract_question(text):
    """
    Extract question text from either:

    Challenge V1:
        QUESTION:
        <question>

        EVIDENCE:
        ...

    or:
        EVIDENCE:
        ...

        QUESTION:
        <question>

    V4.3:
        QUESTION
        ========
        <question>
    """

    lines = text.splitlines()

    # ---------------------------------------------------------
    # Challenge V1 format
    # ---------------------------------------------------------

    question_markers = [
        i
        for i, line in enumerate(lines)
        if line.strip().upper() == "QUESTION:"
    ]

    evidence_markers = [
        i
        for i, line in enumerate(lines)
        if line.strip().upper() == "EVIDENCE:"
    ]

    if question_markers:

        if len(question_markers) != 1:
            raise ValueError(
                f"Expected one QUESTION: marker, "
                f"found {len(question_markers)}"
            )

        q_index = question_markers[0]

        later_evidence = [
            i
            for i in evidence_markers
            if i > q_index
        ]

        if later_evidence:
            end_index = later_evidence[0]
        else:
            end_index = len(lines)

        question = "\n".join(
            lines[
                q_index + 1:
                end_index
            ]
        ).strip()

        if not question:
            raise ValueError(
                "QUESTION: section was empty"
            )

        return question


    # ---------------------------------------------------------
    # V4.3 format
    # ---------------------------------------------------------

    plain_markers = [
        i
        for i, line in enumerate(lines)
        if line.strip().upper() == "QUESTION"
    ]

    if plain_markers:

        if len(plain_markers) != 1:
            raise ValueError(
                f"Expected one QUESTION marker, "
                f"found {len(plain_markers)}"
            )

        q_index = plain_markers[0]

        start_index = q_index + 1

        # Skip ======== separator.
        if (
            start_index < len(lines)
            and lines[start_index].strip()
            and set(
                lines[start_index].strip()
            ) == {"="}
        ):
            start_index += 1

        question = "\n".join(
            lines[start_index:]
        ).strip()

        if not question:
            raise ValueError(
                "QUESTION section was empty"
            )

        return question


    raise ValueError(
        "Could not locate a supported QUESTION section."
    )


def normalize_question(text):
    text = text.lower()

    for asset in ASSETS:
        text = text.replace(
            asset.lower(),
            "<asset>",
        )

    # Remove V4.3's arbitrary request identifiers.
    text = re.sub(
        r"\brequest\s+[a-z]+\b",
        "request <target>",
        text,
    )

    text = re.sub(
        r"[^a-z0-9<>]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_input(text):
    """
    Remove formatting boilerplate while retaining policy semantics.
    """

    text = text.lower()

    for asset in ASSETS:
        text = text.replace(
            asset.lower(),
            "<asset>",
        )

    text = re.sub(
        r"\brequest\s+[a-z]+\b",
        "request <target>",
        text,
    )

    # Remove formatting labels / evidence IDs that are not semantics.
    text = re.sub(
        r"\[[^\]\n]+\]",
        " ",
        text,
    )

    text = re.sub(
        r"\bsource\s*:",
        " ",
        text,
    )

    text = re.sub(
        r"\bsection\s*:",
        " ",
        text,
    )

    text = re.sub(
        r"\btext\s*:",
        " ",
        text,
    )

    text = re.sub(
        r"\bquestion\b",
        " ",
        text,
    )

    text = re.sub(
        r"[=_\-]{2,}",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def token_jaccard(a, b):
    aa = set(a.split())
    bb = set(b.split())

    if not aa or not bb:
        return 0.0

    return len(
        aa & bb
    ) / len(
        aa | bb
    )


print("=" * 78)
print("SENTINEL V4.3 ↔ CHALLENGE V1 CONTAMINATION AUDIT")
print("=" * 78)

print()
print(
    "NOTE: Challenge gold labels are not loaded or used."
)


# ---------------------------------------------------------------------
# Load Challenge.
# ---------------------------------------------------------------------

challenge = json.loads(
    CHALLENGE.read_text()
)

if not isinstance(
    challenge,
    list,
):
    raise SystemExit(
        "ERROR: Challenge V1 root is not a list"
    )

if len(challenge) != 80:
    raise SystemExit(
        f"ERROR: expected 80 Challenge cases, got {len(challenge)}"
    )


challenge_records = []

for item in challenge:

    # Intentionally do NOT access item["gold"].
    case_id = item["case_id"]
    input_text = item["input"]

    if not isinstance(
        input_text,
        str,
    ):
        raise SystemExit(
            f"ERROR: Challenge input for {case_id} is not text"
        )

    challenge_records.append({
        "case_id":
            case_id,

        "question":
            extract_question(
                input_text
            ),

        "input":
            input_text,
    })


print(
    "challenge cases:",
    len(challenge_records),
)


# ---------------------------------------------------------------------
# Load V4.3.
# ---------------------------------------------------------------------

synthetic = []

for split, path in DATASETS.items():

    rows = load_jsonl(path)

    print(
        f"{split} rows:",
        len(rows),
    )

    for row in rows:

        synthetic.append({
            "split":
                split,

            "family":
                row["family"],

            "scenario_id":
                row["scenario_id"],

            "question":
                extract_question(
                    row["input"]
                ),

            "input":
                row["input"],
        })


# Deduplicate synthetic surface material for comparison efficiency.
question_records = {}

for row in synthetic:

    q = normalize_question(
        row["question"]
    )

    question_records.setdefault(
        (
            row["split"],
            row["family"],
            q,
        ),
        row,
    )


synthetic_questions = list(
    question_records.values()
)


print(
    "unique synthetic normalized question records:",
    len(synthetic_questions),
)


# =====================================================================
# QUESTION-LEVEL PARAPHRASE AUDIT
# =====================================================================

print()
print("=" * 78)
print("QUESTION-LEVEL AUDIT")
print("=" * 78)


hard_question = []
review_question = []

max_seq = 0.0
max_jaccard = 0.0


challenge_q = [
    (
        item["case_id"],
        normalize_question(
            item["question"]
        ),
    )
    for item in challenge_records
]


for synth in synthetic_questions:

    sq = normalize_question(
        synth["question"]
    )

    for case_id, cq in challenge_q:

        seq = SequenceMatcher(
            None,
            sq,
            cq,
        ).ratio()

        jac = token_jaccard(
            sq,
            cq,
        )

        max_seq = max(
            max_seq,
            seq,
        )

        max_jaccard = max(
            max_jaccard,
            jac,
        )

        record = {
            "split":
                synth["split"],

            "family":
                synth["family"],

            "scenario_id":
                synth["scenario_id"],

            "challenge_case":
                case_id,

            "sequence":
                seq,

            "jaccard":
                jac,
        }

        if (
            seq >= QUESTION_HARD_SEQ
            and jac >= QUESTION_HARD_JACCARD
        ):
            hard_question.append(
                record
            )

        elif (
            seq >= QUESTION_REVIEW_SEQ
            and jac >= QUESTION_REVIEW_JACCARD
        ):
            review_question.append(
                record
            )


print(
    "maximum sequence similarity:",
    f"{max_seq:.3f}",
)

print(
    "maximum token Jaccard:",
    f"{max_jaccard:.3f}",
)

print(
    "hard question alarms:",
    len(hard_question),
)

print(
    "review question alarms:",
    len(review_question),
)


def show_alarm_metadata(
    title,
    records,
    limit=10,
):
    if not records:
        return

    print()
    print(title)
    print("-" * 78)

    records = sorted(
        records,
        key=lambda x: (
            x.get(
                "sequence",
                x.get(
                    "cosine",
                    0,
                )
            ),
            x.get(
                "jaccard",
                0,
            ),
        ),
        reverse=True,
    )

    for item in records[:limit]:

        # IMPORTANT:
        # no Challenge text is printed.
        print(
            "split=",
            item["split"],
            " family=",
            item["family"],
            " scenario=",
            item["scenario_id"],
            " challenge_case=",
            item["challenge_case"],
            " seq=",
            f"{item.get('sequence', 0):.3f}",
            " jac=",
            f"{item.get('jaccard', 0):.3f}",
            " cosine=",
            f"{item.get('cosine', 0):.3f}",
            sep="",
        )


show_alarm_metadata(
    "HARD QUESTION ALARMS",
    hard_question,
)

show_alarm_metadata(
    "REVIEW QUESTION ALARMS",
    review_question,
)


# =====================================================================
# FULL-CASE LEXICAL SIMILARITY
# =====================================================================

print()
print("=" * 78)
print("FULL-CASE TF-IDF AUDIT")
print("=" * 78)


# One row per scenario is enough; surfaces otherwise triple the same world.
scenario_records = {}

for row in synthetic:

    scenario_records.setdefault(
        row["scenario_id"],
        row,
    )


synthetic_cases = list(
    scenario_records.values()
)


synth_texts = [
    normalize_input(
        row["input"]
    )
    for row in synthetic_cases
]

challenge_texts = [
    normalize_input(
        row["input"]
    )
    for row in challenge_records
]


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=1,
    sublinear_tf=True,
)

matrix = vectorizer.fit_transform(
    synth_texts
    + challenge_texts
)


n_synth = len(synth_texts)

similarity = cosine_similarity(
    matrix[:n_synth],
    matrix[n_synth:],
)


hard_case = []
review_case = []

global_max = 0.0


for i, synth in enumerate(
    synthetic_cases
):

    for j, challenge_item in enumerate(
        challenge_records
    ):

        score = float(
            similarity[i, j]
        )

        global_max = max(
            global_max,
            score,
        )

        record = {
            "split":
                synth["split"],

            "family":
                synth["family"],

            "scenario_id":
                synth["scenario_id"],

            "challenge_case":
                challenge_item["case_id"],

            "cosine":
                score,
        }

        if score >= FULL_INPUT_HARD_COSINE:
            hard_case.append(
                record
            )

        elif score >= FULL_INPUT_REVIEW_COSINE:
            review_case.append(
                record
            )


print(
    "synthetic scenarios compared:",
    len(synthetic_cases),
)

print(
    "challenge cases compared:",
    len(challenge_records),
)

print(
    "maximum full-case cosine:",
    f"{global_max:.3f}",
)

print(
    "hard full-case alarms:",
    len(hard_case),
)

print(
    "review full-case alarms:",
    len(review_case),
)


show_alarm_metadata(
    "HARD FULL-CASE ALARMS",
    hard_case,
)

show_alarm_metadata(
    "REVIEW FULL-CASE ALARMS",
    review_case,
)


# =====================================================================
# SPLIT SUMMARY
# =====================================================================

print()
print("=" * 78)
print("ALARM COUNTS BY SYNTHETIC SPLIT")
print("=" * 78)


for split in [
    "train",
    "validation",
    "test",
]:

    counts = {
        "question_hard":
            sum(
                1
                for x in hard_question
                if x["split"] == split
            ),

        "question_review":
            sum(
                1
                for x in review_question
                if x["split"] == split
            ),

        "case_hard":
            sum(
                1
                for x in hard_case
                if x["split"] == split
            ),

        "case_review":
            sum(
                1
                for x in review_case
                if x["split"] == split
            ),
    }

    print(
        split,
        counts,
    )


# =====================================================================
# FINAL
# =====================================================================

print()
print("=" * 78)

if hard_question or hard_case:

    print(
        "CHALLENGE CONTAMINATION AUDIT: FAILED"
    )

    print()
    print(
        "At least one high-confidence close-paraphrase "
        "alarm was detected."
    )

    print("=" * 78)

    raise SystemExit(1)


if review_question or review_case:

    print(
        "CHALLENGE CONTAMINATION AUDIT: REVIEW REQUIRED"
    )

    print()
    print(
        "No hard contamination was detected, but one or more "
        "borderline similarities require inspection before training."
    )

    print("=" * 78)

    raise SystemExit(2)


print(
    "CHALLENGE CONTAMINATION AUDIT: PASSED"
)

print()
print(
    "No exact/high-confidence close-paraphrase contamination "
    "was detected against frozen Challenge V1."
)

print("=" * 78)
