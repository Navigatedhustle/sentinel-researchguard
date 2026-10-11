from pathlib import Path
from difflib import SequenceMatcher
from itertools import combinations
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE = Path("/workspace")

FILES = {
    "train":
        BASE
        / "training/datasets/"
          "sentinel_train_v44_gate_pilot.jsonl",

    "validation":
        BASE
        / "training/datasets/"
          "sentinel_validation_v44_gate_pilot.jsonl",

    "test":
        BASE
        / "training/datasets/"
          "sentinel_test_v44_gate_pilot.jsonl",
}

CHALLENGE = (
    BASE
    / "experiments/challenge_v1/"
      "sentinel_challenge_v1.json"
)


def load_jsonl(path):

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def normalize(text):

    return re.sub(
        r"\s+",
        " ",
        text.lower().strip(),
    )


def flatten_strings(obj):

    out = []

    if isinstance(obj, str):

        out.append(obj)

    elif isinstance(obj, list):

        for item in obj:
            out.extend(
                flatten_strings(item)
            )

    elif isinstance(obj, dict):

        for value in obj.values():
            out.extend(
                flatten_strings(value)
            )

    return out


def load_challenge(path):

    obj = json.loads(
        path.read_text()
    )

    if isinstance(obj, list):
        cases = obj

    elif isinstance(obj, dict):

        # Robustly locate the first obvious list
        # of benchmark cases.
        candidates = [
            value
            for value in obj.values()
            if isinstance(value, list)
        ]

        if not candidates:
            raise ValueError(
                "Could not locate Challenge case list."
            )

        cases = max(
            candidates,
            key=len,
        )

    else:
        raise TypeError(
            "Unexpected Challenge structure."
        )

    texts = []

    for case in cases:

        text = "\n".join(
            flatten_strings(case)
        )

        texts.append(
            normalize(text)
        )

    return texts


def exact_overlap(
    left,
    right,
    field,
):

    a = {
        normalize(row[field])
        for row in left
    }

    b = {
        normalize(row[field])
        for row in right
    }

    return a & b


def max_cosine(
    left_texts,
    right_texts,
):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
    )

    matrix = vectorizer.fit_transform(
        left_texts + right_texts
    )

    left_matrix = matrix[
        :len(left_texts)
    ]

    right_matrix = matrix[
        len(left_texts):
    ]

    scores = cosine_similarity(
        left_matrix,
        right_matrix,
    )

    index = scores.argmax()

    i, j = divmod(
        index,
        scores.shape[1],
    )

    return (
        float(scores[i, j]),
        i,
        j,
    )


def main():

    print("=" * 88)
    print("V4.4 GATE PILOT — CONTAMINATION / LEAKAGE AUDIT")
    print("=" * 88)

    data = {
        split: load_jsonl(path)
        for split, path in FILES.items()
    }

    errors = []

    # ---------------------------------------------------------
    # Cross-split identifiers
    # ---------------------------------------------------------

    print()
    print("CROSS-SPLIT IDENTITY")
    print("-" * 88)

    for a, b in combinations(
        [
            "train",
            "validation",
            "test",
        ],
        2,
    ):

        scenario_overlap = (
            {
                row["scenario_id"]
                for row in data[a]
            }
            &
            {
                row["scenario_id"]
                for row in data[b]
            }
        )

        question_overlap = exact_overlap(
            data[a],
            data[b],
            "question",
        )

        evidence_overlap = exact_overlap(
            data[a],
            data[b],
            "evidence",
        )

        input_overlap = exact_overlap(
            data[a],
            data[b],
            "input",
        )

        print()
        print(f"{a} ↔ {b}")
        print(
            " scenario overlap:",
            len(scenario_overlap),
        )
        print(
            " exact question overlap:",
            len(question_overlap),
        )
        print(
            " exact evidence overlap:",
            len(evidence_overlap),
        )
        print(
            " exact input overlap:",
            len(input_overlap),
        )

        if scenario_overlap:
            errors.append(
                f"{a}/{b}: scenario leakage"
            )

        if question_overlap:
            errors.append(
                f"{a}/{b}: exact question leakage"
            )

        if evidence_overlap:
            errors.append(
                f"{a}/{b}: exact evidence leakage"
            )

        if input_overlap:
            errors.append(
                f"{a}/{b}: exact input leakage"
            )

    # ---------------------------------------------------------
    # Cross-split lexical similarity
    # Informational — high template similarity is expected.
    # Exact identity above is the hard failure criterion.
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("CROSS-SPLIT MAX TF-IDF SIMILARITY")
    print("=" * 88)

    for eval_split in [
        "validation",
        "test",
    ]:

        train_texts = [
            normalize(
                row["question"]
                + "\n"
                + row["evidence"]
            )
            for row in data["train"]
        ]

        eval_texts = [
            normalize(
                row["question"]
                + "\n"
                + row["evidence"]
            )
            for row in data[eval_split]
        ]

        score, i, j = max_cosine(
            train_texts,
            eval_texts,
        )

        print()
        print(
            f"train ↔ {eval_split}:",
            f"{score:.4f}",
        )

        print(
            " train scenario:",
            data["train"][i][
                "scenario_id"
            ],
        )

        print(
            f" {eval_split} scenario:",
            data[eval_split][j][
                "scenario_id"
            ],
        )

    # ---------------------------------------------------------
    # Challenge V1 contamination
    # ---------------------------------------------------------

    print()
    print("=" * 88)
    print("CHALLENGE V1 CONTAMINATION")
    print("=" * 88)

    challenge_texts = load_challenge(
        CHALLENGE
    )

    print(
        "challenge cases:",
        len(challenge_texts),
    )

    pilot_rows = (
        data["train"]
        + data["validation"]
        + data["test"]
    )

    pilot_texts = [
        normalize(
            row["question"]
            + "\n"
            + row["evidence"]
        )
        for row in pilot_rows
    ]

    challenge_set = set(
        challenge_texts
    )

    exact_full = sum(
        text in challenge_set
        for text in pilot_texts
    )

    print(
        "exact normalized full-case overlaps:",
        exact_full,
    )

    if exact_full:
        errors.append(
            "Challenge exact full-case contamination"
        )

    score, i, j = max_cosine(
        pilot_texts,
        challenge_texts,
    )

    print(
        "max pilot ↔ Challenge cosine:",
        f"{score:.4f}",
    )

    print(
        "pilot scenario:",
        pilot_rows[i][
            "scenario_id"
        ],
    )

    print(
        "challenge index:",
        j,
    )

    # 0.92 is intentionally conservative:
    # similarity below this may simply reflect shared
    # governance vocabulary.
    if score >= 0.92:

        errors.append(
            (
                "Challenge near-duplicate alarm: "
                f"{score:.4f}"
            )
        )

    # ---------------------------------------------------------
    # Challenge question-level similarity
    # ---------------------------------------------------------

    pilot_questions = sorted({
        normalize(
            row["question"]
        )
        for row in pilot_rows
    })

    # Challenge is heterogeneous, so compare pilot questions
    # against flattened case text only as an alarm mechanism.
    max_sequence = 0.0
    max_pair = None

    for question in pilot_questions:

        for index, challenge_text in enumerate(
            challenge_texts
        ):

            score_seq = SequenceMatcher(
                None,
                question,
                challenge_text,
            ).ratio()

            if score_seq > max_sequence:

                max_sequence = score_seq

                max_pair = (
                    question,
                    index,
                )

    print(
        "max question sequence similarity:",
        f"{max_sequence:.4f}",
    )

    if max_sequence >= 0.90:

        errors.append(
            (
                "Challenge question near-duplicate alarm: "
                f"{max_sequence:.4f}"
            )
        )

    print()
    print("=" * 88)

    if errors:

        print(
            "V4.4 GATE CONTAMINATION AUDIT: FAIL"
        )

        for error in errors:
            print(
                " -",
                error,
            )

        raise SystemExit(1)

    print(
        "V4.4 GATE CONTAMINATION AUDIT: PASS"
    )


if __name__ == "__main__":
    main()
