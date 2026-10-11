from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)


DATA = Path("/workspace/training/datasets")

FILES = {
    "train": DATA / "sentinel_train_v43.jsonl",
    "validation": DATA / "sentinel_validation_v43.jsonl",
    "test": DATA / "sentinel_test_v43.jsonl",
}

LABELS = [
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
]

ASSETS = [
    "human-subject interview transcripts",
    "student advising records",
    "genomic sequencing files",
    "unpublished research manuscripts",
    "sponsor-confidential prototype designs",
]

EXPECTED = {
    "train": {
        "rows": 2880,
        "scenarios": 960,
        "families": 30,
        "questions": 720,
        "evidence_packs": 240,
        "per_label": 720,
    },
    "validation": {
        "rows": 960,
        "scenarios": 320,
        "families": 10,
        "questions": 240,
        "evidence_packs": 80,
        "per_label": 240,
    },
    "test": {
        "rows": 960,
        "scenarios": 320,
        "families": 10,
        "questions": 240,
        "evidence_packs": 80,
        "per_label": 240,
    },
}


def load(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


ROWS = {
    split: load(path)
    for split, path in FILES.items()
}


def decision(row):
    return json.loads(
        row["output"]
    )["decision"]


def output_obj(row):
    return json.loads(
        row["output"]
    )


def extract_question(input_text):
    marker = "\n\nQUESTION\n========\n"

    if marker not in input_text:
        raise ValueError(
            "QUESTION marker missing"
        )

    return input_text.split(
        marker,
        1,
    )[1].strip()


def extract_evidence(input_text):
    marker = "\n\nQUESTION\n========\n"

    if marker not in input_text:
        raise ValueError(
            "QUESTION marker missing"
        )

    return input_text.split(
        marker,
        1,
    )[0].strip()


def canonical_evidence(input_text):
    """
    Ignore shuffled passage order while retaining semantics.
    """

    evidence = extract_evidence(
        input_text
    )

    blocks = re.split(
        r"\n(?=\[[^\]\n]+\]\n)",
        evidence,
    )

    normalized = []

    for block in blocks:

        if "Text:" not in block:
            continue

        normalized.append(
            re.sub(
                r"\s+",
                " ",
                block.strip().lower(),
            )
        )

    return tuple(
        sorted(normalized)
    )


def label_counts(rows):
    return Counter(
        decision(row)
        for row in rows
    )


def exact_identity_baseline(groups):
    correct = 0
    total = 0

    for rows in groups.values():
        counts = label_counts(rows)

        correct += max(
            counts.values()
        )

        total += len(rows)

    return correct / total


def slug(text):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        text.lower(),
    ).strip("_")


def normalized_question(q):
    q = q.lower()

    for asset in ASSETS:
        q = q.replace(
            asset.lower(),
            "<asset>",
        )

    # Request labels are deliberately synthetic surface identifiers.
    q = re.sub(
        r"\brequest\s+[a-z]+\b",
        "request <target>",
        q,
    )

    q = re.sub(
        r"\s+",
        " ",
        q,
    )

    return q.strip()


failed = False


print("=" * 78)
print("SENTINEL V4.3 FULL ADVERSARIAL AUDIT")
print("=" * 78)


# =====================================================================
# 1–3. STRUCTURE + QUESTION CAUSALITY + EVIDENCE CAUSALITY
# =====================================================================

for split, rows in ROWS.items():

    target = EXPECTED[split]

    print()
    print("=" * 78)
    print(split.upper())
    print("=" * 78)

    families = {
        row["family"]
        for row in rows
    }

    scenarios = {
        row["scenario_id"]
        for row in rows
    }

    counts = label_counts(rows)

    print(
        "rows:",
        len(rows),
    )

    print(
        "scenarios:",
        len(scenarios),
    )

    print(
        "families:",
        len(families),
    )

    print(
        "labels:",
        dict(
            sorted(
                counts.items()
            )
        ),
    )

    if len(rows) != target["rows"]:
        print("FAIL: wrong row count")
        failed = True

    if len(scenarios) != target["scenarios"]:
        print("FAIL: wrong scenario count")
        failed = True

    if len(families) != target["families"]:
        print("FAIL: wrong family count")
        failed = True

    for label in LABELS:

        if counts[label] != target["per_label"]:
            print(
                f"FAIL: {label} count "
                f"{counts[label]}"
            )

            failed = True


    # -------------------------------------------------------------
    # Scenario consistency:
    # all three real paraphrases must preserve gold semantics.
    # -------------------------------------------------------------

    by_scenario = defaultdict(list)

    for row in rows:
        by_scenario[
            row["scenario_id"]
        ].append(row)

    bad_scenarios = []

    for scenario, group in by_scenario.items():

        decisions = {
            decision(row)
            for row in group
        }

        outputs = {
            row["output"]
            for row in group
        }

        surfaces = {
            row["surface_variant"]
            for row in group
        }

        if (
            len(group) != 3
            or len(decisions) != 1
            or len(outputs) != 1
            or surfaces != {0, 1, 2}
        ):
            bad_scenarios.append(
                scenario
            )

    print()
    print(
        "bad paraphrase/scenario groups:",
        len(bad_scenarios),
    )

    if bad_scenarios:
        failed = True


    # -------------------------------------------------------------
    # QUESTION CAUSALITY
    #
    # Hold exact question constant across four evidence worlds.
    # Must produce all four labels once.
    # -------------------------------------------------------------

    by_question = defaultdict(list)

    for row in rows:
        by_question[
            extract_question(
                row["input"]
            )
        ].append(row)

    bad_questions = []

    expected_q_labels = Counter({
        "YES": 1,
        "NO": 1,
        "CONDITIONAL": 1,
        "INSUFFICIENT": 1,
    })

    for question, group in by_question.items():

        if (
            len(group) != 4
            or label_counts(group)
            != expected_q_labels
        ):
            bad_questions.append(
                question
            )

    q_identity = exact_identity_baseline(
        by_question
    )

    print()
    print("QUESTION CAUSALITY")
    print("-" * 78)

    print(
        "unique exact questions:",
        len(by_question),
    )

    print(
        "bad question groups:",
        len(bad_questions),
    )

    print(
        "best exact-question-only accuracy:",
        f"{q_identity:.3f}",
    )

    if (
        len(by_question)
        != target["questions"]
    ):
        print(
            "FAIL: wrong unique question count"
        )

        failed = True

    if bad_questions:
        failed = True

    if abs(q_identity - 0.25) > 1e-9:
        print(
            "FAIL: exact question carries label signal"
        )

        failed = True


    # -------------------------------------------------------------
    # EVIDENCE CAUSALITY
    #
    # Each evidence pack supports 4 questions × 3 paraphrases.
    # Every label therefore occurs exactly 3 times.
    # -------------------------------------------------------------

    by_evidence = defaultdict(list)

    for row in rows:
        by_evidence[
            canonical_evidence(
                row["input"]
            )
        ].append(row)

    expected_e_labels = Counter({
        "YES": 3,
        "NO": 3,
        "CONDITIONAL": 3,
        "INSUFFICIENT": 3,
    })

    bad_evidence = []

    for evidence, group in by_evidence.items():

        if (
            len(group) != 12
            or label_counts(group)
            != expected_e_labels
        ):
            bad_evidence.append(
                evidence
            )

    e_identity = exact_identity_baseline(
        by_evidence
    )

    print()
    print("EVIDENCE CAUSALITY")
    print("-" * 78)

    print(
        "unique evidence packs:",
        len(by_evidence),
    )

    print(
        "bad evidence groups:",
        len(bad_evidence),
    )

    print(
        "best exact-evidence-only accuracy:",
        f"{e_identity:.3f}",
    )

    if (
        len(by_evidence)
        != target["evidence_packs"]
    ):
        print(
            "FAIL: wrong evidence pack count"
        )

        failed = True

    if bad_evidence:
        failed = True

    if abs(e_identity - 0.25) > 1e-9:
        print(
            "FAIL: exact evidence carries label signal"
        )

        failed = True


    # -------------------------------------------------------------
    # 4. CITATION / CONDITION / TARGET RESOLUTION
    # -------------------------------------------------------------

    semantic_errors = []

    for row in rows:

        obj = output_obj(row)

        target_name = row["target"]

        target_prefix = (
            slug(target_name)
            + "__"
        )

        question = extract_question(
            row["input"]
        )

        evidence = extract_evidence(
            row["input"]
        )

        if (
            f"request {target_name}".lower()
            not in question.lower()
        ):
            semantic_errors.append(
                (
                    row["scenario_id"],
                    "question missing target",
                )
            )

        evidence_ids = set(
            re.findall(
                r"^\[([^\]]+)\]",
                evidence,
                flags=re.MULTILINE,
            )
        )

        citations = set(
            obj["citations"]
        )

        invalid = (
            citations
            - evidence_ids
        )

        if invalid:
            semantic_errors.append(
                (
                    row["scenario_id"],
                    "invalid citation",
                    sorted(invalid),
                )
            )

        # Non-empty citations must resolve to the exact
        # request asked about.
        for citation in citations:

            if not citation.startswith(
                target_prefix
            ):
                semantic_errors.append(
                    (
                        row["scenario_id"],
                        "cross-target citation",
                        citation,
                    )
                )

        if (
            obj["decision"]
            == "INSUFFICIENT"
            and citations
        ):
            semantic_errors.append(
                (
                    row["scenario_id"],
                    "INSUFFICIENT has citations",
                )
            )

        if (
            obj["decision"]
            == "CONDITIONAL"
            and not obj["conditions"]
        ):
            semantic_errors.append(
                (
                    row["scenario_id"],
                    "CONDITIONAL missing conditions",
                )
            )

        if (
            obj["decision"]
            != "CONDITIONAL"
            and obj["conditions"]
        ):
            semantic_errors.append(
                (
                    row["scenario_id"],
                    "non-CONDITIONAL has conditions",
                )
            )

        # Evidence pack must actually contain material for
        # the request in the question.
        if target_prefix not in evidence:
            semantic_errors.append(
                (
                    row["scenario_id"],
                    "target has no evidence chunk",
                )
            )

    print()
    print("TARGET / METADATA INTEGRITY")
    print("-" * 78)

    print(
        "semantic errors:",
        len(semantic_errors),
    )

    if semantic_errors:

        print(
            "first errors:",
            semantic_errors[:10],
        )

        failed = True


# =====================================================================
# 5. CROSS-SPLIT EXACT + NEAR-DUPLICATE QUESTION AUDIT
# =====================================================================

print()
print("=" * 78)
print("CROSS-SPLIT LEAKAGE / NEAR-DUPLICATE AUDIT")
print("=" * 78)


def unique_questions(split):
    return sorted({
        normalized_question(
            extract_question(
                row["input"]
            )
        )
        for row in ROWS[split]
    })


question_sets = {
    split: unique_questions(split)
    for split in ROWS
}


pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]


for left, right in pairs:

    exact = (
        set(question_sets[left])
        & set(question_sets[right])
    )

    alarms = []

    for q1 in question_sets[left]:

        for q2 in question_sets[right]:

            ratio = SequenceMatcher(
                None,
                q1,
                q2,
            ).ratio()

            if ratio >= 0.92:
                alarms.append(
                    (
                        ratio,
                        q1,
                        q2,
                    )
                )

    alarms.sort(
        reverse=True,
        key=lambda x: x[0],
    )

    print()
    print(
        f"{left} vs {right}:"
    )

    print(
        " exact normalized overlap:",
        len(exact),
    )

    print(
        " >=0.92 similarity alarms:",
        len(alarms),
    )

    if alarms:

        print(
            " top alarm:",
            alarms[0],
        )

    if exact or alarms:
        failed = True


# =====================================================================
# 6. QUESTION-ONLY / EVIDENCE-ONLY LEXICAL ATTACKS
# =====================================================================

print()
print("=" * 78)
print("LEXICAL ADVERSARIES")
print("=" * 78)


def run_probe(
    name,
    text_fn,
):

    train_rows = ROWS["train"]

    X_train = [
        text_fn(row)
        for row in train_rows
    ]

    y_train = [
        decision(row)
        for row in train_rows
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    Xtr = vectorizer.fit_transform(
        X_train
    )

    model = LogisticRegression(
        max_iter=3000,
        C=1.0,
        random_state=20260814,
    )

    model.fit(
        Xtr,
        y_train,
    )

    results = {}

    print()
    print(name)
    print("-" * 78)

    for split in [
        "validation",
        "test",
    ]:

        rows = ROWS[split]

        X = vectorizer.transform([
            text_fn(row)
            for row in rows
        ])

        y = [
            decision(row)
            for row in rows
        ]

        pred = model.predict(X)

        acc = accuracy_score(
            y,
            pred,
        )

        results[split] = acc

        print()
        print(
            split.upper(),
            "accuracy:",
            f"{acc:.3f}",
        )

        print(
            "confusion matrix "
            "(rows=gold, cols=pred)"
        )

        print(LABELS)

        print(
            confusion_matrix(
                y,
                pred,
                labels=LABELS,
            )
        )

    return results


question_results = run_probe(
    "QUESTION-ONLY TF-IDF",
    lambda row: extract_question(
        row["input"]
    ),
)

evidence_results = run_probe(
    "EVIDENCE-ONLY TF-IDF",
    lambda row: extract_evidence(
        row["input"]
    ),
)


for probe_name, results in [
    (
        "question-only",
        question_results,
    ),
    (
        "evidence-only",
        evidence_results,
    ),
]:

    for split, acc in results.items():

        if acc > 0.35:

            print(
                f"FAIL: {probe_name} {split} "
                f"{acc:.3f} > 0.350"
            )

            failed = True


# =====================================================================
# 7. COUNTERFACTUAL SWAP / NEGATIVE CONTROL
#
# A. hold question constant, swap evidence world
# B. hold evidence constant, swap question target
#
# The gold decision should change.
# =====================================================================

print()
print("=" * 78)
print("COUNTERFACTUAL SWAP CONTROLS")
print("=" * 78)


for split, rows in ROWS.items():

    # Use one representative per surface for deterministic lookup.
    lookup = {}

    for row in rows:

        key = (
            row["family"],
            row["asset"],
            row["world"],
            row["target"],
            row["surface_variant"],
        )

        lookup[key] = row


    evidence_swap_total = 0
    evidence_swap_changed = 0

    question_swap_total = 0
    question_swap_changed = 0


    # Map family/asset to the four targets and four worlds.
    contexts = defaultdict(
        lambda: {
            "targets": set(),
            "worlds": set(),
        }
    )

    for row in rows:

        key = (
            row["family"],
            row["asset"],
        )

        contexts[key]["targets"].add(
            row["target"]
        )

        contexts[key]["worlds"].add(
            row["world"]
        )


    for (
        family,
        asset,
    ), context in contexts.items():

        targets = sorted(
            context["targets"]
        )

        worlds = sorted(
            context["worlds"]
        )

        assert len(targets) == 4
        assert len(worlds) == 4


        # ---------------------------------------------------------
        # Same question target, different evidence world.
        # ---------------------------------------------------------

        for target in targets:

            for surface in range(3):

                for i, world in enumerate(
                    worlds
                ):

                    next_world = worlds[
                        (i + 1) % 4
                    ]

                    original = lookup[
                        (
                            family,
                            asset,
                            world,
                            target,
                            surface,
                        )
                    ]

                    swapped = lookup[
                        (
                            family,
                            asset,
                            next_world,
                            target,
                            surface,
                        )
                    ]

                    evidence_swap_total += 1

                    if (
                        decision(original)
                        != decision(swapped)
                    ):
                        evidence_swap_changed += 1


        # ---------------------------------------------------------
        # Same evidence world, different question target.
        # ---------------------------------------------------------

        for world in worlds:

            for surface in range(3):

                for i, target in enumerate(
                    targets
                ):

                    next_target = targets[
                        (i + 1) % 4
                    ]

                    original = lookup[
                        (
                            family,
                            asset,
                            world,
                            target,
                            surface,
                        )
                    ]

                    swapped = lookup[
                        (
                            family,
                            asset,
                            world,
                            next_target,
                            surface,
                        )
                    ]

                    question_swap_total += 1

                    if (
                        decision(original)
                        != decision(swapped)
                    ):
                        question_swap_changed += 1


    evidence_rate = (
        evidence_swap_changed
        / evidence_swap_total
    )

    question_rate = (
        question_swap_changed
        / question_swap_total
    )

    print()
    print(split.upper())

    print(
        "same question + swapped evidence "
        "changes decision:",
        f"{evidence_swap_changed}/"
        f"{evidence_swap_total}",
        f"({evidence_rate:.3f})",
    )

    print(
        "same evidence + swapped question "
        "changes decision:",
        f"{question_swap_changed}/"
        f"{question_swap_total}",
        f"({question_rate:.3f})",
    )

    # Every column and every row of the Latin relation matrix
    # contains all four labels, so adjacent swaps should always flip.
    if evidence_rate != 1.0:
        print(
            "FAIL: evidence swap does not "
            "always change decision"
        )

        failed = True

    if question_rate != 1.0:
        print(
            "FAIL: question swap does not "
            "always change decision"
        )

        failed = True


# =====================================================================
# FINAL
# =====================================================================

print()
print("=" * 78)
print("V4.3 FULL DATASET SCORECARD")
print("=" * 78)

print(
    "question-only validation:",
    f"{question_results['validation']:.3f}",
)

print(
    "question-only test:",
    f"{question_results['test']:.3f}",
)

print(
    "evidence-only validation:",
    f"{evidence_results['validation']:.3f}",
)

print(
    "evidence-only test:",
    f"{evidence_results['test']:.3f}",
)

print()

if failed:

    print(
        "V4.3 FULL ADVERSARIAL AUDIT: FAILED"
    )

    print("=" * 78)

    raise SystemExit(1)


print(
    "V4.3 FULL ADVERSARIAL AUDIT: PASSED"
)

print()
print(
    "Neither question nor evidence alone "
    "carries sufficient label signal."
)

print(
    "The decision changes when either side "
    "of the relation is counterfactually changed."
)

print("=" * 78)
