from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import re
import sys


BASE = Path("/workspace")

PUBLIC = (
    BASE
    / "experiments/challenge_v4/candidate_v2/"
      "sentinel_challenge_v4_candidate_v2.json"
)

HIDDEN = (
    BASE
    / "experiments/challenge_v4/candidate_v2/audit_only/"
      "sentinel_challenge_v4_hidden_semantics_candidate_v2.jsonl"
)

OUT_JSON = (
    BASE
    / "experiments/challenge_v4/candidate_v2/audits/"
      "CHALLENGE_V4_CANDIDATE_V2_STRUCTURAL_SHORTCUT_AUDIT.json"
)

OUT_TXT = (
    BASE
    / "experiments/challenge_v4/candidate_v2/audits/"
      "CHALLENGE_V4_CANDIDATE_V2_STRUCTURAL_SHORTCUT_AUDIT.txt"
)


LABELS = [
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
]


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def norm(text):
    return " ".join(text.lower().split())


def fail(msg):
    raise AssertionError(msg)


def majority_accuracy(rows, key_fn):
    buckets = defaultdict(Counter)

    for row in rows:
        buckets[key_fn(row)][row["semantic_state"]] += 1

    correct = sum(
        max(c.values())
        for c in buckets.values()
    )

    total = sum(
        sum(c.values())
        for c in buckets.values()
    )

    return correct / total, buckets


def tfidf_probe(texts, labels, groups):
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import Pipeline
    except Exception as e:
        return {
            "status": "UNAVAILABLE",
            "reason": repr(e),
        }

    pipe = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                max_features=25000,
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                random_state=20260817,
            ),
        ),
    ])

    splitter = GroupKFold(n_splits=5)

    scores = []

    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels, dtype=object)
    groups = np.asarray(groups, dtype=object)

    for train_idx, test_idx in splitter.split(
        texts,
        labels,
        groups,
    ):
        pipe.fit(
            texts[train_idx].tolist(),
            labels[train_idx].tolist(),
        )

        score = pipe.score(
            texts[test_idx].tolist(),
            labels[test_idx].tolist(),
        )

        scores.append(float(score))

    return {
        "status": "OK",
        "fold_scores": scores,
        "mean_accuracy": sum(scores) / len(scores),
        "min_accuracy": min(scores),
        "max_accuracy": max(scores),
    }


def main():
    payload = json.loads(
        PUBLIC.read_text(
            encoding="utf-8"
        )
    )

    public = payload["cases"]

    hidden = [
        json.loads(line)
        for line in HIDDEN.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    # ============================================================
    # BASIC COUNTS
    # ============================================================

    if len(public) != 320:
        fail(
            f"public count {len(public)} != 320"
        )

    if len(hidden) != 320:
        fail(
            f"hidden count {len(hidden)} != 320"
        )

    public_ids = [
        x["case_id"]
        for x in public
    ]

    hidden_ids = [
        x["case_id"]
        for x in hidden
    ]

    if len(public_ids) != len(set(public_ids)):
        fail("duplicate public IDs")

    if len(hidden_ids) != len(set(hidden_ids)):
        fail("duplicate hidden IDs")

    if public_ids != hidden_ids:
        fail("public/hidden ordering mismatch")

    public_by_id = {
        x["case_id"]: x
        for x in public
    }

    hidden_by_id = {
        x["case_id"]: x
        for x in hidden
    }

    # ============================================================
    # LABEL / TYPE / COMPOSITION
    # ============================================================

    labels = Counter(
        x["gold"]
        for x in public
    )

    expected_labels = Counter({
        "YES": 80,
        "NO": 80,
        "CONDITIONAL": 80,
        "INSUFFICIENT": 80,
    })

    if labels != expected_labels:
        fail(
            f"label imbalance: {labels}"
        )

    question_types = Counter(
        x["question_type"]
        for x in public
    )

    if question_types != Counter({
        "relational_counterfactual": 240,
        "stress": 80,
    }):
        fail(
            f"question type mismatch: "
            f"{question_types}"
        )

    compositions = Counter(
        x["composition"]
        for x in public
    )

    if len(compositions) != 16:
        fail(
            f"composition count "
            f"{len(compositions)} != 16"
        )

    if set(compositions.values()) != {20}:
        fail(
            f"composition row counts: "
            f"{compositions}"
        )

    # Every composition must itself be exactly balanced.
    comp_labels = defaultdict(Counter)

    for row in public:
        comp_labels[
            row["composition"]
        ][row["gold"]] += 1

    for comp, counts in comp_labels.items():
        if counts != Counter({
            "YES": 5,
            "NO": 5,
            "CONDITIONAL": 5,
            "INSUFFICIENT": 5,
        }):
            fail(
                f"composition label imbalance "
                f"{comp}: {counts}"
            )

    # ============================================================
    # INPUT ORDER
    # ============================================================

    input_order = Counter(
        x["input_order"]
        for x in public
    )

    if input_order != Counter({
        "evidence_first": 160,
        "question_first": 160,
    }):
        fail(
            f"input order imbalance: "
            f"{input_order}"
        )

    # ============================================================
    # PUBLIC/HIDDEN CONSISTENCY
    # ============================================================

    hidden_field_names = {
        "relational_system",
        "world_index",
        "target_index",
        "target_id",
        "transversal_index",
        "label_shift",
        "oracle_mode",
        "semantic_state",
        "gold_annotation",
    }

    for case in public:
        leaked = (
            set(case)
            & hidden_field_names
        )

        if leaked:
            fail(
                f"hidden metadata leaked into "
                f"{case['case_id']}: {leaked}"
            )

    for cid in public_ids:
        p = public_by_id[cid]
        h = hidden_by_id[cid]

        if p["gold"] != h["semantic_state"]:
            fail(
                f"gold mismatch {cid}"
            )

        ann = h["gold_annotation"]

        if ann["decision"] != p["gold"]:
            fail(
                f"annotation decision mismatch "
                f"{cid}"
            )

        if set(ann) != {
            "decision",
            "citations",
            "conditions",
            "rationale",
            "state",
        }:
            fail(
                f"annotation schema mismatch "
                f"{cid}: {set(ann)}"
            )

        evidence_ids = {
            d["source_id"]
            for d in p["evidence"]
        }

        for citation in ann["citations"]:
            if citation not in evidence_ids:
                fail(
                    f"invalid citation "
                    f"{citation} in {cid}"
                )

        if (
            p["gold"] == "CONDITIONAL"
            and not ann["conditions"]
        ):
            fail(
                f"CONDITIONAL missing conditions "
                f"{cid}"
            )

        if (
            p["gold"] != "CONDITIONAL"
            and ann["conditions"]
        ):
            fail(
                f"non-CONDITIONAL has unresolved "
                f"conditions {cid}"
            )

    # ============================================================
    # RELATIONAL-SYSTEM STRUCTURE
    # ============================================================

    systems = defaultdict(list)

    for h in hidden:
        systems[
            h["relational_system"]
        ].append(h)

    if len(systems) != 80:
        fail(
            f"relational systems "
            f"{len(systems)} != 80"
        )

    for system_id, rows in systems.items():
        if len(rows) != 4:
            fail(
                f"{system_id} has "
                f"{len(rows)} rows"
            )

        states = Counter(
            r["semantic_state"]
            for r in rows
        )

        if states != Counter({
            "YES": 1,
            "NO": 1,
            "CONDITIONAL": 1,
            "INSUFFICIENT": 1,
        }):
            fail(
                f"{system_id} state geometry "
                f"{states}"
            )

        worlds = sorted(
            r["world_index"]
            for r in rows
        )

        if worlds != [0, 1, 2, 3]:
            fail(
                f"{system_id} world geometry "
                f"{worlds}"
            )

        targets = sorted(
            r["target_index"]
            for r in rows
        )

        if targets != [0, 1, 2, 3]:
            fail(
                f"{system_id} target geometry "
                f"{targets}"
            )

    # ============================================================
    # STRUCTURAL SHORTCUTS
    # ============================================================

    world_acc, world_buckets = majority_accuracy(
        hidden,
        lambda r: r["world_index"],
    )

    target_acc, target_buckets = majority_accuracy(
        hidden,
        lambda r: r["target_index"],
    )

    pair_acc, pair_buckets = majority_accuracy(
        hidden,
        lambda r: (
            r["target_index"],
            r["world_index"],
        ),
    )

    transversal_acc, transversal_buckets = majority_accuracy(
        hidden,
        lambda r: r["transversal_index"],
    )

    if abs(world_acc - 0.25) > 1e-12:
        fail(
            f"world coordinate shortcut "
            f"{world_acc}"
        )

    if abs(target_acc - 0.25) > 1e-12:
        fail(
            f"target coordinate shortcut "
            f"{target_acc}"
        )

    if abs(pair_acc - 0.25) > 1e-12:
        fail(
            f"target/world pair shortcut "
            f"{pair_acc}"
        )

    if abs(transversal_acc - 0.25) > 1e-12:
        fail(
            f"transversal-index shortcut "
            f"{transversal_acc}"
        )

    if len(pair_buckets) != 16:
        fail(
            f"coordinate cells "
            f"{len(pair_buckets)} != 16"
        )

    # Every coordinate cell should contain 20 rows:
    # 5 of each label.
    for cell, counts in pair_buckets.items():
        if sum(counts.values()) != 20:
            fail(
                f"cell {cell} count "
                f"{sum(counts.values())}"
            )

        if counts != Counter({
            "YES": 5,
            "NO": 5,
            "CONDITIONAL": 5,
            "INSUFFICIENT": 5,
        }):
            fail(
                f"cell {cell} imbalance "
                f"{counts}"
            )

    # ============================================================
    # DUPLICATES
    # ============================================================

    question_counts = Counter(
        norm(x["question"])
        for x in public
    )

    duplicate_questions = sum(
        v - 1
        for v in question_counts.values()
        if v > 1
    )

    pair_counts = Counter()

    for row in public:
        evidence_text = "\n".join(
            d["text"]
            for d in row["evidence"]
        )

        pair_counts[
            (
                norm(row["question"]),
                norm(evidence_text),
            )
        ] += 1

    duplicate_pairs = sum(
        v - 1
        for v in pair_counts.values()
        if v > 1
    )

    if duplicate_pairs != 0:
        fail(
            f"duplicate question/evidence pairs: "
            f"{duplicate_pairs}"
        )

    # ============================================================
    # OUTPUT-LABEL LEAKAGE
    #
    # YES and NO are ordinary English words, so title-case or
    # lowercase occurrences such as "No completed decision..." are
    # not themselves evidence of output-label leakage.
    #
    # We therefore distinguish:
    #
    #   ordinary prose
    #
    # from:
    #
    #   literal class tokens / explicit answer markers.
    # ============================================================

    direct_class_leaks = []
    explicit_marker_leaks = []

    marker_pattern = re.compile(
        r"\b(?:decision|label|gold|answer|classification)"
        r"\s*(?:is\s+|[:=]\s*)"
        r"[\"']?"
        r"(YES|NO|CONDITIONAL|INSUFFICIENT)"
        r"\b",
        flags=re.IGNORECASE,
    )

    for row in public:
        # Only inspect case-specific question/evidence.
        #
        # The system instruction intentionally contains all four
        # output labels and is therefore excluded from this audit.
        case_text = (
            row["question"]
            + "\n"
            + "\n".join(
                d["text"]
                for d in row["evidence"]
            )
        )

        # CONDITIONAL and INSUFFICIENT are distinctive class words.
        # Any direct occurrence is suspicious enough to fail.
        for label in (
            "CONDITIONAL",
            "INSUFFICIENT",
        ):
            if re.search(
                r"\b"
                + re.escape(label)
                + r"\b",
                case_text,
                flags=re.IGNORECASE,
            ):
                direct_class_leaks.append(
                    (
                        row["case_id"],
                        label,
                    )
                )

        # YES / NO are common English vocabulary.
        #
        # Only literal all-uppercase tokens in the original text are
        # treated as direct class-token leakage.
        for label in (
            "YES",
            "NO",
        ):
            if re.search(
                r"\b"
                + re.escape(label)
                + r"\b",
                case_text,
            ):
                direct_class_leaks.append(
                    (
                        row["case_id"],
                        label,
                    )
                )

        # Explicit answer-like structures are forbidden regardless
        # of capitalization.
        for match in marker_pattern.finditer(
            case_text
        ):
            explicit_marker_leaks.append(
                (
                    row["case_id"],
                    match.group(1).upper(),
                    match.group(0),
                )
            )

    if direct_class_leaks:
        fail(
            "direct output-class tokens found in "
            "case-specific text: "
            f"{direct_class_leaks[:10]}"
        )

    if explicit_marker_leaks:
        fail(
            "explicit output-label markers found in "
            "case-specific text: "
            f"{explicit_marker_leaks[:10]}"
        )

    # ============================================================
    # TF-IDF SHORTCUT PROBES
    #
    # Grouped by relational system so members of the same system
    # never cross train/test folds.
    # ============================================================

    labels_seq = [
        row["gold"]
        for row in public
    ]

    groups = [
        hidden_by_id[
            row["case_id"]
        ]["relational_system"]
        for row in public
    ]

    question_texts = [
        row["question"]
        for row in public
    ]

    evidence_texts = [
        "\n".join(
            d["text"]
            for d in row["evidence"]
        )
        for row in public
    ]

    q_probe = tfidf_probe(
        question_texts,
        labels_seq,
        groups,
    )

    e_probe = tfidf_probe(
        evidence_texts,
        labels_seq,
        groups,
    )

    # A probe above 0.35 is too suspicious for promotion.
    # 0.30–0.35 is retained as a diagnostic warning.
    warnings = []

    for name, probe in [
        ("question_only", q_probe),
        ("evidence_only", e_probe),
    ]:
        if (
            probe.get("status") == "OK"
            and probe["mean_accuracy"] > 0.35
        ):
            fail(
                f"{name} TF-IDF shortcut "
                f"{probe['mean_accuracy']:.4f} "
                f"> 0.35"
            )

        if (
            probe.get("status") == "OK"
            and probe["mean_accuracy"] > 0.30
        ):
            warnings.append(
                f"{name} mean accuracy "
                f"{probe['mean_accuracy']:.4f} "
                f"is above 0.30 and requires review"
            )

    # ============================================================
    # REPORT
    # ============================================================

    report = {
        "benchmark": "Sentinel Challenge V4",
        "candidate_status": (
            "STRUCTURAL_SHORTCUT_AUDIT_PASS"
        ),
        "public_sha256": sha(PUBLIC),
        "hidden_sha256": sha(HIDDEN),
        "case_count": len(public),
        "relational_system_count": len(systems),
        "labels": dict(sorted(labels.items())),
        "question_types": dict(
            sorted(question_types.items())
        ),
        "composition_count": len(compositions),
        "composition_counts": dict(
            sorted(compositions.items())
        ),
        "input_order": dict(
            sorted(input_order.items())
        ),
        "duplicate_questions": duplicate_questions,
        "duplicate_question_evidence_pairs": (
            duplicate_pairs
        ),
        "structural_shortcuts": {
            "world_index_majority_accuracy": (
                world_acc
            ),
            "target_index_majority_accuracy": (
                target_acc
            ),
            "target_world_pair_majority_accuracy": (
                pair_acc
            ),
            "transversal_index_majority_accuracy": (
                transversal_acc
            ),
            "coordinate_cells": len(
                pair_buckets
            ),
        },
        "tfidf_shortcuts": {
            "question_only": q_probe,
            "evidence_only": e_probe,
        },
        "warnings": warnings,
        "model_loads": 0,
        "inference_calls": 0,
        "individual_cases_printed": 0,
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
        "SENTINEL CHALLENGE V4 — CANDIDATE STRUCTURAL / SHORTCUT AUDIT",
        "=" * 76,
        "",
        "STATUS: PASS",
        "",
        f"Cases: {len(public)}",
        f"Relational systems: {len(systems)}",
        f"Compositions: {len(compositions)}",
        "",
        f"Labels: {dict(sorted(labels.items()))}",
        f"Question types: {dict(sorted(question_types.items()))}",
        f"Input order: {dict(sorted(input_order.items()))}",
        "",
        "STRUCTURAL SHORTCUT MAJORITY ACCURACY",
        "-" * 76,
        f"world_index:       {world_acc:.4f}",
        f"target_index:      {target_acc:.4f}",
        f"target/world pair: {pair_acc:.4f}",
        f"transversal_index:     {transversal_acc:.4f}",
        f"coordinate cells:  {len(pair_buckets)}",
        "",
        "DUPLICATES",
        "-" * 76,
        f"duplicate questions: {duplicate_questions}",
        f"duplicate question/evidence pairs: {duplicate_pairs}",
        "",
        "TF-IDF SHORTCUT PROBES",
        "-" * 76,
    ]

    if q_probe.get("status") == "OK":
        lines.append(
            "question-only mean: "
            f"{q_probe['mean_accuracy']:.4f}"
        )
        lines.append(
            "question-only folds: "
            f"{q_probe['fold_scores']}"
        )
    else:
        lines.append(
            "question-only: "
            f"{q_probe}"
        )

    if e_probe.get("status") == "OK":
        lines.append(
            "evidence-only mean: "
            f"{e_probe['mean_accuracy']:.4f}"
        )
        lines.append(
            "evidence-only folds: "
            f"{e_probe['fold_scores']}"
        )
    else:
        lines.append(
            "evidence-only: "
            f"{e_probe}"
        )

    lines.extend([
        "",
        "WARNINGS",
        "-" * 76,
    ])

    if warnings:
        lines.extend(warnings)
    else:
        lines.append("None.")

    lines.extend([
        "",
        f"Public SHA256: {sha(PUBLIC)}",
        f"Hidden SHA256: {sha(HIDDEN)}",
        "",
        "Model loads: 0",
        "Inference: 0",
        "Individual cases printed: 0",
        "",
        "=" * 76,
        "CANDIDATE AUDIT COMPLETE",
    ])

    OUT_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "============================================================"
    )
    print(
        "CHALLENGE V4 CANDIDATE STRUCTURAL / SHORTCUT AUDIT"
    )
    print(
        "============================================================"
    )
    print("PASS")
    print()
    print("Cases:", len(public))
    print("Relational systems:", len(systems))
    print("Compositions:", len(compositions))
    print("Labels:", dict(sorted(labels.items())))
    print(
        "Question types:",
        dict(sorted(question_types.items())),
    )
    print("Input order:", dict(sorted(input_order.items())))
    print()
    print("Structural shortcut accuracies:")
    print(f"  world_index:       {world_acc:.4f}")
    print(f"  target_index:      {target_acc:.4f}")
    print(f"  target/world pair: {pair_acc:.4f}")
    print(f"  transversal_index:     {transversal_acc:.4f}")
    print(f"  coordinate cells:  {len(pair_buckets)}")
    print()
    print("Duplicates:")
    print(
        "  question/evidence pairs:",
        duplicate_pairs,
    )
    print()
    print("TF-IDF shortcut probes:")

    if q_probe.get("status") == "OK":
        print(
            "  question-only:",
            f"{q_probe['mean_accuracy']:.4f}",
        )
    else:
        print(
            "  question-only:",
            q_probe["status"],
        )

    if e_probe.get("status") == "OK":
        print(
            "  evidence-only:",
            f"{e_probe['mean_accuracy']:.4f}",
        )
    else:
        print(
            "  evidence-only:",
            e_probe["status"],
        )

    print()
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(" ", warning)
    else:
        print("Warnings: none")

    print()
    print("Audit JSON SHA256:")
    print(" ", sha(OUT_JSON))
    print()
    print("Audit TXT SHA256:")
    print(" ", sha(OUT_TXT))
    print()
    print("Model loads: 0")
    print("Inference: 0")
    print("Individual cases printed: 0")
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()
