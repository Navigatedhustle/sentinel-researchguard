from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import hashlib
import json
import re


BASE = Path("/workspace")

PUBLIC = (
    BASE
    / "experiments/challenge_v4/candidate/"
      "sentinel_challenge_v4_candidate.json"
)

HIDDEN = (
    BASE
    / "experiments/challenge_v4/candidate/audit_only/"
      "sentinel_challenge_v4_hidden_semantics_candidate.jsonl"
)

OUT_JSON = (
    BASE
    / "experiments/challenge_v4/candidate/audits/"
      "CHALLENGE_V4_CANDIDATE_LEAKAGE_SEMANTIC_CAUSAL_AUDIT_V1.json"
)

OUT_TXT = (
    BASE
    / "experiments/challenge_v4/candidate/audits/"
      "CHALLENGE_V4_CANDIDATE_LEAKAGE_SEMANTIC_CAUSAL_AUDIT_V1.txt"
)


LABELS = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
}


# ====================================================================
# GENERAL UTILITIES
# ====================================================================

def sha(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def norm(text: str) -> str:
    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
    )


def token_set(text: str):
    return set(norm(text).split())


def jaccard(a, b):
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


def shingles(text: str, n=5):
    toks = norm(text).split()

    if len(toks) < n:
        return {tuple(toks)} if toks else set()

    return {
        tuple(toks[i:i+n])
        for i in range(
            len(toks) - n + 1
        )
    }


def fail(msg):
    raise AssertionError(msg)


def string_values(obj):
    if isinstance(obj, str):
        yield obj

    elif isinstance(obj, list):
        for item in obj:
            yield from string_values(item)

    elif isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in {
                "gold",
                "label",
                "decision",
                "answer",
                "expected",
                "expected_answer",
                "rationale",
                "conditions",
            }:
                continue

            yield from string_values(value)


# ====================================================================
# PRIOR-DATA EXTRACTION
# ====================================================================

def extract_user_message(row):
    messages = row.get("messages")

    if not isinstance(messages, list):
        return ""

    users = [
        m.get("content", "")
        for m in messages
        if (
            isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(
                m.get("content"),
                str,
            )
        )
    ]

    return "\n".join(users)


def extract_question_from_user(text):
    if not text:
        return ""

    patterns = [
        r"QUESTION\s*\n=+\s*\n(.+?)(?:\n\nEVIDENCE|\Z)",
        r"QUESTION\s*:\s*(.+?)(?:\n\n|\Z)",
    ]

    for pattern in patterns:
        m = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if m:
            return m.group(1).strip()

    return ""


def extract_evidence(row):
    evidence = row.get("evidence")

    if evidence is None:
        return ""

    texts = []

    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                text = item.get("text")

                if isinstance(text, str):
                    texts.append(text)
                else:
                    texts.extend(
                        string_values(item)
                    )

            elif isinstance(item, str):
                texts.append(item)

    elif isinstance(evidence, dict):
        texts.extend(
            string_values(evidence)
        )

    elif isinstance(evidence, str):
        texts.append(evidence)

    return "\n".join(texts)


def row_to_record(row, source, index):
    if not isinstance(row, dict):
        return None

    user_text = extract_user_message(row)

    question = row.get("question", "")

    if not isinstance(question, str):
        question = ""

    if not question:
        question = extract_question_from_user(
            user_text
        )

    evidence = extract_evidence(row)

    if not evidence and user_text:
        evidence = user_text

    if not question and not evidence:
        return None

    rid = (
        row.get("case_id")
        or row.get("id")
        or row.get("scenario_id")
        or f"row_{index}"
    )

    return {
        "source": str(
            source.relative_to(BASE)
        ),
        "record_id": str(rid),
        "question": question,
        "evidence": evidence,
        "full": (
            question
            + "\n"
            + evidence
        ).strip(),
    }


def load_records(path):
    records = []

    try:
        if path.suffix == ".jsonl":
            rows = []

            for line in path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines():
                if not line.strip():
                    continue

                try:
                    rows.append(
                        json.loads(line)
                    )
                except Exception:
                    continue

        else:
            obj = json.loads(
                path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )

            if isinstance(obj, list):
                rows = obj

            elif isinstance(obj, dict):
                rows = None

                for key in (
                    "cases",
                    "examples",
                    "rows",
                    "data",
                    "items",
                ):
                    candidate = obj.get(key)

                    if isinstance(
                        candidate,
                        list,
                    ):
                        rows = candidate
                        break

                if rows is None:
                    rows = [obj]

            else:
                rows = []

    except Exception:
        return []

    for index, row in enumerate(rows):
        record = row_to_record(
            row,
            path,
            index,
        )

        if record:
            records.append(record)

    return records


def discover_prior_sources():
    files = set()

    roots = [
        BASE / "experiments/challenge_v1",
        BASE / "experiments/challenge_v2",
        BASE / "experiments/challenge_v3/canonical",
        BASE / "training/v6/pilot",
    ]

    for root in roots:
        if not root.exists():
            continue

        if root.is_file():
            files.add(root)
            continue

        for suffix in (
            "*.json",
            "*.jsonl",
        ):
            for path in root.rglob(suffix):
                ptext = str(path)

                if "/audit_only/" in ptext:
                    continue

                if "/evaluation/" in ptext:
                    continue

                if "/archive/" in ptext:
                    continue

                files.add(path)

    return sorted(files)


# ====================================================================
# QUESTION PARSER
# ====================================================================

QUESTION_RE = re.compile(
    r"^Under the supplied evidence, may "
    r"(?P<actor>.+?) "
    r"(?P<action>[a-z]+) "
    r"the "
    r"(?P<asset>.+?) "
    r"through "
    r"(?P<route>.+?) "
    r"for "
    r"(?P<purpose>.+?)\?$",
    flags=re.IGNORECASE,
)


def parse_question(question):
    m = QUESTION_RE.match(
        question.strip()
    )

    if not m:
        raise ValueError(
            "question did not match "
            "Challenge V4 grammar"
        )

    return {
        k: v.strip()
        for k, v in m.groupdict().items()
    }


def target_evidence(case, question):
    q = parse_question(question)

    # Reconstruct the PUBLIC target-track namespace from the
    # question itself.
    #
    # This mirrors only the public source naming convention.
    # It does not use hidden target_index, world_index, gold,
    # semantic_state, or gold_annotation.
    target_key = re.sub(
        r"[^a-z0-9]+",
        "_",
        (
            f"{q['actor']}_"
            f"{q['asset']}_"
            f"{q['route']}"
        ).lower(),
    ).strip("_")[:48]

    selected = [
        doc["text"]
        for doc in case["evidence"]
        if doc["source_id"].startswith(
            target_key + "_"
        )
    ]

    if not selected:
        raise AssertionError(
            "no exact public target-track "
            "evidence found"
        )

    return q, "\n".join(selected).lower()


# ====================================================================
# INDEPENDENT BENCHMARK-LOCAL ADJUDICATOR
#
# IMPORTANT:
#
# This code does NOT import the generator.
# It does NOT read gold labels during adjudication.
# It uses:
#
#   question
#   public evidence
#   hidden oracle_mode only
#
# and independently derives a decision.
# ====================================================================

def adjudicate(
    mode,
    case,
    question=None,
):
    question = (
        question
        if question is not None
        else case["question"]
    )

    q, text = target_evidence(
        case,
        question,
    )

    action = q["action"].lower()


    # ------------------------------------------------------------
    # PERMISSION + CONDITION
    # ------------------------------------------------------------

    if mode == "permission_condition":
        if (
            f"must not {action} " in text
        ):
            return "NO"

        if (
            f"may {action} " in text
        ):
            if (
                "does not establish whether"
                in text
            ):
                return "CONDITIONAL"

            if (
                "is established"
                in text
            ):
                return "YES"

        return "INSUFFICIENT"


    # ------------------------------------------------------------
    # SCOPE / CLASS MEMBERSHIP
    # ------------------------------------------------------------

    if mode == "scope_membership":
        if (
            f"must not {action} records"
            in text
        ):
            return "NO"

        if (
            f"may {action} records"
            in text
        ):
            if (
                "does not establish whether"
                in text
            ):
                return "CONDITIONAL"

            if (
                "is classified as a member"
                in text
            ):
                return "YES"

        return "INSUFFICIENT"


    # ------------------------------------------------------------
    # CURRENT RULE / PRECEDENCE
    # ------------------------------------------------------------

    if mode == "precedence_current_rule":
        if (
            "the current rule replaces the former rule "
            "and prohibits"
            in text
        ):
            return "NO"

        if (
            "the current rule replaces the former rule "
            "and permits"
            in text
        ):
            if (
                "does not establish whether"
                in text
            ):
                return "CONDITIONAL"

            if (
                "establishes"
                in text
            ):
                return "YES"

        return "INSUFFICIENT"


    # ------------------------------------------------------------
    # EXCEPTION ACTIVATION
    # ------------------------------------------------------------

    if mode == "exception_activation":
        if (
            "explicit exception overrides"
            in text
        ):
            if (
                "does not establish whether"
                in text
            ):
                return "CONDITIONAL"

            if (
                "is established"
                in text
            ):
                return "YES"

        if (
            "governing rule prohibits"
            in text
            and
            "does not override"
            in text
        ):
            return "NO"

        return "INSUFFICIENT"


    # ------------------------------------------------------------
    # DELEGATED AUTHORITY
    # ------------------------------------------------------------

    if mode == "delegated_authority":
        if (
            "denied the request"
            in text
            and
            "must not proceed"
            in text
        ):
            return "NO"

        if (
            "approved this exact request"
            in text
        ):
            return "YES"

        if (
            "no completed decision"
            in text
        ):
            return "CONDITIONAL"

        return "INSUFFICIENT"


    # ------------------------------------------------------------
    # DEFINITION BINDING
    # ------------------------------------------------------------

    if mode == "definition_binding":
        action_ed = action + "ed"

        if (
            "must not be "
            + action_ed
            in text
        ):
            return "NO"

        if (
            "may be "
            + action_ed
            in text
        ):
            if (
                "does not establish whether"
                in text
            ):
                return "CONDITIONAL"

            if (
                "satisfies the definition"
                in text
            ):
                return "YES"

        return "INSUFFICIENT"


    # ------------------------------------------------------------
    # TEMPORAL VALIDITY
    # ------------------------------------------------------------

    if mode == "temporal_validity":
        if (
            f"must not {action} "
            in text
            and
            "has expired"
            in text
        ):
            return "NO"

        if (
            f"may {action} "
            in text
        ):
            if (
                "does not establish whether"
                in text
            ):
                return "CONDITIONAL"

            if (
                "is active"
                in text
            ):
                return "YES"

        return "INSUFFICIENT"


    # ------------------------------------------------------------
    # WORKFLOW DEPENDENCY
    # ------------------------------------------------------------

    if mode == "workflow_dependency":
        if (
            "blocking finding"
            in text
            and
            f"must not {action} "
            in text
        ):
            return "NO"

        if (
            f"may {action} "
            in text
        ):
            if (
                "does not establish completion "
                "of release review"
                in text
            ):
                return "CONDITIONAL"

            if (
                "intake review"
                in text
                and
                "release review is complete"
                in text
            ):
                return "YES"

        return "INSUFFICIENT"

    raise ValueError(
        f"unknown oracle mode: {mode}"
    )


# ====================================================================
# MAIN
# ====================================================================

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

    if len(public) != 320:
        fail(
            f"public count != 320: "
            f"{len(public)}"
        )

    if len(hidden) != 320:
        fail(
            f"hidden count != 320: "
            f"{len(hidden)}"
        )

    public_by_id = {
        row["case_id"]: row
        for row in public
    }

    hidden_by_id = {
        row["case_id"]: row
        for row in hidden
    }

    if set(public_by_id) != set(hidden_by_id):
        fail(
            "public/hidden case identity mismatch"
        )


    # ============================================================
    # 1. CROSS-BENCHMARK / V6-PILOT LEAKAGE
    # ============================================================

    prior_sources = discover_prior_sources()

    prior_records = []

    for path in prior_sources:
        prior_records.extend(
            load_records(path)
        )

    if not prior_records:
        fail(
            "no prior benchmark/pilot records "
            "were discovered"
        )

    candidate_records = []

    for row in public:
        candidate_records.append({
            "record_id": row["case_id"],
            "question": row["question"],
            "evidence": "\n".join(
                d["text"]
                for d in row["evidence"]
            ),
            "full": (
                row["question"]
                + "\n"
                + "\n".join(
                    d["text"]
                    for d in row["evidence"]
                )
            ),
        })


    # ------------------------------------------------------------
    # Exact question overlap
    # ------------------------------------------------------------

    prior_question_index = defaultdict(
        list
    )

    for record in prior_records:
        nq = norm(
            record["question"]
        )

        if nq:
            prior_question_index[nq].append(
                (
                    record["source"],
                    record["record_id"],
                )
            )

    exact_question_overlaps = []

    for candidate in candidate_records:
        nq = norm(
            candidate["question"]
        )

        for source, rid in (
            prior_question_index.get(
                nq,
                [],
            )
        ):
            exact_question_overlaps.append({
                "candidate": (
                    candidate["record_id"]
                ),
                "prior_source": source,
                "prior_record": rid,
            })

    if exact_question_overlaps:
        fail(
            "exact prior-question overlap detected: "
            f"{exact_question_overlaps[:5]}"
        )


    # ------------------------------------------------------------
    # Exact evidence-document overlap
    # ------------------------------------------------------------

    prior_string_index = set()

    for record in prior_records:
        evidence = record["evidence"]

        for piece in re.split(
            r"\n{2,}|\n(?=\[)",
            evidence,
        ):
            npiece = norm(piece)

            if len(npiece.split()) >= 8:
                prior_string_index.add(
                    npiece
                )

    exact_evidence_overlaps = []

    for row in public:
        for doc in row["evidence"]:
            ndoc = norm(
                doc["text"]
            )

            if (
                len(ndoc.split()) >= 8
                and
                ndoc in prior_string_index
            ):
                exact_evidence_overlaps.append({
                    "candidate": row["case_id"],
                    "source_id": doc["source_id"],
                })

    if exact_evidence_overlaps:
        fail(
            "exact prior evidence-document "
            "overlap detected: "
            f"{exact_evidence_overlaps[:5]}"
        )


    # ------------------------------------------------------------
    # Close question overlap
    # ------------------------------------------------------------

    prior_questions = [
        r
        for r in prior_records
        if norm(r["question"])
    ]

    near_question_hits = []

    max_question_similarity = 0.0
    max_question_pair = None

    for candidate in candidate_records:
        cq = norm(
            candidate["question"]
        )

        c_tokens = token_set(
            candidate["question"]
        )

        for prior in prior_questions:
            pq = norm(
                prior["question"]
            )

            ratio = SequenceMatcher(
                None,
                cq,
                pq,
            ).ratio()

            tj = jaccard(
                c_tokens,
                token_set(
                    prior["question"]
                ),
            )

            score = max(
                ratio,
                tj,
            )

            if (
                score
                > max_question_similarity
            ):
                max_question_similarity = score
                max_question_pair = {
                    "candidate": (
                        candidate["record_id"]
                    ),
                    "prior_source": (
                        prior["source"]
                    ),
                    "prior_record": (
                        prior["record_id"]
                    ),
                    "sequence_ratio": ratio,
                    "token_jaccard": tj,
                }

            if (
                ratio >= 0.93
                and
                tj >= 0.72
            ):
                near_question_hits.append({
                    "candidate": (
                        candidate["record_id"]
                    ),
                    "prior_source": (
                        prior["source"]
                    ),
                    "prior_record": (
                        prior["record_id"]
                    ),
                    "sequence_ratio": ratio,
                    "token_jaccard": tj,
                })

    if near_question_hits:
        fail(
            "close prior-question overlap "
            "detected: "
            f"{near_question_hits[:5]}"
        )


    # ------------------------------------------------------------
    # Full-case 5-gram overlap
    #
    # This is intentionally stricter than shared policy vocabulary.
    # ------------------------------------------------------------

    prior_full_shingles = []

    for prior in prior_records:
        s = shingles(
            prior["full"],
            n=5,
        )

        if s:
            prior_full_shingles.append(
                (
                    prior,
                    s,
                )
            )

    high_case_overlap = []

    max_case_jaccard = 0.0
    max_case_pair = None

    for candidate in candidate_records:
        cs = shingles(
            candidate["full"],
            n=5,
        )

        for prior, ps in prior_full_shingles:
            score = jaccard(
                cs,
                ps,
            )

            if score > max_case_jaccard:
                max_case_jaccard = score

                max_case_pair = {
                    "candidate": (
                        candidate["record_id"]
                    ),
                    "prior_source": (
                        prior["source"]
                    ),
                    "prior_record": (
                        prior["record_id"]
                    ),
                    "fivegram_jaccard": score,
                }

            if score >= 0.55:
                high_case_overlap.append({
                    "candidate": (
                        candidate["record_id"]
                    ),
                    "prior_source": (
                        prior["source"]
                    ),
                    "prior_record": (
                        prior["record_id"]
                    ),
                    "fivegram_jaccard": score,
                })

    if high_case_overlap:
        fail(
            "high full-case prior overlap "
            "detected: "
            f"{high_case_overlap[:5]}"
        )


    # ============================================================
    # 2. INDEPENDENT PUBLIC-CASE SEMANTIC ADJUDICATION
    # ============================================================

    oracle_confusion = defaultdict(
        Counter
    )

    oracle_errors = []

    for row in public:
        hidden_row = hidden_by_id[
            row["case_id"]
        ]

        predicted = adjudicate(
            hidden_row["oracle_mode"],
            row,
        )

        gold = row["gold"]

        oracle_confusion[
            gold
        ][predicted] += 1

        if predicted != gold:
            oracle_errors.append({
                "case_id": row["case_id"],
                "gold": gold,
                "derived": predicted,
                "oracle_mode": (
                    hidden_row[
                        "oracle_mode"
                    ]
                ),
            })

    if oracle_errors:
        fail(
            "independent semantic adjudicator "
            "disagreed with gold: "
            f"{oracle_errors[:10]}"
        )


    # ============================================================
    # 3. COUNTERFACTUAL / CAUSAL TRACK AUDIT
    #
    # For every relational system:
    #
    # - recover each target's question
    # - hold that target/question fixed
    # - apply it to all four world evidence packs
    #
    # The deterministic oracle must recover:
    #
    # YES
    # NO
    # CONDITIONAL
    # INSUFFICIENT
    #
    # exactly once for EACH target.
    #
    # 80 systems
    # x 4 targets
    # x 4 worlds
    # = 1,280 derived counterfactual adjudications
    # ============================================================

    systems = defaultdict(
        list
    )

    for h in hidden:
        systems[
            h["relational_system"]
        ].append(h)

    if len(systems) != 80:
        fail(
            f"expected 80 systems, "
            f"found {len(systems)}"
        )

    causal_track_count = 0
    derived_adjudications = 0
    derived_label_counts = Counter()
    causal_errors = []

    for system_id, hrows in systems.items():
        if len(hrows) != 4:
            fail(
                f"{system_id}: "
                f"expected 4 worlds"
            )

        mode_set = {
            h["oracle_mode"]
            for h in hrows
        }

        if len(mode_set) != 1:
            fail(
                f"{system_id}: mixed oracle modes"
            )

        mode = next(iter(mode_set))

        world_cases = {}

        target_questions = {}

        for h in hrows:
            case = public_by_id[
                h["case_id"]
            ]

            world_cases[
                h["world_index"]
            ] = case

            target_questions[
                h["target_index"]
            ] = case["question"]

        if set(world_cases) != {
            0, 1, 2, 3
        }:
            fail(
                f"{system_id}: missing world"
            )

        if set(target_questions) != {
            0, 1, 2, 3
        }:
            fail(
                f"{system_id}: missing target"
            )

        for target_index in range(4):
            causal_track_count += 1

            question = target_questions[
                target_index
            ]

            predictions = []

            target_world_hashes = []

            for world_index in range(4):
                case = world_cases[
                    world_index
                ]

                predicted = adjudicate(
                    mode,
                    case,
                    question=question,
                )

                predictions.append(
                    predicted
                )

                derived_label_counts[
                    predicted
                ] += 1

                derived_adjudications += 1

                # Verify the target-relevant evidence
                # genuinely changes across worlds.
                _, target_text = (
                    target_evidence(
                        case,
                        question,
                    )
                )

                target_world_hashes.append(
                    hashlib.sha256(
                        target_text.encode(
                            "utf-8"
                        )
                    ).hexdigest()
                )

            counts = Counter(
                predictions
            )

            expected = Counter({
                "YES": 1,
                "NO": 1,
                "CONDITIONAL": 1,
                "INSUFFICIENT": 1,
            })

            if counts != expected:
                causal_errors.append({
                    "system": system_id,
                    "target_index": (
                        target_index
                    ),
                    "prediction_counts": (
                        dict(counts)
                    ),
                })

            if len(
                set(target_world_hashes)
            ) != 4:
                causal_errors.append({
                    "system": system_id,
                    "target_index": (
                        target_index
                    ),
                    "error": (
                        "target evidence did "
                        "not change uniquely "
                        "across all four worlds"
                    ),
                })

    if causal_errors:
        fail(
            "counterfactual causal audit "
            "failed: "
            f"{causal_errors[:10]}"
        )

    if causal_track_count != 320:
        fail(
            f"causal tracks "
            f"{causal_track_count} != 320"
        )

    if derived_adjudications != 1280:
        fail(
            "derived adjudications "
            f"{derived_adjudications} "
            "!= 1280"
        )

    expected_derived_labels = Counter({
        "YES": 320,
        "NO": 320,
        "CONDITIONAL": 320,
        "INSUFFICIENT": 320,
    })

    if (
        derived_label_counts
        != expected_derived_labels
    ):
        fail(
            "derived counterfactual label "
            "imbalance: "
            f"{derived_label_counts}"
        )


    # ============================================================
    # REPORT
    # ============================================================

    report = {
        "benchmark": (
            "Sentinel Challenge V4"
        ),
        "candidate_status": (
            "LEAKAGE_SEMANTIC_CAUSAL_PASS"
        ),
        "public_sha256": sha(PUBLIC),
        "hidden_sha256": sha(HIDDEN),

        "prior_data": {
            "source_file_count": (
                len(prior_sources)
            ),
            "record_count": (
                len(prior_records)
            ),
            "exact_question_overlaps": (
                len(
                    exact_question_overlaps
                )
            ),
            "exact_evidence_overlaps": (
                len(
                    exact_evidence_overlaps
                )
            ),
            "near_question_hits": (
                len(near_question_hits)
            ),
            "high_full_case_overlap_hits": (
                len(high_case_overlap)
            ),
            "max_question_similarity": (
                max_question_similarity
            ),
            "max_question_pair": (
                max_question_pair
            ),
            "max_full_case_fivegram_jaccard": (
                max_case_jaccard
            ),
            "max_full_case_pair": (
                max_case_pair
            ),
        },

        "semantic_audit": {
            "public_cases_adjudicated": (
                len(public)
            ),
            "oracle_errors": (
                len(oracle_errors)
            ),
            "oracle_confusion": {
                gold: dict(
                    sorted(
                        preds.items()
                    )
                )
                for gold, preds
                in sorted(
                    oracle_confusion.items()
                )
            },
        },

        "causal_audit": {
            "relational_systems": (
                len(systems)
            ),
            "target_tracks": (
                causal_track_count
            ),
            "derived_adjudications": (
                derived_adjudications
            ),
            "derived_label_counts": (
                dict(
                    sorted(
                        derived_label_counts.items()
                    )
                )
            ),
            "causal_errors": (
                len(causal_errors)
            ),
        },

        "sentinel_model_loads": 0,
        "sentinel_inference_calls": 0,
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
        (
            "SENTINEL CHALLENGE V4 — "
            "LEAKAGE + SEMANTIC/CAUSAL AUDIT"
        ),
        "=" * 78,
        "",
        "STATUS: PASS",
        "",
        "CROSS-BENCHMARK / V6 PILOT LEAKAGE",
        "-" * 78,
        (
            "Prior source files: "
            f"{len(prior_sources)}"
        ),
        (
            "Prior records: "
            f"{len(prior_records)}"
        ),
        (
            "Exact question overlaps: "
            f"{len(exact_question_overlaps)}"
        ),
        (
            "Exact evidence overlaps: "
            f"{len(exact_evidence_overlaps)}"
        ),
        (
            "Near question hits: "
            f"{len(near_question_hits)}"
        ),
        (
            "High full-case overlap hits: "
            f"{len(high_case_overlap)}"
        ),
        (
            "Maximum question similarity: "
            f"{max_question_similarity:.4f}"
        ),
        (
            "Maximum full-case 5-gram Jaccard: "
            f"{max_case_jaccard:.4f}"
        ),
        "",
        "INDEPENDENT SEMANTIC AUDIT",
        "-" * 78,
        (
            "Public cases adjudicated: "
            f"{len(public)}"
        ),
        (
            "Oracle disagreements: "
            f"{len(oracle_errors)}"
        ),
        "",
        "COUNTERFACTUAL CAUSAL AUDIT",
        "-" * 78,
        (
            "Relational systems: "
            f"{len(systems)}"
        ),
        (
            "Target causal tracks: "
            f"{causal_track_count}"
        ),
        (
            "Derived adjudications: "
            f"{derived_adjudications}"
        ),
        (
            "Derived labels: "
            f"{dict(sorted(derived_label_counts.items()))}"
        ),
        (
            "Causal failures: "
            f"{len(causal_errors)}"
        ),
        "",
        f"Public SHA256: {sha(PUBLIC)}",
        f"Hidden SHA256: {sha(HIDDEN)}",
        "",
        "Sentinel model loads: 0",
        "Sentinel inference: 0",
        "Individual cases printed: 0",
        "",
        "=" * 78,
        "AUDIT COMPLETE",
    ]

    OUT_TXT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


    # ============================================================
    # AGGREGATE TERMINAL OUTPUT ONLY
    # ============================================================

    print()
    print(
        "============================================================"
    )
    print(
        "CHALLENGE V4 — LEAKAGE + SEMANTIC/CAUSAL AUDIT"
    )
    print(
        "============================================================"
    )
    print("PASS")

    print()
    print(
        "Cross-benchmark / V6 pilot leakage:"
    )
    print(
        "  prior source files:",
        len(prior_sources),
    )
    print(
        "  prior records:",
        len(prior_records),
    )
    print(
        "  exact question overlaps:",
        len(exact_question_overlaps),
    )
    print(
        "  exact evidence overlaps:",
        len(exact_evidence_overlaps),
    )
    print(
        "  near question hits:",
        len(near_question_hits),
    )
    print(
        "  high full-case overlaps:",
        len(high_case_overlap),
    )
    print(
        "  max question similarity:",
        f"{max_question_similarity:.4f}",
    )
    print(
        "  max full-case 5gram Jaccard:",
        f"{max_case_jaccard:.4f}",
    )

    print()
    print(
        "Independent semantic audit:"
    )
    print(
        "  public cases adjudicated:",
        len(public),
    )
    print(
        "  oracle disagreements:",
        len(oracle_errors),
    )

    print()
    print(
        "Counterfactual causal audit:"
    )
    print(
        "  relational systems:",
        len(systems),
    )
    print(
        "  target causal tracks:",
        causal_track_count,
    )
    print(
        "  derived adjudications:",
        derived_adjudications,
    )
    print(
        "  derived labels:",
        dict(
            sorted(
                derived_label_counts.items()
            )
        ),
    )
    print(
        "  causal failures:",
        len(causal_errors),
    )

    print()
    print(
        "Audit JSON SHA256:"
    )
    print(
        " ",
        sha(OUT_JSON),
    )

    print()
    print(
        "Audit TXT SHA256:"
    )
    print(
        " ",
        sha(OUT_TXT),
    )

    print()
    print(
        "Sentinel model loads: 0"
    )
    print(
        "Sentinel inference: 0"
    )
    print(
        "Individual cases printed: 0"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()
