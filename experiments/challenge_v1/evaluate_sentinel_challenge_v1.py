import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE = Path(__file__).resolve().parents[2]

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

V2_PATH = (
    BASE /
    "training/output/llama-sentinel-8b-lora-v2"
)

V3_PATH = (
    BASE /
    "training/output/llama-sentinel-8b-lora-v3"
)

CHALLENGE_PATH = (
    Path(__file__).resolve().parent /
    "sentinel_challenge_v1.json"
)

OUTPUT_PATH = (
    BASE /
    "results/sentinel_challenge_v1_base_v2_v3.json"
)

EXPECTED_CHALLENGE_SHA256 = (
    "02b49320e3b46b87af8841344044d1d51"
    "bfc0eb4af2d285e09cce6a43811894d"
)

MAX_NEW_TOKENS = 140

DECISIONS = [
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
]

SCHEMA_SUFFIX = """

OUTPUT CONTRACT:

Return valid JSON only in this structure:

{
  "evidence_sufficient": true or false,
  "decision": "YES" or "NO" or "CONDITIONAL" or "INSUFFICIENT",
  "conditions": [],
  "answer": "brief evidence-grounded explanation",
  "citations": ["S1"]
}

Rules:
- If decision is INSUFFICIENT, set evidence_sufficient=false.
- Otherwise set evidence_sufficient=true.
- For CONDITIONAL, conditions should list unresolved material requirements.
- For all other decisions, conditions should be [].
- citations must contain only supporting source labels such as S1 or S2.
- For INSUFFICIENT, citations should be [].
- Do not wrap the JSON in Markdown.
""".strip()


def sha256_file(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def verify_challenge():

    actual = sha256_file(
        CHALLENGE_PATH
    )

    if actual != EXPECTED_CHALLENGE_SHA256:

        raise RuntimeError(
            "CHALLENGE HASH MISMATCH\n"
            f"Expected: {EXPECTED_CHALLENGE_SHA256}\n"
            f"Actual:   {actual}\n"
            "Refusing to evaluate modified benchmark."
        )

    print(
        "Challenge SHA-256 verified:",
        actual
    )


def load_cases():

    return json.loads(
        CHALLENGE_PATH.read_text()
    )


def build_messages(case):

    system_prompt = (
        case["instruction"]
        + "\n\n"
        + SCHEMA_SUFFIX
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": case["input"],
        },
    ]


def extract_json(text):

    cleaned = text.strip()

    try:

        obj = json.loads(
            cleaned
        )

        if isinstance(
            obj,
            dict,
        ):
            return obj, True

    except Exception:
        pass

    fenced = cleaned

    if fenced.startswith("```"):

        fenced = re.sub(
            r"^```(?:json)?\s*",
            "",
            fenced,
            flags=re.IGNORECASE,
        )

        fenced = re.sub(
            r"\s*```$",
            "",
            fenced,
        )

        try:

            obj = json.loads(
                fenced
            )

            if isinstance(
                obj,
                dict,
            ):
                return obj, False

        except Exception:
            pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if (
        start != -1
        and end > start
    ):

        try:

            obj = json.loads(
                cleaned[
                    start:end + 1
                ]
            )

            if isinstance(
                obj,
                dict,
            ):
                return obj, False

        except Exception:
            pass

    return None, False


def decision_from(obj):

    if not isinstance(
        obj,
        dict,
    ):
        return "INVALID"

    d = obj.get(
        "decision"
    )

    if not isinstance(
        d,
        str,
    ):
        return "INVALID"

    d = d.strip().upper()

    if d not in DECISIONS:
        return "INVALID"

    return d


def schema_ok(obj):

    if not isinstance(
        obj,
        dict,
    ):
        return False

    required = {
        "evidence_sufficient",
        "decision",
        "conditions",
        "answer",
        "citations",
    }

    if not required.issubset(
        obj.keys()
    ):
        return False

    if not isinstance(
        obj["evidence_sufficient"],
        bool,
    ):
        return False

    if decision_from(obj) == "INVALID":
        return False

    if not isinstance(
        obj["conditions"],
        list,
    ):
        return False

    if not isinstance(
        obj["answer"],
        str,
    ):
        return False

    if not isinstance(
        obj["citations"],
        list,
    ):
        return False

    return True


def citations_from(obj):

    if not isinstance(
        obj,
        dict,
    ):
        return []

    citations = obj.get(
        "citations",
        [],
    )

    if not isinstance(
        citations,
        list,
    ):
        return []

    return [
        x
        for x in citations
        if isinstance(x, str)
    ]


def conditions_from(obj):

    if not isinstance(
        obj,
        dict,
    ):
        return []

    conditions = obj.get(
        "conditions",
        [],
    )

    if not isinstance(
        conditions,
        list,
    ):
        return []

    return conditions


def select_mode(
    model,
    mode,
):

    if mode == "base":
        return model.disable_adapter()

    model.set_adapter(
        mode
    )

    return nullcontext()


def generate(
    model,
    tokenizer,
    case,
    mode,
):

    messages = build_messages(
        case
    )

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    prompt_tokens = (
        encoded[
            "input_ids"
        ].shape[1]
    )

    context = select_mode(
        model,
        mode,
    )

    torch.cuda.synchronize()

    start = time.perf_counter()

    with context:

        with torch.inference_mode():

            output = model.generate(
                **encoded,
                max_new_tokens=
                    MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=
                    tokenizer.pad_token_id,
                eos_token_id=
                    tokenizer.eos_token_id,
                use_cache=True,
            )

    torch.cuda.synchronize()

    latency = (
        time.perf_counter()
        - start
    )

    generated = output[
        0,
        prompt_tokens:
    ]

    output_tokens = len(
        generated
    )

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return {
        "text": text,
        "latency": latency,
        "output_tokens":
            output_tokens,
        "tokens_per_second":
            (
                output_tokens / latency
                if latency > 0
                else 0.0
            ),
    }


def pack_metrics(rows):

    packs = defaultdict(list)

    for row in rows:

        packs[
            row["policy_pack"]
        ].append(
            row
        )

    all_correct = 0
    at_least_one = 0

    details = {}

    for pack, items in sorted(
        packs.items()
    ):

        correct = sum(
            x["correct"]
            for x in items
        )

        pack_all = (
            correct
            == len(items)
        )

        pack_some = (
            correct > 0
        )

        all_correct += int(
            pack_all
        )

        at_least_one += int(
            pack_some
        )

        details[
            pack
        ] = {
            "correct": correct,
            "total": len(items),
            "all_correct": pack_all,
        }

    n = len(packs)

    return {
        "pack_count": n,
        "all_correct_rate":
            all_correct / n,
        "at_least_one_correct_rate":
            at_least_one / n,
        "details": details,
    }


def evaluate(
    name,
    mode,
    model,
    tokenizer,
    cases,
):

    print(
        "\n"
        + "=" * 80
    )

    print(name)

    print(
        "=" * 80
    )

    rows = []

    label_total = Counter()
    label_correct = Counter()

    qtype_total = Counter()
    qtype_correct = Counter()

    evidence_total = Counter()
    evidence_correct = Counter()

    tag_total = Counter()
    tag_correct = Counter()

    confusion = Counter()

    strict_json_count = 0
    schema_count = 0
    sufficiency_correct = 0
    citation_exact = 0
    citation_hit = 0

    latencies = []
    output_tokens = []
    token_rates = []

    for i, case in enumerate(
        cases,
        start=1,
    ):

        gold = case["gold"]

        gold_decision = (
            gold["decision"]
        )

        generated = generate(
            model,
            tokenizer,
            case,
            mode,
        )

        obj, strict_json = (
            extract_json(
                generated["text"]
            )
        )

        pred = decision_from(
            obj
        )

        correct = (
            pred == gold_decision
        )

        label_total[
            gold_decision
        ] += 1

        label_correct[
            gold_decision
        ] += int(correct)

        qtype = case[
            "question_type"
        ]

        qtype_total[
            qtype
        ] += 1

        qtype_correct[
            qtype
        ] += int(correct)

        evidence_count = case[
            "evidence_count"
        ]

        evidence_total[
            evidence_count
        ] += 1

        evidence_correct[
            evidence_count
        ] += int(correct)

        for tag in case[
            "reasoning_tags"
        ]:

            tag_total[
                tag
            ] += 1

            tag_correct[
                tag
            ] += int(correct)

        confusion[
            (
                gold_decision,
                pred,
            )
        ] += 1

        strict_json_count += int(
            strict_json
        )

        schema_count += int(
            schema_ok(obj)
        )

        pred_sufficient = (
            obj.get(
                "evidence_sufficient"
            )
            if isinstance(
                obj,
                dict,
            )
            else None
        )

        sufficiency_correct += int(
            pred_sufficient
            ==
            gold[
                "evidence_sufficient"
            ]
        )

        pred_cites = set(
            citations_from(
                obj
            )
        )

        gold_cites = set(
            gold["citations"]
        )

        citation_exact += int(
            pred_cites
            == gold_cites
        )

        if gold_cites:

            citation_hit += int(
                bool(
                    pred_cites
                    & gold_cites
                )
            )

        else:

            citation_hit += int(
                len(
                    pred_cites
                ) == 0
            )

        latencies.append(
            generated[
                "latency"
            ]
        )

        output_tokens.append(
            generated[
                "output_tokens"
            ]
        )

        token_rates.append(
            generated[
                "tokens_per_second"
            ]
        )

        rows.append({
            "case_id":
                case["case_id"],

            "policy_pack":
                case["policy_pack"],

            "question_type":
                qtype,

            "reasoning_tags":
                case[
                    "reasoning_tags"
                ],

            "evidence_count":
                evidence_count,

            "gold_decision":
                gold_decision,

            "predicted_decision":
                pred,

            "correct":
                correct,

            "gold_sufficient":
                gold[
                    "evidence_sufficient"
                ],

            "predicted_sufficient":
                pred_sufficient,

            "gold_citations":
                sorted(
                    gold_cites
                ),

            "predicted_citations":
                sorted(
                    pred_cites
                ),

            "predicted_conditions":
                conditions_from(
                    obj
                ),

            "strict_json":
                strict_json,

            "schema_ok":
                schema_ok(
                    obj
                ),

            "latency":
                generated[
                    "latency"
                ],

            "output_tokens":
                generated[
                    "output_tokens"
                ],

            "text":
                generated[
                    "text"
                ],
        })

        if (
            i % 10 == 0
            or i == len(cases)
        ):

            correct_now = sum(
                x["correct"]
                for x in rows
            )

            print(
                f"{i:2d}/"
                f"{len(cases)} "
                f"accuracy="
                f"{correct_now / i:.3f}"
            )

    n = len(rows)

    total_correct = sum(
        x["correct"]
        for x in rows
    )

    by_label = {}

    for d in DECISIONS:

        total = label_total[d]
        correct = label_correct[d]

        by_label[d] = {
            "correct": correct,
            "total": total,
            "accuracy":
                correct / total,
        }

    by_question_type = {}

    for qtype in sorted(
        qtype_total
    ):

        total = qtype_total[
            qtype
        ]

        correct = qtype_correct[
            qtype
        ]

        by_question_type[
            qtype
        ] = {
            "correct": correct,
            "total": total,
            "accuracy":
                correct / total,
        }

    by_evidence_count = {}

    for count in sorted(
        evidence_total
    ):

        total = evidence_total[
            count
        ]

        correct = evidence_correct[
            count
        ]

        by_evidence_count[
            str(count)
        ] = {
            "correct": correct,
            "total": total,
            "accuracy":
                correct / total,
        }

    by_tag = {}

    for tag in sorted(
        tag_total
    ):

        total = tag_total[
            tag
        ]

        correct = tag_correct[
            tag
        ]

        by_tag[
            tag
        ] = {
            "correct": correct,
            "total": total,
            "accuracy":
                correct / total,
        }

    confusion_dict = {}

    for gold in DECISIONS:

        confusion_dict[
            gold
        ] = {}

        for pred in (
            DECISIONS
            + ["INVALID"]
        ):

            confusion_dict[
                gold
            ][
                pred
            ] = confusion[
                (
                    gold,
                    pred,
                )
            ]

    packs = pack_metrics(
        rows
    )

    result = {
        "name":
            name,

        "mode":
            mode,

        "cases":
            n,

        "decision_accuracy":
            total_correct / n,

        "correct_decisions":
            total_correct,

        "pack_all_correct_rate":
            packs[
                "all_correct_rate"
            ],

        "sufficiency_accuracy":
            sufficiency_correct / n,

        "strict_json_rate":
            strict_json_count / n,

        "schema_rate":
            schema_count / n,

        "citation_hit_rate":
            citation_hit / n,

        "citation_exact_rate":
            citation_exact / n,

        "avg_latency":
            sum(latencies)
            / len(latencies),

        "avg_output_tokens":
            sum(output_tokens)
            / len(output_tokens),

        "avg_tokens_per_second":
            sum(token_rates)
            / len(token_rates),

        "by_label":
            by_label,

        "by_question_type":
            by_question_type,

        "by_evidence_count":
            by_evidence_count,

        "by_reasoning_tag":
            by_tag,

        "confusion":
            confusion_dict,

        "pack_metrics":
            packs,

        "rows":
            rows,
    }

    print()
    print(
        "DECISION ACCURACY:       "
        f"{result['decision_accuracy']:.3f} "
        f"({total_correct}/{n})"
    )

    print(
        "PACK ALL-CORRECT RATE:   "
        f"{result['pack_all_correct_rate']:.3f}"
    )

    print(
        "SUFFICIENCY ACCURACY:    "
        f"{result['sufficiency_accuracy']:.3f}"
    )

    print(
        "STRICT JSON RATE:        "
        f"{result['strict_json_rate']:.3f}"
    )

    print(
        "SCHEMA RATE:             "
        f"{result['schema_rate']:.3f}"
    )

    print(
        "CITATION HIT RATE:       "
        f"{result['citation_hit_rate']:.3f}"
    )

    print(
        "CITATION EXACT RATE:     "
        f"{result['citation_exact_rate']:.3f}"
    )

    print(
        "AVG LATENCY:             "
        f"{result['avg_latency']:.2f}s"
    )

    print(
        "AVG OUTPUT TOKENS:       "
        f"{result['avg_output_tokens']:.1f}"
    )

    print()
    print("BY LABEL:")

    for d in DECISIONS:

        s = by_label[d]

        print(
            f"  {d:12s} "
            f"{s['accuracy']:.3f} "
            f"({s['correct']}/"
            f"{s['total']})"
        )

    print()
    print("BY QUESTION TYPE:")

    for qtype, s in (
        by_question_type.items()
    ):

        print(
            f"  {qtype:22s} "
            f"{s['accuracy']:.3f} "
            f"({s['correct']}/"
            f"{s['total']})"
        )

    print()
    print("BY EVIDENCE COUNT:")

    for count, s in (
        by_evidence_count.items()
    ):

        print(
            f"  {count} passages: "
            f"{s['accuracy']:.3f} "
            f"({s['correct']}/"
            f"{s['total']})"
        )

    print()
    print("CONFUSION MATRIX:")

    print(
        "GOLD \\ PRED       "
        "YES   NO   COND   INSUFF   INVALID"
    )

    for gold in DECISIONS:

        c = confusion_dict[
            gold
        ]

        print(
            f"{gold:12s}"
            f"{c['YES']:5d}"
            f"{c['NO']:5d}"
            f"{c['CONDITIONAL']:7d}"
            f"{c['INSUFFICIENT']:9d}"
            f"{c['INVALID']:10d}"
        )

    failures = [
        r
        for r in rows
        if not r["correct"]
    ]

    print()
    print(
        "FAILURES:",
        len(failures)
    )

    for row in failures[:20]:

        print(
            f"  {row['case_id']} "
            f"[{row['policy_pack']}] "
            f"expected="
            f"{row['gold_decision']} "
            f"predicted="
            f"{row['predicted_decision']}"
        )

    return result


def warmup(
    model,
    tokenizer,
    case,
    mode,
):

    print(
        f"Warmup: {mode}..."
    )

    generate(
        model,
        tokenizer,
        case,
        mode,
    )


def main():

    print(
        "=" * 80
    )

    print(
        "SENTINEL CHALLENGE V1 — FROZEN BENCHMARK"
    )

    print(
        "=" * 80
    )

    verify_challenge()

    cases = load_cases()

    print(
        f"Loaded {len(cases)} "
        "frozen cases."
    )

    print(
        "Policy packs:",
        len({
            x["policy_pack"]
            for x in cases
        })
    )

    print()
    print(
        "Loading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print()
    print(
        "Loading BF16 Llama 3.1 8B..."
    )

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=
                torch.bfloat16,
            device_map={
                "": 0
            },
        )
    )

    print()
    print(
        "Loading Sentinel V2..."
    )

    model = (
        PeftModel
        .from_pretrained(
            base_model,
            V2_PATH,
            adapter_name="v2",
            is_trainable=False,
        )
    )

    print(
        "Loading Sentinel V3..."
    )

    model.load_adapter(
        V3_PATH,
        adapter_name="v3",
        is_trainable=False,
    )

    model.eval()

    print(
        "Loaded adapters:",
        sorted(
            model.peft_config.keys()
        )
    )

    warmup(
        model,
        tokenizer,
        cases[0],
        "base",
    )

    warmup(
        model,
        tokenizer,
        cases[0],
        "v2",
    )

    warmup(
        model,
        tokenizer,
        cases[0],
        "v3",
    )

    results = {}

    results["base"] = evaluate(
        "HF LLAMA 3.1 8B BASE",
        "base",
        model,
        tokenizer,
        cases,
    )

    results["v2"] = evaluate(
        "HF LLAMA-SENTINEL-8B LoRA V2",
        "v2",
        model,
        tokenizer,
        cases,
    )

    results["v3"] = evaluate(
        "HF LLAMA-SENTINEL-8B LoRA V3",
        "v3",
        model,
        tokenizer,
        cases,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL CHALLENGE V1 HEAD-TO-HEAD"
    )

    print(
        "=" * 80
    )

    print(
        f"{'MODEL':32s}"
        f"{'ACC':>10s}"
        f"{'PACKS':>10s}"
        f"{'SUFF':>10s}"
        f"{'JSON':>10s}"
        f"{'LATENCY':>11s}"
    )

    for key in [
        "base",
        "v2",
        "v3",
    ]:

        r = results[key]

        print(
            f"{r['name'][:32]:32s}"
            f"{r['decision_accuracy']:10.3f}"
            f"{r['pack_all_correct_rate']:10.3f}"
            f"{r['sufficiency_accuracy']:10.3f}"
            f"{r['strict_json_rate']:10.3f}"
            f"{r['avg_latency']:10.2f}s"
        )

    base = results[
        "base"
    ]["decision_accuracy"]

    v2 = results[
        "v2"
    ]["decision_accuracy"]

    v3 = results[
        "v3"
    ]["decision_accuracy"]

    print()
    print(
        "V2 - Base delta: "
        f"{v2 - base:+.3f}"
    )

    print(
        "V3 - Base delta: "
        f"{v3 - base:+.3f}"
    )

    print(
        "V3 - V2 delta:   "
        f"{v3 - v2:+.3f}"
    )

    output = {
        "challenge_sha256":
            EXPECTED_CHALLENGE_SHA256,

        "model":
            MODEL_NAME,

        "results":
            results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        )
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
