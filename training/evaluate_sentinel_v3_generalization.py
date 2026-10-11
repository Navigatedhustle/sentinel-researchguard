import json
import re
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE = Path(__file__).resolve().parent.parent

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

V2_PATH = (
    BASE /
    "training/output/llama-sentinel-8b-lora-v2"
)

V3_PATH = (
    BASE /
    "training/output/llama-sentinel-8b-lora-v3"
)

TEST_PATH = (
    BASE /
    "training/datasets/sentinel_test_v3.jsonl"
)

OUTPUT_PATH = (
    BASE /
    "results/sentinel_v3_generalization.json"
)

MAX_NEW_TOKENS = 120

ALLOWED_DECISIONS = [
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
]


def load_cases():

    rows = []

    for line in TEST_PATH.read_text().splitlines():

        if line.strip():
            rows.append(
                json.loads(line)
            )

    return rows


def extract_json(text):

    cleaned = text.strip()

    strict = False

    try:
        obj = json.loads(cleaned)

        if isinstance(obj, dict):
            strict = True
            return obj, strict

    except Exception:
        pass

    # Remove Markdown fences if present.
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
            obj = json.loads(fenced)

            if isinstance(obj, dict):
                return obj, False

        except Exception:
            pass

    # Last-resort JSON object extraction.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        try:
            obj = json.loads(
                cleaned[
                    start:end + 1
                ]
            )

            if isinstance(obj, dict):
                return obj, False

        except Exception:
            pass

    return None, False


def normalize_decision(obj):

    if not isinstance(obj, dict):
        return "INVALID"

    decision = obj.get(
        "decision"
    )

    if not isinstance(
        decision,
        str,
    ):
        return "INVALID"

    decision = (
        decision
        .strip()
        .upper()
    )

    if decision not in ALLOWED_DECISIONS:
        return "INVALID"

    return decision


def valid_citations(obj):

    if not isinstance(obj, dict):
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


def build_messages(case):

    # This exactly mirrors V3 SFT formatting:
    # instruction = system
    # input       = user
    return [
        {
            "role": "system",
            "content":
                case["instruction"],
        },
        {
            "role": "user",
            "content":
                case["input"],
        },
    ]


def select_mode(
    model,
    mode,
):

    if mode == "base":

        return model.disable_adapter()

    model.set_adapter(mode)

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

    encoded = (
        tokenizer
        .apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        .to("cuda")
    )

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

    generated_ids = output[
        0,
        prompt_tokens:
    ]

    output_tokens = len(
        generated_ids
    )

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return {
        "text":
            text,

        "latency":
            latency,

        "output_tokens":
            output_tokens,

        "tokens_per_second":
            (
                output_tokens
                / latency
                if latency > 0
                else 0.0
            ),
    }


def scenario_metrics(rows):

    groups = defaultdict(list)

    for row in rows:

        groups[
            row["scenario_id"]
        ].append(
            row
        )

    majority_correct = 0
    all_correct = 0
    consistent = 0

    family_stats = defaultdict(
        lambda: {
            "scenarios": 0,
            "majority_correct": 0,
            "all_correct": 0,
        }
    )

    details = []

    for (
        scenario_id,
        items,
    ) in groups.items():

        gold = items[0][
            "gold_decision"
        ]

        family = items[0][
            "logical_family"
        ]

        preds = [
            x["predicted_decision"]
            for x in items
        ]

        counts = Counter(preds)

        top_count = max(
            counts.values()
        )

        winners = [
            decision
            for decision, count
            in counts.items()
            if count == top_count
        ]

        if len(winners) == 1:
            majority = winners[0]
        else:
            majority = "TIE"

        majority_ok = (
            majority == gold
        )

        all_ok = all(
            pred == gold
            for pred in preds
        )

        same_answer = (
            len(set(preds)) == 1
        )

        majority_correct += int(
            majority_ok
        )

        all_correct += int(
            all_ok
        )

        consistent += int(
            same_answer
        )

        family_stats[
            family
        ]["scenarios"] += 1

        family_stats[
            family
        ][
            "majority_correct"
        ] += int(
            majority_ok
        )

        family_stats[
            family
        ][
            "all_correct"
        ] += int(
            all_ok
        )

        details.append({
            "scenario_id":
                scenario_id,

            "logical_family":
                family,

            "gold":
                gold,

            "predictions":
                preds,

            "majority":
                majority,

            "majority_correct":
                majority_ok,

            "all_correct":
                all_ok,
        })

    n = len(groups)

    by_family = {}

    for (
        family,
        stats,
    ) in sorted(
        family_stats.items()
    ):

        count = stats[
            "scenarios"
        ]

        by_family[
            family
        ] = {
            **stats,

            "majority_accuracy":
                (
                    stats[
                        "majority_correct"
                    ] / count
                ),

            "all_correct_rate":
                (
                    stats[
                        "all_correct"
                    ] / count
                ),
        }

    return {
        "scenario_count":
            n,

        "majority_accuracy":
            (
                majority_correct / n
                if n
                else 0.0
            ),

        "all_correct_rate":
            (
                all_correct / n
                if n
                else 0.0
            ),

        "decision_consistency_rate":
            (
                consistent / n
                if n
                else 0.0
            ),

        "by_family":
            by_family,

        "details":
            details,
    }


def evaluate(
    display_name,
    mode,
    model,
    tokenizer,
    cases,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        display_name
    )

    print(
        "=" * 80
    )

    rows = []

    class_total = Counter()
    class_correct = Counter()

    family_total = Counter()
    family_correct = Counter()

    confusion = Counter()

    strict_json_count = 0
    sufficiency_correct = 0

    citation_hit_count = 0
    citation_exact_count = 0

    latencies = []
    output_tokens = []
    token_rates = []

    for i, case in enumerate(
        cases,
        start=1,
    ):

        gold_obj = json.loads(
            case["output"]
        )

        gold_decision = gold_obj[
            "decision"
        ]

        gold_sufficient = gold_obj[
            "evidence_sufficient"
        ]

        gold_citations = set(
            gold_obj.get(
                "citations",
                [],
            )
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

        predicted_decision = (
            normalize_decision(
                obj
            )
        )

        correct = (
            predicted_decision
            == gold_decision
        )

        family = case[
            "logical_family"
        ]

        class_total[
            gold_decision
        ] += 1

        class_correct[
            gold_decision
        ] += int(correct)

        family_total[
            family
        ] += 1

        family_correct[
            family
        ] += int(correct)

        confusion[
            (
                gold_decision,
                predicted_decision,
            )
        ] += 1

        strict_json_count += int(
            strict_json
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
            == gold_sufficient
        )

        pred_citations = set(
            valid_citations(
                obj
            )
        )

        if gold_citations:

            citation_hit = bool(
                pred_citations
                & gold_citations
            )

        else:

            citation_hit = (
                len(
                    pred_citations
                ) == 0
            )

        citation_exact = (
            pred_citations
            == gold_citations
        )

        citation_hit_count += int(
            citation_hit
        )

        citation_exact_count += int(
            citation_exact
        )

        latencies.append(
            generated["latency"]
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
            "scenario_id":
                case["scenario_id"],

            "logical_family":
                family,

            "gold_decision":
                gold_decision,

            "predicted_decision":
                predicted_decision,

            "correct":
                correct,

            "strict_json":
                strict_json,

            "gold_sufficient":
                gold_sufficient,

            "predicted_sufficient":
                pred_sufficient,

            "gold_citations":
                sorted(
                    gold_citations
                ),

            "predicted_citations":
                sorted(
                    pred_citations
                ),

            "citation_hit":
                citation_hit,

            "citation_exact":
                citation_exact,

            "latency":
                generated[
                    "latency"
                ],

            "output_tokens":
                generated[
                    "output_tokens"
                ],

            "text":
                generated["text"],
        })

        if (
            i % 20 == 0
            or i == len(cases)
        ):

            correct_so_far = sum(
                x["correct"]
                for x in rows
            )

            print(
                f"{i:3d}/"
                f"{len(cases)} "
                f"row_accuracy="
                f"{correct_so_far / i:.3f}"
            )

    n = len(rows)

    correct_total = sum(
        x["correct"]
        for x in rows
    )

    by_class = {}

    for decision in (
        ALLOWED_DECISIONS
    ):

        total = class_total[
            decision
        ]

        correct = class_correct[
            decision
        ]

        by_class[
            decision
        ] = {
            "correct":
                correct,

            "total":
                total,

            "accuracy":
                (
                    correct / total
                    if total
                    else 0.0
                ),
        }

    by_family = {}

    for family in sorted(
        family_total
    ):

        total = family_total[
            family
        ]

        correct = family_correct[
            family
        ]

        by_family[
            family
        ] = {
            "correct":
                correct,

            "total":
                total,

            "accuracy":
                correct / total,
        }

    scenario = scenario_metrics(
        rows
    )

    confusion_dict = {}

    for gold in (
        ALLOWED_DECISIONS
    ):

        confusion_dict[
            gold
        ] = {}

        for pred in (
            ALLOWED_DECISIONS
            + ["INVALID"]
        ):

            confusion_dict[
                gold
            ][pred] = (
                confusion[
                    (
                        gold,
                        pred,
                    )
                ]
            )

    result = {
        "name":
            display_name,

        "mode":
            mode,

        "rows":
            n,

        "row_decision_accuracy":
            correct_total / n,

        "correct_decisions":
            correct_total,

        "sufficiency_accuracy":
            sufficiency_correct / n,

        "strict_json_rate":
            strict_json_count / n,

        "citation_hit_rate":
            citation_hit_count / n,

        "citation_exact_rate":
            citation_exact_count / n,

        "avg_latency":
            sum(latencies)
            / len(latencies),

        "avg_output_tokens":
            sum(output_tokens)
            / len(output_tokens),

        "avg_tokens_per_second":
            sum(token_rates)
            / len(token_rates),

        "by_class":
            by_class,

        "by_family":
            by_family,

        "confusion":
            confusion_dict,

        "scenario_metrics":
            scenario,

        "rows_detail":
            rows,
    }

    print()
    print(
        "ROW DECISION ACCURACY:   "
        f"{result['row_decision_accuracy']:.3f} "
        f"({correct_total}/{n})"
    )

    print(
        "SCENARIO MAJORITY ACC:   "
        f"{scenario['majority_accuracy']:.3f} "
        f"({scenario['scenario_count']} scenarios)"
    )

    print(
        "SCENARIO ALL-CORRECT:    "
        f"{scenario['all_correct_rate']:.3f}"
    )

    print(
        "SCENARIO CONSISTENCY:    "
        f"{scenario['decision_consistency_rate']:.3f}"
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
    print("BY CLASS:")

    for decision in (
        ALLOWED_DECISIONS
    ):

        stats = by_class[
            decision
        ]

        print(
            f"  {decision:12s} "
            f"{stats['accuracy']:.3f} "
            f"({stats['correct']}/"
            f"{stats['total']})"
        )

    print()
    print("BY HELD-OUT FAMILY:")

    for (
        family,
        stats,
    ) in by_family.items():

        print(
            f"  {family:30s} "
            f"{stats['accuracy']:.3f} "
            f"({stats['correct']}/"
            f"{stats['total']})"
        )

    print()
    print("CONFUSION MATRIX:")
    print(
        "GOLD \\ PRED       "
        "YES   NO   COND   INSUFF   INVALID"
    )

    for gold in (
        ALLOWED_DECISIONS
    ):

        c = confusion_dict[
            gold
        ]

        print(
            f"{gold:12s} "
            f"{c['YES']:5d}"
            f"{c['NO']:5d}"
            f"{c['CONDITIONAL']:7d}"
            f"{c['INSUFFICIENT']:9d}"
            f"{c['INVALID']:10d}"
        )

    failures = [
        row
        for row in rows
        if not row["correct"]
    ]

    print()
    print(
        "FAILURES:",
        len(failures),
    )

    for row in failures[:12]:

        print(
            f"  {row['scenario_id']}: "
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

    _ = generate(
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
        "SENTINEL V3 GENERALIZATION BENCHMARK"
    )

    print(
        "=" * 80
    )

    cases = load_cases()

    print(
        f"Loaded {len(cases)} "
        "V3 held-out test rows."
    )

    scenario_count = len({
        case["scenario_id"]
        for case in cases
    })

    family_count = len({
        case["logical_family"]
        for case in cases
    })

    print(
        f"Unique scenarios: "
        f"{scenario_count}"
    )

    print(
        f"Held-out families: "
        f"{family_count}"
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
        "Loading Sentinel V2 adapter..."
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
        "Loading Sentinel V3 adapter..."
    )

    model.load_adapter(
        V3_PATH,
        adapter_name="v3",
        is_trainable=False,
    )

    model.eval()

    # Confirm adapters actually exist.
    print()
    print(
        "Loaded adapters:",
        sorted(
            model.peft_config.keys()
        ),
    )

    # Warm each execution mode once.
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

    results[
        "base"
    ] = evaluate(
        "HF LLAMA 3.1 8B BASE",
        "base",
        model,
        tokenizer,
        cases,
    )

    results[
        "v2"
    ] = evaluate(
        "HF LLAMA-SENTINEL-8B LoRA V2",
        "v2",
        model,
        tokenizer,
        cases,
    )

    results[
        "v3"
    ] = evaluate(
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
        "FINAL GENERALIZATION HEAD-TO-HEAD"
    )

    print(
        "=" * 80
    )

    print(
        f"{'MODEL':32s}"
        f"{'ROW ACC':>10s}"
        f"{'SCENARIO':>12s}"
        f"{'ALL-4':>10s}"
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
            f"{r['row_decision_accuracy']:10.3f}"
            f"{r['scenario_metrics']['majority_accuracy']:12.3f}"
            f"{r['scenario_metrics']['all_correct_rate']:10.3f}"
            f"{r['strict_json_rate']:10.3f}"
            f"{r['avg_latency']:10.2f}s"
        )

    base_acc = results[
        "base"
    ][
        "row_decision_accuracy"
    ]

    v2_acc = results[
        "v2"
    ][
        "row_decision_accuracy"
    ]

    v3_acc = results[
        "v3"
    ][
        "row_decision_accuracy"
    ]

    print()
    print(
        "V2 - Base row delta: "
        f"{v2_acc - base_acc:+.3f}"
    )

    print(
        "V3 - Base row delta: "
        f"{v3_acc - base_acc:+.3f}"
    )

    print(
        "V3 - V2 row delta:   "
        f"{v3_acc - v2_acc:+.3f}"
    )

    output = {
        "model":
            MODEL_NAME,

        "test_path":
            str(TEST_PATH),

        "test_rows":
            len(cases),

        "unique_scenarios":
            scenario_count,

        "held_out_families":
            sorted({
                case[
                    "logical_family"
                ]
                for case in cases
            }),

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
