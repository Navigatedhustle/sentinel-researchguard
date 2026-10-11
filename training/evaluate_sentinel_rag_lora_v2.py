import json
import re
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE = Path(__file__).resolve().parent.parent

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

ADAPTER_PATH = (
    BASE /
    "training/output/llama-sentinel-8b-lora-v2"
)

FIXTURE_PATH = (
    BASE /
    "results/sentinel_retrieval_fixture_v1.json"
)

OUTPUT_PATH = (
    BASE /
    "results/sentinel_rag_hf_base_vs_lora_v2.json"
)

MAX_NEW_TOKENS = 300

ALLOWED_DECISIONS = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT"
}


SYSTEM_PROMPT = """
You are Sentinel, a research-governance decision system.

Use ONLY the supplied evidence.

Return one of four decisions:

YES
The evidence clearly permits the exact action asked about, and all
material conditions required by the cited policy are already satisfied
in the user's question. There must be no additional approval,
authorization, review, environment restriction, protocol requirement,
or other unresolved condition.

NO
The evidence clearly prohibits the exact action asked about, or clearly
establishes that the proposition in the question is false.

CONDITIONAL
The action may be permitted, BUT one or more material conditions remain.
Use CONDITIONAL whenever the evidence uses requirements such as:
- when
- provided that
- only if
- subject to
- consistent with
- requires approval
- requires authorization
- requires review
- must remain within
- must comply with

If even one required condition is not already established as satisfied
in the user's question, choose CONDITIONAL rather than YES.

INSUFFICIENT
The supplied evidence does not actually establish whether the exact
action is permitted, prohibited, or conditionally permitted.

Important rules:

Before assigning a decision, apply this evidence-boundary test:

STEP 1 — EXACT ISSUE
Determine the exact action, technology, data use, or proposition asked
about.

STEP 2 — DIRECT SUPPORT
Ask whether the supplied evidence actually governs that exact issue.

A policy about a related subject is not enough.

Examples:
- permission to use approved local AI does NOT establish whether
  personally owned USB drives are permitted or prohibited
- a requirement for approval does NOT identify who must give approval
  unless the evidence says so
- a policy requiring security controls does NOT establish a specific
  encryption algorithm unless the evidence names it
- silence about an action is NOT a prohibition

STEP 3 — DECISION
Only choose YES, NO, or CONDITIONAL when the evidence directly supports
that classification.

If determining the answer requires an unstated assumption, inference
about an unmentioned technology, or extending a rule beyond what the
evidence explicitly governs, choose INSUFFICIENT.

Additional rules:

- Semantic similarity is NOT evidence.
- Do not infer university policy from general knowledge.
- Do not invent missing rules.
- A passage about a related technology or data category does not
  automatically answer the question.
- If evidence is insufficient, set evidence_sufficient=false and
  decision=INSUFFICIENT.
- Otherwise set evidence_sufficient=true.
- citations must contain only source labels such as "S1".
- Cite only evidence that actually supports the decision.

OUTPUT CONTRACT:

Return valid JSON only.

The benchmark fields are:

{
  "evidence_sufficient": true or false,
  "decision": "YES" or "NO" or "CONDITIONAL" or "INSUFFICIENT",
  "answer": "brief evidence-grounded explanation",
  "citations": ["S1"]
}

Do not wrap the JSON in Markdown.
""".strip()


def source_key(result):
    return (
        result["file"],
        result["section"]
    )


def build_messages(case):

    evidence = []

    for i, result in enumerate(
        case["retrieved"],
        start=1
    ):
        evidence.append(
            f"""[S{i}]
SOURCE: {result['file']}
SECTION: {result['section']}
TEXT: {result['text']}"""
        )

    user_content = f"""
EVIDENCE:

{chr(10).join(evidence)}

QUESTION:

{case["query"]}
""".strip()

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_content
        }
    ]


def extract_json(text):

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned
        )

    try:
        obj = json.loads(cleaned)

        if isinstance(obj, dict):
            return obj, True

    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):
        try:
            obj = json.loads(
                cleaned[start:end + 1]
            )

            if isinstance(obj, dict):
                return obj, False

        except Exception:
            pass

    return {}, False


def semantic_decision(obj, raw_text):

    value = obj.get("decision")

    # Fair semantic fallback for an unfine-tuned
    # model that uses "classification".
    if value is None:
        value = obj.get("classification")

    if isinstance(value, str):

        value = value.strip().upper()

        if value in ALLOWED_DECISIONS:
            return value

    match = re.search(
        r'"(?:decision|classification)"\s*:\s*"'
        r'(YES|NO|CONDITIONAL|INSUFFICIENT)"',
        raw_text,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return "PARSE_ERROR"


def original_schema_ok(obj, strict_json):

    if not strict_json:
        return False

    required = {
        "evidence_sufficient",
        "decision",
        "answer",
        "citations"
    }

    if set(obj.keys()) != required:
        return False

    if not isinstance(
        obj["evidence_sufficient"],
        bool
    ):
        return False

    if (
        not isinstance(
            obj["decision"],
            str
        )
        or
        obj["decision"] not in ALLOWED_DECISIONS
    ):
        return False

    if not isinstance(
        obj["answer"],
        str
    ):
        return False

    if not isinstance(
        obj["citations"],
        list
    ):
        return False

    if not all(
        isinstance(x, str)
        for x in obj["citations"]
    ):
        return False

    return True


def generate(
    model,
    tokenizer,
    case,
    use_adapter
):

    messages = build_messages(case)

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to("cuda")

    prompt_tokens = (
        encoded["input_ids"].shape[1]
    )

    context = (
        nullcontext()
        if use_adapter
        else model.disable_adapter()
    )

    torch.cuda.synchronize()

    start = time.perf_counter()

    with context:

        with torch.inference_mode():

            output = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

    torch.cuda.synchronize()

    latency = (
        time.perf_counter() - start
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
        clean_up_tokenization_spaces=False
    ).strip()

    tokens_per_second = (
        output_tokens / latency
        if latency > 0
        else 0.0
    )

    return {
        "text": text,
        "latency": latency,
        "output_tokens": output_tokens,
        "tokens_per_second":
            tokens_per_second
    }


def evaluate(
    name,
    model,
    tokenizer,
    cases,
    use_adapter
):

    correct_decision = 0
    correct_sufficiency = 0

    retrieval_hits = 0
    citation_hits = 0

    hallucination_traps_passed = 0

    answerable = 0
    unknown = 0

    strict_json_count = 0
    original_schema_count = 0

    latencies = []
    output_token_counts = []
    token_rates = []

    rows = []

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    for i, case in enumerate(
        cases,
        start=1
    ):

        retrieved = case["retrieved"]

        relevant = {
            tuple(x)
            for x in case["relevant"]
        }

        retrieved_keys = [
            source_key(x)
            for x in retrieved
        ]

        if relevant:

            answerable += 1

            retrieval_hit = any(
                key in relevant
                for key in retrieved_keys
            )

            retrieval_hits += int(
                retrieval_hit
            )

        else:

            unknown += 1
            retrieval_hit = None

        generated = generate(
            model,
            tokenizer,
            case,
            use_adapter
        )

        obj, strict_json = extract_json(
            generated["text"]
        )

        predicted = semantic_decision(
            obj,
            generated["text"]
        )

        expected = case[
            "expected_decision"
        ]

        decision_ok = (
            predicted == expected
        )

        correct_decision += int(
            decision_ok
        )

        expected_sufficiency = (
            expected != "INSUFFICIENT"
        )

        predicted_sufficiency = (
            obj.get(
                "evidence_sufficient",
                None
            )
        )

        sufficiency_ok = (
            predicted_sufficiency
            == expected_sufficiency
        )

        correct_sufficiency += int(
            sufficiency_ok
        )

        strict_json_count += int(
            strict_json
        )

        schema_ok = original_schema_ok(
            obj,
            strict_json
        )

        original_schema_count += int(
            schema_ok
        )

        citations = obj.get(
            "citations",
            []
        )

        if not isinstance(
            citations,
            list
        ):
            citations = []

        cited_indexes = []

        for citation in citations:

            if not isinstance(
                citation,
                str
            ):
                continue

            match = re.fullmatch(
                r"S([1-9][0-9]*)",
                citation.strip()
            )

            if match:

                index = (
                    int(
                        match.group(1)
                    )
                    - 1
                )

                if (
                    0 <= index
                    < len(retrieved)
                ):
                    cited_indexes.append(
                        index
                    )

        citation_hit = False

        if relevant:

            for index in cited_indexes:

                if (
                    source_key(
                        retrieved[index]
                    )
                    in relevant
                ):
                    citation_hit = True
                    break

            citation_hits += int(
                citation_hit
            )

        if expected == "INSUFFICIENT":

            trap_pass = (
                predicted
                == "INSUFFICIENT"
                and
                predicted_sufficiency
                is False
            )

            hallucination_traps_passed += int(
                trap_pass
            )

        latency = generated[
            "latency"
        ]

        output_tokens = generated[
            "output_tokens"
        ]

        tps = generated[
            "tokens_per_second"
        ]

        latencies.append(latency)
        output_token_counts.append(
            output_tokens
        )
        token_rates.append(tps)

        marker = (
            "✓"
            if decision_ok
            else "✗"
        )

        print("\n" + "-" * 80)

        print(
            f"{marker} {i:02d}/"
            f"{len(cases)} "
            f"{case['id']}"
        )

        print(
            f"Expected:    {expected}"
        )

        print(
            f"Predicted:   {predicted}"
        )

        print(
            f"Sufficient:  "
            f"{predicted_sufficiency}"
        )

        print(
            f"Citations:   {citations}"
        )

        print(
            f"Strict JSON: {strict_json}"
        )

        print(
            f"Orig schema: {schema_ok}"
        )

        print(
            f"Latency:     "
            f"{latency:.2f}s"
        )

        print(
            f"Output toks: {output_tokens}"
        )

        print(
            f"Tokens/sec:  {tps:.2f}"
        )

        if not decision_ok:

            print(
                f"OUTPUT:      "
                f"{generated['text']}"
            )

        rows.append({
            "id":
                case["id"],
            "query":
                case["query"],
            "expected":
                expected,
            "predicted":
                predicted,
            "decision_correct":
                decision_ok,
            "expected_sufficiency":
                expected_sufficiency,
            "predicted_sufficiency":
                predicted_sufficiency,
            "sufficiency_correct":
                sufficiency_ok,
            "strict_json":
                strict_json,
            "original_schema_ok":
                schema_ok,
            "citations":
                citations,
            "retrieval_hit":
                retrieval_hit,
            "citation_hit":
                citation_hit
                if relevant
                else None,
            "latency_seconds":
                latency,
            "output_tokens":
                output_tokens,
            "tokens_per_second":
                tps,
            "raw_output":
                generated["text"]
        })

    total = len(cases)

    summary = {
        "model": name,
        "total_cases": total,

        "decision_accuracy":
            correct_decision / total,

        "sufficiency_accuracy":
            correct_sufficiency / total,

        "retrieval_hit_rate":
            retrieval_hits / answerable,

        "gold_citation_hit_rate":
            citation_hits / answerable,

        "hallucination_trap_pass":
            hallucination_traps_passed
            / unknown,

        "strict_json_rate":
            strict_json_count / total,

        "original_schema_rate":
            original_schema_count / total,

        "avg_latency_seconds":
            float(
                np.mean(latencies)
            ),

        "avg_output_tokens":
            float(
                np.mean(
                    output_token_counts
                )
            ),

        "avg_tokens_per_second":
            float(
                np.mean(token_rates)
            )
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Decision accuracy:       "
        f"{summary['decision_accuracy']:.3f}"
    )

    print(
        f"Sufficiency accuracy:    "
        f"{summary['sufficiency_accuracy']:.3f}"
    )

    print(
        f"Retrieval hit rate:      "
        f"{summary['retrieval_hit_rate']:.3f}"
    )

    print(
        f"Gold citation hit rate:  "
        f"{summary['gold_citation_hit_rate']:.3f}"
    )

    print(
        f"Hallucination trap pass: "
        f"{summary['hallucination_trap_pass']:.3f}"
    )

    print(
        f"Strict JSON rate:        "
        f"{summary['strict_json_rate']:.3f}"
    )

    print(
        f"Original schema rate:    "
        f"{summary['original_schema_rate']:.3f}"
    )

    print(
        f"Avg latency:             "
        f"{summary['avg_latency_seconds']:.2f}s"
    )

    print(
        f"Avg output tokens:       "
        f"{summary['avg_output_tokens']:.1f}"
    )

    print(
        f"Avg tokens/sec:          "
        f"{summary['avg_tokens_per_second']:.2f}"
    )

    return {
        "summary": summary,
        "rows": rows
    }


def main():

    cases = json.loads(
        FIXTURE_PATH.read_text()
    )

    print(
        f"Loaded {len(cases)} "
        f"frozen Sentinel cases."
    )

    print("\nLoading tokenizer...")

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    print(
        "\nLoading BF16 "
        "Llama 3.1 8B..."
    )

    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map={"": 0}
        )
    )

    base_model.eval()
    base_model.config.use_cache = True

    print(
        "\nLoading Sentinel "
        "LoRA adapter..."
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False
    )

    model.eval()

    print("\nWarmup: BASE...")

    generate(
        model,
        tokenizer,
        cases[0],
        use_adapter=False
    )

    print("Warmup: LoRA...")

    generate(
        model,
        tokenizer,
        cases[0],
        use_adapter=True
    )

    print(
        "\nEvaluating HF BF16 "
        "8B BASE..."
    )

    base_results = evaluate(
        "HF LLAMA 3.1 8B BASE",
        model,
        tokenizer,
        cases,
        use_adapter=False
    )

    print(
        "\nEvaluating HF BF16 "
        "8B + SENTINEL LoRA..."
    )

    lora_results = evaluate(
        "HF LLAMA-SENTINEL-8B LoRA V2",
        model,
        tokenizer,
        cases,
        use_adapter=True
    )

    output = {
        "experiment": {
            "fixture":
                str(FIXTURE_PATH),
            "base_model":
                MODEL_NAME,
            "adapter":
                str(ADAPTER_PATH),
            "cases":
                len(cases),
            "max_new_tokens":
                MAX_NEW_TOKENS
        },

        "base":
            base_results,

        "lora":
            lora_results
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2
        )
    )

    b = base_results[
        "summary"
    ]

    l = lora_results[
        "summary"
    ]

    print("\n" + "=" * 80)
    print("FINAL HEAD-TO-HEAD")
    print("=" * 80)

    print(
        f"HF Base decision accuracy: "
        f"{b['decision_accuracy']:.3f}"
    )

    print(
        f"HF LoRA decision accuracy: "
        f"{l['decision_accuracy']:.3f}"
    )

    print(
        f"Accuracy delta:             "
        f"{l['decision_accuracy'] - b['decision_accuracy']:+.3f}"
    )

    print()

    print(
        f"HF Base hallucination:      "
        f"{b['hallucination_trap_pass']:.3f}"
    )

    print(
        f"HF LoRA hallucination:      "
        f"{l['hallucination_trap_pass']:.3f}"
    )

    print()

    print(
        f"HF Base latency:            "
        f"{b['avg_latency_seconds']:.2f}s"
    )

    print(
        f"HF LoRA latency:            "
        f"{l['avg_latency_seconds']:.2f}s"
    )

    print()

    print(
        f"HF Base tokens/sec:         "
        f"{b['avg_tokens_per_second']:.2f}"
    )

    print(
        f"HF LoRA tokens/sec:         "
        f"{l['avg_tokens_per_second']:.2f}"
    )

    print()

    print(
        f"HF Base schema adherence:   "
        f"{b['original_schema_rate']:.3f}"
    )

    print(
        f"HF LoRA schema adherence:   "
        f"{l['original_schema_rate']:.3f}"
    )

    print(
        f"\nSaved:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
