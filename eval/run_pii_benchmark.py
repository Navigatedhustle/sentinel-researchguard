import json
import csv
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "tinyllama-pii",
    "qwen:32b",
    "pii-detector",
    "llama3.1:70b"
]

ALLOWED_TYPES = [
    "NAME",
    "EMAIL",
    "DOB",
    "PHONE",
    "STUDENT_ID",
    "STUDY_ID",
    "MEDICAL_RECORD_ID",
    "IP_ADDRESS"
]

SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ALLOWED_TYPES
                    }
                },
                "required": ["text", "type"],
                "additionalProperties": False
            }
        }
    },
    "required": ["entities"],
    "additionalProperties": False
}

BASE = Path(__file__).resolve().parent.parent
cases = json.loads((BASE / "eval/pii_cases.json").read_text())

def canonical(entity):
    return (
        entity["text"].strip().lower(),
        entity["type"].strip().upper()
    )

def safe_rate(num, denom):
    return num / denom if denom else 0.0

rows = []

for model in MODELS:

    print("\n" + "=" * 72)
    print(f"MODEL: {model}")
    print("=" * 72)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    valid_json = 0
    inference_times = []
    token_rates = []

    for case in cases:

        prompt = f"""
You are performing exact PII entity extraction.

Return ONLY entities that literally appear in the supplied text.

Allowed entity types:
{", ".join(ALLOWED_TYPES)}

Do not infer entities that are not explicitly present.
Do not classify general medical information as an identifier.
Do not explain your answer.

TEXT:
{case["text"]}
""".strip()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": SCHEMA,
            "keep_alive": "15m",
            "options": {
                "temperature": 0,
                "num_predict": 256
            }
        }

        try:
            r = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=300
            )
            r.raise_for_status()
            result = r.json()

            response_text = result.get("response", "")

            try:
                parsed = json.loads(response_text)
                predicted = parsed.get("entities", [])
                valid_json += 1
            except Exception:
                predicted = []

            expected_set = {
                canonical(x)
                for x in case["expected"]
            }

            predicted_set = {
                canonical(x)
                for x in predicted
                if isinstance(x, dict)
                and "text" in x
                and "type" in x
            }

            tp = len(expected_set & predicted_set)
            fp = len(predicted_set - expected_set)
            fn = len(expected_set - predicted_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            total_ns = result.get("total_duration", 0)
            load_ns = result.get("load_duration", 0)
            eval_ns = result.get("eval_duration", 0)
            eval_count = result.get("eval_count", 0)

            inference_s = max(total_ns - load_ns, 0) / 1e9

            tokens_s = (
                eval_count / (eval_ns / 1e9)
                if eval_ns > 0
                else 0
            )

            inference_times.append(inference_s)

            if tokens_s:
                token_rates.append(tokens_s)

            print(f"\n[{case['id']}]")
            print(f"Expected: {sorted(expected_set)}")
            print(f"Predicted: {sorted(predicted_set)}")
            print(f"TP={tp} FP={fp} FN={fn}")
            print(
                f"Inference={inference_s:.2f}s | "
                f"Generation={tokens_s:.2f} tok/s"
            )

            rows.append({
                "model": model,
                "case": case["id"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "json_valid": 1 if predicted or response_text else 0,
                "inference_seconds": round(inference_s, 4),
                "tokens_per_second": round(tokens_s, 2),
                "response": response_text
            })

        except Exception as e:

            print(f"\n[{case['id']}] ERROR: {e}")

            rows.append({
                "model": model,
                "case": case["id"],
                "tp": 0,
                "fp": 0,
                "fn": len(case["expected"]),
                "json_valid": 0,
                "inference_seconds": "",
                "tokens_per_second": "",
                "response": f"ERROR: {e}"
            })

    precision = safe_rate(
        total_tp,
        total_tp + total_fp
    )

    recall = safe_rate(
        total_tp,
        total_tp + total_fn
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    avg_latency = (
        sum(inference_times) / len(inference_times)
        if inference_times
        else 0
    )

    avg_tps = (
        sum(token_rates) / len(token_rates)
        if token_rates
        else 0
    )

    print("\n--- SUMMARY ---")
    print(f"Precision:       {precision:.3f}")
    print(f"Recall:          {recall:.3f}")
    print(f"F1:              {f1:.3f}")
    print(f"Valid JSON:      {valid_json}/{len(cases)}")
    print(f"Avg inference:   {avg_latency:.2f}s")
    print(f"Avg generation:  {avg_tps:.2f} tok/s")

results_file = BASE / "results/pii_benchmark.csv"

with results_file.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:")
print(results_file)
