import json
import re
import requests
import numpy as np
from pathlib import Path

from manual_retrieval import (
    load_chunks,
    embed,
    cosine_similarity
)

BASE = Path(__file__).resolve().parent.parent

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL = "llama3.1:70b"
TOP_K = 4

SCHEMA = {
    "type": "object",
    "properties": {
        "evidence_sufficient": {
            "type": "boolean"
        },
        "decision": {
            "type": "string",
            "enum": [
                "YES",
                "NO",
                "CONDITIONAL",
                "INSUFFICIENT"
            ]
        },
        "answer": {
            "type": "string"
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "evidence_sufficient",
        "decision",
        "answer",
        "citations"
    ],
    "additionalProperties": False
}


def retrieve(query, chunks, vectors):

    query_vector = embed([
        "search_query: " + query
    ])[0]

    scored = []

    for chunk, vector in zip(chunks, vectors):

        score = cosine_similarity(
            query_vector,
            vector
        )

        scored.append({
            **chunk,
            "score": float(score)
        })

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored[:TOP_K]


def source_key(result):

    return (
        result["file"],
        result["section"]
    )


def generate(query, results):

    evidence = []

    for i, result in enumerate(
        results,
        start=1
    ):

        evidence.append(
            f"""
[S{i}]
SOURCE: {result['file']}
SECTION: {result['section']}
TEXT: {result['text']}
""".strip()
        )

    prompt = f"""
You are Sentinel, a research-governance decision system.

Use ONLY the supplied evidence.

Return one of four decisions:

YES
The supplied evidence clearly permits what the user asks.

NO
The supplied evidence clearly prohibits it or clearly says the proposition is false.

CONDITIONAL
The action may be permitted, but the supplied policy explicitly requires conditions, approval, authorization, review, or other requirements.

INSUFFICIENT
The supplied evidence does not actually answer the question.

Important rules:

- Semantic similarity is NOT evidence.
- Do not infer university policy from general knowledge.
- Do not invent missing rules.
- A passage about a related technology or data category does not automatically answer the question.
- If evidence is insufficient, set evidence_sufficient=false and decision=INSUFFICIENT.
- Otherwise set evidence_sufficient=true.
- citations must contain only source labels such as "S1".
- Cite only evidence that actually supports the decision.

EVIDENCE:

{chr(10).join(evidence)}

QUESTION:

{query}
""".strip()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": SCHEMA,
            "keep_alive": "15m",
            "options": {
                "temperature": 0,
                "num_predict": 350
            }
        },
        timeout=300
    )

    response.raise_for_status()

    raw = response.json()

    return (
        json.loads(raw["response"]),
        raw
    )


def main():

    tests = json.loads(
        (BASE / "rag/generation_eval.json")
        .read_text()
    )

    chunks = load_chunks()

    print(
        f"Embedding {len(chunks)} policy chunks..."
    )

    vectors = embed([
        "search_document: " + chunk["text"]
        for chunk in chunks
    ])

    correct_decision = 0
    correct_sufficiency = 0
    retrieval_hits = 0
    citation_hits = 0
    valid_citations = 0
    hallucination_traps_passed = 0

    answerable = 0
    unknown = 0

    latencies = []

    results_output = []

    print("\n" + "=" * 80)
    print("SENTINEL END-TO-END RAG EVALUATION")
    print("=" * 80)

    for test in tests:

        retrieved = retrieve(
            test["query"],
            chunks,
            vectors
        )

        relevant = {
            tuple(x)
            for x in test["relevant"]
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

        result, raw = generate(
            test["query"],
            retrieved
        )

        expected = test[
            "expected_decision"
        ]

        decision_ok = (
            result["decision"] == expected
        )

        correct_decision += int(
            decision_ok
        )

        expected_sufficiency = (
            expected != "INSUFFICIENT"
        )

        sufficiency_ok = (
            result["evidence_sufficient"]
            == expected_sufficiency
        )

        correct_sufficiency += int(
            sufficiency_ok
        )

        cited_indexes = []

        for citation in result["citations"]:

            match = re.fullmatch(
                r"S([1-9][0-9]*)",
                citation.strip()
            )

            if match:

                index = int(
                    match.group(1)
                ) - 1

                if 0 <= index < len(retrieved):
                    cited_indexes.append(index)
                    valid_citations += 1

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
                result["decision"]
                == "INSUFFICIENT"
                and
                result["evidence_sufficient"]
                is False
            )

            hallucination_traps_passed += int(
                trap_pass
            )

        total_ns = raw.get(
            "total_duration",
            0
        )

        load_ns = raw.get(
            "load_duration",
            0
        )

        latency = max(
            total_ns - load_ns,
            0
        ) / 1e9

        latencies.append(latency)

        print("\n" + "-" * 80)
        print(test["id"])
        print(f"QUERY: {test['query']}")

        print("\nTOP RETRIEVAL:")

        for i, item in enumerate(
            retrieved,
            start=1
        ):

            marker = (
                "✓"
                if source_key(item)
                in relevant
                else " "
            )

            print(
                f"{marker} [S{i}] "
                f"{item['score']:.4f} | "
                f"{item['file']} | "
                f"{item['section']}"
            )

        print("\nMODEL:")

        print(
            f"Expected:   {expected}"
        )

        print(
            f"Predicted:  "
            f"{result['decision']}"
        )

        print(
            f"Sufficient: "
            f"{result['evidence_sufficient']}"
        )

        print(
            f"Citations:  "
            f"{result['citations']}"
        )

        print(
            f"Answer:     "
            f"{result['answer']}"
        )

        print(
            f"Decision correct: "
            f"{decision_ok}"
        )

        if relevant:

            print(
                f"Retrieval hit:    "
                f"{retrieval_hit}"
            )

            print(
                f"Gold citation hit:"
                f" {citation_hit}"
            )

        print(
            f"Latency:          "
            f"{latency:.2f}s"
        )

        results_output.append({
            "id": test["id"],
            "query": test["query"],
            "expected": expected,
            "predicted": result["decision"],
            "evidence_sufficient":
                result["evidence_sufficient"],
            "citations":
                result["citations"],
            "answer":
                result["answer"],
            "latency_seconds":
                latency
        })

    total = len(tests)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Decision accuracy:       "
        f"{correct_decision / total:.3f}"
    )

    print(
        f"Sufficiency accuracy:    "
        f"{correct_sufficiency / total:.3f}"
    )

    print(
        f"Retrieval hit rate:      "
        f"{retrieval_hits / answerable:.3f}"
    )

    print(
        f"Gold citation hit rate:  "
        f"{citation_hits / answerable:.3f}"
    )

    print(
        f"Hallucination trap pass: "
        f"{hallucination_traps_passed / unknown:.3f}"
    )

    print(
        f"Avg generation latency:  "
        f"{np.mean(latencies):.2f}s"
    )

    output = (
        BASE /
        "results/rag_generation_v1.json"
    )

    output.write_text(
        json.dumps(
            results_output,
            indent=2
        )
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
