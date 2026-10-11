import requests

from manual_retrieval import (
    load_chunks,
    embed,
    cosine_similarity
)

OLLAMA_GENERATE = "http://localhost:11434/api/generate"

LLM_MODEL = "llama3.1:70b"
TOP_K = 4


def generate(prompt):

    response = requests.post(
        OLLAMA_GENERATE,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "15m",
            "options": {
                "temperature": 0,
                "num_predict": 500
            }
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["response"]


def retrieve(query, chunks, vectors):

    query_vector = embed([
        "search_query: " + query
    ])[0]

    scored = []

    for chunk, vector in zip(
        chunks,
        vectors
    ):

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


def build_rag_prompt(query, results):

    evidence_blocks = []

    for i, result in enumerate(
        results,
        start=1
    ):

        evidence_blocks.append(
            f"""
[S{i}]
SOURCE: {result['file']}
SECTION: {result['section']}
TEXT:
{result['text']}
""".strip()
        )

    evidence = "\n\n".join(
        evidence_blocks
    )

    return f"""
You are Sentinel, a research governance assistant.

Answer the user's question using ONLY the supplied policy evidence.

Rules:

1. Do not use outside knowledge to determine Sentinel University policy.
2. Do not invent requirements, exceptions, approvals, or procedures.
3. If the supplied evidence does not answer the question, respond:
   INSUFFICIENT EVIDENCE
4. Cite every policy claim using [S1], [S2], etc.
5. Distinguish between what is prohibited, what is permitted, and what requires approval.
6. If multiple policies apply, explain how they interact.
7. Keep the answer concise but precise.

POLICY EVIDENCE:

{evidence}

USER QUESTION:

{query}
""".strip()


def main():

    print("=" * 78)
    print("SENTINEL RAG — 70B CONTROLLED COMPARISON")
    print("=" * 78)

    chunks = load_chunks()

    print(
        f"\nEmbedding {len(chunks)} policy chunks..."
    )

    vectors = embed([
        "search_document: " + chunk["text"]
        for chunk in chunks
    ])

    print(
        f"Vector matrix: {vectors.shape}"
    )

    query = input(
        "\nAsk Sentinel University a policy question:\n> "
    ).strip()

    print("\n" + "=" * 78)
    print("RETRIEVED EVIDENCE")
    print("=" * 78)

    results = retrieve(
        query,
        chunks,
        vectors
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n[S{i}] "
            f"similarity={result['score']:.4f}"
        )

        print(
            f"{result['file']} | "
            f"{result['section']}"
        )

        print(result["text"])

    print("\n" + "=" * 78)
    print("LLAMA 70B — NO RAG")
    print("=" * 78)

    plain_prompt = f"""
Answer this question:

{query}

If the question depends on a specific Sentinel University policy
that you do not actually know, explain that limitation rather than
inventing the policy.
""".strip()

    plain_answer = generate(
        plain_prompt
    )

    print("\n" + plain_answer)

    print("\n" + "=" * 78)
    print("LLAMA 70B — WITH SENTINEL RAG")
    print("=" * 78)

    rag_prompt = build_rag_prompt(
        query,
        results
    )

    rag_answer = generate(
        rag_prompt
    )

    print("\n" + rag_answer)

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
