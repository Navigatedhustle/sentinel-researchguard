import json
from pathlib import Path

from manual_retrieval import (
    load_chunks,
    embed,
    cosine_similarity
)

BASE = Path(__file__).resolve().parent.parent
TOP_K = 4

tests = json.loads(
    (BASE / "rag/generation_eval.json").read_text()
)

chunks = load_chunks()

print(f"Embedding {len(chunks)} policy chunks...")

vectors = embed([
    "search_document: " + chunk["text"]
    for chunk in chunks
])

output = []

for test in tests:

    query_vector = embed([
        "search_query: " + test["query"]
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

    retrieved = scored[:TOP_K]

    output.append({
        **test,
        "retrieved": retrieved
    })

    print(f"\n{test['id']}")

    for i, item in enumerate(
        retrieved,
        start=1
    ):
        print(
            f"  S{i} "
            f"{item['score']:.4f} | "
            f"{item['file']} | "
            f"{item['section']}"
        )

path = (
    BASE /
    "results/sentinel_retrieval_fixture_v1.json"
)

path.parent.mkdir(
    parents=True,
    exist_ok=True
)

path.write_text(
    json.dumps(
        output,
        indent=2
    )
)

print(f"\n✅ Frozen retrieval fixture:")
print(path)
print(f"Cases: {len(output)}")
