import re
import requests
import numpy as np
from pathlib import Path

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"

BASE = Path(__file__).resolve().parent.parent
POLICY_DIR = BASE / "rag/policies"


def embed(texts):
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return np.array(
        data["embeddings"],
        dtype=np.float32
    )


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


def load_chunks():
    chunks = []

    for file in sorted(POLICY_DIR.glob("*.txt")):

        text = file.read_text()

        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", text)
            if p.strip()
        ]

        current_section = "DOCUMENT HEADER"

        for paragraph in paragraphs:

            if paragraph.startswith("SECTION "):
                lines = paragraph.splitlines()

                current_section = lines[0]

                body = "\n".join(lines[1:]).strip()

                if body:
                    chunks.append({
                        "file": file.name,
                        "section": current_section,
                        "text": body
                    })

            else:
                # Skip short document headers
                if (
                    paragraph.startswith("SENTINEL UNIVERSITY")
                    or paragraph.startswith("Version ")
                    or paragraph.startswith("Protocol ")
                    or paragraph.startswith("Agreement ")
                ):
                    continue

                chunks.append({
                    "file": file.name,
                    "section": current_section,
                    "text": paragraph
                })

    return chunks


def main():

    chunks = load_chunks()

    print("=" * 72)
    print("SENTINEL MANUAL RETRIEVAL")
    print("=" * 72)

    print(f"\nLoaded {len(chunks)} policy chunks.")

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\nEmbedding policy chunks...")

    chunk_vectors = embed(chunk_texts)

    print(
        f"Embedding matrix shape: "
        f"{chunk_vectors.shape}"
    )

    print(
        f"One chunk = "
        f"{chunk_vectors.shape[1]} dimensions"
    )

    query = input(
        "\nAsk a research-policy question:\n> "
    ).strip()

    print("\nEmbedding query...")

    query_vector = embed([query])[0]

    print(
        f"Query vector dimensions: "
        f"{len(query_vector)}"
    )

    scored = []

    for chunk, vector in zip(
        chunks,
        chunk_vectors
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

    print("\n" + "=" * 72)
    print("TOP RETRIEVED POLICY CHUNKS")
    print("=" * 72)

    for rank, result in enumerate(
        scored[:5],
        start=1
    ):

        print(
            f"\n#{rank}  "
            f"similarity={result['score']:.4f}"
        )

        print(
            f"{result['file']} | "
            f"{result['section']}"
        )

        print("-" * 72)

        print(result["text"])

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
