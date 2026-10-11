import json
import numpy as np
from pathlib import Path

from manual_retrieval import (
    load_chunks,
    embed,
    cosine_similarity
)

BASE = Path(__file__).resolve().parent.parent

EVAL_FILE = BASE / "rag/retrieval_eval.json"


def is_relevant(result, relevant):
    key = (
        result["file"],
        result["section"]
    )

    return key in {
        tuple(item)
        for item in relevant
    }


def main():

    tests = json.loads(
        EVAL_FILE.read_text()
    )

    chunks = load_chunks()

    print(
        f"Embedding {len(chunks)} policy chunks..."
    )

    vectors = embed([
        chunk["text"]
        for chunk in chunks
    ])

    recall1 = 0
    recall3 = 0
    recall5 = 0

    reciprocal_ranks = []

    margins = []

    print("\n" + "=" * 80)
    print("SENTINEL RETRIEVAL EVALUATION")
    print("=" * 80)

    for test in tests:

        query_vector = embed([
            test["query"]
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

        relevant_ranks = [
            rank
            for rank, result in enumerate(
                scored,
                start=1
            )
            if is_relevant(
                result,
                test["relevant"]
            )
        ]

        first_rank = (
            min(relevant_ranks)
            if relevant_ranks
            else None
        )

        if first_rank is not None:

            reciprocal_ranks.append(
                1 / first_rank
            )

            if first_rank <= 1:
                recall1 += 1

            if first_rank <= 3:
                recall3 += 1

            if first_rank <= 5:
                recall5 += 1

        else:
            reciprocal_ranks.append(0)

        top1 = scored[0]["score"]
        top2 = scored[1]["score"]

        margin = top1 - top2

        margins.append(margin)

        print("\n" + "-" * 80)

        print(
            f"{test['id']}"
        )

        print(
            f"QUERY: {test['query']}"
        )

        print(
            f"First relevant rank: "
            f"{first_rank}"
        )

        print(
            f"Top1={top1:.4f} "
            f"Top2={top2:.4f} "
            f"Margin={margin:.4f}"
        )

        print("\nTOP 5:")

        for rank, result in enumerate(
            scored[:5],
            start=1
        ):

            marker = (
                "✓"
                if is_relevant(
                    result,
                    test["relevant"]
                )
                else " "
            )

            print(
                f"{marker} #{rank} "
                f"{result['score']:.4f} | "
                f"{result['file']} | "
                f"{result['section']}"
            )

    total = len(tests)

    mrr = np.mean(
        reciprocal_ranks
    )

    avg_margin = np.mean(
        margins
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"Queries:          {total}"
    )

    print(
        f"Recall@1:         "
        f"{recall1 / total:.3f}"
    )

    print(
        f"Recall@3:         "
        f"{recall3 / total:.3f}"
    )

    print(
        f"Recall@5:         "
        f"{recall5 / total:.3f}"
    )

    print(
        f"MRR:              "
        f"{mrr:.3f}"
    )

    print(
        f"Avg Top1-Top2 gap:"
        f" {avg_margin:.4f}"
    )


if __name__ == "__main__":
    main()
