from pathlib import Path
from collections import Counter
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


DATA = Path("/workspace/training/datasets")

FILES = {
    "train": DATA / "sentinel_train_v42.jsonl",
    "validation": DATA / "sentinel_validation_v42.jsonl",
    "test": DATA / "sentinel_test_v42.jsonl",
}


def evidence(text):
    marker = "\n\nQUESTION\n========\n"

    return text.split(
        marker,
        1,
    )[0]


def load(path):
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    # One representative per underlying scenario.
    reps = {}

    for row in rows:
        reps.setdefault(
            row["scenario_id"],
            row,
        )

    X = []
    y = []

    for row in reps.values():
        X.append(
            evidence(
                row["input"]
            )
        )

        y.append(
            json.loads(
                row["output"]
            )["decision"]
        )

    return X, y


X_train, y_train = load(FILES["train"])
X_val, y_val = load(FILES["validation"])
X_test, y_test = load(FILES["test"])


print("=" * 72)
print("SENTINEL V4.2 EVIDENCE-ONLY LEXICAL PROBE")
print("=" * 72)

print("train:", Counter(y_train))
print("validation:", Counter(y_val))
print("test:", Counter(y_test))


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

Xtr = vectorizer.fit_transform(X_train)
Xv = vectorizer.transform(X_val)
Xt = vectorizer.transform(X_test)


model = LogisticRegression(
    max_iter=3000,
    C=1.0,
    random_state=20260814,
)

model.fit(
    Xtr,
    y_train,
)


labels = [
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
]


for name, X, y in [
    ("VALIDATION", Xv, y_val),
    ("TEST", Xt, y_test),
]:
    pred = model.predict(X)

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        "evidence-only accuracy:",
        f"{accuracy_score(y, pred):.3f}",
    )

    print()

    print(
        classification_report(
            y,
            pred,
            digits=3,
            zero_division=0,
        )
    )

    print(
        "confusion matrix "
        "(rows=gold, cols=pred):"
    )

    print(labels)

    print(
        confusion_matrix(
            y,
            pred,
            labels=labels,
        )
    )


print()
print("=" * 72)
print("INTERPRETATION")
print("=" * 72)

print(
    "This probe intentionally ignores the question."
)

print(
    "Very high transfer accuracy indicates that label-specific "
    "evidence vocabulary may still provide synthetic shortcuts."
)
