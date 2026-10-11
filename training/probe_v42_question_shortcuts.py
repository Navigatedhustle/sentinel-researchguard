from pathlib import Path
from collections import Counter
import json

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


BASE = Path("/workspace")
DATA = BASE / "training/datasets"

FILES = {
    "train": DATA / "sentinel_train_v42.jsonl",
    "validation": DATA / "sentinel_validation_v42.jsonl",
    "test": DATA / "sentinel_test_v42.jsonl",
}


def question(text):
    marker = "\n\nQUESTION\n========\n"

    q = text.split(
        marker,
        1,
    )[1].strip()

    prefixes = [
        "Decide the exact policy question below:\n",
        "Using only the supplied evidence, classify this question:\n",
    ]

    for prefix in prefixes:
        if q.startswith(prefix):
            q = q[len(prefix):].strip()

    return q


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

        gold = json.loads(
            row["output"]
        )["decision"]

        X.append(
            question(
                row["input"]
            )
        )

        y.append(gold)

    return X, y


X_train, y_train = load(
    FILES["train"]
)

X_val, y_val = load(
    FILES["validation"]
)

X_test, y_test = load(
    FILES["test"]
)


print("=" * 72)
print("SENTINEL V4.2 QUESTION-ONLY LEXICAL PROBE")
print("=" * 72)

print()
print(
    "train labels:",
    Counter(y_train),
)

print(
    "validation labels:",
    Counter(y_val),
)

print(
    "test labels:",
    Counter(y_test),
)


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)

Xtr = vectorizer.fit_transform(
    X_train
)

Xv = vectorizer.transform(
    X_val
)

Xt = vectorizer.transform(
    X_test
)


model = LogisticRegression(
    max_iter=3000,
    C=1.0,
    class_weight=None,
    random_state=20260814,
)

model.fit(
    Xtr,
    y_train,
)


for name, X, y in [
    (
        "VALIDATION",
        Xv,
        y_val,
    ),
    (
        "TEST",
        Xt,
        y_test,
    ),
]:
    pred = model.predict(X)

    acc = accuracy_score(
        y,
        pred,
    )

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        f"question-only accuracy: "
        f"{acc:.3f}"
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

    labels = [
        "YES",
        "NO",
        "CONDITIONAL",
        "INSUFFICIENT",
    ]

    print(
        "confusion matrix "
        "(rows=gold, cols=pred):"
    )

    print(
        labels
    )

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
    "Chance for four balanced classes is 0.250."
)

print(
    "A strong question-only score means the synthetic "
    "surface grammar is leaking labels."
)
