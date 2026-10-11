from collections import Counter
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import sys


BASE = Path("/workspace")

SPEC_PATH = (
    BASE
    / "training/v5/"
      "v5_family_specs.py"
)

CHALLENGE_V2 = (
    BASE
    / "experiments/challenge_v2/"
      "sentinel_challenge_v2.json"
)

CHALLENGE_V2_MANIFEST = (
    BASE
    / "experiments/challenge_v2/"
      "FROZEN_SHA256.txt"
)

EXPECTED_CHALLENGE_SHA = (
    "30eb37ff39382f0d205adb7c17383b240"
    "25601ea3db1dd61f22b715af6c1e482"
)

ERRORS = []


def fail(message):
    ERRORS.append(message)


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_specs():

    spec = importlib.util.spec_from_file_location(
        "v5_family_specs",
        SPEC_PATH,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def normalized_words(name):

    return set(
        re.findall(
            r"[a-z]+",
            name.lower(),
        )
    )


def main():

    print("=" * 88)
    print(
        "SENTINEL V5 — FAMILY ONTOLOGY AUDIT"
    )
    print("=" * 88)

    module = load_specs()

    operators = module.OPERATORS
    expected_operators = module.EXPECTED_OPERATORS
    forbidden = module.FORBIDDEN_EXTERNAL_FAMILY_NAMES


    # ========================================================
    # CHALLENGE V2 HASH ONLY
    # ========================================================
    #
    # We do NOT read benchmark contents.
    # ========================================================

    challenge_sha = sha256(
        CHALLENGE_V2
    )

    print(
        "Challenge V2 SHA:",
        challenge_sha,
    )

    if (
        challenge_sha
        != EXPECTED_CHALLENGE_SHA
    ):

        fail(
            "Challenge V2 SHA mismatch."
        )


    # ========================================================
    # OPERATORS
    # ========================================================

    print()
    print(
        "operators:",
        len(
            operators
        ),
    )

    if set(
        operators
    ) != expected_operators:

        fail(
            "Operator identity mismatch."
        )


    train_families = []
    dev_families = []

    per_operator = {}


    for operator, spec in operators.items():

        train = spec.get(
            "train",
            []
        )

        development = spec.get(
            "development",
            []
        )

        per_operator[
            operator
        ] = {
            "train":
                len(train),

            "development":
                len(development),
        }


        if len(
            train
        ) != 4:

            fail(
                f"{operator}: expected 4 train families, "
                f"got {len(train)}."
            )


        if len(
            development
        ) != 1:

            fail(
                f"{operator}: expected 1 development family, "
                f"got {len(development)}."
            )


        for family in train:

            name = family.get(
                "family"
            )

            if not isinstance(
                name,
                str,
            ) or not name:

                fail(
                    f"{operator}: invalid train family name."
                )

                continue


            train_families.append(
                (
                    operator,
                    name,
                )
            )


        for family in development:

            name = family.get(
                "family"
            )

            if not isinstance(
                name,
                str,
            ) or not name:

                fail(
                    f"{operator}: invalid development family name."
                )

                continue


            dev_families.append(
                (
                    operator,
                    name,
                )
            )


    print(
        "per operator:",
        json.dumps(
            per_operator,
            indent=2,
            sort_keys=True,
        ),
    )


    print()
    print(
        "train families:",
        len(
            train_families
        ),
    )

    print(
        "development families:",
        len(
            dev_families
        ),
    )


    if len(
        train_families
    ) != 32:

        fail(
            "Expected exactly 32 train families."
        )


    if len(
        dev_families
    ) != 8:

        fail(
            "Expected exactly 8 development families."
        )


    # ========================================================
    # UNIQUENESS
    # ========================================================

    train_names = [
        name
        for _, name
        in train_families
    ]

    dev_names = [
        name
        for _, name
        in dev_families
    ]

    all_names = (
        train_names
        + dev_names
    )


    duplicate_names = [
        name
        for name, count
        in Counter(
            all_names
        ).items()
        if count > 1
    ]


    print()
    print(
        "duplicate family names:",
        duplicate_names,
    )


    if duplicate_names:

        fail(
            "Duplicate family names exist."
        )


    overlap = (
        set(
            train_names
        )
        & set(
            dev_names
        )
    )


    print(
        "train/dev exact name overlap:",
        sorted(
            overlap
        ),
    )


    if overlap:

        fail(
            "Train/dev family name overlap."
        )


    # ========================================================
    # EXTERNAL NAME COLLISIONS
    # ========================================================

    external_collisions = (
        set(
            all_names
        )
        & forbidden
    )


    print()
    print(
        "Challenge V2 composition-name collisions:",
        sorted(
            external_collisions
        ),
    )


    if external_collisions:

        fail(
            "V5 family name reuses Challenge V2 composition name."
        )


    # ========================================================
    # TRAIN / DEV FAMILY LEXICAL SEPARATION
    # ========================================================
    #
    # Family names should not be trivial variants of each other.
    # We use Jaccard overlap of name tokens as an ontology-level
    # warning/failure signal.
    # ========================================================

    suspicious = []


    for train_operator, train_name in train_families:

        a = normalized_words(
            train_name
        )

        for dev_operator, dev_name in dev_families:

            if (
                train_operator
                != dev_operator
            ):
                continue

            b = normalized_words(
                dev_name
            )

            union = (
                a | b
            )

            score = (
                len(
                    a & b
                )
                / len(
                    union
                )
                if union
                else 0.0
            )

            if score >= 0.67:

                suspicious.append(
                    (
                        train_operator,
                        train_name,
                        dev_name,
                        score,
                    )
                )


    print()
    print(
        "suspicious same-operator train/dev "
        "name similarities:",
        len(
            suspicious
        ),
    )


    for item in suspicious:

        print(
            " ",
            item,
        )


    if suspicious:

        fail(
            "Held-out development family names are too "
            "lexically close to training family names."
        )


    # ========================================================
    # SOURCE SAFETY
    # ========================================================
    #
    # V5 specification code must not read Challenge V2.
    # The only permitted textual reference is the constant
    # forbidden-name list in this file.
    # ========================================================

    source = SPEC_PATH.read_text()

    forbidden_io_tokens = [
        "sentinel_challenge_v2.json",
        "FROZEN_SHA256.txt",
        "json.load(",
        "json.loads(",
        "read_text(",
        ".open(",
    ]


    source_violations = [
        token
        for token in forbidden_io_tokens
        if token in source
    ]


    print()
    print(
        "spec-layer Challenge-read violations:",
        source_violations,
    )


    if source_violations:

        fail(
            "V5 family-spec module appears capable of reading "
            "Challenge V2."
        )


    # ========================================================
    # VERDICT
    # ========================================================

    print()
    print("=" * 88)
    print(
        "ONTOLOGY AUDIT VERDICT"
    )
    print("=" * 88)


    if ERRORS:

        print(
            "FAILURES:",
            len(
                ERRORS
            ),
        )


        for error in ERRORS:

            print(
                "FAIL:",
                error,
            )


        print()
        print(
            "V5 FAMILY ONTOLOGY: FAIL"
        )

        sys.exit(1)


    print(
        "FAILURES: 0"
    )

    print()
    print(
        "V5 FAMILY ONTOLOGY: PASS"
    )

    print(
        "32 train families + 8 held-out development families."
    )

    print(
        "No V5 training examples have been generated."
    )


if __name__ == "__main__":
    main()
