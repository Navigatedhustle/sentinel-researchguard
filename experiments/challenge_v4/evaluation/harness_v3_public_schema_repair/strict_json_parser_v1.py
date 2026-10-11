from __future__ import annotations

import json


LABELS = {
    "YES",
    "NO",
    "CONDITIONAL",
    "INSUFFICIENT",
}

EXPECTED_KEYS = {
    "decision",
    "conditions",
    "citations",
    "rationale",
}


def parse_completion(raw_completion):
    """
    Strict Challenge V3 parser.

    The entire stripped completion must be one JSON value.

    Explicitly prohibited:
    - markdown-fence stripping
    - JSON substring extraction
    - brace recovery
    - malformed-JSON repair
    - label guessing
    - case normalization of labels

    Decision validity and full-contract validity are tracked
    independently.
    """

    if not isinstance(
        raw_completion,
        str,
    ):
        raise TypeError(
            "raw_completion must be str"
        )

    text = raw_completion.strip()

    result = {
        "json_valid": False,
        "decision_valid": False,
        "contract_valid": False,
        "decision": None,
        "parsed_json": None,
        "error": None,
    }

    if not text:
        result["error"] = "empty_completion"
        return result

    try:
        parsed = json.loads(
            text
        )

    except json.JSONDecodeError as exc:
        result["error"] = (
            "json_decode_error:"
            + exc.msg
        )
        return result


    result[
        "json_valid"
    ] = True

    result[
        "parsed_json"
    ] = parsed


    if not isinstance(
        parsed,
        dict,
    ):
        result["error"] = "json_not_object"
        return result


    decision = parsed.get(
        "decision"
    )

    if (
        isinstance(
            decision,
            str,
        )
        and decision in LABELS
    ):
        result[
            "decision"
        ] = decision

        result[
            "decision_valid"
        ] = True

    else:
        result[
            "error"
        ] = "invalid_decision"


    exact_keys = (
        set(
            parsed.keys()
        )
        == EXPECTED_KEYS
    )

    conditions_valid = isinstance(
        parsed.get(
            "conditions"
        ),
        list,
    )

    citations_valid = isinstance(
        parsed.get(
            "citations"
        ),
        list,
    )

    rationale_valid = (
        isinstance(
            parsed.get(
                "rationale"
            ),
            str,
        )
        and bool(
            parsed[
                "rationale"
            ].strip()
        )
    )


    result[
        "contract_valid"
    ] = bool(
        result[
            "decision_valid"
        ]
        and exact_keys
        and conditions_valid
        and citations_valid
        and rationale_valid
    )


    if (
        result[
            "decision_valid"
        ]
        and not result[
            "contract_valid"
        ]
        and result[
            "error"
        ] is None
    ):
        result[
            "error"
        ] = "json_contract_invalid"


    return result
