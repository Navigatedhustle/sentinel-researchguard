from __future__ import annotations

from pathlib import Path
import ast
import json
import sys


if len(
    sys.argv
) != 3:

    raise SystemExit(
        "usage: audit_blind_harness_v1.py EVALUATOR POLICY"
    )


evaluator_path = Path(
    sys.argv[
        1
    ]
)

policy_path = Path(
    sys.argv[
        2
    ]
)


source = evaluator_path.read_text()

tree = ast.parse(
    source
)

policy = json.loads(
    policy_path.read_text()
)

issues = []


# =====================================================================
# PROHIBITED BENCHMARK-INTERNAL REFERENCES
# =====================================================================

forbidden_terms = [
    "audit_only",
    "hidden_semantics",
    "semantic_state",
    "target_index",
    "world_index",
    "question_hash",
    "evidence_hash",
    "relational_system",
    "target_evidence_ref",
    "target_entity",
]


for term in forbidden_terms:

    if term in source:

        issues.append(
            "forbidden benchmark-internal term: "
            + term
        )


# =====================================================================
# PROHIBITED TRAINING / ADAPTATION OPERATIONS
# =====================================================================

prohibited_calls = {
    "backward",
    "step",
    "fit",
    "train",
    "save_pretrained",
    "merge_and_unload",
}


for node in ast.walk(
    tree
):

    if not isinstance(
        node,
        ast.Call,
    ):
        continue


    function_name = None


    if isinstance(
        node.func,
        ast.Name,
    ):

        function_name = node.func.id


    elif isinstance(
        node.func,
        ast.Attribute,
    ):

        function_name = node.func.attr


    if function_name in prohibited_calls:

        issues.append(
            "prohibited adaptive/write call: "
            + str(
                function_name
            )
        )


# =====================================================================
# GOLD ACCESS MUST BE SCORING-ONLY
# =====================================================================

class GoldVisitor(ast.NodeVisitor):

    def __init__(self):
        self.function_stack = []
        self.accesses = []


    def visit_FunctionDef(
        self,
        node,
    ):

        self.function_stack.append(
            node.name
        )

        self.generic_visit(
            node
        )

        self.function_stack.pop()


    def visit_Subscript(
        self,
        node,
    ):

        slice_node = node.slice

        key = None


        if isinstance(
            slice_node,
            ast.Constant,
        ):

            key = slice_node.value


        if key == "gold":

            current_function = (
                self.function_stack[
                    -1
                ]
                if self.function_stack
                else None
            )

            self.accesses.append(
                (
                    current_function,
                    node.lineno,
                )
            )


        self.generic_visit(
            node
        )


visitor = GoldVisitor()

visitor.visit(
    tree
)


if not visitor.accesses:

    issues.append(
        "no gold access found for final scoring"
    )


for function_name, lineno in (
    visitor.accesses
):

    if function_name != "score_outputs":

        issues.append(
            "gold accessed outside score_outputs "
            f"at line {lineno}"
        )


# =====================================================================
# MODEL-FACING FIELD ACCESS
# =====================================================================

def find_function(name):

    for node in tree.body:

        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name
            == name
        ):

            return node


    return None


def string_subscript_keys(node):

    keys = set()


    for child in ast.walk(
        node
    ):

        if not isinstance(
            child,
            ast.Subscript,
        ):

            continue


        slice_node = child.slice


        if (
            isinstance(
                slice_node,
                ast.Constant,
            )
            and isinstance(
                slice_node.value,
                str,
            )
        ):

            keys.add(
                slice_node.value
            )


    return keys


blind_loader = find_function(
    "load_blind_cases"
)

renderer = find_function(
    "render_case"
)


if blind_loader is None:

    issues.append(
        "load_blind_cases missing"
    )

else:

    loader_keys = string_subscript_keys(
        blind_loader
    )

    allowed_loader_keys = {
        "case_id",
        "instruction",
        "input",
    }

    unexpected = (
        loader_keys
        - allowed_loader_keys
    )


    if unexpected:

        issues.append(
            "blind loader accesses unexpected case fields: "
            + repr(
                sorted(
                    unexpected
                )
            )
        )


if renderer is None:

    issues.append(
        "render_case missing"
    )

else:

    render_keys = string_subscript_keys(
        renderer
    )

    allowed_render_keys = {
        "instruction",
        "input",
    }

    unexpected = (
        render_keys
        - allowed_render_keys
    )


    if unexpected:

        issues.append(
            "render_case accesses unexpected fields: "
            + repr(
                sorted(
                    unexpected
                )
            )
        )


# =====================================================================
# DECODING POLICY GATES
# =====================================================================

expected_policy = {
    "batch_size": 1,
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 256,
    "pad_token_id": 128009,
}


for key, expected in (
    expected_policy.items()
):

    if policy.get(
        key
    ) != expected:

        issues.append(
            "decoding policy mismatch: "
            + key
        )


if policy.get(
    "case_order"
) != "canonical_file_order":

    issues.append(
        "canonical case order is not frozen"
    )


isolation = policy.get(
    "case_isolation",
    {}
)


for key in [
    "batch_size_one",
    "no_prior_output_in_prompt",
    "no_cross_case_cache",
    "no_adaptation",
]:

    if isolation.get(
        key
    ) is not True:

        issues.append(
            "case-isolation rule missing: "
            + key
        )


# =====================================================================
# AUTHORIZATION + OFFLINE TRIPWIRES
# =====================================================================

required_source_fragments = [
    "--authorization-file",
    "ledger_sha256",
    "AUTHORIZED_FOR_INFERENCE",
    "local_files_only=True",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
]


for fragment in required_source_fragments:

    if fragment not in source:

        issues.append(
            "required evaluator tripwire missing: "
            + fragment
        )


# =====================================================================
# REPORT
# =====================================================================

print(
    "Gold accesses:",
    visitor.accesses,
)

print(
    "Frozen decoding:",
    {
        "batch_size":
            policy.get(
                "batch_size"
            ),

        "do_sample":
            policy.get(
                "do_sample"
            ),

        "num_beams":
            policy.get(
                "num_beams"
            ),

        "max_new_tokens":
            policy.get(
                "max_new_tokens"
            ),

        "pad_token_id":
            policy.get(
                "pad_token_id"
            ),

        "eos_token_id":
            policy.get(
                "eos_token_id"
            ),
    },
)

print(
    "Issues:",
    len(
        issues
    ),
)


if issues:

    for issue in issues:

        print(
            "FAIL:",
            issue,
        )

    raise SystemExit(
        2
    )


print()
print(
    "BLIND HARNESS STATIC AUDIT: PASS"
)

print(
    "Model-facing case fields: instruction + input only"
)

print(
    "Gold access: score_outputs only"
)

print(
    "Benchmark-internal metadata access: 0"
)

print(
    "Adaptive model operations: 0"
)

print(
    "Mandatory authorization tripwire: PRESENT"
)

print(
    "Offline/local-only model loading: REQUIRED"
)
