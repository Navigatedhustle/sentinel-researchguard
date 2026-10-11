import json
import random
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "training/datasets"

INSTRUCTION = """
You are a policy-governance decision model.

Use ONLY the supplied evidence.

Classify the exact question as one of:

YES
NO
CONDITIONAL
INSUFFICIENT

YES:
The evidence directly permits the action and every material condition
is already satisfied by the question.

NO:
The evidence directly prohibits the action or directly establishes
that the proposition is false.

CONDITIONAL:
The action may be permitted, but one or more stated requirements,
approvals, authorizations, reviews, or restrictions remain unresolved.

INSUFFICIENT:
The evidence does not directly govern the exact issue. Related evidence
is not enough. Silence is not prohibition.

Return valid JSON only.
""".strip()


QUESTION_PATTERNS = [
    "May {action}?",
    "Is it permitted to {action}?",
    "Does the policy allow me to {action}?",
    "Can someone {action}?",
    "Would the policy permit someone to {action}?",
    "Is {action} allowed under this policy?",
    "Based only on this evidence, may someone {action}?",
    "Does this evidence authorize someone to {action}?",
    "Would it comply with policy to {action}?",
    "Is there policy support to {action}?",
    "Can I conclude that someone may {action}?",
    "Does this rule permit the action to {action}?",
    "Under the supplied evidence, may someone {action}?",
    "Does the evidence clearly allow someone to {action}?",
    "Would someone be authorized to {action}?",
    "What is the decision on whether someone may {action}?"
]


YES = [
    {
        "action": "publish records formally classified as public",
        "evidence": "Records formally classified as public may be published externally.",
        "answer": "The evidence directly permits publication of records classified as public."
    },
    {
        "action": "use an approved external service for public documents",
        "evidence": "Public documents may be processed using approved external services.",
        "answer": "The supplied rule directly permits approved external services for public documents."
    },
    {
        "action": "share aggregate statistics containing no individual-level information",
        "evidence": "Aggregate statistics containing no individual-level information may be shared without further approval.",
        "answer": "The rule directly permits sharing these aggregate statistics."
    },
    {
        "action": "copy unrestricted documentation to the approved public repository",
        "evidence": "Unrestricted documentation may be copied to the approved public repository.",
        "answer": "The requested action is directly permitted."
    },
    {
        "action": "process public benchmark data on an approved cloud platform",
        "evidence": "Public benchmark data may be processed on approved cloud platforms.",
        "answer": "The evidence expressly permits this processing."
    },
    {
        "action": "send publicly released reports to an approved collaborator",
        "evidence": "Publicly released reports may be sent to approved collaborators.",
        "answer": "The action is expressly permitted by the supplied evidence."
    },
    {
        "action": "archive nonrestricted project summaries in the public archive",
        "evidence": "Nonrestricted project summaries may be archived in the public archive.",
        "answer": "The evidence directly permits the requested archival action."
    },
    {
        "action": "use approved visualization software for public datasets",
        "evidence": "Public datasets may be analyzed with approved visualization software.",
        "answer": "The evidence directly permits the requested analysis."
    }
]


NO = [
    {
        "action": "send restricted records to an unapproved cloud service",
        "evidence": "Restricted records must not be transmitted to unapproved cloud services.",
        "answer": "The evidence directly prohibits transmission to an unapproved cloud service."
    },
    {
        "action": "rely on an automated risk assessment without human review",
        "evidence": "Automated risk assessments must be reviewed by a qualified person before they are relied upon.",
        "answer": "The evidence establishes that the assessment cannot be relied upon without human review."
    },
    {
        "action": "treat coded records as automatically unrestricted",
        "evidence": "Replacing direct identifiers with codes does not automatically make records unrestricted.",
        "answer": "The evidence directly establishes that coded records are not automatically unrestricted."
    },
    {
        "action": "share authentication credentials with an external vendor",
        "evidence": "Authentication credentials must not be shared with external vendors.",
        "answer": "The requested sharing is expressly prohibited."
    },
    {
        "action": "upload confidential designs to an unapproved AI service",
        "evidence": "Confidential designs must not be submitted to unapproved generative AI services.",
        "answer": "The requested upload is prohibited."
    },
    {
        "action": "grant an unauthorized collaborator access to restricted files",
        "evidence": "Restricted files may not be accessed by unauthorized collaborators.",
        "answer": "The evidence expressly prohibits the requested access."
    },
    {
        "action": "publish participant-level records identified as restricted",
        "evidence": "Participant-level records classified as restricted may not be publicly released.",
        "answer": "The evidence directly prohibits public release."
    },
    {
        "action": "assume removal of names eliminates all re-identification risk",
        "evidence": "Removal of names does not establish that re-identification risk has been eliminated.",
        "answer": "The proposition is directly contradicted by the evidence."
    }
]


CONDITIONAL = [
    {
        "action": "process restricted files on local compute",
        "evidence": "Restricted files may be processed locally when the system is approved for the data classification and access controls are enabled.",
        "conditions": [
            "system must be approved for the data classification",
            "access controls must be enabled"
        ]
    },
    {
        "action": "share de-identified research results externally",
        "evidence": "De-identified research results may be shared externally provided the applicable protocol and contractual restrictions permit sharing.",
        "conditions": [
            "applicable protocol must permit sharing",
            "contractual restrictions must permit sharing"
        ]
    },
    {
        "action": "use an external processor for internal records",
        "evidence": "Internal records may be processed externally only after the processing environment has been approved.",
        "conditions": [
            "processing environment must be approved"
        ]
    },
    {
        "action": "provide a collaborator access to controlled project files",
        "evidence": "Controlled project files may be shared with collaborators who have been specifically authorized.",
        "conditions": [
            "collaborator must be specifically authorized"
        ]
    },
    {
        "action": "move regulated files to a new computing environment",
        "evidence": "Regulated files may be moved to a new computing environment after security review and written authorization.",
        "conditions": [
            "security review must be completed",
            "written authorization must be obtained"
        ]
    },
    {
        "action": "use an AI-generated analysis in a final report",
        "evidence": "AI-generated analysis may be used in a final report after qualified human review.",
        "conditions": [
            "qualified human review must occur"
        ]
    },
    {
        "action": "release derived statistics from controlled data",
        "evidence": "Derived statistics may be released if they cannot reasonably identify an individual and the project agreement permits release.",
        "conditions": [
            "statistics must not reasonably identify an individual",
            "project agreement must permit release"
        ]
    },
    {
        "action": "run restricted workloads on institutionally managed AI",
        "evidence": "Restricted workloads may run on institutionally managed AI when authorization, security controls, and applicable project requirements are satisfied.",
        "conditions": [
            "authorization must be satisfied",
            "security controls must be satisfied",
            "applicable project requirements must be satisfied"
        ]
    }
]


INSUFFICIENT = [
    {
        "action": "store controlled files on a personally owned USB drive",
        "evidence": "Controlled files may be analyzed on approved institutional computing systems.",
        "answer": "The evidence discusses approved computing systems but does not govern personally owned USB storage."
    },
    {
        "action": "use AES-256 encryption for every restricted file",
        "evidence": "Restricted files must be protected using approved security controls.",
        "answer": "The evidence requires approved security controls but does not specify AES-256."
    },
    {
        "action": "delete all project records exactly seven years after completion",
        "evidence": "Project records must be managed according to applicable retention requirements.",
        "answer": "The evidence does not specify a seven-year retention period."
    },
    {
        "action": "obtain approval specifically from the vice president",
        "evidence": "External processing requires institutional approval.",
        "answer": "The evidence requires approval but does not identify the vice president as the required approver."
    },
    {
        "action": "store confidential information on a personal smartphone",
        "evidence": "Confidential information must not be submitted to unapproved external AI services.",
        "answer": "The evidence addresses external AI services, not personal smartphone storage."
    },
    {
        "action": "use biometric authentication on every restricted system",
        "evidence": "Restricted systems must enforce approved access controls.",
        "answer": "The evidence requires approved access controls but does not specifically require biometric authentication."
    },
    {
        "action": "retain backup copies for exactly ten years",
        "evidence": "Backup copies remain subject to the organization's data governance requirements.",
        "answer": "The evidence does not specify a ten-year retention requirement."
    },
    {
        "action": "receive written approval from the department chair before local processing",
        "evidence": "Local processing is permitted on approved systems when authorization requirements are satisfied.",
        "answer": "The evidence does not identify the department chair or require written approval from that role."
    }
]


def make_output(label, scenario):
    sufficient = label != "INSUFFICIENT"

    result = {
        "evidence_sufficient": sufficient,
        "decision": label,
        "conditions": scenario.get("conditions", []),
        "answer": scenario.get(
            "answer",
            "The action may be permitted, but one or more material conditions remain."
        ),
        "citations": ["S1"] if sufficient else []
    }

    return json.dumps(
        result,
        ensure_ascii=False,
        separators=(",", ":")
    )


def build_example(label, scenario, question):
    # Add a distractor to teach evidence discipline.
    distractor = (
        "[S2]\n"
        "SOURCE: unrelated_policy.txt\n"
        "TEXT: Public training materials may be distributed internally."
    )

    inp = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n"
        f"[S1]\nSOURCE: policy.txt\nTEXT: {scenario['evidence']}\n\n"
        f"{distractor}"
    )

    return {
        "instruction": INSTRUCTION,
        "input": inp,
        "output": make_output(label, scenario)
    }


def generate_class(label, scenarios):

    data = []

    # Scenario-level split:
    # 0-5 training
    # 6 validation
    # 7 held-out test
    splits = {
        "train": scenarios[:6],
        "validation": scenarios[6:7],
        "test": scenarios[7:8]
    }

    for split, subset in splits.items():

        for scenario in subset:

            for pattern in QUESTION_PATTERNS:

                question = pattern.format(
                    action=scenario["action"]
                )

                data.append((
                    split,
                    build_example(
                        label,
                        scenario,
                        question
                    )
                ))

    return data


all_rows = []

for label, scenarios in [
    ("YES", YES),
    ("NO", NO),
    ("CONDITIONAL", CONDITIONAL),
    ("INSUFFICIENT", INSUFFICIENT)
]:
    all_rows.extend(
        generate_class(label, scenarios)
    )


for split in ["train", "validation", "test"]:

    rows = [
        item
        for row_split, item in all_rows
        if row_split == split
    ]

    random.shuffle(rows)

    path = OUT / f"sentinel_{split}_v1.jsonl"

    with path.open("w") as f:
        for item in rows:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )

    counts = {}

    for item in rows:
        label = json.loads(
            item["output"]
        )["decision"]

        counts[label] = (
            counts.get(label, 0) + 1
        )

    print(
        f"{split:10s} "
        f"{len(rows):3d} examples | "
        f"{counts}"
    )

print("\nDataset written to:")
print(OUT)
