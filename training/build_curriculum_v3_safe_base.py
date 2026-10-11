import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
PARAPHRASES = 20

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "training/datasets"
OUT.mkdir(parents=True, exist_ok=True)

INSTRUCTION = """You are a policy-governance decision model.

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

Return valid JSON only."""

QUESTION_PATTERNS = [
    "Is it permitted to {core}?",
    "Does the policy allow someone to {core}?",
    "Based only on this evidence, may a researcher {core}?",
    "Would it comply with the supplied policy to {core}?",
    "Does the evidence directly support permission to {core}?",
    "Under these rules, can someone {core}?",
    "May a researcher {core}?",
    "Is there policy support for someone to {core}?",
    "Can I conclude that someone may {core}?",
    "Would this action be allowed: {core}?",
    "Does this rule permit the action: {core}?",
    "According to only these passages, may someone {core}?",
    "What is the policy decision on whether someone may {core}?",
    "Should Sentinel treat it as permitted to {core}?",
    "Do these passages establish that someone may {core}?",
    "Would the evidence authorize a researcher to {core}?",
    "Is there direct evidence that permits someone to {core}?",
    "Under the supplied evidence, is someone allowed to {core}?",
    "Can the action '{core}' be treated as permitted?",
    "What decision follows from the evidence about whether someone may {core}?",
]

ASSETS = [
    "human-subject interview transcripts",
    "student advising records",
    "unpublished research manuscripts",
    "genomic sequencing files",
    "sponsor-confidential prototype designs",
]

DISTRACTORS = [
    (
        "training_materials_policy.txt",
        "GENERAL",
        "Public training materials may be distributed internally."
    ),
    (
        "account_security_policy.txt",
        "GENERAL",
        "Institutional accounts must use multifactor authentication for remote access."
    ),
    (
        "publication_policy.txt",
        "GENERAL",
        "Published institutional web pages must follow accessibility requirements."
    ),
    (
        "device_inventory_policy.txt",
        "GENERAL",
        "Institution-owned devices must remain listed in the central inventory."
    ),
]

# evidence tuple:
# source, section, text, decisive

ARCHETYPES = {

"YES": [

{
"name": "direct_public",
"core":
"analyze the publicly released version of {asset} with approved institutional software",
"conditions": [],
"evidence": [
(
"public_data_policy.txt",
"PUBLIC RELEASES",
"Publicly released versions of {asset} may be analyzed with approved institutional software.",
True
),
(
"research_access_policy.txt",
"RESTRICTED MATERIALS",
"Restricted versions of {asset} require project authorization before access.",
False
)
]
},

{
"name": "authorization_satisfied",
"core":
"process restricted {asset} on managed local AI after written project authorization has been recorded",
"conditions": [],
"evidence": [
(
"local_ai_policy.txt",
"AUTHORIZED LOCAL PROCESSING",
"Restricted {asset} may be processed on institutionally managed local AI when written project authorization is recorded.",
True
),
(
"external_services_policy.txt",
"THIRD-PARTY SERVICES",
"Restricted {asset} may not be submitted to unapproved third-party services.",
False
)
]
},

{
"name": "multicondition_satisfied",
"core":
"share {asset} with an approved collaborator after the governing agreement permits sharing and the collaborator has been added to the authorized access roster",
"conditions": [],
"evidence": [
(
"collaboration_policy.txt",
"AUTHORIZED SHARING",
"{Asset} may be shared with approved collaborators only when the governing agreement permits the sharing.",
True
),
(
"access_roster_policy.txt",
"COLLABORATOR ACCESS",
"A collaborator must be listed on the authorized access roster before receiving project data.",
True
),
(
"publication_policy.txt",
"PUBLIC OUTPUTS",
"Aggregate public findings may be published after normal scholarly review.",
False
)
]
},

{
"name": "exception_satisfied",
"core":
"use an approved deidentified export of {asset} externally after the export has passed the required re-identification review",
"conditions": [],
"evidence": [
(
"external_export_policy.txt",
"APPROVED EXCEPTION",
"External use of {asset} is prohibited except for an approved deidentified export that has passed the required re-identification review.",
True
),
(
"classification_policy.txt",
"RESTRICTED SOURCE DATA",
"The original restricted {asset} remain subject to institutional controls.",
False
)
]
},

],

"NO": [

{
"name": "unapproved_external",
"core":
"upload restricted {asset} to an unapproved external generative AI service",
"conditions": [],
"evidence": [
(
"external_ai_policy.txt",
"UNAPPROVED SERVICES",
"Restricted {asset} must not be uploaded to an unapproved external generative AI service.",
True
),
(
"local_ai_policy.txt",
"LOCAL SYSTEMS",
"Approved local systems may process {asset} when project requirements are met.",
False
)
]
},

{
"name": "deidentification_not_automatic",
"core":
"assume that removing names automatically makes restricted {asset} unrestricted",
"conditions": [],
"evidence": [
(
"deidentification_policy.txt",
"CLASSIFICATION AFTER DEIDENTIFICATION",
"Removing direct names from restricted {asset} does not by itself make the data unrestricted.",
True
),
(
"privacy_guidance.txt",
"INDIRECT IDENTIFIERS",
"Indirect identifiers and linkage information may still create re-identification risk.",
False
)
]
},

{
"name": "skip_human_review",
"core":
"release AI-generated conclusions derived from {asset} without qualified human review",
"conditions": [],
"evidence": [
(
"ai_output_policy.txt",
"HUMAN REVIEW",
"AI-generated conclusions derived from {asset} must receive qualified human review before release.",
True
),
(
"model_logging_policy.txt",
"LOGGING",
"Approved AI systems must retain normal operational logs.",
False
)
]
},

{
"name": "local_not_third_party",
"core":
"treat processing {asset} on institutionally managed local compute as a transfer to a third-party processor when the data remains inside the approved environment",
"conditions": [],
"evidence": [
(
"data_transfer_policy.txt",
"LOCAL COMPUTE",
"Processing {asset} on institutionally managed local compute is not a transfer to a third-party processor when the data remains inside the approved environment.",
True
),
(
"vendor_policy.txt",
"EXTERNAL PROCESSORS",
"A third-party processor is an external organization receiving project data for processing.",
False
)
]
},

],

"CONDITIONAL": [

{
"name": "authorization_unresolved",
"core":
"process restricted {asset} on institutionally managed local AI",
"conditions": [
"written project authorization must be recorded"
],
"evidence": [
(
"local_ai_policy.txt",
"AUTHORIZED LOCAL PROCESSING",
"Restricted {asset} may be processed on institutionally managed local AI when written project authorization is recorded.",
True
),
(
"external_ai_policy.txt",
"EXTERNAL SERVICES",
"Unapproved external AI services may not receive restricted {asset}.",
False
)
]
},

{
"name": "agreement_protocol",
"core":
"share a deidentified version of {asset} with an external collaborator",
"conditions": [
"the governing agreement must permit the sharing",
"the project protocol must permit the sharing"
],
"evidence": [
(
"data_sharing_policy.txt",
"DEIDENTIFIED SHARING",
"A deidentified version of {asset} may be shared with an external collaborator only if the governing agreement permits the sharing.",
True
),
(
"protocol_policy.txt",
"PROJECT PROTOCOL",
"External project sharing must also remain consistent with the approved project protocol.",
True
),
(
"publication_policy.txt",
"AGGREGATE RESULTS",
"Aggregate public results may be published through normal scholarly channels.",
False
)
]
},

{
"name": "review_approval",
"core":
"use an AI-generated recommendation derived from {asset} in an operational decision",
"conditions": [
"qualified human review must be completed",
"documented project approval must be recorded"
],
"evidence": [
(
"ai_decision_policy.txt",
"OPERATIONAL USE",
"An AI-generated recommendation derived from {asset} may be used in an operational decision only after qualified human review.",
True
),
(
"project_approval_policy.txt",
"DOCUMENTED APPROVAL",
"Operational use of an AI-generated recommendation also requires documented project approval.",
True
),
(
"model_testing_policy.txt",
"TEST ENVIRONMENT",
"Prototype models may be evaluated in an approved test environment.",
False
)
]
},

{
"name": "external_deidentified",
"core":
"send a deidentified export of {asset} to an external analysis service",
"conditions": [
"a re-identification assessment must be completed",
"the governing contract must allow external processing"
],
"evidence": [
(
"deidentified_export_policy.txt",
"EXTERNAL ANALYSIS",
"A deidentified export of {asset} may be sent to an external analysis service after a re-identification assessment is completed.",
True
),
(
"contract_policy.txt",
"EXTERNAL PROCESSING",
"External processing is permitted only when the governing contract allows it.",
True
),
(
"local_processing_policy.txt",
"LOCAL ANALYSIS",
"Institutionally managed local analysis does not constitute external processing.",
False
)
]
},

],

"INSUFFICIENT": [

{
"name": "specific_approver",
"core":
"require written approval from the department chair before local processing of {asset}",
"conditions": [],
"evidence": [
(
"authorization_policy.txt",
"LOCAL PROCESSING",
"Local processing of {asset} requires documented authorization before work begins.",
False
),
(
"research_roles_policy.txt",
"PROJECT ROLES",
"Project personnel must perform only activities assigned to their approved roles.",
False
),
(
"department_policy.txt",
"DEPARTMENT RECORDS",
"Departments maintain normal administrative records for active projects.",
False
)
]
},

{
"name": "specific_encryption",
"core":
"require AES-256 encryption for storage of {asset}",
"conditions": [],
"evidence": [
(
"security_controls_policy.txt",
"STORAGE SECURITY",
"Storage systems containing {asset} must use approved security controls and encryption where required by the project.",
False
),
(
"access_control_policy.txt",
"ACCESS",
"Access to restricted project data must be limited to authorized personnel.",
False
),
(
"network_policy.txt",
"REMOTE ACCESS",
"Remote institutional access must use approved authentication controls.",
False
)
]
},

{
"name": "usb",
"core":
"store {asset} on a personally owned USB drive",
"conditions": [],
"evidence": [
(
"local_compute_policy.txt",
"APPROVED COMPUTE",
"{Asset} may be processed on approved institutionally managed local compute.",
False
),
(
"data_classification_policy.txt",
"RESTRICTED DATA",
"Restricted {asset} must remain within approved project controls.",
False
),
(
"device_inventory_policy.txt",
"INSTITUTION DEVICES",
"Institution-owned devices must remain listed in the central inventory.",
False
)
]
},

{
"name": "exact_retention",
"core":
"retain {asset} for exactly seven years",
"conditions": [],
"evidence": [
(
"retention_policy.txt",
"PROJECT RETENTION",
"{Asset} must be retained according to the approved project retention schedule.",
False
),
(
"closeout_policy.txt",
"PROJECT CLOSEOUT",
"Project teams must complete normal closeout procedures when work ends.",
False
),
(
"archive_policy.txt",
"ARCHIVES",
"Approved archival materials may be transferred to institutional archives.",
False
)
]
},

]
}

ANSWERS = {

"YES": [
"The supplied evidence directly permits the action and the question already satisfies every stated material condition.",
"The action is expressly permitted with no unresolved material requirement.",
"The evidence directly authorizes the action as described."
],

"NO": [
"The supplied evidence directly prohibits the action or establishes that the proposition is false.",
"The evidence directly supports a NO decision for the exact issue asked about.",
"The requested action or proposition conflicts with the supplied evidence."
],

"CONDITIONAL": [
"The action may be permitted, but material policy conditions remain unresolved.",
"Permission is conditional because stated requirements are not established as satisfied.",
"The evidence allows the action only subject to unresolved requirements."
],

"INSUFFICIENT": [
"The evidence addresses related requirements but does not directly govern the exact issue.",
"The supplied passages do not establish whether the exact action is permitted, prohibited, or conditional.",
"Answering the exact question would require an unstated inference beyond the supplied evidence."
]
}


def fmt(text, asset):
    return text.format(
        asset=asset,
        Asset=asset.capitalize()
    )


def choose_split(archetype_idx, asset_idx):

    # Every archetype:
    # 3 domain scenarios train
    # 1 domain scenario validation
    # 1 domain scenario test

    val_idx = archetype_idx % len(ASSETS)
    test_idx = (
        archetype_idx + 1
    ) % len(ASSETS)

    if asset_idx == val_idx:
        return "validation"

    if asset_idx == test_idx:
        return "test"

    return "train"


def make_row(
    decision,
    archetype,
    archetype_idx,
    asset,
    asset_idx,
    pattern_idx
):

    scenario_id = (
        f"{decision.lower()}_"
        f"{archetype['name']}_"
        f"{asset_idx}"
    )

    rng = random.Random(
        f"{SEED}|{scenario_id}|{pattern_idx}"
    )

    evidence = []

    for (
        source,
        section,
        text,
        decisive
    ) in archetype["evidence"]:

        evidence.append({
            "source": source,
            "section": section,
            "text": fmt(text, asset),
            "decisive": decisive
        })

    source, section, text = (
        DISTRACTORS[
            pattern_idx
            % len(DISTRACTORS)
        ]
    )

    evidence.append({
        "source": source,
        "section": section,
        "text": text,
        "decisive": False
    })

    # Important:
    # decisive evidence is not always S1.
    rng.shuffle(evidence)

    blocks = []
    citations = []

    for i, item in enumerate(
        evidence,
        start=1
    ):

        label = f"S{i}"

        blocks.append(
            f"[{label}]\n"
            f"SOURCE: {item['source']}\n"
            f"SECTION: {item['section']}\n"
            f"TEXT: {item['text']}"
        )

        if item["decisive"]:
            citations.append(label)

    # INSUFFICIENT should not cite a passage
    # as affirmative support.
    if decision == "INSUFFICIENT":
        citations = []

    core = fmt(
        archetype["core"],
        asset
    )

    question = (
        QUESTION_PATTERNS[pattern_idx]
        .format(core=core)
    )

    output = {
        "evidence_sufficient":
            decision != "INSUFFICIENT",

        "decision":
            decision,

        "conditions":
            archetype["conditions"]
            if decision == "CONDITIONAL"
            else [],

        "answer":
            rng.choice(
                ANSWERS[decision]
            ),

        "citations":
            citations
    }

    return {
        "scenario_id":
            scenario_id,

        "archetype":
            archetype["name"],

        "instruction":
            INSTRUCTION,

        "input":
            f"QUESTION:\n{question}\n\n"
            f"EVIDENCE:\n"
            + "\n\n".join(blocks),

        "output":
            json.dumps(
                output,
                separators=(",", ":")
            )
    }


def main():

    splits = {
        "train": [],
        "validation": [],
        "test": []
    }

    scenario_sets = {
        "train": set(),
        "validation": set(),
        "test": set()
    }

    for (
        decision,
        archetypes
    ) in ARCHETYPES.items():

        for (
            archetype_idx,
            archetype
        ) in enumerate(archetypes):

            for (
                asset_idx,
                asset
            ) in enumerate(ASSETS):

                split = choose_split(
                    archetype_idx,
                    asset_idx
                )

                scenario_id = (
                    f"{decision.lower()}_"
                    f"{archetype['name']}_"
                    f"{asset_idx}"
                )

                scenario_sets[
                    split
                ].add(
                    scenario_id
                )

                for pattern_idx in range(
                    PARAPHRASES
                ):

                    splits[
                        split
                    ].append(
                        make_row(
                            decision,
                            archetype,
                            archetype_idx,
                            asset,
                            asset_idx,
                            pattern_idx
                        )
                    )

    # Deterministic row shuffle.
    for offset, split in enumerate(
        [
            "train",
            "validation",
            "test"
        ]
    ):

        rng = random.Random(
            SEED + offset
        )

        rng.shuffle(
            splits[split]
        )

    # Critical leakage checks.
    assert (
        scenario_sets["train"]
        .isdisjoint(
            scenario_sets[
                "validation"
            ]
        )
    )

    assert (
        scenario_sets["train"]
        .isdisjoint(
            scenario_sets["test"]
        )
    )

    assert (
        scenario_sets["validation"]
        .isdisjoint(
            scenario_sets["test"]
        )
    )

    paths = {

        "train":
            OUT /
            "sentinel_train_v3.jsonl",

        "validation":
            OUT /
            "sentinel_validation_v3.jsonl",

        "test":
            OUT /
            "sentinel_test_v3.jsonl"
    }

    for split, path in paths.items():

        with path.open("w") as f:

            for row in splits[split]:

                f.write(
                    json.dumps(row)
                    + "\n"
                )

    print(
        "=== SENTINEL "
        "CURRICULUM V3 ==="
    )

    for split in [
        "train",
        "validation",
        "test"
    ]:

        labels = Counter(
            json.loads(
                row["output"]
            )["decision"]
            for row
            in splits[split]
        )

        print(
            f"{split:10s} "
            f"rows="
            f"{len(splits[split]):4d} "
            f"scenarios="
            f"{len(scenario_sets[split]):2d} "
            f"labels="
            f"{dict(sorted(labels.items()))}"
        )

    print(
        "\nTOTAL:",
        sum(
            len(x)
            for x in splits.values()
        )
    )

    print(
        "\nSample TEST row:"
    )

    print(
        json.dumps(
            splits["test"][0],
            indent=2
        )
    )


if __name__ == "__main__":
    main()
