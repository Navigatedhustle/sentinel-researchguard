from pathlib import Path
from collections import Counter
import hashlib
import json
import random
import re

BASE = Path("/workspace")
OUT = BASE / "training/datasets"

SEED = 20260814
PARAPHRASES = 4

ASSETS = [
    "human-subject interview transcripts",
    "student advising records",
    "genomic sequencing files",
    "unpublished research manuscripts",
    "sponsor-confidential prototype designs",
]

QUESTIONS = [
    "Under the supplied evidence, may someone {core}?",
    "Would the supplied policy permit someone to {core}?",
    "What decision follows from the evidence about whether someone may {core}?",
    "Based only on these passages, is it permitted to {core}?",
]

DISTRACTORS = [
    (
        "device_inventory_policy.txt",
        "GENERAL",
        "Institution-owned devices must remain listed in the central inventory.",
    ),
    (
        "account_security_policy.txt",
        "GENERAL",
        "Institutional accounts must use multifactor authentication for remote access.",
    ),
    (
        "accessibility_policy.txt",
        "GENERAL",
        "Published institutional web pages must follow accessibility requirements.",
    ),
    (
        "training_materials_policy.txt",
        "GENERAL",
        "Public training materials may be distributed internally.",
    ),
]

INSTRUCTION = """You are Sentinel, a policy-governance decision model.

Use ONLY the supplied evidence.

Classify the exact question as one of:

YES
NO
CONDITIONAL
INSUFFICIENT

Apply these rules in order:

INSUFFICIENT:
Choose INSUFFICIENT when the supplied evidence does not directly govern
the exact issue asked about. Related evidence is not enough. Do not invent
missing approvers, technologies, durations, thresholds, locations, or rules.

NO:
Choose NO when the evidence directly prohibits the exact action or directly
establishes that the proposition in the question is false.

CONDITIONAL:
Choose CONDITIONAL when the action may be permitted, but one or more
material requirements remain unresolved.

YES:
Choose YES only when the evidence directly permits the exact action and
every material condition is already established as satisfied by the question.

Silence is not prohibition.
Semantic similarity is not evidence.

Return valid JSON only.
""".strip()


def F(
    name,
    permission,
    prohibition,
    yes,
    conditional,
    no,
    insufficient,
    conditions,
):
    return {
        "name": name,
        "permission": permission,
        "prohibition": prohibition,
        "core": {
            "YES": yes,
            "CONDITIONAL": conditional,
            "NO": no,
            "INSUFFICIENT": insufficient,
        },
        "conditions": conditions,
    }


# --------------------------------------------------------------------
# TRAIN FAMILIES
# --------------------------------------------------------------------

TRAIN = [

    F(
        "project_authorization",
        (
            "local_ai_policy.txt",
            "AUTHORIZED PROCESSING",
            "Restricted {asset} may be processed on institutionally managed "
            "local AI only while written project authorization is active.",
        ),
        (
            "authorization_policy.txt",
            "REVOKED AUTHORIZATION",
            "Restricted {asset} must not be processed after written project "
            "authorization has been denied or revoked.",
        ),
        "process restricted {asset} on institutionally managed local AI "
        "while written project authorization is active",
        "process restricted {asset} on institutionally managed local AI",
        "process restricted {asset} on institutionally managed local AI "
        "after written project authorization has been revoked",
        "require the written authorization for {asset} to be issued by "
        "the university vice president",
        ["written project authorization must be active"],
    ),

    F(
        "qualified_human_review",
        (
            "ai_output_policy.txt",
            "HUMAN REVIEW",
            "AI-generated conclusions derived from {asset} may be released "
            "after qualified human review is completed.",
        ),
        (
            "ai_output_policy.txt",
            "UNREVIEWED OUTPUT",
            "AI-generated conclusions derived from {asset} must not be "
            "released before qualified human review.",
        ),
        "release AI-generated conclusions derived from {asset} after "
        "qualified human review has been completed",
        "release AI-generated conclusions derived from {asset}",
        "release AI-generated conclusions derived from {asset} before "
        "qualified human review",
        "require the qualified reviewer for {asset} to hold a PhD",
        ["qualified human review must be completed"],
    ),

    F(
        "contract_and_dpa",
        (
            "external_processing_policy.txt",
            "APPROVED PROCESSING",
            "An external processor may receive {asset} only when the governing "
            "contract permits processing and an approved data-processing "
            "agreement is active.",
        ),
        (
            "external_processing_policy.txt",
            "PROHIBITED PROCESSING",
            "An external processor must not receive {asset} when the governing "
            "contract prohibits external processing.",
        ),
        "send {asset} to an external processor after the governing contract "
        "permits processing and an approved data-processing agreement is active",
        "send {asset} to an external processor",
        "send {asset} to an external processor when the governing contract "
        "prohibits external processing",
        "require the data-processing agreement for {asset} to be signed by "
        "the university president",
        [
            "the governing contract must permit external processing",
            "an approved data-processing agreement must be active",
        ],
    ),

    F(
        "authorized_access_roster",
        (
            "collaboration_policy.txt",
            "AUTHORIZED COLLABORATORS",
            "{Asset} may be shared with an approved collaborator only when "
            "the collaborator is listed on the authorized access roster.",
        ),
        (
            "collaboration_policy.txt",
            "REMOVED COLLABORATORS",
            "{Asset} must not be shared with a collaborator who has been "
            "removed from the authorized access roster.",
        ),
        "share {asset} with an approved collaborator who is listed on the "
        "authorized access roster",
        "share {asset} with an approved collaborator",
        "share {asset} with a collaborator who has been removed from the "
        "authorized access roster",
        "require every approved collaborator receiving {asset} to be a "
        "full-time university employee",
        ["the collaborator must be listed on the authorized access roster"],
    ),

    F(
        "reidentification_assessment",
        (
            "deidentified_export_policy.txt",
            "EXTERNAL EXPORT",
            "A deidentified export of {asset} may be sent externally after "
            "the required re-identification assessment has passed.",
        ),
        (
            "deidentified_export_policy.txt",
            "FAILED ASSESSMENT",
            "A deidentified export of {asset} must not be sent externally "
            "after failing the required re-identification assessment.",
        ),
        "send a deidentified export of {asset} externally after the required "
        "re-identification assessment has passed",
        "send a deidentified export of {asset} externally",
        "send a deidentified export of {asset} externally after the required "
        "re-identification assessment has failed",
        "require the re-identification assessment for {asset} to use a "
        "specific commercial privacy tool",
        ["the required re-identification assessment must pass"],
    ),

    F(
        "participant_consent_scope",
        (
            "consent_policy.txt",
            "APPROVED USE",
            "{Asset} may be used for a secondary research purpose when that "
            "purpose falls within the participant's documented consent.",
        ),
        (
            "consent_policy.txt",
            "OUTSIDE CONSENT",
            "{Asset} must not be used for a secondary research purpose that "
            "is explicitly outside the participant's documented consent.",
        ),
        "use {asset} for a secondary research purpose that is within the "
        "participant's documented consent",
        "use {asset} for a secondary research purpose",
        "use {asset} for a secondary research purpose explicitly outside "
        "the participant's documented consent",
        "require participant consent covering {asset} to have been signed "
        "with blue ink",
        ["the secondary purpose must fall within documented participant consent"],
    ),

    F(
        "data_residency",
        (
            "data_residency_policy.txt",
            "APPROVED REGION",
            "{Asset} may be processed in an approved geographic region when "
            "the project residency requirement identifies that region.",
        ),
        (
            "data_residency_policy.txt",
            "PROHIBITED REGION",
            "{Asset} must not be processed in a geographic region explicitly "
            "prohibited by the project residency requirement.",
        ),
        "process {asset} in a geographic region identified as approved by "
        "the project residency requirement",
        "process {asset} in another geographic region",
        "process {asset} in a geographic region explicitly prohibited by "
        "the project residency requirement",
        "require {asset} to be processed specifically in the state of Nevada",
        ["the processing region must be approved by the project residency requirement"],
    ),

    F(
        "publication_embargo",
        (
            "publication_policy.txt",
            "POST-EMBARGO RELEASE",
            "Research findings derived from {asset} may be published after "
            "the applicable publication embargo has expired.",
        ),
        (
            "publication_policy.txt",
            "ACTIVE EMBARGO",
            "Research findings derived from {asset} must not be published "
            "while the applicable publication embargo remains active.",
        ),
        "publish research findings derived from {asset} after the applicable "
        "publication embargo has expired",
        "publish research findings derived from {asset}",
        "publish research findings derived from {asset} while the applicable "
        "publication embargo remains active",
        "require the publication embargo for {asset} to last exactly 18 months",
        ["the applicable publication embargo must have expired"],
    ),

    F(
        "license_scope",
        (
            "license_policy.txt",
            "PERMITTED USE",
            "{Asset} may be used for model development when the governing "
            "license expressly permits model-development use.",
        ),
        (
            "license_policy.txt",
            "PROHIBITED USE",
            "{Asset} must not be used for model development when the governing "
            "license expressly excludes model-development use.",
        ),
        "use {asset} for model development when the governing license expressly "
        "permits model-development use",
        "use {asset} for model development",
        "use {asset} for model development when the governing license expressly "
        "excludes model-development use",
        "require the license governing {asset} to be an Apache 2.0 license",
        ["the governing license must permit model-development use"],
    ),

    F(
        "retention_schedule",
        (
            "retention_policy.txt",
            "AUTHORIZED RETENTION",
            "{Asset} may be retained while the approved project retention "
            "schedule requires continued retention.",
        ),
        (
            "retention_policy.txt",
            "MANDATORY DELETION",
            "{Asset} must not be retained after the approved project retention "
            "schedule requires deletion.",
        ),
        "retain {asset} while the approved project retention schedule requires "
        "continued retention",
        "retain {asset}",
        "retain {asset} after the approved project retention schedule requires deletion",
        "retain {asset} for exactly seven years",
        ["the approved project retention schedule must require continued retention"],
    ),

    F(
        "export_control_clearance",
        (
            "export_control_policy.txt",
            "CLEARED ACCESS",
            "{Asset} may be provided to an external research participant after "
            "required export-control clearance has been granted.",
        ),
        (
            "export_control_policy.txt",
            "DENIED CLEARANCE",
            "{Asset} must not be provided to an external research participant "
            "after export-control clearance has been denied.",
        ),
        "provide {asset} to an external research participant after required "
        "export-control clearance has been granted",
        "provide {asset} to an external research participant",
        "provide {asset} to an external research participant after export-control "
        "clearance has been denied",
        "require export-control clearance for {asset} to be approved by the dean",
        ["required export-control clearance must be granted"],
    ),

    F(
        "security_review",
        (
            "security_review_policy.txt",
            "APPROVED DEPLOYMENT",
            "A system processing {asset} may enter production after the required "
            "security review has been approved.",
        ),
        (
            "security_review_policy.txt",
            "FAILED REVIEW",
            "A system processing {asset} must not enter production after the "
            "required security review has failed.",
        ),
        "move a system processing {asset} into production after the required "
        "security review has been approved",
        "move a system processing {asset} into production",
        "move a system processing {asset} into production after the required "
        "security review has failed",
        "require the security review for a system processing {asset} to use "
        "a particular vulnerability scanner",
        ["the required security review must be approved"],
    ),
]


# --------------------------------------------------------------------
# VALIDATION-ONLY FAMILIES
# --------------------------------------------------------------------

VALIDATION = [

    F(
        "secondary_use_approval",
        (
            "secondary_use_policy.txt",
            "APPROVED SECONDARY USE",
            "{Asset} may be reused for an additional approved research purpose "
            "after secondary-use approval has been recorded.",
        ),
        (
            "secondary_use_policy.txt",
            "DENIED SECONDARY USE",
            "{Asset} must not be reused for a proposed secondary purpose after "
            "secondary-use approval has been denied.",
        ),
        "reuse {asset} for an additional research purpose after secondary-use "
        "approval has been recorded",
        "reuse {asset} for an additional research purpose",
        "reuse {asset} for an additional research purpose after secondary-use "
        "approval has been denied",
        "require secondary-use approval for {asset} to come from the provost",
        ["secondary-use approval must be recorded"],
    ),

    F(
        "derived_data_release",
        (
            "derived_data_policy.txt",
            "APPROVED RELEASE",
            "Derived data produced from {asset} may be released externally "
            "after required disclosure review has approved the release.",
        ),
        (
            "derived_data_policy.txt",
            "BLOCKED RELEASE",
            "Derived data produced from {asset} must not be released externally "
            "after disclosure review has rejected the release.",
        ),
        "release derived data produced from {asset} externally after required "
        "disclosure review has approved the release",
        "release derived data produced from {asset} externally",
        "release derived data produced from {asset} externally after disclosure "
        "review has rejected the release",
        "require disclosure review of derived data from {asset} to be performed "
        "by outside counsel",
        ["required disclosure review must approve the release"],
    ),

    F(
        "cross_border_transfer",
        (
            "cross_border_policy.txt",
            "AUTHORIZED TRANSFER",
            "{Asset} may be transferred across a national border after the "
            "required cross-border transfer authorization is active.",
        ),
        (
            "cross_border_policy.txt",
            "PROHIBITED TRANSFER",
            "{Asset} must not cross a national border when cross-border transfer "
            "authorization has been denied.",
        ),
        "transfer {asset} across a national border after required cross-border "
        "transfer authorization is active",
        "transfer {asset} across a national border",
        "transfer {asset} across a national border after cross-border transfer "
        "authorization has been denied",
        "require cross-border transfer authorization for {asset} to be renewed "
        "every 30 days",
        ["required cross-border transfer authorization must be active"],
    ),
]


# --------------------------------------------------------------------
# TEST-ONLY FAMILIES
# --------------------------------------------------------------------

TEST = [

    F(
        "incident_preservation_hold",
        (
            "incident_policy.txt",
            "RELEASE FROM HOLD",
            "{Asset} may return to normal deletion procedures after an incident "
            "preservation hold has been formally released.",
        ),
        (
            "incident_policy.txt",
            "ACTIVE HOLD",
            "{Asset} must not be deleted while an incident preservation hold "
            "remains active.",
        ),
        "delete {asset} after the incident preservation hold has been formally released",
        "delete {asset}",
        "delete {asset} while an incident preservation hold remains active",
        "require an incident preservation hold covering {asset} to be released "
        "by the chief financial officer",
        ["the incident preservation hold must be formally released"],
    ),

    F(
        "patent_disclosure_clearance",
        (
            "invention_policy.txt",
            "CLEARED DISCLOSURE",
            "Technical findings derived from {asset} may be publicly disclosed "
            "after required patent-disclosure clearance has been granted.",
        ),
        (
            "invention_policy.txt",
            "UNCLEARED DISCLOSURE",
            "Technical findings derived from {asset} must not be publicly "
            "disclosed after patent-disclosure clearance has been denied.",
        ),
        "publicly disclose technical findings derived from {asset} after required "
        "patent-disclosure clearance has been granted",
        "publicly disclose technical findings derived from {asset}",
        "publicly disclose technical findings derived from {asset} after "
        "patent-disclosure clearance has been denied",
        "require patent-disclosure clearance for {asset} to be issued within "
        "five business days",
        ["required patent-disclosure clearance must be granted"],
    ),

    F(
        "federated_external_access",
        (
            "federated_access_policy.txt",
            "AUTHORIZED FEDERATED ACCESS",
            "An external collaborator may remotely access {asset} only after "
            "federated identity verification is complete and multifactor "
            "authentication has been successfully completed.",
        ),
        (
            "federated_access_policy.txt",
            "BLOCKED FEDERATED ACCESS",
            "An external collaborator must not remotely access {asset} after "
            "federated identity verification has failed.",
        ),
        "allow an external collaborator to remotely access {asset} after "
        "federated identity verification and multifactor authentication have "
        "both been successfully completed",
        "allow an external collaborator to remotely access {asset}",
        "allow an external collaborator to remotely access {asset} after "
        "federated identity verification has failed",
        "require the external collaborator accessing {asset} to use a specific "
        "brand of hardware security key",
        [
            "federated identity verification must be complete",
            "multifactor authentication must be successfully completed",
        ],
    ),
]


FAMILIES = {
    "train": TRAIN,
    "validation": VALIDATION,
    "test": TEST,
}

ANSWERS = {
    "YES":
        "The evidence directly permits the action and every material condition is satisfied.",
    "NO":
        "The evidence directly prohibits the action as described.",
    "CONDITIONAL":
        "The action may be permitted, but material policy conditions remain unresolved.",
    "INSUFFICIENT":
        "The supplied evidence does not establish the exact detail asked about.",
}


def fmt(text, asset):
    return text.format(
        asset=asset,
        Asset=asset.capitalize(),
    )


def norm(text):
    return re.sub(
        r"\s+",
        " ",
        text.strip().lower(),
    )


def make_row(
    split,
    fam,
    asset,
    asset_idx,
    decision,
    pattern_idx,
):
    scenario_id = (
        f"{split}_{fam['name']}_{asset_idx}_{decision.lower()}"
    )

    rng = random.Random(
        f"{SEED}|{scenario_id}|{pattern_idx}"
    )

    evidence = [
        {
            "source": fam["permission"][0],
            "section": fam["permission"][1],
            "text": fmt(fam["permission"][2], asset),
            "kind": "permission",
        },
        {
            "source": fam["prohibition"][0],
            "section": fam["prohibition"][1],
            "text": fmt(fam["prohibition"][2], asset),
            "kind": "prohibition",
        },
    ]

    source, section, text = DISTRACTORS[
        (asset_idx + pattern_idx)
        % len(DISTRACTORS)
    ]

    evidence.append({
        "source": source,
        "section": section,
        "text": text,
        "kind": "distractor",
    })

    rng.shuffle(evidence)

    decisive_kind = {
        "YES": "permission",
        "CONDITIONAL": "permission",
        "NO": "prohibition",
        "INSUFFICIENT": None,
    }[decision]

    blocks = []
    citations = []

    for i, item in enumerate(evidence, start=1):

        label = f"S{i}"

        blocks.append(
            f"[{label}]\n"
            f"SOURCE: {item['source']}\n"
            f"SECTION: {item['section']}\n"
            f"TEXT: {item['text']}"
        )

        if item["kind"] == decisive_kind:
            citations.append(label)

    if decision == "INSUFFICIENT":
        citations = []

    core = fmt(
        fam["core"][decision],
        asset,
    )

    question = QUESTIONS[
        pattern_idx
    ].format(
        core=core
    )

    output = {
        "evidence_sufficient":
            decision != "INSUFFICIENT",

        "decision":
            decision,

        "conditions":
            fam["conditions"]
            if decision == "CONDITIONAL"
            else [],

        "answer":
            ANSWERS[decision],

        "citations":
            citations,
    }

    return {
        "scenario_id":
            scenario_id,

        "logical_family":
            fam["name"],

        "split":
            split,

        "instruction":
            INSTRUCTION,

        "input":
            f"QUESTION:\n{question}\n\n"
            f"EVIDENCE:\n"
            + "\n\n".join(blocks),

        "output":
            json.dumps(
                output,
                separators=(",", ":"),
            ),
    }


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    family_sets = {
        split: {
            f["name"]
            for f in families
        }
        for split, families
        in FAMILIES.items()
    }

    # Family leakage must be zero.
    assert family_sets["train"].isdisjoint(
        family_sets["validation"]
    )

    assert family_sets["train"].isdisjoint(
        family_sets["test"]
    )

    assert family_sets["validation"].isdisjoint(
        family_sets["test"]
    )

    splits = {
        split: []
        for split in FAMILIES
    }

    scenarios = {
        split: set()
        for split in FAMILIES
    }

    decisions = [
        "YES",
        "NO",
        "CONDITIONAL",
        "INSUFFICIENT",
    ]

    for split, families in FAMILIES.items():

        for fam in families:

            for asset_idx, asset in enumerate(
                ASSETS
            ):

                for decision in decisions:

                    scenario_id = (
                        f"{split}_"
                        f"{fam['name']}_"
                        f"{asset_idx}_"
                        f"{decision.lower()}"
                    )

                    scenarios[
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
                                split,
                                fam,
                                asset,
                                asset_idx,
                                decision,
                                pattern_idx,
                            )
                        )

    # Deterministic shuffle.
    for offset, split in enumerate(
        ["train", "validation", "test"]
    ):

        random.Random(
            SEED + offset
        ).shuffle(
            splits[split]
        )

    # Scenario leakage.
    assert scenarios["train"].isdisjoint(
        scenarios["validation"]
    )

    assert scenarios["train"].isdisjoint(
        scenarios["test"]
    )

    assert scenarios["validation"].isdisjoint(
        scenarios["test"]
    )

    # Exact-input leakage.
    fps = {}

    for split, rows in splits.items():

        fps[split] = {
            hashlib.sha256(
                norm(
                    row["input"]
                ).encode()
            ).hexdigest()

            for row in rows
        }

    assert fps["train"].isdisjoint(
        fps["validation"]
    )

    assert fps["train"].isdisjoint(
        fps["test"]
    )

    assert fps["validation"].isdisjoint(
        fps["test"]
    )

    # Require balanced labels.
    for split, rows in splits.items():

        labels = Counter(
            json.loads(
                row["output"]
            )["decision"]
            for row in rows
        )

        assert len(
            set(labels.values())
        ) == 1, (
            split,
            labels,
        )

    paths = {
        "train":
            OUT / "sentinel_train_v3.jsonl",

        "validation":
            OUT / "sentinel_validation_v3.jsonl",

        "test":
            OUT / "sentinel_test_v3.jsonl",
    }

    for split, path in paths.items():

        with path.open("w") as f:

            for row in splits[split]:

                f.write(
                    json.dumps(row)
                    + "\n"
                )

    print("=" * 72)
    print("SENTINEL CONTRASTIVE CURRICULUM V3")
    print("=" * 72)

    for split in [
        "train",
        "validation",
        "test",
    ]:

        labels = Counter(
            json.loads(
                row["output"]
            )["decision"]
            for row in splits[split]
        )

        print(
            f"{split:10s} "
            f"rows={len(splits[split]):4d} "
            f"scenarios={len(scenarios[split]):3d} "
            f"families={len(family_sets[split]):2d} "
            f"labels={dict(sorted(labels.items()))}"
        )

        print(
            "  families: "
            + ", ".join(
                sorted(
                    family_sets[split]
                )
            )
        )

    print()
    print(
        "LEAKAGE CHECKS: "
        "family=0 scenario=0 exact_input=0"
    )

    print()
    print("Sample TEST example:")
    print(
        json.dumps(
            splits["test"][0],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
