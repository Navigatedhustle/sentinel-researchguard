from pathlib import Path
from collections import Counter
import hashlib
import json
import re

BASE = Path(__file__).resolve().parents[2]

OUT = (
    BASE /
    "training/challenges/sentinel_challenge_v1.json"
)

INSTRUCTION = """You are Sentinel, a policy-governance decision system.

Use ONLY the supplied evidence.

Classify the exact question as one of:

YES
NO
CONDITIONAL
INSUFFICIENT

YES:
The evidence directly permits the exact action with all material conditions
satisfied, or directly establishes that the proposition asked about is true.

NO:
The evidence directly prohibits the exact action, or directly establishes
that the proposition asked about is false.

CONDITIONAL:
The action or proposition may be supported, but one or more material
requirements established by the evidence remain unresolved.

INSUFFICIENT:
The supplied evidence does not directly establish the exact issue.
Related evidence is not enough. Do not invent missing policy.

Apply specificity, precedence, definitions, exceptions, scope, timing,
and stated conditions exactly as supplied by the evidence.

Silence is not prohibition.
Semantic similarity is not evidence.

Return valid JSON only.
""".strip()


CASES = []


def Q(
    decision,
    qtype,
    question,
    citations,
    rationale,
    tags,
    conditions=None,
):
    return {
        "decision": decision,
        "question_type": qtype,
        "question": question,
        "citations": citations,
        "rationale": rationale,
        "tags": tags,
        "conditions": conditions or [],
    }


def P(pack_id, evidence, *questions):

    for idx, q in enumerate(
        questions,
        start=1,
    ):
        case_id = (
            f"cv1_{pack_id.lower()}_{idx:02d}"
        )

        blocks = []

        for i, item in enumerate(
            evidence,
            start=1,
        ):
            source, section, text = item

            blocks.append(
                f"[S{i}]\n"
                f"SOURCE: {source}\n"
                f"SECTION: {section}\n"
                f"TEXT: {text}"
            )

        # Deliberately vary question/evidence order.
        digest = hashlib.sha256(
            case_id.encode()
        ).digest()

        if digest[0] % 2 == 0:
            order = "question_first"
            model_input = (
                f"QUESTION:\n"
                f"{q['question']}\n\n"
                f"EVIDENCE:\n"
                + "\n\n".join(blocks)
            )
        else:
            order = "evidence_first"
            model_input = (
                f"EVIDENCE:\n"
                + "\n\n".join(blocks)
                + f"\n\nQUESTION:\n"
                f"{q['question']}"
            )

        decision = q["decision"]

        citations = [
            f"S{i}"
            for i in q["citations"]
        ]

        if decision == "INSUFFICIENT":
            citations = []

        output = {
            "evidence_sufficient":
                decision != "INSUFFICIENT",

            "decision":
                decision,

            "conditions":
                q["conditions"]
                if decision == "CONDITIONAL"
                else [],

            "answer":
                q["rationale"],

            "citations":
                citations,
        }

        CASES.append({
            "case_id": case_id,
            "policy_pack": pack_id,
            "question_type":
                q["question_type"],
            "reasoning_tags":
                q["tags"],
            "input_order":
                order,
            "instruction":
                INSTRUCTION,
            "input":
                model_input,
            "gold":
                output,
            "evidence_count":
                len(evidence),
        })


# ================================================================
# A: YES + CONDITIONAL
# ================================================================

P(
    "A01",
    [
        (
            "research_sharing_manual.txt",
            "GENERAL SHARING",
            "Research data may be shared externally after institutional approval."
        ),
        (
            "genomics_addendum.txt",
            "SPECIFIC CONTROL",
            "Genomic research data may be shared externally only after IRB approval and an active data-use agreement."
        ),
        (
            "policy_precedence.txt",
            "SPECIFICITY",
            "When a specific research-data rule conflicts with a general rule, the specific rule controls."
        ),
        (
            "collaboration_record.txt",
            "STATUS",
            "The collaborating laboratory's data-use agreement is active."
        ),
    ],
    Q(
        "YES",
        "permission",
        "The IRB has approved the genomic-data sharing and the collaborator's data-use agreement is active. May the genomic dataset be shared externally?",
        [2, 3, 4],
        "The specific genomics rule controls and both required conditions are satisfied.",
        ["policy_precedence", "multiple_evidence", "specificity"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "The IRB has approved sharing of the genomic dataset. May it now be sent to the external laboratory?",
        [2, 3],
        "External sharing may be permitted, but the evidence does not establish that the required data-use agreement is active.",
        ["partial_condition", "policy_precedence"],
        ["an active data-use agreement is required"],
    ),
)

P(
    "A02",
    [
        (
            "sponsor_confidentiality.txt",
            "DEFAULT RULE",
            "Sponsor-confidential technical material must not be exported outside the project."
        ),
        (
            "sponsor_confidentiality.txt",
            "REDACTED SUMMARY EXCEPTION",
            "A redacted technical summary may be exported after written sponsor clearance confirms that protected technical details have been removed."
        ),
    ],
    Q(
        "YES",
        "permission",
        "A redacted summary has written sponsor clearance confirming that protected technical details were removed. May the summary be exported?",
        [2],
        "The stated exception applies and its clearance requirement is satisfied.",
        ["exception", "condition_satisfied"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "May a redacted technical summary be exported outside the project?",
        [2],
        "The exception may permit export, but written sponsor clearance remains unresolved.",
        ["exception", "unresolved_approval"],
        ["written sponsor clearance must confirm removal of protected details"],
    ),
)

P(
    "A03",
    [
        (
            "publication_terms.txt",
            "EMBARGO",
            "Project findings may be published only after the publication embargo ends."
        ),
        (
            "publication_terms.txt",
            "SPONSOR NOTICE",
            "The sponsor must receive the final manuscript at least five business days before publication."
        ),
        (
            "project_calendar.txt",
            "STATUS",
            "The publication embargo ended on July 1, 2026."
        ),
    ],
    Q(
        "YES",
        "permission",
        "The embargo ended July 1 and the sponsor received the final manuscript eight business days ago. May the findings be published today?",
        [1, 2, 3],
        "The embargo has expired and the required advance sponsor notice is satisfied.",
        ["temporal", "multiple_conditions"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "The publication embargo ended July 1. May the findings be published now?",
        [1, 2, 3],
        "The embargo condition is satisfied, but timely sponsor notice is not established.",
        ["temporal", "partial_condition"],
        ["the sponsor must receive the final manuscript at least five business days before publication"],
    ),
)

P(
    "A04",
    [
        (
            "approval_delegation.txt",
            "DELEGATED AUTHORITY",
            "A delegated approver may issue department approval only while the delegation is recorded in the institutional delegation registry."
        ),
        (
            "department_policy.txt",
            "RESTRICTED ACCESS",
            "Restricted project records may be accessed after valid department approval."
        ),
        (
            "delegation_registry.txt",
            "ROLE",
            "The dean may delegate department-approval authority to an acting chair."
        ),
    ],
    Q(
        "YES",
        "permission",
        "The acting chair approved access, and the dean's delegation to that acting chair is currently recorded in the delegation registry. May the records be accessed?",
        [1, 2, 3],
        "The acting chair has valid delegated authority and the required department approval exists.",
        ["delegated_authority", "indirect_condition"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "The acting chair approved access to the restricted records. Is access permitted?",
        [1, 2, 3],
        "Approval may be valid, but the evidence does not establish that the acting chair's delegation is currently recorded.",
        ["delegated_authority", "unresolved_status"],
        ["the acting chair's delegation must be recorded in the delegation registry"],
    ),
)

P(
    "A05",
    [
        (
            "compute_definitions.txt",
            "QUALIFIED LOCAL COMPUTE",
            "Qualified local compute means institution-owned hardware using managed institutional identity and approved network registration."
        ),
        (
            "restricted_compute_policy.txt",
            "LOCAL PROCESSING",
            "Restricted project data may be processed on qualified local compute."
        ),
    ],
    Q(
        "YES",
        "classification",
        "A workstation is institution-owned, uses managed institutional identity, and has approved network registration. Does the evidence establish that restricted project data may be processed on it?",
        [1, 2],
        "The workstation satisfies every element of the definition of qualified local compute.",
        ["definition", "multi_hop"],
    ),
    Q(
        "CONDITIONAL",
        "classification",
        "A workstation is institution-owned and uses managed institutional identity. May restricted project data be processed on it?",
        [1, 2],
        "Processing may be permitted, but approved network registration is not established.",
        ["definition", "partial_definition"],
        ["approved network registration must be established"],
    ),
)

P(
    "A06",
    [
        (
            "secondary_use_policy.txt",
            "SECONDARY RESEARCH",
            "Participant data may be reused for a secondary research purpose when documented consent covers that purpose and ethics review is complete."
        ),
        (
            "consent_record.txt",
            "SCOPE",
            "The participant consent covers cardiovascular-outcomes research."
        ),
        (
            "ethics_record.txt",
            "STATUS",
            "The secondary-use ethics review for the cardiovascular project is complete."
        ),
    ],
    Q(
        "YES",
        "permission",
        "May the participant data be reused for the cardiovascular-outcomes project?",
        [1, 2, 3],
        "The stated purpose is within consent scope and the required ethics review is complete.",
        ["purpose_limitation", "multiple_evidence"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "The participant consent covers cardiovascular-outcomes research. May the data be reused for that research?",
        [1, 2],
        "Consent scope is satisfied, but completion of the required ethics review is unresolved.",
        ["purpose_limitation", "partial_condition"],
        ["secondary-use ethics review must be complete"],
    ),
)

P(
    "A07",
    [
        (
            "records_schedule.txt",
            "DELETION",
            "Project records may be deleted after the applicable retention date only if no preservation hold remains active."
        ),
        (
            "project_schedule.txt",
            "DATE",
            "The applicable retention date for Project Cedar was June 30, 2026."
        ),
        (
            "legal_hold_registry.txt",
            "STATUS",
            "Project Cedar has no active preservation hold."
        ),
    ],
    Q(
        "YES",
        "permission",
        "Today is after June 30, 2026. May Project Cedar records be deleted?",
        [1, 2, 3],
        "The retention date has passed and the evidence establishes that no preservation hold is active.",
        ["temporal", "legal_hold", "multiple_evidence"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "The retention date for Project Cedar has passed. May its records be deleted?",
        [1],
        "Deletion may be permitted, but the absence of an active preservation hold is not established by the question.",
        ["temporal", "legal_hold"],
        ["no preservation hold may remain active"],
    ),
)

P(
    "A08",
    [
        (
            "derived_results_policy.txt",
            "AGGREGATE RELEASE",
            "Aggregate results may be released when every reported group contains at least ten individuals and disclosure review approves the release."
        ),
        (
            "disclosure_review.txt",
            "STATUS",
            "Disclosure review approved release of the aggregate results."
        ),
    ],
    Q(
        "YES",
        "permission",
        "Every reported group contains at least ten individuals, and disclosure review approved release. May the aggregate results be released?",
        [1, 2],
        "Both the minimum-group-size requirement and disclosure-review requirement are satisfied.",
        ["derived_data", "multiple_conditions"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "Every reported group contains at least ten individuals. May the aggregate results be released?",
        [1],
        "Release may be permitted, but disclosure-review approval remains unresolved.",
        ["derived_data", "partial_condition"],
        ["disclosure review must approve release"],
    ),
)

P(
    "A09",
    [
        (
            "international_transfer.txt",
            "ADEQUACY TRANSFER",
            "Restricted research data may be transferred to an approved adequacy jurisdiction when approved encryption is used and the required contractual clauses are active."
        ),
        (
            "jurisdiction_list.txt",
            "APPROVED",
            "Germany is an approved adequacy jurisdiction."
        ),
        (
            "security_record.txt",
            "TRANSFER CONTROL",
            "The planned transfer uses institutionally approved encryption."
        ),
    ],
    Q(
        "YES",
        "permission",
        "The required contractual clauses are active. May the restricted data be transferred to Germany using the planned transfer method?",
        [1, 2, 3],
        "Germany is approved, the encryption requirement is satisfied, and the required contractual clauses are active.",
        ["jurisdiction", "multiple_conditions"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "May the restricted data be transferred to Germany using the planned transfer method?",
        [1, 2, 3],
        "Jurisdiction and encryption requirements are satisfied, but active contractual clauses are not established.",
        ["jurisdiction", "partial_condition"],
        ["the required contractual clauses must be active"],
    ),
)

P(
    "A10",
    [
        (
            "cloud_policy.txt",
            "GENERAL",
            "Approved cloud providers may process institutional data."
        ),
        (
            "sensitive_research_addendum.txt",
            "SENSITIVE DATA",
            "Sensitive participant-level research data may be processed by an approved cloud provider only inside the project's approved isolated enclave."
        ),
        (
            "precedence_policy.txt",
            "SPECIFIC RULE",
            "A data-specific restriction controls over a general technology permission."
        ),
    ],
    Q(
        "YES",
        "permission",
        "The provider is approved and the sensitive participant data will remain inside the project's approved isolated enclave. May the provider process the data?",
        [1, 2, 3],
        "The specific sensitive-data rule controls and its enclave requirement is satisfied.",
        ["specificity", "policy_precedence"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "The cloud provider is approved. May it process the sensitive participant-level research data?",
        [2, 3],
        "The general provider approval is insufficient because the specific enclave requirement remains unresolved.",
        ["specificity", "policy_precedence"],
        ["processing must occur inside the project's approved isolated enclave"],
    ),
)


# ================================================================
# B: YES + INSUFFICIENT
# ================================================================

P(
    "B01",
    [
        (
            "approval_policy.txt",
            "REQUIRED APPROVAL",
            "Restricted-data export requires written approval from an authorized institutional authority."
        ),
        (
            "approval_record.txt",
            "STATUS",
            "Written approval from an authorized institutional authority is recorded for this export."
        ),
    ],
    Q(
        "YES",
        "permission",
        "May the restricted-data export proceed?",
        [1, 2],
        "The required written institutional approval is recorded.",
        ["authorization"],
    ),
    Q(
        "INSUFFICIENT",
        "requirement",
        "Must the written approval for this export specifically come from the university provost?",
        [],
        "The evidence requires an authorized institutional authority but does not identify the provost specifically.",
        ["specific_approver", "evidence_boundary"],
    ),
)

P(
    "B02",
    [
        (
            "storage_security.txt",
            "ENCRYPTION",
            "Restricted project files may be stored when institutionally approved encryption is enabled."
        ),
    ],
    Q(
        "YES",
        "permission",
        "Institutionally approved encryption is enabled. May the restricted project files be stored?",
        [1],
        "The evidence directly permits storage when approved encryption is enabled.",
        ["security_control"],
    ),
    Q(
        "INSUFFICIENT",
        "requirement",
        "Does the policy require AES-256 specifically for those stored files?",
        [],
        "The evidence requires approved encryption but does not name a specific algorithm.",
        ["specific_technology", "evidence_boundary"],
    ),
)

P(
    "B03",
    [
        (
            "project_retention.txt",
            "PROJECT R",
            "Project R records must be retained through December 31, 2028."
        ),
        (
            "retention_policy.txt",
            "AUTHORIZED RETENTION",
            "Records may be retained for the period required by the applicable approved schedule."
        ),
    ],
    Q(
        "YES",
        "permission",
        "May Project R records be retained through December 31, 2028?",
        [1, 2],
        "The approved schedule requires retention through that date.",
        ["retention", "temporal"],
    ),
    Q(
        "INSUFFICIENT",
        "requirement",
        "Does the evidence establish that Project R has an exactly seven-year retention period?",
        [],
        "The evidence supplies an end date but not an exactly seven-year duration.",
        ["exact_duration", "evidence_boundary"],
    ),
)

P(
    "B04",
    [
        (
            "local_compute_definition.txt",
            "MANAGED WORKSTATION",
            "A managed local workstation is institution-owned computing hardware enrolled in institutional device management."
        ),
        (
            "local_processing.txt",
            "PERMITTED SYSTEM",
            "Restricted research data may be processed on a managed local workstation."
        ),
    ],
    Q(
        "YES",
        "classification",
        "A workstation is institution-owned and enrolled in institutional device management. Does the evidence permit restricted-data processing on it?",
        [1, 2],
        "The workstation meets the supplied definition of a managed local workstation.",
        ["definition", "classification"],
    ),
    Q(
        "INSUFFICIENT",
        "requirement",
        "Must that workstation contain an NVIDIA GPU to qualify as managed local compute?",
        [],
        "No GPU brand or hardware accelerator requirement appears in the supplied definition.",
        ["specific_technology", "scope_mismatch"],
    ),
)

P(
    "B05",
    [
        (
            "jurisdiction_definition.txt",
            "COVERED JURISDICTION",
            "Covered jurisdiction means a jurisdiction named in Annex A."
        ),
        (
            "annex_a.txt",
            "LIST",
            "Annex A lists Germany, France, and the Netherlands."
        ),
        (
            "regional_processing.txt",
            "PROCEDURE",
            "The covered-jurisdiction processing procedure applies to projects operating in a covered jurisdiction."
        ),
    ],
    Q(
        "YES",
        "classification",
        "Does the covered-jurisdiction processing procedure apply to a project operating in Germany?",
        [1, 2, 3],
        "Germany is explicitly named in Annex A and therefore meets the definition.",
        ["definition", "jurisdiction"],
    ),
    Q(
        "INSUFFICIENT",
        "classification",
        "Is Switzerland treated as a covered jurisdiction because of a separate bilateral arrangement?",
        [],
        "The supplied evidence does not address any bilateral arrangement involving Switzerland.",
        ["jurisdiction", "missing_rule"],
    ),
)

P(
    "B06",
    [
        (
            "public_outputs_policy.txt",
            "ARCHIVING",
            "Researchers may archive publicly released project outputs in the institutional repository."
        ),
    ],
    Q(
        "YES",
        "permission",
        "May researchers archive publicly released project outputs in the institutional repository?",
        [1],
        "The policy expressly permits that action.",
        ["direct_permission"],
    ),
    Q(
        "INSUFFICIENT",
        "requirement",
        "Are researchers required to archive every publicly released project output in the institutional repository?",
        [],
        "The evidence grants permission but does not establish a mandatory archiving requirement.",
        ["mandatory_vs_permitted", "evidence_boundary"],
    ),
)

P(
    "B07",
    [
        (
            "access_approval.txt",
            "AUTHORIZED APPROVERS",
            "Internal project access may be approved by the principal investigator or by a delegated data steward."
        ),
        (
            "delegation_record.txt",
            "STATUS",
            "Jordan Lee is recorded as the delegated data steward for Project Orion."
        ),
    ],
    Q(
        "YES",
        "permission",
        "Jordan Lee approved internal access to Project Orion. Is that approval authorized by the supplied evidence?",
        [1, 2],
        "Jordan Lee is expressly recorded as the delegated data steward.",
        ["delegated_authority"],
    ),
    Q(
        "INSUFFICIENT",
        "classification",
        "Does being the laboratory manager automatically make someone a delegated data steward?",
        [],
        "The supplied evidence does not define laboratory managers as delegated data stewards.",
        ["actor_scope", "evidence_boundary"],
    ),
)

P(
    "B08",
    [
        (
            "classification_definitions.txt",
            "DIRECT IDENTIFIERS",
            "Social Security numbers are direct identifiers."
        ),
        (
            "classification_policy.txt",
            "RESTRICTED",
            "Records containing direct identifiers are classified as restricted."
        ),
    ],
    Q(
        "YES",
        "classification",
        "A dataset contains Social Security numbers. Does the evidence establish that the dataset is restricted?",
        [1, 2],
        "The supplied definition makes Social Security numbers direct identifiers, and such records are restricted.",
        ["definition", "classification"],
    ),
    Q(
        "INSUFFICIENT",
        "classification",
        "Does a salted hash of an internal user identifier count as a direct identifier under this policy?",
        [],
        "The supplied definition does not classify salted hashes of internal identifiers.",
        ["classification_boundary", "unmentioned_detail"],
    ),
)

P(
    "B09",
    [
        (
            "policy_2024.txt",
            "LOCAL PROCESSING",
            "The 2024 policy allowed approved local research processing."
        ),
        (
            "policy_2026.txt",
            "LOCAL PROCESSING",
            "The 2026 policy permits restricted local processing after security review is completed."
        ),
        (
            "supersession_notice.txt",
            "EFFECT",
            "The 2026 policy supersedes the 2024 policy in full."
        ),
    ],
    Q(
        "YES",
        "permission",
        "Security review is complete. May restricted local processing proceed under the current policy?",
        [2, 3],
        "The superseding 2026 rule controls and its review condition is satisfied.",
        ["supersession", "temporal_precedence"],
    ),
    Q(
        "INSUFFICIENT",
        "proposition",
        "Was the 2026 policy approved unanimously by the university governance committee?",
        [],
        "The supplied evidence contains no information about the vote approving the policy.",
        ["irrelevant_detail", "evidence_boundary"],
    ),
)

P(
    "B10",
    [
        (
            "dataset_license.txt",
            "PERMITTED PURPOSE",
            "The dataset license permits use for noncommercial academic research."
        ),
        (
            "project_description.txt",
            "PURPOSE",
            "The proposed project is an unfunded university research study with no commercial customer or product use."
        ),
    ],
    Q(
        "YES",
        "permission",
        "May the dataset be used for the proposed project?",
        [1, 2],
        "The project is described as noncommercial academic research within the licensed purpose.",
        ["license_scope", "purpose_limitation"],
    ),
    Q(
        "INSUFFICIENT",
        "classification",
        "Would conference sponsorship by a private company automatically make any resulting academic research commercial under this license?",
        [],
        "The evidence does not define how conference sponsorship affects commercial status.",
        ["license_scope", "undefined_term"],
    ),
)


# ================================================================
# C: NO + CONDITIONAL
# ================================================================

P(
    "C01",
    [
        (
            "access_policy.txt",
            "ACTIVE AUTHORIZATION",
            "Restricted records may be accessed only while project authorization remains active."
        ),
        (
            "revocation_policy.txt",
            "EFFECT",
            "Revocation immediately terminates authorization to access restricted records."
        ),
    ],
    Q(
        "NO",
        "permission",
        "The project authorization was revoked this morning. May the researcher continue accessing the restricted records?",
        [1, 2],
        "Revocation terminates the authorization required for access.",
        ["revocation", "temporal"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "May the researcher access the restricted records?",
        [1, 2],
        "Access may be permitted, but the current authorization status is not established.",
        ["authorization_status"],
        ["project authorization must remain active"],
    ),
)

P(
    "C02",
    [
        (
            "cloud_standard.txt",
            "GENERAL",
            "Approved cloud services may handle institutional information."
        ),
        (
            "participant_data_policy.txt",
            "PUBLIC CLOUD",
            "Participant-level restricted research data must not be stored in a public-cloud environment."
        ),
        (
            "participant_data_policy.txt",
            "PRIVATE EXTERNAL ENVIRONMENT",
            "Participant-level restricted data may be processed in an approved private external environment after security review."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May participant-level restricted research data be stored in an approved public-cloud service?",
        [2],
        "The data-specific rule expressly prohibits public-cloud storage.",
        ["specificity", "direct_prohibition"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "May participant-level restricted research data be processed in an approved private external environment?",
        [3],
        "The private-environment route may be permitted, but security review completion is unresolved.",
        ["specificity", "unresolved_review"],
        ["security review must be completed"],
    ),
)

P(
    "C03",
    [
        (
            "external_release.txt",
            "DEFAULT",
            "Identifiable participant records must not be released to an external collaborator."
        ),
        (
            "external_release.txt",
            "DEIDENTIFIED EXCEPTION",
            "A deidentified version may be released after disclosure review approves the release."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May identifiable participant records be released directly to the external collaborator?",
        [1],
        "The evidence directly prohibits release of identifiable participant records.",
        ["direct_prohibition"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "The records have been deidentified. May the deidentified version be released to the collaborator?",
        [2],
        "The exception may apply, but disclosure-review approval is not established.",
        ["exception", "partial_condition"],
        ["disclosure review must approve the release"],
    ),
)

P(
    "C04",
    [
        (
            "export_policy_2024.txt",
            "OLD RULE",
            "The 2024 rule allowed export after principal-investigator approval."
        ),
        (
            "export_policy_2026.txt",
            "CURRENT RULE",
            "The 2026 rule permits export only after sponsor signoff is recorded."
        ),
        (
            "precedence.txt",
            "LATEST RULE",
            "When successive versions conflict, the latest effective policy controls."
        ),
        (
            "status.txt",
            "EFFECTIVE DATE",
            "The 2026 export policy is currently effective."
        ),
    ],
    Q(
        "NO",
        "permission",
        "Sponsor signoff was denied, but the principal investigator approved export. May the export proceed?",
        [2, 3, 4],
        "The current 2026 rule controls and requires sponsor signoff, which was denied.",
        ["policy_precedence", "conflicting_policy"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "The principal investigator approved export. May the export proceed under the current policy?",
        [2, 3, 4],
        "The current rule requires sponsor signoff, whose status is not established.",
        ["policy_precedence", "unresolved_condition"],
        ["sponsor signoff must be recorded"],
    ),
)

P(
    "C05",
    [
        (
            "invention_policy.txt",
            "PATENT REVIEW",
            "Technical findings must not be publicly disclosed while patent review remains open."
        ),
        (
            "invention_policy.txt",
            "EXCEPTION",
            "During an open patent review, disclosure may occur only when written counsel waiver is granted and sponsor consent is recorded."
        ),
        (
            "review_status.txt",
            "STATUS",
            "Patent review remains open."
        ),
    ],
    Q(
        "NO",
        "permission",
        "Written counsel waiver was granted, but the sponsor denied consent. May the findings be publicly disclosed?",
        [1, 2, 3],
        "The nested exception requires both waiver and sponsor consent; sponsor consent is not satisfied.",
        ["nested_exception", "multiple_conditions"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "Written counsel waiver was granted during the open patent review. May the findings be disclosed?",
        [1, 2, 3],
        "The exception may apply, but sponsor consent remains unresolved.",
        ["nested_exception", "partial_condition"],
        ["sponsor consent must be recorded"],
    ),
)

P(
    "C06",
    [
        (
            "remote_access.txt",
            "ACCESS CONDITIONS",
            "External remote access is allowed only when multifactor authentication is complete and no incident-access block is active."
        ),
        (
            "incident_controls.txt",
            "ACTIVE BLOCK",
            "An active incident-access block prohibits all external remote access."
        ),
    ],
    Q(
        "NO",
        "permission",
        "An incident-access block is active. May an external collaborator remotely access the system even if MFA is complete?",
        [1, 2],
        "The active incident-access block independently prohibits remote access.",
        ["multiple_conditions", "independent_prohibition"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "No incident-access block is active. May the external collaborator remotely access the system?",
        [1],
        "Access may be allowed, but completion of multifactor authentication remains unresolved.",
        ["partial_condition"],
        ["multifactor authentication must be complete"],
    ),
)

P(
    "C07",
    [
        (
            "deployment_policy.txt",
            "HUMAN REVIEW",
            "An AI-generated operational recommendation must receive qualified human review before deployment."
        ),
        (
            "deployment_policy.txt",
            "NO-BYPASS RULE",
            "Deployment without the required qualified human review is prohibited."
        ),
    ],
    Q(
        "NO",
        "negative_permission",
        "May the AI-generated operational recommendation be deployed without qualified human review?",
        [1, 2],
        "The evidence expressly prohibits bypassing the required review.",
        ["human_review", "negative_question"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "May the AI-generated operational recommendation be deployed?",
        [1, 2],
        "Deployment may be permitted, but completion of qualified human review is not established.",
        ["human_review", "unresolved_review"],
        ["qualified human review must be completed"],
    ),
)

P(
    "C08",
    [
        (
            "export_control.txt",
            "CONTROLLED DATA",
            "Controlled technical data may leave the United States only while the required export license is active."
        ),
        (
            "export_control.txt",
            "DENIAL",
            "When the required export license is denied, export of the controlled technical data is prohibited."
        ),
    ],
    Q(
        "NO",
        "permission",
        "The required export license was denied. May the controlled technical data be sent outside the United States?",
        [1, 2],
        "The denied license directly triggers the prohibition.",
        ["export_control", "jurisdiction"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "May the controlled technical data be sent outside the United States?",
        [1],
        "Export may be permitted only if the required license is active, which is unresolved.",
        ["export_control", "unresolved_authorization"],
        ["the required export license must be active"],
    ),
)

P(
    "C09",
    [
        (
            "purpose_policy.txt",
            "ADVERTISING",
            "Participant research data must not be used to target commercial advertising."
        ),
        (
            "purpose_policy.txt",
            "ACADEMIC RESEARCH",
            "Participant research data may be reused for academic research after secondary-use ethics approval."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May the participant research data be used to target commercial advertising?",
        [1],
        "The evidence directly prohibits advertising use.",
        ["purpose_limitation", "direct_prohibition"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "May the participant research data be reused for a new academic study?",
        [2],
        "Academic reuse may be permitted, but secondary-use ethics approval remains unresolved.",
        ["purpose_limitation", "unresolved_approval"],
        ["secondary-use ethics approval is required"],
    ),
)

P(
    "C10",
    [
        (
            "derived_records.txt",
            "CLASSIFICATION",
            "Row-level records derived from restricted source data remain restricted until privacy review formally reclassifies them."
        ),
        (
            "derived_records.txt",
            "PUBLIC RELEASE",
            "Restricted row-level derived records must not be released publicly."
        ),
        (
            "external_sharing.txt",
            "RECLASSIFIED RECORDS",
            "Reclassified derived records may be shared externally after an active data-use agreement is confirmed."
        ),
    ],
    Q(
        "NO",
        "permission",
        "Privacy review has not reclassified the row-level derived records. May they be released publicly?",
        [1, 2],
        "The records remain restricted and public release is expressly prohibited.",
        ["derived_data", "classification"],
    ),
    Q(
        "CONDITIONAL",
        "permission",
        "Privacy review has formally reclassified the derived records. May they be shared externally?",
        [3],
        "External sharing may be permitted, but an active data-use agreement is not established.",
        ["derived_data", "partial_condition"],
        ["an active data-use agreement must be confirmed"],
    ),
)


# ================================================================
# D: NO + INSUFFICIENT
# ================================================================

P(
    "D01",
    [
        (
            "external_ai_policy.txt",
            "RESTRICTED DATA",
            "Restricted research data must not be submitted to external generative-AI services."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May restricted research data be submitted to an external generative-AI service?",
        [1],
        "The supplied policy directly prohibits the action.",
        ["direct_prohibition"],
    ),
    Q(
        "INSUFFICIENT",
        "permission",
        "Does this policy prohibit researchers from using an ordinary handheld calculator?",
        [],
        "The evidence governs external generative-AI services, not handheld calculators.",
        ["scope_mismatch", "related_technology"],
    ),
)

P(
    "D02",
    [
        (
            "key_release.txt",
            "AUTHORIZED ACTOR",
            "Only the project's principal investigator may release the study linkage key."
        ),
        (
            "key_release.txt",
            "DELEGATION",
            "The linkage-key release authority may not be delegated."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May a postdoctoral researcher release the study linkage key on behalf of the principal investigator?",
        [1, 2],
        "Release authority is limited to the principal investigator and may not be delegated.",
        ["actor_scope", "direct_prohibition"],
    ),
    Q(
        "INSUFFICIENT",
        "classification",
        "Can the department vice chair be formally designated as the project's principal investigator?",
        [],
        "The evidence does not govern how principal investigators are designated.",
        ["actor_scope", "missing_rule"],
    ),
)

P(
    "D03",
    [
        (
            "country_controls.txt",
            "PROHIBITED DESTINATION",
            "Controlled research software must not be exported to Country X."
        ),
        (
            "country_controls.txt",
            "SCOPE",
            "The prohibition applies to destinations legally within Country X."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May the controlled research software be exported directly to Country X?",
        [1],
        "Country X is expressly prohibited.",
        ["jurisdiction", "direct_prohibition"],
    ),
    Q(
        "INSUFFICIENT",
        "classification",
        "Is Territory Y legally within Country X for purposes of this policy?",
        [],
        "The supplied evidence does not establish Territory Y's legal status.",
        ["jurisdiction", "external_fact_required"],
    ),
)

P(
    "D04",
    [
        (
            "dataset_license.txt",
            "COMMERCIAL REDISTRIBUTION",
            "Commercial redistribution of the licensed dataset is prohibited."
        ),
        (
            "dataset_license.txt",
            "ACADEMIC USE",
            "Noncommercial academic analysis is permitted."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May the licensed dataset be redistributed as part of a commercial data product?",
        [1],
        "Commercial redistribution is expressly prohibited.",
        ["license_scope", "direct_prohibition"],
    ),
    Q(
        "INSUFFICIENT",
        "classification",
        "Does presenting research at a nonprofit conference sponsored by a corporation make the research 'commercial' under this license?",
        [],
        "The supplied license does not define the commercial status of such sponsorship.",
        ["license_scope", "undefined_term"],
    ),
)

P(
    "D05",
    [
        (
            "participant_consent.txt",
            "EXCLUDED PURPOSE",
            "The participant's documented consent explicitly excludes genetic-sequence analysis."
        ),
        (
            "consent_policy.txt",
            "PURPOSE LIMIT",
            "Participant data must not be used for a purpose explicitly excluded by documented consent."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May the participant data be used for whole-genome sequence analysis?",
        [1, 2],
        "The proposed genetic-sequence analysis is explicitly excluded by consent.",
        ["purpose_limitation", "consent"],
    ),
    Q(
        "INSUFFICIENT",
        "classification",
        "Does methylation analysis count as 'genetic-sequence analysis' under these supplied policies?",
        [],
        "The evidence does not define whether methylation analysis falls within that term.",
        ["undefined_term", "classification_boundary"],
    ),
)

P(
    "D06",
    [
        (
            "transmission_security.txt",
            "PLAINTEXT",
            "Restricted project data must not be transmitted in plaintext."
        ),
        (
            "transmission_security.txt",
            "APPROVED ENCRYPTION",
            "Electronic transmission is permitted when institutionally approved transport encryption is used."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May restricted project data be transmitted over an unencrypted plaintext connection?",
        [1],
        "Plaintext transmission is directly prohibited.",
        ["security_control", "direct_prohibition"],
    ),
    Q(
        "INSUFFICIENT",
        "requirement",
        "Does the supplied policy establish that TLS 1.3 specifically is the institutionally approved transport-encryption standard?",
        [],
        "The evidence requires approved transport encryption but does not identify TLS 1.3.",
        ["specific_technology", "evidence_boundary"],
    ),
)

P(
    "D07",
    [
        (
            "retention_schedule.txt",
            "EXPIRATION",
            "Records must be deleted after the approved retention period expires unless an active legal hold requires preservation."
        ),
        (
            "project_status.txt",
            "STATUS",
            "The retention period has expired and no legal hold is active."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May the project continue retaining the records?",
        [1, 2],
        "The retention period has expired and no preservation exception applies.",
        ["retention", "temporal"],
    ),
    Q(
        "INSUFFICIENT",
        "requirement",
        "Must the records be destroyed using exactly three overwrite passes?",
        [],
        "The evidence requires deletion but does not specify a destruction method.",
        ["specific_method", "evidence_boundary"],
    ),
)

P(
    "D08",
    [
        (
            "collaboration_policy.txt",
            "GENERAL SHARING",
            "Approved collaborators may receive project outputs."
        ),
        (
            "controlled_dataset_rule.txt",
            "PUBLIC REPOSITORIES",
            "Controlled row-level datasets must not be uploaded to a public repository."
        ),
        (
            "specificity_rule.txt",
            "SPECIFICITY",
            "A restriction specific to a controlled dataset overrides a general collaboration permission."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May an approved collaborator upload the controlled row-level dataset to a public repository?",
        [2, 3],
        "The specific controlled-dataset prohibition overrides the general collaborator permission.",
        ["policy_precedence", "specificity"],
    ),
    Q(
        "INSUFFICIENT",
        "requirement",
        "Which university office, if any, has authority to waive the public-repository restriction?",
        [],
        "The supplied evidence does not identify a waiver authority or even establish that the restriction is waivable.",
        ["delegated_authority", "missing_rule"],
    ),
)

P(
    "D09",
    [
        (
            "identifier_communication.txt",
            "EMAIL",
            "Direct participant identifiers must not be transmitted by email."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May direct participant identifiers be transmitted by email?",
        [1],
        "Email transmission of direct participant identifiers is expressly prohibited.",
        ["direct_prohibition", "communication_channel"],
    ),
    Q(
        "INSUFFICIENT",
        "permission",
        "Does this evidence establish whether direct participant identifiers may be transmitted by SMS text message?",
        [],
        "The supplied rule governs email and does not establish the rule for SMS.",
        ["scope_mismatch", "communication_channel"],
    ),
)

P(
    "D10",
    [
        (
            "automated_decision_policy.txt",
            "ADVERSE ACTION",
            "An automated recommendation must not be the sole basis for an adverse decision affecting an individual."
        ),
        (
            "automated_decision_policy.txt",
            "HUMAN ROLE",
            "A qualified human decision-maker must independently review the recommendation before an adverse decision is finalized."
        ),
    ],
    Q(
        "NO",
        "permission",
        "May an automated recommendation be used as the sole basis for finalizing an adverse decision?",
        [1, 2],
        "The evidence expressly prohibits sole reliance on the automated recommendation.",
        ["human_review", "direct_prohibition"],
    ),
    Q(
        "INSUFFICIENT",
        "requirement",
        "Does the policy require the automated model to exceed a 95% confidence score before its recommendation may be reviewed?",
        [],
        "No model-confidence threshold is specified in the evidence.",
        ["specific_threshold", "evidence_boundary"],
    ),
)


def normalized(case):
    return re.sub(
        r"\s+",
        " ",
        (
            case["instruction"]
            + " "
            + case["input"]
        ).lower().strip()
    )


def main():

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    assert len(CASES) == 80, (
        "Expected 80 cases, got",
        len(CASES),
    )

    ids = [
        c["case_id"]
        for c in CASES
    ]

    assert len(ids) == len(set(ids))

    labels = Counter(
        c["gold"]["decision"]
        for c in CASES
    )

    expected = {
        "YES": 20,
        "NO": 20,
        "CONDITIONAL": 20,
        "INSUFFICIENT": 20,
    }

    assert labels == expected, (
        labels,
        expected,
    )

    for case in CASES:

        gold = case["gold"]

        assert (
            gold["evidence_sufficient"]
            ==
            (
                gold["decision"]
                != "INSUFFICIENT"
            )
        )

        if gold["decision"] == "INSUFFICIENT":
            assert gold["citations"] == []

        if gold["decision"] == "CONDITIONAL":
            assert len(
                gold["conditions"]
            ) >= 1

    fingerprints = [
        hashlib.sha256(
            normalized(c).encode()
        ).hexdigest()
        for c in CASES
    ]

    assert (
        len(fingerprints)
        ==
        len(set(fingerprints))
    )

    OUT.write_text(
        json.dumps(
            CASES,
            indent=2,
        )
        + "\n"
    )

    qtypes = Counter(
        c["question_type"]
        for c in CASES
    )

    evidence_counts = Counter(
        c["evidence_count"]
        for c in CASES
    )

    orders = Counter(
        c["input_order"]
        for c in CASES
    )

    packs = Counter(
        c["policy_pack"]
        for c in CASES
    )

    print("=" * 72)
    print("SENTINEL CHALLENGE V1 BUILT")
    print("=" * 72)

    print("Cases:", len(CASES))
    print("Policy packs:", len(packs))
    print("Labels:", dict(sorted(labels.items())))
    print("Question types:", dict(sorted(qtypes.items())))
    print(
        "Evidence counts:",
        dict(sorted(evidence_counts.items()))
    )
    print(
        "Input ordering:",
        dict(sorted(orders.items()))
    )

    print()
    print("Output:")
    print(OUT)


if __name__ == "__main__":
    main()
