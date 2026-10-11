# Sentinel V5 — Challenge V2 Decision Failure Dossier

Frozen one-shot result: **216/240 = 90.00% decision accuracy.**

This dossier contains only the 24 already-recorded decision failures. No model was rerun.

## Failure transitions

- **YES->NO: 10**
- **INSUFFICIENT->CONDITIONAL: 7**
- **CONDITIONAL->NO: 2**
- **NO->CONDITIONAL: 2**
- **CONDITIONAL->YES: 1**
- **YES->CONDITIONAL: 1**
- **YES->INSUFFICIENT: 1**

## Failure compositions

- **exception_activation: 6**
- **nested_exception_requirements: 5**
- **purpose_scope_authorization: 5**
- **superseding_rule_precedence: 3**
- **jurisdiction_then_requirement: 2**
- **temporal_authorization: 1**
- **permission_with_material_condition: 1**
- **definition_then_authorization: 1**

---

# CONDITIONAL->NO

## Failure 01 — challenge_v2__jurisdiction_then_requirement__rel__p1__q1

- **Composition:** `jurisdiction_then_requirement`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['jurisdiction_then_requirement', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `jurisdiction_then_requirement__latin4`
- **Evidence count:** 2
- **Input order:** `evidence_first`
- **Gold decision:** **CONDITIONAL**
- **Predicted decision:** **NO**
- **Citation set exact:** False
- **Condition relation:** `missing_all_gold_conditions`

### Question

Under the supplied evidence, may someone transfer the Summit corpus to Territory Q?

### Evidence

```text
[E1] Territory Q is classified as an approved destination for the Umber dataset. Transfers to approved destinations are permitted when contractual clauses are active. The required clauses are active. Territory Q is classified as an approved destination for the Summit corpus. Transfer is permitted only while contractual clauses are active. Clause status is not established.

[E2] Territory Q is classified as a prohibited destination for the Quartz archive. Transfers to prohibited destinations are not permitted. The transfer policy distinguishes approved and prohibited destinations for the Talon files, but the supplied evidence does not classify Territory Q in either category.
```

### Gold

**Decision:** CONDITIONAL

**Answer:** Jurisdiction is approved, but the contractual requirement remains unresolved.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[
  "contractual clauses unresolved"
]
```

### Sentinel V5

**Decision:** NO

**Rationale:** The evidence directly classifies the destination as prohibited and the action is controlled by that classification.

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[]
```

---

## Failure 02 — challenge_v2__nested_exception_requirements__rel__p2__q0

- **Composition:** `nested_exception_requirements`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['nested_exception_requirements', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `nested_exception_requirements__latin4`
- **Evidence count:** 2
- **Input order:** `evidence_first`
- **Gold decision:** **CONDITIONAL**
- **Predicted decision:** **NO**
- **Citation set exact:** True
- **Condition relation:** `missing_all_gold_conditions`

### Question

Under the supplied evidence, may someone transfer the Quartz archive to an external research partner?

### Evidence

```text
[E1] External transfer of the Umber dataset is prohibited. The only exception requires both an active institutional waiver and sponsor consent. The institutional waiver was denied, while sponsor consent is recorded. The Summit corpus is subject to collaboration reporting and annual inventory reconciliation. The supplied evidence does not establish any rule governing external research-partner transfer.

[E2] External transfer of the Talon files is prohibited. A collaboration exception applies when an institutional waiver is active and sponsor consent is recorded. The waiver is active and sponsor consent is recorded. External transfer of the Quartz archive is prohibited. A collaboration exception requires an active institutional waiver and sponsor consent. The waiver is active, but sponsor-consent status is not established.
```

### Gold

**Decision:** CONDITIONAL

**Answer:** The exception may permit transfer, but sponsor consent remains unresolved.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[
  "sponsor consent unresolved"
]
```

### Sentinel V5

**Decision:** NO

**Rationale:** The exception cannot authorize the action because a required exception element has failed.

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[]
```

---

# CONDITIONAL->YES

## Failure 03 — challenge_v2__exception_activation__rel__p1__q1

- **Composition:** `exception_activation`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['exception_activation', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `exception_activation__latin4`
- **Evidence count:** 1
- **Input order:** `evidence_first`
- **Gold decision:** **CONDITIONAL**
- **Predicted decision:** **YES**
- **Citation set exact:** False
- **Condition relation:** `missing_all_gold_conditions`

### Question

Under the supplied evidence, may someone export the Noble files outside the approved environment?

### Evidence

```text
[E1] Export of the Prism records outside the approved environment is prohibited. An exception applies during a declared recovery event if written security clearance is recorded. A recovery event is declared and written clearance is recorded. Export of the Noble files outside the approved environment is prohibited. A declared recovery event activates an exception allowing export with written security clearance. A recovery event is declared, but written-clearance status is not established. Export of the Maple corpus outside the approved environment is prohibited. An exception becomes available only during a declared recovery event. No recovery event has been declared, although security clearance is already recorded. The Orchid dataset is covered by incident notification and inventory rules. The supplied evidence does not state whether export outside an approved environment is permitted or prohibited.
```

### Gold

**Decision:** CONDITIONAL

**Answer:** The exception is active, but its required clearance remains unresolved.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[
  "written security clearance unresolved"
]
```

### Sentinel V5

**Decision:** YES

**Rationale:** The exception is active and every material exception requirement is satisfied.

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[]
```

---

# INSUFFICIENT->CONDITIONAL

## Failure 04 — challenge_v2__exception_activation__stress__insufficient

- **Composition:** `exception_activation`
- **Question type:** `stress`
- **Reasoning tags:** `['exception_activation', 'stress_case']`
- **Relational group:** `None`
- **Evidence count:** 5
- **Input order:** `question_first`
- **Gold decision:** **INSUFFICIENT**
- **Predicted decision:** **CONDITIONAL**
- **Citation set exact:** True
- **Condition relation:** `exact`

### Question

Based only on these passages, is it permitted to export the Exception Activation stress asset 4 outside the approved environment?

### Evidence

```text
[E1] Institution-owned devices must remain listed in the central inventory.

[E2] Public training materials may be distributed internally.

[E3] The Exception Activation stress asset 4 is covered by incident notification and inventory rules. The supplied evidence does not state whether export outside an approved environment is permitted or prohibited.

[E4] Travel reimbursement requests must include itemized receipts.

[E5] Annual asset inventories are due before the end of the fiscal year.
```

### Gold

**Decision:** INSUFFICIENT

**Answer:** The evidence concerns the asset but does not govern the requested export.

**Evidence sufficient:** False

**Citations:**

```json
[
  "E3"
]
```

**Conditions:**

```json
[]
```

### Sentinel V5

**Decision:** CONDITIONAL

**Rationale:** The exact proposition is governed, but a material permission requirement remains unresolved.

**Citations:**

```json
[
  "E3"
]
```

**Conditions:**

```json
[]
```

---

## Failure 05 — challenge_v2__jurisdiction_then_requirement__stress__insufficient

- **Composition:** `jurisdiction_then_requirement`
- **Question type:** `stress`
- **Reasoning tags:** `['jurisdiction_then_requirement', 'stress_case']`
- **Relational group:** `None`
- **Evidence count:** 5
- **Input order:** `question_first`
- **Gold decision:** **INSUFFICIENT**
- **Predicted decision:** **CONDITIONAL**
- **Citation set exact:** True
- **Condition relation:** `extra_when_gold_empty`

### Question

Based only on these passages, is it permitted to transfer the Jurisdiction Then Requirement stress asset 4 to Territory Q?

### Evidence

```text
[E1] Public training materials may be distributed internally.

[E2] The transfer policy distinguishes approved and prohibited destinations for the Jurisdiction Then Requirement stress asset 4, but the supplied evidence does not classify Territory Q in either category.

[E3] Published institutional web pages must follow accessibility requirements.

[E4] Institution-owned devices must remain listed in the central inventory.

[E5] Travel reimbursement requests must include itemized receipts.
```

### Gold

**Decision:** INSUFFICIENT

**Answer:** The framework is supplied, but Territory Q's required classification is missing.

**Evidence sufficient:** False

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[]
```

### Sentinel V5

**Decision:** CONDITIONAL

**Rationale:** The action may be authorized, but the material destination requirement remains unresolved.

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[
  "destination unresolved"
]
```

---

## Failure 06 — challenge_v2__purpose_scope_authorization__rel__p0__q3

- **Composition:** `purpose_scope_authorization`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['purpose_scope_authorization', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `purpose_scope_authorization__latin4`
- **Evidence count:** 2
- **Input order:** `evidence_first`
- **Gold decision:** **INSUFFICIENT**
- **Predicted decision:** **CONDITIONAL**
- **Citation set exact:** True
- **Condition relation:** `extra_when_gold_empty`

### Question

Under the supplied evidence, may someone reuse the Prism records for secondary academic research?

### Evidence

```text
[E1] The authorized purpose for the Orchid dataset includes secondary academic research. Such reuse is permitted only after ethics review. Ethics-review status is not established. The authorization for the Noble files permits primary analysis but explicitly excludes secondary academic reuse.

[E2] The authorized purpose for the Maple corpus includes noncommercial secondary academic research. Reuse within that purpose is permitted after ethics review, and ethics review is complete. The license for the Prism records permits research use but does not define whether the proposed secondary academic activity falls within the licensed purpose, and no supplied definition resolves that term.
```

### Gold

**Decision:** INSUFFICIENT

**Answer:** The evidence does not establish whether the requested purpose falls within the governed scope.

**Evidence sufficient:** False

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[]
```

### Sentinel V5

**Decision:** CONDITIONAL

**Rationale:** The evidence governs a neighboring purpose and does not directly address the exact proposition asked.

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[
  "licensed purpose unresolved"
]
```

---

## Failure 07 — challenge_v2__purpose_scope_authorization__rel__p1__q2

- **Composition:** `purpose_scope_authorization`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['purpose_scope_authorization', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `purpose_scope_authorization__latin4`
- **Evidence count:** 3
- **Input order:** `question_first`
- **Gold decision:** **INSUFFICIENT**
- **Predicted decision:** **CONDITIONAL**
- **Citation set exact:** True
- **Condition relation:** `extra_when_gold_empty`

### Question

Under the supplied evidence, may someone reuse the Orchid dataset for secondary academic research?

### Evidence

```text
[E1] The authorization for the Maple corpus permits primary analysis but explicitly excludes secondary academic reuse.

[E2] The authorized purpose for the Noble files includes secondary academic research. Such reuse is permitted only after ethics review. Ethics-review status is not established.

[E3] The license for the Orchid dataset permits research use but does not define whether the proposed secondary academic activity falls within the licensed purpose, and no supplied definition resolves that term. The authorized purpose for the Prism records includes noncommercial secondary academic research. Reuse within that purpose is permitted after ethics review, and ethics review is complete.
```

### Gold

**Decision:** INSUFFICIENT

**Answer:** The evidence does not establish whether the requested purpose falls within the governed scope.

**Evidence sufficient:** False

**Citations:**

```json
[
  "E3"
]
```

**Conditions:**

```json
[]
```

### Sentinel V5

**Decision:** CONDITIONAL

**Rationale:** The evidence governs a neighboring purpose and does not directly address the exact proposition in the question.

**Citations:**

```json
[
  "E3"
]
```

**Conditions:**

```json
[
  "purpose unresolved"
]
```

---

## Failure 08 — challenge_v2__purpose_scope_authorization__rel__p2__q1

- **Composition:** `purpose_scope_authorization`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['purpose_scope_authorization', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `purpose_scope_authorization__latin4`
- **Evidence count:** 2
- **Input order:** `evidence_first`
- **Gold decision:** **INSUFFICIENT**
- **Predicted decision:** **CONDITIONAL**
- **Citation set exact:** True
- **Condition relation:** `extra_when_gold_empty`

### Question

Under the supplied evidence, may someone reuse the Noble files for secondary academic research?

### Evidence

```text
[E1] The authorization for the Prism records permits primary analysis but explicitly excludes secondary academic reuse. The authorized purpose for the Orchid dataset includes noncommercial secondary academic research. Reuse within that purpose is permitted after ethics review, and ethics review is complete.

[E2] The license for the Noble files permits research use but does not define whether the proposed secondary academic activity falls within the licensed purpose, and no supplied definition resolves that term. The authorized purpose for the Maple corpus includes secondary academic research. Such reuse is permitted only after ethics review. Ethics-review status is not established.
```

### Gold

**Decision:** INSUFFICIENT

**Answer:** The evidence does not establish whether the requested purpose falls within the governed scope.

**Evidence sufficient:** False

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[]
```

### Sentinel V5

**Decision:** CONDITIONAL

**Rationale:** The purpose may be permitted, but its material requirement remains unresolved.

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[
  "purpose unresolved"
]
```

---

## Failure 09 — challenge_v2__purpose_scope_authorization__rel__p3__q0

- **Composition:** `purpose_scope_authorization`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['purpose_scope_authorization', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `purpose_scope_authorization__latin4`
- **Evidence count:** 2
- **Input order:** `question_first`
- **Gold decision:** **INSUFFICIENT**
- **Predicted decision:** **CONDITIONAL**
- **Citation set exact:** True
- **Condition relation:** `extra_when_gold_empty`

### Question

Under the supplied evidence, may someone reuse the Maple corpus for secondary academic research?

### Evidence

```text
[E1] The authorized purpose for the Prism records includes secondary academic research. Such reuse is permitted only after ethics review. Ethics-review status is not established. The license for the Maple corpus permits research use but does not define whether the proposed secondary academic activity falls within the licensed purpose, and no supplied definition resolves that term.

[E2] The authorized purpose for the Noble files includes noncommercial secondary academic research. Reuse within that purpose is permitted after ethics review, and ethics review is complete. The authorization for the Orchid dataset permits primary analysis but explicitly excludes secondary academic reuse.
```

### Gold

**Decision:** INSUFFICIENT

**Answer:** The evidence does not establish whether the requested purpose falls within the governed scope.

**Evidence sufficient:** False

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[]
```

### Sentinel V5

**Decision:** CONDITIONAL

**Rationale:** The evidence governs a neighboring purpose and does not directly address the exact proposition in the question.

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[
  "licensed purpose unresolved"
]
```

---

## Failure 10 — challenge_v2__purpose_scope_authorization__stress__insufficient

- **Composition:** `purpose_scope_authorization`
- **Question type:** `stress`
- **Reasoning tags:** `['purpose_scope_authorization', 'stress_case']`
- **Relational group:** `None`
- **Evidence count:** 2
- **Input order:** `evidence_first`
- **Gold decision:** **INSUFFICIENT**
- **Predicted decision:** **CONDITIONAL**
- **Citation set exact:** True
- **Condition relation:** `extra_when_gold_empty`

### Question

Based only on these passages, is it permitted to reuse the Purpose Scope Authorization stress asset 4 for secondary academic research?

### Evidence

```text
[E1] The license for the Purpose Scope Authorization stress asset 4 permits research use but does not define whether the proposed secondary academic activity falls within the licensed purpose, and no supplied definition resolves that term.

[E2] Annual asset inventories are due before the end of the fiscal year.
```

### Gold

**Decision:** INSUFFICIENT

**Answer:** The evidence does not establish whether the requested purpose falls within the governed scope.

**Evidence sufficient:** False

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[]
```

### Sentinel V5

**Decision:** CONDITIONAL

**Rationale:** The exact proposition is governed, but a material permission requirement remains unresolved.

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[
  "the proposed secondary academic activity may or may not fall within the licensed purpose"
]
```

---

# NO->CONDITIONAL

## Failure 11 — challenge_v2__exception_activation__rel__p3__q2

- **Composition:** `exception_activation`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['exception_activation', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `exception_activation__latin4`
- **Evidence count:** 2
- **Input order:** `evidence_first`
- **Gold decision:** **NO**
- **Predicted decision:** **CONDITIONAL**
- **Citation set exact:** True
- **Condition relation:** `extra_when_gold_empty`

### Question

Under the supplied evidence, may someone export the Orchid dataset outside the approved environment?

### Evidence

```text
[E1] Export of the Prism records outside the approved environment is prohibited. A declared recovery event activates an exception allowing export with written security clearance. A recovery event is declared, but written-clearance status is not established. Export of the Orchid dataset outside the approved environment is prohibited. An exception becomes available only during a declared recovery event. No recovery event has been declared, although security clearance is already recorded.

[E2] Export of the Noble files outside the approved environment is prohibited. An exception applies during a declared recovery event if written security clearance is recorded. A recovery event is declared and written clearance is recorded. The Maple corpus is covered by incident notification and inventory rules. The supplied evidence does not state whether export outside an approved environment is permitted or prohibited.
```

### Gold

**Decision:** NO

**Answer:** The exception is inactive, so the base prohibition controls despite downstream clearance.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[]
```

### Sentinel V5

**Decision:** CONDITIONAL

**Rationale:** The exception could permit the action, but a material exception requirement remains unresolved.

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[
  "recovery event declared, but exception not active"
]
```

---

## Failure 12 — challenge_v2__superseding_rule_precedence__rel__p2__q3

- **Composition:** `superseding_rule_precedence`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['superseding_rule_precedence', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `superseding_rule_precedence__latin4`
- **Evidence count:** 3
- **Input order:** `evidence_first`
- **Gold decision:** **NO**
- **Predicted decision:** **CONDITIONAL**
- **Citation set exact:** False
- **Condition relation:** `extra_when_gold_empty`

### Question

Under the supplied evidence, may someone release the Kestrel archive through the sponsor portal?

### Evidence

```text
[E1] A 2026 rule supersedes the prior retention schedule for the Harbor dataset and changes its archival review date. No supplied provision states whether the Harbor dataset may be released through a sponsor portal. A 2024 rule permitted sponsor-portal release of the Kestrel archive. A 2026 rule expressly supersedes it and requires sponsor signoff before release. Sponsor signoff was denied.

[E2] A 2024 rule addressed release of the Ember files. A 2026 rule expressly supersedes it and permits sponsor-portal release only with sponsor signoff. Current signoff status is not established.

[E3] A 2024 release rule restricted sponsor-portal release of the Juniper records. A 2026 rule expressly supersedes the 2024 rule and permits release after sponsor review. Sponsor review is complete.
```

### Gold

**Decision:** NO

**Answer:** The current superseding rule controls, and the denied signoff prevents release.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[]
```

### Sentinel V5

**Decision:** CONDITIONAL

**Rationale:** The action may be authorized, but its material requirement remains unresolved.

**Citations:**

```json
[
  "E1",
  "E2"
]
```

**Conditions:**

```json
[
  "sponsor signoff unresolved"
]
```

---

# YES->CONDITIONAL

## Failure 13 — challenge_v2__temporal_authorization__stress__yes

- **Composition:** `temporal_authorization`
- **Question type:** `stress`
- **Reasoning tags:** `['temporal_authorization', 'stress_case']`
- **Relational group:** `None`
- **Evidence count:** 5
- **Input order:** `evidence_first`
- **Gold decision:** **YES**
- **Predicted decision:** **CONDITIONAL**
- **Citation set exact:** True
- **Condition relation:** `different_or_reworded`

### Question

Based only on these passages, is it permitted to access the Temporal Authorization stress asset 1 remotely?

### Evidence

```text
[E1] Travel reimbursement requests must include itemized receipts.

[E2] Published institutional web pages must follow accessibility requirements.

[E3] Institution-owned devices must remain listed in the central inventory.

[E4] Remote access to the Temporal Authorization stress asset 1 is permitted only while project authorization is active and annual review is complete. Authorization runs March 1 through September 30, the request occurs June 15, and annual review is complete.

[E5] Public training materials may be distributed internally.
```

### Gold

**Decision:** YES

**Answer:** The request falls within the active authorization period and the review condition is satisfied.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E4"
]
```

**Conditions:**

```json
[
  "authorization active",
  "annual review complete"
]
```

### Sentinel V5

**Decision:** CONDITIONAL

**Rationale:** The action may be authorized, but one material ledger requirement remains unresolved.

**Citations:**

```json
[
  "E4"
]
```

**Conditions:**

```json
[
  "project authorization active",
  "annual review complete"
]
```

---

# YES->INSUFFICIENT

## Failure 14 — challenge_v2__permission_with_material_condition__rel__p2__q2

- **Composition:** `permission_with_material_condition`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['permission_with_material_condition', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `permission_with_material_condition__latin4`
- **Evidence count:** 1
- **Input order:** `question_first`
- **Gold decision:** **YES**
- **Predicted decision:** **INSUFFICIENT**
- **Citation set exact:** False
- **Condition relation:** `missing_all_gold_conditions`

### Question

Under the supplied evidence, may someone publish the Cedar records in a public repository?

### Evidence

```text
[E1] Publication of the Cedar records in a public repository is permitted after disclosure review and sponsor notice. Disclosure review is approved and sponsor notice was sent on time. The evidence describes internal archival retention for the Birch archive and internal catalog publication. It does not state whether the asset may be published in a public repository. Publication of the Alder dataset in a public repository is permitted after disclosure review and sponsor notice. Disclosure review is approved, but sponsor-notice status is not established. The public-release rule for the Dune corpus expressly prohibits publication in public repositories.
```

### Gold

**Decision:** YES

**Answer:** Direct publication permission applies and every material requirement is established.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[
  "disclosure review approved",
  "sponsor notice satisfied"
]
```

### Sentinel V5

**Decision:** INSUFFICIENT

**Rationale:** The evidence governs a neighboring route or class rather than the exact requested action.

**Citations:**

```json
[]
```

**Conditions:**

```json
[]
```

---

# YES->NO

## Failure 15 — challenge_v2__definition_then_authorization__rel__p3__q1

- **Composition:** `definition_then_authorization`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['definition_then_authorization', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `definition_then_authorization__latin4`
- **Evidence count:** 3
- **Input order:** `evidence_first`
- **Gold decision:** **YES**
- **Predicted decision:** **NO**
- **Citation set exact:** True
- **Condition relation:** `missing_all_gold_conditions`

### Question

Under the supplied evidence, may someone process the Harbor dataset on a local workstation?

### Evidence

```text
[E1] The Kestrel archive satisfies the definition of a qualified local workload. Qualified local workloads may be processed on a local workstation only after endpoint security approval. Endpoint-security approval status is not established.

[E2] A restricted local workload is defined to include the Juniper records. Restricted local workloads are expressly prohibited from workstation processing. A qualified local workload is defined as institutional research data with no external service call. Processing the Harbor dataset meets that definition. Qualified local workloads may run on managed local workstations, and the workstation is recorded as managed.

[E3] The evidence defines the Ember files as a qualified local workload and describes its technical characteristics. No supplied provision states whether qualified local workloads may be processed on a local workstation.
```

### Gold

**Decision:** YES

**Answer:** The workload falls within the defined category and the authorization rule permits it on the managed workstation.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[
  "qualified local workload",
  "managed workstation"
]
```

### Sentinel V5

**Decision:** NO

**Rationale:** The evidence directly excludes the requested action.

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[]
```

---

## Failure 16 — challenge_v2__exception_activation__rel__p0__q0

- **Composition:** `exception_activation`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['exception_activation', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `exception_activation__latin4`
- **Evidence count:** 3
- **Input order:** `question_first`
- **Gold decision:** **YES**
- **Predicted decision:** **NO**
- **Citation set exact:** True
- **Condition relation:** `missing_all_gold_conditions`

### Question

Under the supplied evidence, may someone export the Maple corpus outside the approved environment?

### Evidence

```text
[E1] Export of the Maple corpus outside the approved environment is prohibited. An exception applies during a declared recovery event if written security clearance is recorded. A recovery event is declared and written clearance is recorded. The Prism records is covered by incident notification and inventory rules. The supplied evidence does not state whether export outside an approved environment is permitted or prohibited.

[E2] Export of the Orchid dataset outside the approved environment is prohibited. A declared recovery event activates an exception allowing export with written security clearance. A recovery event is declared, but written-clearance status is not established.

[E3] Export of the Noble files outside the approved environment is prohibited. An exception becomes available only during a declared recovery event. No recovery event has been declared, although security clearance is already recorded.
```

### Gold

**Decision:** YES

**Answer:** The exception is active and its clearance requirement is satisfied.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[
  "recovery event declared",
  "security clearance recorded"
]
```

### Sentinel V5

**Decision:** NO

**Rationale:** The exception cannot authorize the action because a required exception element remains unresolved.

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[]
```

---

## Failure 17 — challenge_v2__exception_activation__rel__p2__q2

- **Composition:** `exception_activation`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['exception_activation', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `exception_activation__latin4`
- **Evidence count:** 4
- **Input order:** `question_first`
- **Gold decision:** **YES**
- **Predicted decision:** **NO**
- **Citation set exact:** True
- **Condition relation:** `missing_all_gold_conditions`

### Question

Under the supplied evidence, may someone export the Orchid dataset outside the approved environment?

### Evidence

```text
[E1] Export of the Maple corpus outside the approved environment is prohibited. A declared recovery event activates an exception allowing export with written security clearance. A recovery event is declared, but written-clearance status is not established.

[E2] Export of the Orchid dataset outside the approved environment is prohibited. An exception applies during a declared recovery event if written security clearance is recorded. A recovery event is declared and written clearance is recorded.

[E3] Export of the Prism records outside the approved environment is prohibited. An exception becomes available only during a declared recovery event. No recovery event has been declared, although security clearance is already recorded.

[E4] The Noble files is covered by incident notification and inventory rules. The supplied evidence does not state whether export outside an approved environment is permitted or prohibited.
```

### Gold

**Decision:** YES

**Answer:** The exception is active and its clearance requirement is satisfied.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[
  "recovery event declared",
  "security clearance recorded"
]
```

### Sentinel V5

**Decision:** NO

**Rationale:** The exception cannot activate because the base prohibition remains active.

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[]
```

---

## Failure 18 — challenge_v2__exception_activation__stress__yes

- **Composition:** `exception_activation`
- **Question type:** `stress`
- **Reasoning tags:** `['exception_activation', 'stress_case']`
- **Relational group:** `None`
- **Evidence count:** 5
- **Input order:** `question_first`
- **Gold decision:** **YES**
- **Predicted decision:** **NO**
- **Citation set exact:** True
- **Condition relation:** `missing_all_gold_conditions`

### Question

Based only on these passages, is it permitted to export the Exception Activation stress asset 1 outside the approved environment?

### Evidence

```text
[E1] Annual asset inventories are due before the end of the fiscal year.

[E2] Export of the Exception Activation stress asset 1 outside the approved environment is prohibited. An exception applies during a declared recovery event if written security clearance is recorded. A recovery event is declared and written clearance is recorded.

[E3] Travel reimbursement requests must include itemized receipts.

[E4] Institution-owned devices must remain listed in the central inventory.

[E5] Public training materials may be distributed internally.
```

### Gold

**Decision:** YES

**Answer:** The exception is active and its clearance requirement is satisfied.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[
  "recovery event declared",
  "security clearance recorded"
]
```

### Sentinel V5

**Decision:** NO

**Rationale:** The exception cannot authorize the action because the exception trigger is inactive.

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[]
```

---

## Failure 19 — challenge_v2__nested_exception_requirements__rel__p1__q3

- **Composition:** `nested_exception_requirements`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['nested_exception_requirements', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `nested_exception_requirements__latin4`
- **Evidence count:** 2
- **Input order:** `question_first`
- **Gold decision:** **YES**
- **Predicted decision:** **NO**
- **Citation set exact:** True
- **Condition relation:** `missing_all_gold_conditions`

### Question

Under the supplied evidence, may someone transfer the Umber dataset to an external research partner?

### Evidence

```text
[E1] External transfer of the Quartz archive is prohibited. The only exception requires both an active institutional waiver and sponsor consent. The institutional waiver was denied, while sponsor consent is recorded. External transfer of the Summit corpus is prohibited. A collaboration exception requires an active institutional waiver and sponsor consent. The waiver is active, but sponsor-consent status is not established.

[E2] External transfer of the Umber dataset is prohibited. A collaboration exception applies when an institutional waiver is active and sponsor consent is recorded. The waiver is active and sponsor consent is recorded. The Talon files is subject to collaboration reporting and annual inventory reconciliation. The supplied evidence does not establish any rule governing external research-partner transfer.
```

### Gold

**Decision:** YES

**Answer:** The nested exception applies and both required conditions are satisfied.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[
  "institutional waiver active",
  "sponsor consent recorded"
]
```

### Sentinel V5

**Decision:** NO

**Rationale:** The evidence directly governs the exact proposition and prohibits it.

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[]
```

---

## Failure 20 — challenge_v2__nested_exception_requirements__rel__p2__q2

- **Composition:** `nested_exception_requirements`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['nested_exception_requirements', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `nested_exception_requirements__latin4`
- **Evidence count:** 2
- **Input order:** `evidence_first`
- **Gold decision:** **YES**
- **Predicted decision:** **NO**
- **Citation set exact:** True
- **Condition relation:** `missing_all_gold_conditions`

### Question

Under the supplied evidence, may someone transfer the Talon files to an external research partner?

### Evidence

```text
[E1] External transfer of the Umber dataset is prohibited. The only exception requires both an active institutional waiver and sponsor consent. The institutional waiver was denied, while sponsor consent is recorded. The Summit corpus is subject to collaboration reporting and annual inventory reconciliation. The supplied evidence does not establish any rule governing external research-partner transfer.

[E2] External transfer of the Talon files is prohibited. A collaboration exception applies when an institutional waiver is active and sponsor consent is recorded. The waiver is active and sponsor consent is recorded. External transfer of the Quartz archive is prohibited. A collaboration exception requires an active institutional waiver and sponsor consent. The waiver is active, but sponsor-consent status is not established.
```

### Gold

**Decision:** YES

**Answer:** The nested exception applies and both required conditions are satisfied.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[
  "institutional waiver active",
  "sponsor consent recorded"
]
```

### Sentinel V5

**Decision:** NO

**Rationale:** The exception cannot authorize the action because a required exception element has failed.

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[]
```

---

## Failure 21 — challenge_v2__nested_exception_requirements__rel__p3__q1

- **Composition:** `nested_exception_requirements`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['nested_exception_requirements', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `nested_exception_requirements__latin4`
- **Evidence count:** 4
- **Input order:** `question_first`
- **Gold decision:** **YES**
- **Predicted decision:** **NO**
- **Citation set exact:** True
- **Condition relation:** `missing_all_gold_conditions`

### Question

Under the supplied evidence, may someone transfer the Summit corpus to an external research partner?

### Evidence

```text
[E1] External transfer of the Umber dataset is prohibited. A collaboration exception requires an active institutional waiver and sponsor consent. The waiver is active, but sponsor-consent status is not established.

[E2] External transfer of the Talon files is prohibited. The only exception requires both an active institutional waiver and sponsor consent. The institutional waiver was denied, while sponsor consent is recorded.

[E3] The Quartz archive is subject to collaboration reporting and annual inventory reconciliation. The supplied evidence does not establish any rule governing external research-partner transfer.

[E4] External transfer of the Summit corpus is prohibited. A collaboration exception applies when an institutional waiver is active and sponsor consent is recorded. The waiver is active and sponsor consent is recorded.
```

### Gold

**Decision:** YES

**Answer:** The nested exception applies and both required conditions are satisfied.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E4"
]
```

**Conditions:**

```json
[
  "institutional waiver active",
  "sponsor consent recorded"
]
```

### Sentinel V5

**Decision:** NO

**Rationale:** The exception cannot authorize the action because a required exception element is not active.

**Citations:**

```json
[
  "E4"
]
```

**Conditions:**

```json
[]
```

---

## Failure 22 — challenge_v2__nested_exception_requirements__stress__yes

- **Composition:** `nested_exception_requirements`
- **Question type:** `stress`
- **Reasoning tags:** `['nested_exception_requirements', 'stress_case']`
- **Relational group:** `None`
- **Evidence count:** 2
- **Input order:** `evidence_first`
- **Gold decision:** **YES**
- **Predicted decision:** **NO**
- **Citation set exact:** True
- **Condition relation:** `missing_all_gold_conditions`

### Question

Based only on these passages, is it permitted to transfer the Nested Exception Requirements stress asset 1 to an external research partner?

### Evidence

```text
[E1] External transfer of the Nested Exception Requirements stress asset 1 is prohibited. A collaboration exception applies when an institutional waiver is active and sponsor consent is recorded. The waiver is active and sponsor consent is recorded.

[E2] Annual asset inventories are due before the end of the fiscal year.
```

### Gold

**Decision:** YES

**Answer:** The nested exception applies and both required conditions are satisfied.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[
  "institutional waiver active",
  "sponsor consent recorded"
]
```

### Sentinel V5

**Decision:** NO

**Rationale:** The exception cannot authorize the action because a controlling prohibition remains in effect.

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[]
```

---

## Failure 23 — challenge_v2__superseding_rule_precedence__rel__p0__q0

- **Composition:** `superseding_rule_precedence`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['superseding_rule_precedence', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `superseding_rule_precedence__latin4`
- **Evidence count:** 2
- **Input order:** `evidence_first`
- **Gold decision:** **YES**
- **Predicted decision:** **NO**
- **Citation set exact:** False
- **Condition relation:** `missing_all_gold_conditions`

### Question

Under the supplied evidence, may someone release the Ember files through the sponsor portal?

### Evidence

```text
[E1] A 2024 rule addressed release of the Juniper records. A 2026 rule expressly supersedes it and permits sponsor-portal release only with sponsor signoff. Current signoff status is not established. A 2024 release rule restricted sponsor-portal release of the Ember files. A 2026 rule expressly supersedes the 2024 rule and permits release after sponsor review. Sponsor review is complete.

[E2] A 2026 rule supersedes the prior retention schedule for the Kestrel archive and changes its archival review date. No supplied provision states whether the Kestrel archive may be released through a sponsor portal. A 2024 rule permitted sponsor-portal release of the Harbor dataset. A 2026 rule expressly supersedes it and requires sponsor signoff before release. Sponsor signoff was denied.
```

### Gold

**Decision:** YES

**Answer:** The superseding 2026 rule controls and its review requirement is satisfied.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[
  "2026 rule operative",
  "sponsor review complete"
]
```

### Sentinel V5

**Decision:** NO

**Rationale:** The higher-priority superseding rule controls and prohibits the requested action.

**Citations:**

```json
[
  "E2"
]
```

**Conditions:**

```json
[]
```

---

## Failure 24 — challenge_v2__superseding_rule_precedence__rel__p1__q3

- **Composition:** `superseding_rule_precedence`
- **Question type:** `relational_counterfactual`
- **Reasoning tags:** `['superseding_rule_precedence', 'dual_counterfactual', 'operative_rule_selection']`
- **Relational group:** `superseding_rule_precedence__latin4`
- **Evidence count:** 1
- **Input order:** `question_first`
- **Gold decision:** **YES**
- **Predicted decision:** **NO**
- **Citation set exact:** True
- **Condition relation:** `missing_all_gold_conditions`

### Question

Under the supplied evidence, may someone release the Kestrel archive through the sponsor portal?

### Evidence

```text
[E1] A 2024 rule permitted sponsor-portal release of the Ember files. A 2026 rule expressly supersedes it and requires sponsor signoff before release. Sponsor signoff was denied. A 2024 release rule restricted sponsor-portal release of the Kestrel archive. A 2026 rule expressly supersedes the 2024 rule and permits release after sponsor review. Sponsor review is complete. A 2026 rule supersedes the prior retention schedule for the Juniper records and changes its archival review date. No supplied provision states whether the Juniper records may be released through a sponsor portal. A 2024 rule addressed release of the Harbor dataset. A 2026 rule expressly supersedes it and permits sponsor-portal release only with sponsor signoff. Current signoff status is not established.
```

### Gold

**Decision:** YES

**Answer:** The superseding 2026 rule controls and its review requirement is satisfied.

**Evidence sufficient:** True

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[
  "2026 rule operative",
  "sponsor review complete"
]
```

### Sentinel V5

**Decision:** NO

**Rationale:** The evidence directly governs the exact proposition and prohibits it.

**Citations:**

```json
[
  "E1"
]
```

**Conditions:**

```json
[]
```

---

