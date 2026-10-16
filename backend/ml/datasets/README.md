# NewsProof Claim-Evidence Datasets

This folder stores training data for the claim-evidence verifier.

Required JSONL columns:

- `claim_text`
- `evidence_title`
- `evidence_snippet`
- `evidence_body`
- `claim_language`
- `claim_location`
- `evidence_location`
- `claim_entities`
- `evidence_entities`
- `event_type`
- `relation_label`
- `mismatch_reason`
- `source_url`
- `source_reliability`

Supported labels:

- `supports`
- `contradicts`
- `partially_related`
- `background_only`
- `unrelated`
- `insufficient`

Seed data in `newsproof_claim_evidence_seed.jsonl` focuses on hard negatives:
same topic with wrong location, same keywords with wrong entity/event, fake
claims with no direct support, and matching Bangla claim/evidence pairs.

Public datasets such as FEVER and X-FACT should be converted into this schema
before fine-tuning. FEVER labels can be mapped as:

- `SUPPORTS` -> `supports`
- `REFUTES` -> `contradicts`
- `NOT ENOUGH INFO` -> `insufficient`

NewsProof-specific hard negatives should be kept in the training mix because
generic NLI data often under-penalizes wrong-location evidence.
