export type VerificationStatus =
  | "Verified"
  | "Likely Verified"
  | "Not Verifiable"
  | "Unverified"
  | "Timeout";

export interface AuditEntry {
  event_type: string;
  event_payload: Record<string, unknown>;
  previous_hash: string | null;
  entry_hash: string;
  created_at: string;
}

export interface LayerResult {
  status: string;
  confidence?: number | null;
  explanation: string;
  details: Record<string, unknown>;
}

export interface VerificationResult {
  job_id: string;
  input_type: string;
  status: VerificationStatus;
  headline_message: string;
  short_reason: string;
  content_preview: Record<string, unknown>;
  proof_summary: Record<string, unknown>;
  analysis_report: Record<string, unknown>;
  layer_results: Record<string, LayerResult>;
  audit_trail: AuditEntry[];
  created_at: string;
}

export interface ConsensusVote {
  id: number;
  reporter_name: string | null;
  role: "user";
  vote_value: number;
  confidence: number;
  weight: number;
  report_reason: string | null;
  evidence_url: string | null;
  moderation_flag: boolean;
  created_at: string;
}

export interface ConsensusSummary {
  job_id: string;
  totals: Record<string, unknown>;
  votes: ConsensusVote[];
}

export interface FeedbackEntry {
  id: string;
  verification_record_id: string;
  manifest_id: string | null;
  feedback_type: "correction" | "dispute" | "source_suggestion" | "clarification" | "review_note";
  feedback_text: string;
  user_role: "user" | "expert" | "moderator";
  submitted_by: string;
  linked_source_url: string | null;
  created_at: string;
  status: "open" | "reviewed" | "resolved";
  moderator_note: string | null;
}

export interface FeedbackSummary {
  job_id: string;
  verification_record_id: string;
  current_manifest_id: string | null;
  totals: Record<string, unknown>;
  feedback_entries: FeedbackEntry[];
}

export interface HistoryItem {
  job_id: string;
  status: VerificationStatus;
  input_type: string;
  headline_message: string;
  created_at: string;
}

export interface PublisherKey {
  key_id: string;
  algorithm: string;
  public_key: string;
  is_active: boolean;
}

export interface ContentRecord {
  id: number;
  publisher_id: number;
  canonical_url: string | null;
  title: string | null;
  author: string | null;
  publish_time: string | null;
  normalized_text: string;
  content_hash: string;
  signature: string | null;
  key_id: string | null;
  source_metadata: Record<string, unknown>;
  created_at: string;
}

export interface Publisher {
  id: number;
  name: string;
  domain: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  public_keys: PublisherKey[];
}

export type CryptoObjectType =
  | "article_snapshot"
  | "evidence_bundle"
  | "review_event"
  | "verification_record"
  | "image_verification_record"
  | "audio_verification_record"
  | "video_verification_record";
export type IssuerType = "platform" | "publisher" | "journalist" | "fact_checker" | "moderator";
export type ProvenanceVerificationStatus =
  | "valid"
  | "hash_mismatch"
  | "signature_invalid"
  | "issuer_revoked"
  | "timestamp_invalid"
  | "version_chain_broken";

export interface IssuerCredential {
  id: string;
  issuer_type: IssuerType;
  issuer_name: string;
  public_key: string;
  status: string;
  issued_at: string;
  expires_at: string | null;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface ArticleSnapshot {
  id: string;
  source_url: string;
  publisher_name: string;
  author_name: string | null;
  title: string;
  published_at: string | null;
  captured_at: string;
  normalized_article_text: string;
  media_urls: unknown[];
  media_hashes: unknown[];
  metadata: Record<string, unknown>;
  previous_version_id: string | null;
  created_at: string;
}

export interface EvidenceBundle {
  id: string;
  linked_article_id: string;
  source_links: unknown[];
  extracted_claims: unknown[];
  supporting_notes: string | null;
  created_by: string;
  created_at: string;
  metadata: Record<string, unknown>;
  previous_version_id: string | null;
}

export interface ReviewEvent {
  id: string;
  linked_article_id: string;
  event_type: "fact_check" | "correction" | "retraction" | "update" | "moderation_note";
  event_payload: Record<string, unknown>;
  created_by: string;
  created_at: string;
  previous_version_id: string | null;
}

export interface ProvenanceManifest {
  id: string;
  object_type: CryptoObjectType;
  object_id: string;
  canonical_json: string;
  sha256_hash: string;
  signature: string;
  signing_algorithm: string;
  signer_id: string;
  signer_certificate_or_public_key_reference: string;
  timestamp_token: Record<string, unknown>;
  created_at: string;
  previous_manifest_id: string | null;
  verification_status: ProvenanceVerificationStatus;
}

export interface ManifestVerificationResult {
  manifest_id: string;
  object_type: CryptoObjectType;
  object_id: string;
  status: ProvenanceVerificationStatus;
  valid: boolean;
  checks: Record<string, unknown>;
}

export interface ProvenanceHistory {
  object_type: CryptoObjectType;
  record_id: string;
  versions: Record<string, unknown>[];
  manifests: Record<string, unknown>[];
}

export interface VersionCompareResult {
  object_type: CryptoObjectType;
  left_version_id: string;
  right_version_id: string;
  differences: Array<Record<string, unknown>>;
  changed_fields: string[];
}

export interface SimilarityMatch {
  title: string | null;
  url: string | null;
  score: number;
  explanation: string;
  source: string;
  snippet: string | null;
  domain: string | null;
  relation?: string | null;
  evidence_relation?: string | null;
  aboutness?: string | null;
  stance?: string | null;
  exact_claim_support?: boolean | null;
  evidence_quote?: string | null;
  relation_confidence?: number | null;
  rerank_score?: number | null;
  embedding_similarity?: number | null;
  entity_match_score?: number | null;
  location_match_score?: number | null;
  location_match_pass?: boolean | null;
  event_match_score?: number | null;
  context_match_pass?: boolean | null;
  mismatch_reason?: string | null;
  reason_accepted_or_rejected?: string | null;
}

export interface SimilarityMatchesPage {
  items: SimilarityMatch[];
  total: number;
  offset: number;
  limit: number;
  has_more: boolean;
}
