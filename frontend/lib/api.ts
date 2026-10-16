import {
  ArticleSnapshot,
  ConsensusSummary,
  ConsensusVote,
  ContentRecord,
  EvidenceBundle,
  FeedbackEntry,
  FeedbackSummary,
  HistoryItem,
  IssuerCredential,
  ManifestVerificationResult,
  ProvenanceHistory,
  ProvenanceManifest,
  Publisher,
  ReviewEvent,
  SimilarityMatchesPage,
  VerificationResult,
  VersionCompareResult
} from "@/lib/types";

const DEFAULT_LOCAL_API_BASE_URL = "http://localhost:8000";

function withApiPrefix(baseUrl: string | undefined) {
  const raw = (baseUrl || DEFAULT_LOCAL_API_BASE_URL).trim();
  if (raw === "same-origin" || raw === "/") {
    return "/api/v1";
  }
  const trimmed = raw.replace(/\/+$/, "");
  return trimmed.endsWith("/api/v1") ? trimmed : `${trimmed}/api/v1`;
}

function getApiBase() {
  if (typeof window === "undefined") {
    return withApiPrefix(process.env.INTERNAL_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL);
  }
  return withApiPrefix(process.env.NEXT_PUBLIC_API_BASE_URL);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBase()}${path}`, {
    ...init,
    cache: "no-store",
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers
    }
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  return response.json();
}

export async function verifyUrl(url: string): Promise<VerificationResult> {
  return request<VerificationResult>("/verify/url?fast=true", {
    method: "POST",
    body: JSON.stringify({ url })
  });
}

export async function verifyText(text: string): Promise<VerificationResult> {
  return request<VerificationResult>("/verify/text", {
    method: "POST",
    body: JSON.stringify({ text })
  });
}

export async function verifyImage(file: File, captionOrClaim?: string): Promise<VerificationResult> {
  const formData = new FormData();
  formData.append("file", file);
  if (captionOrClaim?.trim()) {
    formData.append("caption_or_claim", captionOrClaim.trim());
  }
  return request<VerificationResult>("/verify/image", {
    method: "POST",
    body: formData
  });
}

export async function verifyAudio(file: File, captionOrClaim?: string, claimedSpeaker?: string): Promise<VerificationResult> {
  const formData = new FormData();
  formData.append("file", file);
  if (captionOrClaim?.trim()) {
    formData.append("caption_or_claim", captionOrClaim.trim());
  }
  if (claimedSpeaker?.trim()) {
    formData.append("claimed_speaker", claimedSpeaker.trim());
  }
  return request<VerificationResult>("/verify/audio", {
    method: "POST",
    body: formData
  });
}

export async function verifyVideo(file: File, captionOrClaim?: string): Promise<VerificationResult> {
  const formData = new FormData();
  formData.append("file", file);
  if (captionOrClaim?.trim()) {
    formData.append("caption_or_claim", captionOrClaim.trim());
  }
  return request<VerificationResult>("/verify/video", {
    method: "POST",
    body: formData
  });
}

export async function getResult(jobId: string): Promise<VerificationResult> {
  return request<VerificationResult>(`/results/${jobId}`);
}

export async function getSimilarityMatches(jobId: string, offset = 0, limit = 10): Promise<SimilarityMatchesPage> {
  return request<SimilarityMatchesPage>(`/results/${jobId}/matches?offset=${offset}&limit=${limit}`);
}

export async function getHistory(): Promise<{ items: HistoryItem[] }> {
  return request<{ items: HistoryItem[] }>("/history");
}

export async function getPublishers(): Promise<Publisher[]> {
  return request<Publisher[]>("/publishers");
}

export async function createPublisher(payload: Omit<Publisher, "id" | "created_at" | "updated_at">): Promise<Publisher> {
  return request<Publisher>("/publishers", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function updatePublisher(
  id: number,
  payload: Omit<Publisher, "id" | "created_at" | "updated_at">
): Promise<Publisher> {
  return request<Publisher>(`/publishers/${id}`, {
    method: "PUT",
    body: JSON.stringify(payload)
  });
}

export async function deletePublisher(id: number): Promise<void> {
  await request(`/publishers/${id}`, {
    method: "DELETE"
  });
}

export async function getContentRecords(publisherId?: number): Promise<ContentRecord[]> {
  const query = publisherId ? `?publisher_id=${publisherId}` : "";
  return request<ContentRecord[]>(`/content-records${query}`);
}

export async function createContentRecord(payload: Omit<ContentRecord, "id" | "created_at">): Promise<ContentRecord> {
  return request<ContentRecord>("/content-records", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function updateContentRecord(
  id: number,
  payload: Omit<ContentRecord, "id" | "created_at" | "publisher_id">
): Promise<ContentRecord> {
  return request<ContentRecord>(`/content-records/${id}`, {
    method: "PUT",
    body: JSON.stringify(payload)
  });
}

export async function deleteContentRecord(id: number): Promise<void> {
  await request(`/content-records/${id}`, {
    method: "DELETE"
  });
}

export async function getConsensus(jobId: string): Promise<ConsensusSummary> {
  return request<ConsensusSummary>(`/consensus/${jobId}`);
}

export async function createConsensusVote(
  jobId: string,
  payload: {
    reporter_name?: string;
    vote_value: number;
    report_reason?: string;
  }
): Promise<ConsensusVote> {
  return request<ConsensusVote>(`/consensus/${jobId}/votes`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function getFeedback(jobId: string): Promise<FeedbackSummary> {
  return request<FeedbackSummary>(`/feedback/${jobId}`);
}

export async function submitFeedback(
  jobId: string,
  payload: {
    feedback_type: "correction" | "dispute" | "source_suggestion" | "clarification" | "review_note";
    feedback_text: string;
    submitted_by: string;
    user_role: "user" | "expert" | "moderator";
    linked_source_url?: string;
    moderator_note?: string;
  }
): Promise<FeedbackEntry> {
  return request<FeedbackEntry>(`/feedback/${jobId}`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function getCryptoIssuers(): Promise<IssuerCredential[]> {
  return request<IssuerCredential[]>("/crypto/issuers");
}

export async function createArticleSnapshot(payload: {
  source_url: string;
  publisher_name: string;
  author_name?: string;
  title: string;
  published_at?: string | null;
  captured_at?: string | null;
  normalized_article_text: string;
  media_urls: string[];
  metadata?: Record<string, unknown>;
  previous_version_id?: string | null;
}): Promise<ArticleSnapshot> {
  return request<ArticleSnapshot>("/crypto/article-snapshots", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function createEvidenceBundle(payload: {
  linked_article_id: string;
  source_links: string[];
  extracted_claims: string[];
  supporting_notes?: string;
  created_by: string;
  created_at?: string | null;
  metadata?: Record<string, unknown>;
  previous_version_id?: string | null;
}): Promise<EvidenceBundle> {
  return request<EvidenceBundle>("/crypto/evidence-bundles", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function createReviewEvent(payload: {
  linked_article_id: string;
  event_type: "fact_check" | "correction" | "retraction" | "update" | "moderation_note";
  event_payload: Record<string, unknown>;
  created_by: string;
  created_at?: string | null;
  previous_version_id?: string | null;
}): Promise<ReviewEvent> {
  return request<ReviewEvent>("/crypto/review-events", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function signProvenanceObject(payload: {
  object_type: "article_snapshot" | "evidence_bundle" | "review_event";
  object_id: string;
  signer_id?: string | null;
}): Promise<ProvenanceManifest> {
  return request<ProvenanceManifest>("/crypto/manifests/sign", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function verifyProvenanceManifest(payload: {
  manifest_id: string;
  verified_by?: string;
}): Promise<ManifestVerificationResult> {
  return request<ManifestVerificationResult>("/crypto/manifests/verify", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export async function getProvenanceHistory(
  objectType: "article_snapshot" | "evidence_bundle" | "review_event",
  recordId: string
): Promise<ProvenanceHistory> {
  return request<ProvenanceHistory>(`/crypto/records/${objectType}/${recordId}/history`);
}

export async function compareProvenanceVersions(
  objectType: "article_snapshot" | "evidence_bundle" | "review_event",
  leftVersionId: string,
  rightVersionId: string,
  recordId: string
): Promise<VersionCompareResult> {
  return request<VersionCompareResult>(
    `/crypto/records/${objectType}/${recordId}/compare?left_version_id=${leftVersionId}&right_version_id=${rightVersionId}`
  );
}
