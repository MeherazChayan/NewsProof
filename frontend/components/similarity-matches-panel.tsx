"use client";

import { useState } from "react";

import { getSimilarityMatches } from "@/lib/api";
import { SimilarityMatch } from "@/lib/types";

function asText(value: unknown) {
  return typeof value === "string" ? value : "";
}

function formatSimilarityScore(value: unknown) {
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : "Not calculated";
}

function formatBoolean(value: unknown) {
  if (value === true) return "Pass";
  if (value === false) return "Fail";
  return "Not calculated";
}

export function SimilarityMatchesPanel({
  jobId,
  initialMatches,
  initialHasMore,
  initialTotal,
}: {
  jobId: string;
  initialMatches: SimilarityMatch[];
  initialHasMore: boolean;
  initialTotal: number;
}) {
  const [matches, setMatches] = useState<SimilarityMatch[]>(initialMatches);
  const [hasMore, setHasMore] = useState(initialHasMore);
  const [total, setTotal] = useState(initialTotal);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadMore() {
    try {
      setLoading(true);
      setError(null);
      const page = await getSimilarityMatches(jobId, matches.length, 5);
      setMatches((current) => [...current, ...page.items]);
      setHasMore(page.has_more);
      setTotal(page.total);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load more similar sources");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mt-4 space-y-3">
      {matches.length ? (
        matches.map((item, index) => (
          <div key={`${asText(item.url) || index}-${index}`} className="rounded-2xl border border-black/10 bg-white/70 p-4 text-sm text-black/75">
            {item.title ? <p className="font-medium text-black/85">{item.title}</p> : null}
            {item.url ? (
              <a className="mt-1 block break-all text-accent underline" href={item.url} target="_blank" rel="noreferrer">
                {item.url}
              </a>
            ) : null}
            {item.source ? <p className="mt-2"><strong>Source:</strong> {item.source}</p> : null}
            {item.domain ? <p><strong>Domain:</strong> {item.domain}</p> : null}
            <div className="mt-2 grid gap-1 sm:grid-cols-2">
              <p><strong>Relation:</strong> {item.evidence_relation || item.relation || "insufficient"}</p>
              <p><strong>Aboutness:</strong> {item.aboutness || "unknown"}</p>
              <p><strong>Source stance:</strong> {item.stance || "unknown"}</p>
              <p><strong>Exact claim support:</strong> {item.exact_claim_support === true ? "Yes" : "No"}</p>
              <p><strong>Relation confidence:</strong> {formatSimilarityScore(item.relation_confidence)}</p>
              <p><strong>Rerank score:</strong> {formatSimilarityScore(item.rerank_score)}</p>
              <p><strong>Embedding similarity:</strong> {formatSimilarityScore(item.embedding_similarity ?? item.score)}</p>
              <p><strong>Entity match:</strong> {formatSimilarityScore(item.entity_match_score)}</p>
              <p><strong>Location match:</strong> {formatBoolean(item.location_match_pass)} ({formatSimilarityScore(item.location_match_score)})</p>
              <p><strong>Event match:</strong> {formatSimilarityScore(item.event_match_score)}</p>
            </div>
            {item.snippet ? <p className="mt-2">{item.snippet}</p> : null}
            {item.evidence_quote ? <p className="mt-2 rounded-xl border border-black/10 bg-paper/50 p-3 text-xs text-black/70"><strong>Evidence quote:</strong> {item.evidence_quote}</p> : null}
            {item.reason_accepted_or_rejected ? <p className="mt-2 text-xs text-black/65">{item.reason_accepted_or_rejected}</p> : null}
            {item.mismatch_reason ? <p className="mt-1 text-xs text-warning">{item.mismatch_reason}</p> : null}
            {item.explanation ? <p className="mt-2 text-xs text-black/60">{item.explanation}</p> : null}
          </div>
        ))
      ) : (
        <p className="text-sm text-black/65">No corroborating article matches were returned for this run.</p>
      )}

      {error ? <p className="text-sm text-warning">{error}</p> : null}

      {matches.length > 0 ? (
        <p className="text-xs text-black/55">Showing {matches.length} of {total} similar sources.</p>
      ) : null}

      {hasMore ? (
        <button
          type="button"
          onClick={loadMore}
          disabled={loading}
          className="rounded-full border border-black/10 px-4 py-2 text-sm font-semibold text-black disabled:opacity-50"
        >
          {loading ? "Loading more..." : "Load 5 more similar"}
        </button>
      ) : matches.length > 0 ? (
        <p className="text-xs text-black/55">No more similar sources were found.</p>
      ) : null}
    </div>
  );
}
