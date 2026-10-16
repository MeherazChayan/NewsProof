import Link from "next/link";

import { AnalysisReport } from "@/components/analysis-report";
import { ConsensusPanel } from "@/components/consensus-panel";
import { FeedbackPanel } from "@/components/feedback-panel";
import { ResultHeader } from "@/components/result-header";
import { getConsensus, getFeedback, getResult } from "@/lib/api";

export default async function ResultPage({ params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await params;
  const result = await getResult(jobId);
  const hasReportLayers = Object.keys(result.layer_results ?? {}).length > 0;
  const canShowFeedback = hasReportLayers && (result.input_type === "text" || result.input_type === "image" || result.input_type === "audio" || result.input_type === "video");
  const feedback = canShowFeedback ? await getFeedback(jobId) : null;
  const consensus = hasReportLayers && !canShowFeedback ? await getConsensus(jobId) : null;

  return (
    <main className="report-page">
      <div className="no-print mb-4 flex justify-between text-sm">
        <Link href="/">New verification</Link>
        <Link href="/history">History</Link>
      </div>
      <ResultHeader result={result} />
      <section className="panel mt-8 p-6">
        <h2 className="font-display text-2xl">{result.input_type === "text" ? "Text Preview" : "Content Preview"}</h2>
        <div className="mt-4 space-y-3 text-sm text-black/80">
          <p className="text-xl font-medium">{String(result.content_preview.title ?? "Untitled content")}</p>
          <p>{String(result.content_preview.text_excerpt ?? "No preview available.")}</p>
          {result.input_type !== "text" && result.content_preview.canonical_url ? (
            <div className="flex flex-wrap gap-3">
              <a
                className="inline-flex items-center gap-2 rounded-full border border-black/10 px-4 py-2 text-sm font-medium text-black transition hover:border-black/30"
                href={String(result.content_preview.canonical_url)}
                target="_blank"
                rel="noreferrer"
              >
                <span aria-hidden="true">↗</span>
                Open full article
              </a>
            </div>
          ) : null}
        </div>
      </section>
      {hasReportLayers ? (
        <AnalysisReport result={result} />
      ) : (
        <section className="panel mt-8 p-6">
          <h2 className="font-display text-2xl">Report unavailable</h2>
          <div className="mt-4 space-y-3 text-sm text-black/75">
            <p>The verification did not complete, so no analysis layers were generated for this result.</p>
            <p>{result.short_reason || "Try running the verification again, or use a source that can be fetched without timing out."}</p>
          </div>
        </section>
      )}
      {canShowFeedback ? (
        feedback ? <FeedbackPanel jobId={jobId} feedback={feedback} /> : null
      ) : (
        consensus ? <ConsensusPanel jobId={jobId} consensus={consensus} /> : null
      )}
    </main>
  );
}
