"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { submitFeedback } from "@/lib/api";
import { FeedbackEntry, FeedbackSummary } from "@/lib/types";

export function FeedbackPanel({ jobId, feedback }: { jobId: string; feedback: FeedbackSummary }) {
  const router = useRouter();
  const [submittedBy, setSubmittedBy] = useState("");
  const [feedbackType, setFeedbackType] = useState<FeedbackEntry["feedback_type"]>("clarification");
  const [userRole, setUserRole] = useState<FeedbackEntry["user_role"]>("user");
  const [feedbackText, setFeedbackText] = useState("");
  const [linkedSourceUrl, setLinkedSourceUrl] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [entries, setEntries] = useState<FeedbackEntry[]>(feedback.feedback_entries);

  const totals = useMemo(() => {
    return {
      total: entries.length,
      open: entries.filter((entry) => entry.status === "open").length,
      reviewed: entries.filter((entry) => entry.status === "reviewed").length,
      resolved: entries.filter((entry) => entry.status === "resolved").length,
      expert: entries.filter((entry) => entry.user_role === "expert").length,
      moderator: entries.filter((entry) => entry.user_role === "moderator").length
    };
  }, [entries]);

  async function handleSubmit() {
    try {
      setSubmitting(true);
      setError(null);
      const saved = await submitFeedback(jobId, {
        feedback_type: feedbackType,
        feedback_text: feedbackText,
        submitted_by: submittedBy || "Anonymous reviewer",
        user_role: userRole,
        linked_source_url: linkedSourceUrl || undefined
      });
      setEntries((current) => [saved, ...current]);
      setSubmittedBy("");
      setFeedbackType("clarification");
      setUserRole("user");
      setFeedbackText("");
      setLinkedSourceUrl("");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit feedback");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="panel mt-8 p-6">
      <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
        <div>
          <h2 className="font-display text-2xl">Feedback and Review</h2>
          <p className="mt-3 text-sm text-black/70">
            NewsProof stores community input here as structured feedback, review notes, disputes, and correction requests.
            It does not act as a truth-by-voting signal or affect the numeric score directly.
          </p>
          <div className="mt-4 space-y-2 text-sm text-black/75">
            <p>Total feedback: {String(totals.total)}</p>
            <p>Open: {String(totals.open)}</p>
            <p>Reviewed: {String(totals.reviewed)}</p>
            <p>Resolved: {String(totals.resolved)}</p>
            <p>Expert notes: {String(totals.expert)}</p>
            <p>Moderator notes: {String(totals.moderator)}</p>
          </div>
          <div className="mt-5 space-y-3">
            {entries.map((entry) => (
              <div key={entry.id} className="rounded-2xl border border-black/10 bg-paper/60 p-3 text-sm">
                <div className="flex items-center justify-between gap-3">
                  <div className="font-medium">{entry.submitted_by}</div>
                  <div className="text-xs uppercase text-black/55">{entry.feedback_type.replaceAll("_", " ")}</div>
                </div>
                <p className="mt-2 text-black/75">{entry.feedback_text}</p>
                <div className="mt-3 flex flex-wrap gap-3 text-xs text-black/55">
                  <span>Role: {entry.user_role}</span>
                  <span>Status: {entry.status}</span>
                  <span>{new Date(entry.created_at).toLocaleString()}</span>
                </div>
                {entry.linked_source_url ? (
                  <a className="mt-2 block text-xs text-accent underline" href={entry.linked_source_url} target="_blank" rel="noreferrer">
                    Linked source
                  </a>
                ) : null}
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-lg font-medium">Submit feedback</h3>
          <div className="mt-4 space-y-3">
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Your name"
              value={submittedBy}
              onChange={(event) => setSubmittedBy(event.target.value)}
            />
            <select
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              value={userRole}
              onChange={(event) => setUserRole(event.target.value as FeedbackEntry["user_role"])}
            >
              <option value="user">User</option>
              <option value="expert">Expert</option>
              <option value="moderator">Moderator</option>
            </select>
            <select
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              value={feedbackType}
              onChange={(event) => setFeedbackType(event.target.value as FeedbackEntry["feedback_type"])}
            >
              <option value="clarification">Clarification</option>
              <option value="source_suggestion">Source suggestion</option>
              <option value="correction">Correction request</option>
              <option value="dispute">Dispute report</option>
              <option value="review_note">Review note</option>
            </select>
            <textarea
              className="min-h-32 w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Explain the correction, dispute, additional source, or review note."
              value={feedbackText}
              onChange={(event) => setFeedbackText(event.target.value)}
            />
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Linked source URL (optional)"
              value={linkedSourceUrl}
              onChange={(event) => setLinkedSourceUrl(event.target.value)}
            />
            {error ? <p className="text-sm text-warning">{error}</p> : null}
            <button
              type="button"
              onClick={handleSubmit}
              disabled={submitting || feedbackText.trim().length < 3}
              className="rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white disabled:opacity-50"
            >
              {submitting ? "Submitting..." : "Submit feedback"}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
