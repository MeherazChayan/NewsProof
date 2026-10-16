"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";

import { createConsensusVote } from "@/lib/api";
import { ConsensusSummary, ConsensusVote } from "@/lib/types";

export function ConsensusPanel({ jobId, consensus }: { jobId: string; consensus: ConsensusSummary }) {
  const router = useRouter();
  const [reporterName, setReporterName] = useState("");
  const [voteValue, setVoteValue] = useState(1);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [localVotes, setLocalVotes] = useState<ConsensusVote[]>(consensus.votes);

  const totals = useMemo(() => {
    const trueVotes = localVotes.filter((vote) => vote.vote_value > 0).length;
    const falseVotes = localVotes.filter((vote) => vote.vote_value < 0).length;
    const notSureVotes = localVotes.filter((vote) => vote.vote_value === 0).length;
    const totalVotes = localVotes.length;
    const majority = [
      ["true", trueVotes],
      ["false", falseVotes],
      ["not sure", notSureVotes],
    ].sort((left, right) => Number(right[1]) - Number(left[1]))[0];
    return {
      trueVotes,
      falseVotes,
      notSureVotes,
      totalVotes,
      majorityVote: totalVotes ? String(majority[0]) : "none",
      majorityConfidence: totalVotes ? Number(majority[1]) / totalVotes : 0,
    };
  }, [localVotes]);

  async function submitVote() {
    try {
      setSubmitting(true);
      setError(null);
      const recordedVote = await createConsensusVote(jobId, {
        reporter_name: reporterName || undefined,
        vote_value: voteValue
      });
      setLocalVotes((current) => [recordedVote, ...current]);
      setReporterName("");
      setVoteValue(1);
      router.refresh();
      window.setTimeout(() => {
        window.location.reload();
      }, 150);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit vote");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section className="panel mt-8 p-6">
      <div className="grid gap-6 lg:grid-cols-[0.95fr_1.05fr]">
        <div>
          <h2 className="font-display text-2xl">Community Verification</h2>
          <div className="mt-4 space-y-2 text-sm text-black/75">
            <p>Votes: {String(totals.totalVotes)}</p>
            <p>True: {String(totals.trueVotes)}</p>
            <p>False: {String(totals.falseVotes)}</p>
            <p>Not sure: {String(totals.notSureVotes)}</p>
            <p>Majority: {totals.majorityVote}</p>
            <p>Majority confidence: {totals.majorityConfidence.toFixed(3)}</p>
          </div>
          <div className="mt-5 space-y-3">
            {localVotes.map((vote) => (
              <div key={vote.id} className="rounded-2xl border border-black/10 bg-paper/60 p-3 text-sm">
                <div className="flex items-center justify-between gap-3">
                  <div className="font-medium">{vote.reporter_name || "Anonymous user"}</div>
                  <div className="text-xs uppercase text-black/55">
                    {vote.vote_value > 0 ? "True" : vote.vote_value < 0 ? "False" : "Not sure"}
                  </div>
                </div>
                <p className="mt-2 text-xs text-black/60">Recorded user vote</p>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-lg font-medium">Cast Vote</h3>
          <div className="mt-4 space-y-3">
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Your name"
              value={reporterName}
              onChange={(event) => setReporterName(event.target.value)}
            />
            <select
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              value={voteValue}
              onChange={(event) => setVoteValue(Number(event.target.value))}
            >
              <option value={1}>Content is true</option>
              <option value={-1}>Content is false</option>
              <option value={0}>Not sure</option>
            </select>
            {error ? <p className="text-sm text-warning">{error}</p> : null}
            <button
              type="button"
              onClick={submitVote}
              disabled={submitting}
              className="rounded-full bg-ink px-5 py-3 text-sm font-semibold text-white disabled:opacity-50"
            >
              {submitting ? "Submitting..." : "Submit vote"}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
