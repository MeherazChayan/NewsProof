import { VerificationResult } from "@/lib/types";

export function ResultFooter({ result }: { result: VerificationResult }) {
  const auditLayer =
    result.layer_results.layer4 && typeof result.layer_results.layer4 === "object"
      ? result.layer_results.layer4
      : null;

  return (
    <footer className="panel mt-8 p-6">
      <div className="grid gap-6 md:grid-cols-2">
        <section>
          <h2 className="font-display text-xl">Proof Summary</h2>
          <pre className="mt-3 overflow-auto rounded-2xl bg-ink p-4 text-xs text-white">
            {JSON.stringify(result.proof_summary, null, 2)}
          </pre>
        </section>
        <section>
          <div className="flex items-center justify-between gap-4">
            <h2 className="font-display text-xl">Audit Trail</h2>
            {auditLayer ? (
              <span className="rounded-full border border-black/10 bg-paper/70 px-3 py-1 text-xs font-medium uppercase tracking-[0.2em] text-black/70">
                {String(auditLayer.status)}
              </span>
            ) : null}
          </div>
          {auditLayer ? <p className="mt-2 text-sm text-black/70">{String(auditLayer.explanation)}</p> : null}
          <div className="mt-3 space-y-3">
            {result.audit_trail.map((entry) => (
              <div key={entry.entry_hash} className="rounded-2xl border border-black/10 bg-paper/60 p-3 text-sm">
                <div className="font-medium">{entry.event_type}</div>
                <div className="text-xs text-black/60">{new Date(entry.created_at).toLocaleString()}</div>
                <div className="mt-2 text-xs break-all">hash: {entry.entry_hash}</div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </footer>
  );
}
