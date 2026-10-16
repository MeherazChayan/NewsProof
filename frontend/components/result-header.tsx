import { DownloadReportButton } from "@/components/download-report-button";
import { StatusBadge } from "@/components/status-badge";
import { VerificationResult } from "@/lib/types";

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

export function ResultHeader({ result }: { result: VerificationResult }) {
  const analysisReport = asRecord(result.analysis_report);
  const proofSummary = asRecord(result.proof_summary);
  const systemScoring = asRecord(analysisReport.system_scoring ?? proofSummary.system_scoring);
  const derivedLabel = systemScoring.final_label;
  const derivedReason = systemScoring.reason ?? systemScoring.explanation;
  const headerLabel =
    typeof derivedLabel === "string" && derivedLabel.trim().length > 0 ? derivedLabel : result.headline_message;
  const headerReason =
    typeof derivedReason === "string" && derivedReason.trim().length > 0 ? derivedReason : result.short_reason;
  const badgeStatus =
    typeof derivedLabel === "string" && derivedLabel.trim().length > 0 ? (derivedLabel as VerificationResult["status"]) : result.status;

  return (
    <header className="panel p-6 md:p-8">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-black/60">Verification outcome</p>
          <h1 className="mt-2 font-display text-4xl">{headerLabel}</h1>
          <p className="mt-3 max-w-2xl text-sm text-black/70">{headerReason}</p>
        </div>
        <div className="flex flex-col items-start gap-3 md:items-end">
          <StatusBadge status={badgeStatus} />
          <div className="no-print">
            <DownloadReportButton reportTitle={`${headerLabel} report`} />
          </div>
        </div>
      </div>
    </header>
  );
}
