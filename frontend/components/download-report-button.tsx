"use client";

type DownloadReportButtonProps = {
  reportTitle: string;
};

function sanitizeTitle(value: string): string {
  const normalized = value.trim().replace(/\s+/g, " ");
  if (!normalized) {
    return "newsproof-report";
  }

  return normalized
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80) || "newsproof-report";
}

export function DownloadReportButton({ reportTitle }: DownloadReportButtonProps) {
  const handleDownload = () => {
    const previousTitle = document.title;
    document.title = sanitizeTitle(reportTitle);
    window.print();
    window.setTimeout(() => {
      document.title = previousTitle;
    }, 0);
  };

  return (
    <button
      type="button"
      onClick={handleDownload}
      className="inline-flex items-center gap-2 rounded-full border border-black/10 bg-white px-4 py-2 text-sm font-medium text-black transition hover:border-black/30 hover:bg-paper/80"
    >
      <span aria-hidden="true">↓</span>
      Download report as PDF
    </button>
  );
}
