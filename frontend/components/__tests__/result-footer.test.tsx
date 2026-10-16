import { render, screen } from "@testing-library/react";

import { ResultFooter } from "@/components/result-footer";

describe("ResultFooter", () => {
  it("renders proof summary and audit trail", () => {
    render(
      <ResultFooter
        result={{
          job_id: "job-1",
          status: "Not Verifiable",
          headline_message: "Not verifiable",
          short_reason: "No proof",
          content_preview: {},
          proof_summary: { content_hash: "abc" },
          analysis_report: {},
          layer_results: {},
          audit_trail: [
            {
              event_type: "job_created",
              event_payload: {},
              previous_hash: null,
              entry_hash: "hash-1",
              created_at: new Date().toISOString()
            }
          ],
          created_at: new Date().toISOString()
        }}
      />
    );

    expect(screen.getByText("Proof Summary")).toBeInTheDocument();
    expect(screen.getByText("Audit Trail")).toBeInTheDocument();
    expect(screen.getByText("job_created")).toBeInTheDocument();
  });
});
