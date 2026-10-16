import { render, screen } from "@testing-library/react";

import { StatusBadge } from "@/components/status-badge";

describe("StatusBadge", () => {
  it("renders the verification status", () => {
    render(<StatusBadge status="Verified" />);
    expect(screen.getByText("Verified")).toBeInTheDocument();
  });
});

