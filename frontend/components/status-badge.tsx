import clsx from "clsx";

import { VerificationStatus } from "@/lib/types";

const toneMap: Record<VerificationStatus, string> = {
  Verified: "bg-pine text-white",
  "Likely Verified": "bg-accent text-white",
  "Not Verifiable": "bg-mist text-ink",
  Unverified: "bg-warning text-white",
  Timeout: "bg-black text-white"
};

export function StatusBadge({ status }: { status: VerificationStatus }) {
  return (
    <span className={clsx("inline-flex rounded-full px-4 py-2 text-sm font-semibold", toneMap[status])}>
      {status}
    </span>
  );
}
