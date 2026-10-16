import Link from "next/link";

import { ProvenanceWorkbench } from "@/components/provenance-workbench";

export default function ProvenancePage() {
  return (
    <main className="space-y-6">
      <div className="flex justify-between">
        <div>
          <h1 className="font-display text-4xl">Cryptographic Provenance</h1>
          <p className="mt-2 max-w-3xl text-sm text-black/65">
            Create signed article snapshots, verify manifests, and inspect immutable version history for platform-issued provenance records.
          </p>
        </div>
        <Link href="/">Back home</Link>
      </div>
      <ProvenanceWorkbench />
    </main>
  );
}
