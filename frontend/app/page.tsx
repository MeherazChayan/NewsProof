import Link from "next/link";

import { HomeForm } from "@/components/home-form";

export default function HomePage() {
  return (
    <main className="space-y-8">
      <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="panel p-6 md:p-8">
          <p className="text-sm uppercase tracking-[0.3em] text-black/60">NewsProof</p>
          <h1 className="mt-4 font-display text-5xl leading-tight">
            Verify where a story came from before you trust where it is going.
          </h1>
          <p className="mt-4 max-w-2xl text-base text-black/70">
            Submit a URL, paste article text, or upload image, audio, and video files. NewsProof extracts what it can,
            checks trusted publisher proofs, adds supporting semantic evidence, and records every step in a
            tamper-evident audit trail.
          </p>
          <div className="mt-6 flex gap-3 text-sm">
            <Link className="rounded-full bg-ink px-4 py-2 text-white" href="/history">
              Verification history
            </Link>
            <Link className="rounded-full border border-black/10 px-4 py-2" href="/admin">
              Trusted publishers
            </Link>
            <Link className="rounded-full border border-black/10 px-4 py-2" href="/how-it-works">
              How it works
            </Link>
          </div>
        </div>
        <HomeForm />
      </section>
    </main>
  );
}
