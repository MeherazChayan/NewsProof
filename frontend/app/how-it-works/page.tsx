import Link from "next/link";

const sections = [
  {
    title: "What NewsProof checks",
    body:
      "NewsProof compares the content you submit with source signals, trusted publisher records, and supporting evidence from other reporting to see whether the claim holds up.",
  },
  {
    title: "How provenance helps",
    body:
      "When a publisher or platform record includes signed provenance, NewsProof can confirm whether the content matches a recorded source snapshot. That helps us tell apart a traceable source record from missing or altered origin details.",
  },
  {
    title: "What evidence means",
    body:
      "Evidence comes from matching source records, related reporting, and other supporting materials that point to the same event or claim. Stronger results usually need more than one useful source.",
  },
  {
    title: "What Not Verifiable means",
    body:
      "Not Verifiable does not automatically mean false. It usually means the available source record, corroborating evidence, or content clarity was too limited for a stronger judgment.",
  },
  {
    title: "What the audit trail is",
    body:
      "Each verification stores the key steps taken by the system so you can review what was checked, what evidence was found, and how the report was assembled.",
  },
];

export default function HowItWorksPage() {
  return (
    <main className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-black/60">How It Works</p>
          <h1 className="mt-3 font-display text-4xl">Simple checks, clearer proof, and readable reports.</h1>
          <p className="mt-3 max-w-3xl text-sm text-black/70">
            NewsProof is designed to help everyday readers understand what was checked, what evidence was found,
            and where provenance fits into the picture without asking you to learn internal jargon first.
          </p>
        </div>
        <Link className="text-sm text-black/70 hover:text-black" href="/">
          Back home
        </Link>
      </div>

      <section className="grid gap-4 md:grid-cols-2">
        {sections.map((section) => (
          <article key={section.title} className="panel p-6">
            <h2 className="font-display text-2xl">{section.title}</h2>
            <p className="mt-3 text-sm leading-7 text-black/70">{section.body}</p>
          </article>
        ))}
      </section>

      <section className="panel p-6">
        <h2 className="font-display text-2xl">Where provenance lives now</h2>
        <p className="mt-3 max-w-3xl text-sm leading-7 text-black/70">
          Provenance is still part of NewsProof, but it now appears where it is most useful: inside report details,
          proof records, and deeper verification views rather than as a primary homepage action.
        </p>
        <div className="mt-5 flex gap-3 text-sm">
          <Link className="rounded-full bg-ink px-4 py-2 text-white" href="/history">
            Verification history
          </Link>
          <Link className="rounded-full border border-black/10 px-4 py-2" href="/provenance">
            Advanced provenance tools
          </Link>
        </div>
      </section>
    </main>
  );
}
