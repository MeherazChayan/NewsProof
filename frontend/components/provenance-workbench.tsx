"use client";

import { FormEvent, useEffect, useState } from "react";

import {
  compareProvenanceVersions,
  createArticleSnapshot,
  getCryptoIssuers,
  getProvenanceHistory,
  signProvenanceObject,
  verifyProvenanceManifest
} from "@/lib/api";
import { IssuerCredential, ManifestVerificationResult, ProvenanceHistory, ProvenanceManifest } from "@/lib/types";

type SnapshotState = {
  source_url: string;
  publisher_name: string;
  author_name: string;
  title: string;
  normalized_article_text: string;
  media_urls: string;
};

const initialSnapshot: SnapshotState = {
  source_url: "",
  publisher_name: "",
  author_name: "",
  title: "",
  normalized_article_text: "",
  media_urls: ""
};

export function ProvenanceWorkbench() {
  const [issuers, setIssuers] = useState<IssuerCredential[]>([]);
  const [form, setForm] = useState<SnapshotState>(initialSnapshot);
  const [createdSnapshot, setCreatedSnapshot] = useState<{ id: string; title: string } | null>(null);
  const [manifest, setManifest] = useState<ProvenanceManifest | null>(null);
  const [verification, setVerification] = useState<ManifestVerificationResult | null>(null);
  const [history, setHistory] = useState<ProvenanceHistory | null>(null);
  const [comparison, setComparison] = useState<{ changed_fields: string[] } | null>(null);
  const [historyId, setHistoryId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getCryptoIssuers().then(setIssuers).catch((err: Error) => setError(err.message));
  }, []);

  async function handleCreateSnapshot(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const snapshot = await createArticleSnapshot({
        source_url: form.source_url,
        publisher_name: form.publisher_name,
        author_name: form.author_name || undefined,
        title: form.title,
        normalized_article_text: form.normalized_article_text,
        media_urls: form.media_urls
          .split("\n")
          .map((item) => item.trim())
          .filter(Boolean)
      });
      setCreatedSnapshot({ id: snapshot.id, title: snapshot.title });
      setHistoryId(snapshot.id);
      const nextManifest = await signProvenanceObject({
        object_type: "article_snapshot",
        object_id: snapshot.id,
        signer_id: issuers[0]?.id
      });
      setManifest(nextManifest);
      const nextVerification = await verifyProvenanceManifest({
        manifest_id: nextManifest.id,
        verified_by: "NewsProof UI"
      });
      setVerification(nextVerification);
      const nextHistory = await getProvenanceHistory("article_snapshot", snapshot.id);
      setHistory(nextHistory);
      if (nextHistory.versions.length >= 2) {
        const left = String(nextHistory.versions[nextHistory.versions.length - 2].id);
        const right = String(nextHistory.versions[nextHistory.versions.length - 1].id);
        const diff = await compareProvenanceVersions("article_snapshot", left, right, snapshot.id);
        setComparison({ changed_fields: diff.changed_fields });
      } else {
        setComparison(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create snapshot");
    } finally {
      setBusy(false);
    }
  }

  async function handleLoadHistory() {
    if (!historyId) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const nextHistory = await getProvenanceHistory("article_snapshot", historyId);
      setHistory(nextHistory);
      if (nextHistory.manifests.length) {
        const latestManifest = nextHistory.manifests[nextHistory.manifests.length - 1];
        const manifestId = String(latestManifest.manifest_id);
        setVerification(await verifyProvenanceManifest({ manifest_id: manifestId, verified_by: "NewsProof UI" }));
      }
      if (nextHistory.versions.length >= 2) {
        const left = String(nextHistory.versions[nextHistory.versions.length - 2].id);
        const right = String(nextHistory.versions[nextHistory.versions.length - 1].id);
        const diff = await compareProvenanceVersions("article_snapshot", left, right, historyId);
        setComparison({ changed_fields: diff.changed_fields });
      } else {
        setComparison(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load provenance history");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <section className="rounded-[2rem] border border-black/10 bg-white/90 p-6 shadow-sm">
        <h2 className="font-display text-3xl">Create Signed Article Snapshot</h2>
        <p className="mt-2 max-w-3xl text-sm text-black/65">
          This creates a canonical article snapshot, hashes it, signs it with the platform issuer, timestamps it, and stores the immutable version.
        </p>
        <form className="mt-6 grid gap-3 md:grid-cols-2" onSubmit={handleCreateSnapshot}>
          <input className="rounded-2xl border border-black/10 px-4 py-3" placeholder="Source URL" value={form.source_url} onChange={(event) => setForm((current) => ({ ...current, source_url: event.target.value }))} />
          <input className="rounded-2xl border border-black/10 px-4 py-3" placeholder="Publisher name" value={form.publisher_name} onChange={(event) => setForm((current) => ({ ...current, publisher_name: event.target.value }))} />
          <input className="rounded-2xl border border-black/10 px-4 py-3" placeholder="Author name" value={form.author_name} onChange={(event) => setForm((current) => ({ ...current, author_name: event.target.value }))} />
          <input className="rounded-2xl border border-black/10 px-4 py-3" placeholder="Title" value={form.title} onChange={(event) => setForm((current) => ({ ...current, title: event.target.value }))} />
          <textarea className="min-h-40 rounded-2xl border border-black/10 px-4 py-3 md:col-span-2" placeholder="Normalized article text" value={form.normalized_article_text} onChange={(event) => setForm((current) => ({ ...current, normalized_article_text: event.target.value }))} />
          <textarea className="min-h-28 rounded-2xl border border-black/10 px-4 py-3 md:col-span-2" placeholder="Media URLs, one per line" value={form.media_urls} onChange={(event) => setForm((current) => ({ ...current, media_urls: event.target.value }))} />
          <button className="w-fit rounded-full bg-black px-5 py-3 text-sm font-semibold text-white disabled:opacity-50" disabled={busy} type="submit">
            {busy ? "Signing snapshot..." : "Create and sign snapshot"}
          </button>
        </form>
      </section>

      <section className="rounded-[2rem] border border-black/10 bg-white/90 p-6 shadow-sm">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="font-display text-3xl">Provenance History</h2>
            <p className="mt-2 text-sm text-black/65">Load the version chain, latest manifest, and changed fields for an article snapshot.</p>
          </div>
          <div className="flex gap-2">
            <input className="min-w-[20rem] rounded-2xl border border-black/10 px-4 py-3" placeholder="Article snapshot ID" value={historyId} onChange={(event) => setHistoryId(event.target.value)} />
            <button className="rounded-full border border-black/10 px-4 py-3 text-sm font-semibold" onClick={handleLoadHistory} type="button">
              Load history
            </button>
          </div>
        </div>
      </section>

      {error ? <p className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p> : null}

      <div className="grid gap-4 lg:grid-cols-3">
        <section className="rounded-[2rem] border border-black/10 bg-white/90 p-5 shadow-sm">
          <p className="text-xs uppercase tracking-[0.35em] text-black/45">Snapshot</p>
          {createdSnapshot ? (
            <div className="mt-3 space-y-2 text-sm text-black/75">
              <p><strong>ID:</strong> {createdSnapshot.id}</p>
              <p><strong>Title:</strong> {createdSnapshot.title}</p>
            </div>
          ) : (
            <p className="mt-3 text-sm text-black/55">No snapshot created in this session yet.</p>
          )}
        </section>

        <section className="rounded-[2rem] border border-black/10 bg-white/90 p-5 shadow-sm">
          <p className="text-xs uppercase tracking-[0.35em] text-black/45">Manifest</p>
          {manifest ? (
            <div className="mt-3 space-y-2 text-sm text-black/75">
              <p><strong>ID:</strong> {manifest.id}</p>
              <p><strong>Hash:</strong> {manifest.sha256_hash}</p>
              <p><strong>Signer:</strong> {manifest.signer_id}</p>
              <p><strong>Status:</strong> {manifest.verification_status}</p>
            </div>
          ) : (
            <p className="mt-3 text-sm text-black/55">No manifest signed yet.</p>
          )}
        </section>

        <section className="rounded-[2rem] border border-black/10 bg-white/90 p-5 shadow-sm">
          <p className="text-xs uppercase tracking-[0.35em] text-black/45">Verification</p>
          {verification ? (
            <div className="mt-3 space-y-2 text-sm text-black/75">
              <p><strong>Valid:</strong> {verification.valid ? "yes" : "no"}</p>
              <p><strong>Status:</strong> {verification.status}</p>
              <p><strong>Timestamp:</strong> {String(verification.checks.timestamp_valid ?? "n/a")}</p>
              <p><strong>Version chain:</strong> {String(verification.checks.version_chain_valid ?? "n/a")}</p>
            </div>
          ) : (
            <p className="mt-3 text-sm text-black/55">No verification result yet.</p>
          )}
        </section>
      </div>

      {history ? (
        <section className="rounded-[2rem] border border-black/10 bg-white/90 p-6 shadow-sm">
          <h2 className="font-display text-3xl">Version Chain</h2>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="space-y-3">
              {history.versions.map((version) => (
                <div key={String(version.id)} className="rounded-2xl border border-black/10 bg-black/[0.02] p-4 text-sm text-black/75">
                  <p><strong>ID:</strong> {String(version.id)}</p>
                  {"title" in version ? <p><strong>Title:</strong> {String(version.title)}</p> : null}
                  {"publisher_name" in version ? <p><strong>Publisher:</strong> {String(version.publisher_name)}</p> : null}
                  {"created_at" in version ? <p><strong>Recorded:</strong> {String(version.created_at)}</p> : null}
                </div>
              ))}
            </div>
            <div className="space-y-3">
              {history.manifests.map((item) => (
                <div key={String(item.manifest_id)} className="rounded-2xl border border-black/10 bg-black/[0.02] p-4 text-sm text-black/75">
                  <p><strong>Manifest:</strong> {String(item.manifest_id)}</p>
                  <p><strong>Status:</strong> {String(item.verification_status)}</p>
                  <p><strong>Hash:</strong> {String(item.sha256_hash)}</p>
                </div>
              ))}
            </div>
          </div>
          {comparison ? (
            <div className="mt-4 rounded-2xl border border-black/10 bg-black/[0.02] p-4 text-sm text-black/75">
              <p><strong>Changed fields:</strong> {comparison.changed_fields.length ? comparison.changed_fields.join(", ") : "none"}</p>
            </div>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
