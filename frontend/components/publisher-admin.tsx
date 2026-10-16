"use client";

import { FormEvent, useEffect, useState } from "react";

import {
  createContentRecord,
  createPublisher,
  deleteContentRecord,
  deletePublisher,
  getContentRecords,
  getPublishers,
  updateContentRecord,
  updatePublisher
} from "@/lib/api";
import { ContentRecord, Publisher } from "@/lib/types";

const emptyPublisherForm = {
  name: "",
  domain: "",
  aliasDomains: "",
  aliasNames: "",
  public_keys: [{ key_id: "", algorithm: "Ed25519", public_key: "", is_active: true }]
};

const emptyRecordForm = {
  publisher_id: 0,
  canonical_url: "",
  title: "",
  author: "",
  publish_time: "",
  normalized_text: "",
  content_hash: "",
  signature: "",
  key_id: "",
  source_metadata: {}
};

export function PublisherAdmin() {
  const [publishers, setPublishers] = useState<Publisher[]>([]);
  const [records, setRecords] = useState<ContentRecord[]>([]);
  const [editingPublisherId, setEditingPublisherId] = useState<number | null>(null);
  const [editingRecordId, setEditingRecordId] = useState<number | null>(null);
  const [publisherForm, setPublisherForm] = useState(emptyPublisherForm);
  const [recordForm, setRecordForm] = useState(emptyRecordForm);

  async function refresh() {
    const nextPublishers = await getPublishers();
    setPublishers(nextPublishers);
    setRecords(await getContentRecords());
    if (!recordForm.publisher_id && nextPublishers[0]) {
      setRecordForm((current) => ({ ...current, publisher_id: nextPublishers[0].id }));
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  async function handlePublisherSubmit(event: FormEvent) {
    event.preventDefault();
    const payload = {
      name: publisherForm.name,
      domain: publisherForm.domain,
      metadata: {
        domains: splitAliases(publisherForm.aliasDomains),
        names: splitAliases(publisherForm.aliasNames)
      },
      public_keys: publisherForm.public_keys
    };

    if (editingPublisherId) {
      await updatePublisher(editingPublisherId, payload);
    } else {
      await createPublisher(payload);
    }

    setPublisherForm(emptyPublisherForm);
    setEditingPublisherId(null);
    await refresh();
  }

  async function handleRecordSubmit(event: FormEvent) {
    event.preventDefault();
    const payload = {
      publisher_id: recordForm.publisher_id,
      canonical_url: recordForm.canonical_url || null,
      title: recordForm.title || null,
      author: recordForm.author || null,
      publish_time: recordForm.publish_time || null,
      normalized_text: recordForm.normalized_text,
      content_hash: recordForm.content_hash,
      signature: recordForm.signature || null,
      key_id: recordForm.key_id || null,
      source_metadata: {}
    };

    if (editingRecordId) {
      await updateContentRecord(editingRecordId, {
        canonical_url: payload.canonical_url,
        title: payload.title,
        author: payload.author,
        publish_time: payload.publish_time,
        normalized_text: payload.normalized_text,
        content_hash: payload.content_hash,
        signature: payload.signature,
        key_id: payload.key_id,
        source_metadata: {}
      });
    } else {
      await createContentRecord(payload);
    }

    setRecordForm((current) => ({ ...emptyRecordForm, publisher_id: current.publisher_id }));
    setEditingRecordId(null);
    await refresh();
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-6 xl:grid-cols-2">
        <form onSubmit={handlePublisherSubmit} className="panel p-6">
          <h2 className="font-display text-2xl">{editingPublisherId ? "Edit trusted publisher" : "Add trusted publisher"}</h2>
          <div className="mt-4 space-y-4">
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Publisher name"
              value={publisherForm.name}
              onChange={(event) => setPublisherForm({ ...publisherForm, name: event.target.value })}
            />
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="bbc.com"
              value={publisherForm.domain}
              onChange={(event) => setPublisherForm({ ...publisherForm, domain: event.target.value })}
            />
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Alias domains: www.bbc.com, m.bbc.com"
              value={publisherForm.aliasDomains}
              onChange={(event) => setPublisherForm({ ...publisherForm, aliasDomains: event.target.value })}
            />
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Alias names: BBC News, BBC"
              value={publisherForm.aliasNames}
              onChange={(event) => setPublisherForm({ ...publisherForm, aliasNames: event.target.value })}
            />
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Key id"
              value={publisherForm.public_keys[0].key_id}
              onChange={(event) =>
                setPublisherForm({ ...publisherForm, public_keys: [{ ...publisherForm.public_keys[0], key_id: event.target.value }] })
              }
            />
            <textarea
              className="min-h-32 w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Base64 public key"
              value={publisherForm.public_keys[0].public_key}
              onChange={(event) =>
                setPublisherForm({
                  ...publisherForm,
                  public_keys: [{ ...publisherForm.public_keys[0], public_key: event.target.value }]
                })
              }
            />
          </div>
          <button className="mt-4 rounded-full bg-ink px-4 py-2 text-white" type="submit">
            {editingPublisherId ? "Update publisher" : "Create publisher"}
          </button>
        </form>

        <form onSubmit={handleRecordSubmit} className="panel p-6">
          <h2 className="font-display text-2xl">{editingRecordId ? "Edit signed proof record" : "Add signed proof record"}</h2>
          <div className="mt-4 space-y-4">
            <select
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              value={recordForm.publisher_id}
              onChange={(event) => setRecordForm({ ...recordForm, publisher_id: Number(event.target.value) })}
            >
              <option value={0}>Select publisher</option>
              {publishers.map((publisher) => (
                <option key={publisher.id} value={publisher.id}>
                  {publisher.name}
                </option>
              ))}
            </select>
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Canonical URL"
              value={recordForm.canonical_url}
              onChange={(event) => setRecordForm({ ...recordForm, canonical_url: event.target.value })}
            />
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Title"
              value={recordForm.title}
              onChange={(event) => setRecordForm({ ...recordForm, title: event.target.value })}
            />
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Author"
              value={recordForm.author}
              onChange={(event) => setRecordForm({ ...recordForm, author: event.target.value })}
            />
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              type="datetime-local"
              value={recordForm.publish_time}
              onChange={(event) => setRecordForm({ ...recordForm, publish_time: event.target.value })}
            />
            <textarea
              className="min-h-24 w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Normalized article text used to derive the signed hash"
              value={recordForm.normalized_text}
              onChange={(event) => setRecordForm({ ...recordForm, normalized_text: event.target.value })}
            />
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="SHA-256 content hash"
              value={recordForm.content_hash}
              onChange={(event) => setRecordForm({ ...recordForm, content_hash: event.target.value })}
            />
            <input
              className="w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Key id"
              value={recordForm.key_id}
              onChange={(event) => setRecordForm({ ...recordForm, key_id: event.target.value })}
            />
            <textarea
              className="min-h-24 w-full rounded-2xl border border-black/10 px-4 py-3"
              placeholder="Base64 signature"
              value={recordForm.signature}
              onChange={(event) => setRecordForm({ ...recordForm, signature: event.target.value })}
            />
          </div>
          <button className="mt-4 rounded-full bg-accent px-4 py-2 text-white" type="submit">
            {editingRecordId ? "Update proof record" : "Create proof record"}
          </button>
        </form>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <div className="panel p-6">
          <h2 className="font-display text-2xl">Trusted publishers</h2>
          <div className="mt-4 space-y-3">
            {publishers.map((publisher) => {
              const aliasDomains = Array.isArray(publisher.metadata?.domains) ? (publisher.metadata.domains as string[]) : [];
              const aliasNames = Array.isArray(publisher.metadata?.names) ? (publisher.metadata.names as string[]) : [];

              return (
                <div key={publisher.id} className="rounded-2xl border border-black/10 bg-paper/70 p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="font-medium">{publisher.name}</div>
                      <div className="text-sm text-black/60">{publisher.domain}</div>
                      {aliasDomains.length ? <div className="mt-2 text-xs text-black/55">Domains: {aliasDomains.join(", ")}</div> : null}
                      {aliasNames.length ? <div className="mt-1 text-xs text-black/55">Names: {aliasNames.join(", ")}</div> : null}
                    </div>
                    <div className="flex gap-2 text-sm">
                      <button
                        type="button"
                        onClick={() => {
                          setEditingPublisherId(publisher.id);
                          setPublisherForm({
                            name: publisher.name,
                            domain: publisher.domain,
                            aliasDomains: aliasDomains.join(", "),
                            aliasNames: aliasNames.join(", "),
                            public_keys: publisher.public_keys.length > 0 ? publisher.public_keys : emptyPublisherForm.public_keys
                          });
                        }}
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        onClick={async () => {
                          await deletePublisher(publisher.id);
                          await refresh();
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="panel p-6">
          <h2 className="font-display text-2xl">Signed proof records</h2>
          <div className="mt-4 space-y-3">
            {records.map((record) => (
              <div key={record.id} className="rounded-2xl border border-black/10 bg-paper/70 p-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="font-medium">{record.title || "Untitled proof record"}</div>
                    <div className="text-sm text-black/60">{record.canonical_url || "No canonical URL"}</div>
                    <div className="mt-1 text-xs text-black/55">Hash: {record.content_hash}</div>
                    <div className="mt-1 text-xs text-black/55">Author: {record.author || "Unknown"}</div>
                  </div>
                  <div className="flex gap-2 text-sm">
                    <button
                      type="button"
                      onClick={() => {
                        setEditingRecordId(record.id);
                        setRecordForm({
                          publisher_id: record.publisher_id,
                          canonical_url: record.canonical_url || "",
                          title: record.title || "",
                          author: record.author || "",
                          publish_time: record.publish_time ? record.publish_time.slice(0, 16) : "",
                          normalized_text: record.normalized_text,
                          content_hash: record.content_hash,
                          signature: record.signature || "",
                          key_id: record.key_id || "",
                          source_metadata: record.source_metadata
                        });
                      }}
                    >
                      Edit
                    </button>
                    <button
                      type="button"
                      onClick={async () => {
                        await deleteContentRecord(record.id);
                        await refresh();
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function splitAliases(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}
