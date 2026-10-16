"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

import { verifyAudio, verifyImage, verifyText, verifyUrl, verifyVideo } from "@/lib/api";

type VerificationMode = "url" | "text" | "image" | "audio" | "video";

export function HomeForm() {
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");
  const [imageContext, setImageContext] = useState("");
  const [audioContext, setAudioContext] = useState("");
  const [videoContext, setVideoContext] = useState("");
  const [claimedSpeaker, setClaimedSpeaker] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<VerificationMode>("url");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result =
        mode === "url"
          ? await verifyUrl(url)
          : mode === "text"
            ? await verifyText(text)
            : mode === "image"
              ? await verifyImage(file as File, imageContext)
              : mode === "audio"
                ? await verifyAudio(file as File, audioContext, claimedSpeaker)
                : await verifyVideo(file as File, videoContext);
      router.push(`/results/${result.job_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verification failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="panel p-6 md:p-8">
      <div className="flex flex-wrap gap-2">
        {(["url", "text", "image", "audio", "video"] as const).map((value) => (
          <button
            key={value}
            type="button"
            onClick={() => setMode(value)}
            className={`rounded-full px-4 py-2 text-sm ${mode === value ? "bg-ink text-white" : "bg-black/5"}`}
          >
            {value === "url"
              ? "URL"
              : value === "text"
                ? "Text"
                : value === "image"
                  ? "Image"
                  : value === "audio"
                    ? "Audio"
                    : "Video"}
          </button>
        ))}
      </div>

      {mode === "url" && (
        <label className="mt-6 block">
          <span className="mb-2 block text-sm font-medium">Article URL</span>
          <input
            className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 outline-none"
            placeholder="https://example.com/news/story"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            required
          />
        </label>
      )}

      {mode === "text" && (
        <label className="mt-6 block">
          <span className="mb-2 block text-sm font-medium">Paste text or claim</span>
          <textarea
            className="min-h-48 w-full rounded-2xl border border-black/10 bg-white px-4 py-3 outline-none"
            placeholder="Paste a claim, headline, paragraph, or article context here"
            value={text}
            onChange={(event) => setText(event.target.value)}
            required
          />
          <p className="mt-2 text-xs text-black/55">
            We analyze the pasted text using claim extraction, evidence retrieval, and fact-checking.
          </p>
        </label>
      )}

      {mode === "image" && (
        <div className="mt-6 space-y-4">
          <label className="block">
            <span className="mb-2 block text-sm font-medium">Upload image</span>
            <input
              className="w-full rounded-2xl border border-dashed border-black/20 bg-white px-4 py-3"
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              required
            />
          </label>
          <label className="block">
            <span className="mb-2 block text-sm font-medium">Caption or claim</span>
            <textarea
              className="min-h-28 w-full rounded-2xl border border-black/10 bg-white px-4 py-3 outline-none"
              placeholder="Add caption, claim, or event context (optional)"
              value={imageContext}
              onChange={(event) => setImageContext(event.target.value)}
            />
          </label>
          <p className="text-xs text-black/55">
            Upload an image and optionally add a caption or claim for verification.
          </p>
        </div>
      )}

      {mode === "audio" && (
        <div className="mt-6 space-y-4">
          <label className="block">
            <span className="mb-2 block text-sm font-medium">Upload audio</span>
              <input
                className="w-full rounded-2xl border border-dashed border-black/20 bg-white px-4 py-3"
                type="file"
                accept="audio/mpeg,audio/mp3,audio/wav,audio/x-wav,audio/mp4,audio/aac,audio/ogg,audio/m4a,audio/x-m4a,audio/x-mpeg-4,.m4a"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                required
              />
          </label>
          <label className="block">
            <span className="mb-2 block text-sm font-medium">Caption or claim</span>
            <textarea
              className="min-h-28 w-full rounded-2xl border border-black/10 bg-white px-4 py-3 outline-none"
              placeholder="Add caption, claim, or event context (optional)"
              value={audioContext}
              onChange={(event) => setAudioContext(event.target.value)}
            />
          </label>
          <label className="block">
            <span className="mb-2 block text-sm font-medium">Claimed speaker name</span>
            <input
              className="w-full rounded-2xl border border-black/10 bg-white px-4 py-3 outline-none"
              placeholder="Claimed speaker name (optional)"
              value={claimedSpeaker}
              onChange={(event) => setClaimedSpeaker(event.target.value)}
            />
          </label>
          <p className="text-xs text-black/55">
            Upload an audio file and optionally add context for verification.
          </p>
        </div>
      )}

      {mode === "video" && (
        <div className="mt-6 space-y-4">
          <label className="block">
            <span className="mb-2 block text-sm font-medium">Upload video</span>
            <input
              className="w-full rounded-2xl border border-dashed border-black/20 bg-white px-4 py-3"
              type="file"
              accept="video/mp4,video/quicktime,video/webm,video/x-matroska,video/mpeg"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              required
            />
          </label>
          <label className="block">
            <span className="mb-2 block text-sm font-medium">Caption or claim</span>
            <textarea
              className="min-h-28 w-full rounded-2xl border border-black/10 bg-white px-4 py-3 outline-none"
              placeholder="Add caption, claim, or event context (optional)"
              value={videoContext}
              onChange={(event) => setVideoContext(event.target.value)}
            />
          </label>
          <p className="text-xs text-black/55">
            Upload a video and optionally add a caption or claim for verification.
          </p>
        </div>
      )}

      {error ? <p className="mt-4 text-sm text-warning">{error}</p> : null}

      <button
        type="submit"
        disabled={loading}
        className="mt-6 rounded-full bg-accent px-6 py-3 text-sm font-semibold text-white disabled:opacity-60"
      >
        {loading
          ? "Verifying..."
          : mode === "text"
            ? "Verify text"
            : mode === "image"
              ? "Verify image"
              : mode === "audio"
                ? "Verify audio"
                : mode === "video"
                  ? "Verify video"
                  : "Verify provenance"}
      </button>
    </form>
  );
}
