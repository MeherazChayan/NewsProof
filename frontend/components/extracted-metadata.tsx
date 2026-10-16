type ExtractedMetadataProps = {
  details: Record<string, unknown>;
  compact?: boolean;
};

export function ExtractedMetadata({ details, compact = false }: ExtractedMetadataProps) {
  const mediaAssets = Array.isArray(details.extracted_media_assets) ? details.extracted_media_assets : [];
  const c2paCredentials = Array.isArray(details.c2pa_credentials) ? details.c2pa_credentials : [];
  const remoteMediaProvenance = Array.isArray(details.remote_media_provenance) ? details.remote_media_provenance : [];

  return (
    <div className={`space-y-2 text-xs text-black/70 ${compact ? "" : ""}`}>
      {details.extracted_publisher_name ? <p><strong>Publisher:</strong> {String(details.extracted_publisher_name)}</p> : null}
      {details.extracted_author ? <p><strong>Author:</strong> {String(details.extracted_author)}</p> : null}
      {details.extracted_publish_time ? (
        <p><strong>Published:</strong> {new Date(String(details.extracted_publish_time)).toLocaleString()}</p>
      ) : null}
      {details.extracted_source_domain ? <p><strong>Domain:</strong> {String(details.extracted_source_domain)}</p> : null}
      {details.extracted_section ? <p><strong>Section:</strong> {String(details.extracted_section)}</p> : null}
      {Array.isArray(details.extracted_keywords) && details.extracted_keywords.length ? (
        <p><strong>Keywords:</strong> {details.extracted_keywords.map((item) => String(item)).join(", ")}</p>
      ) : null}
      {details.extracted_hero_image_url ? (
        <p>
          <strong>Hero image:</strong>{" "}
          <a className="break-all text-accent underline" href={String(details.extracted_hero_image_url)} target="_blank" rel="noreferrer">
            {String(details.extracted_hero_image_url)}
          </a>
        </p>
      ) : null}
      {c2paCredentials.length ? (
        <div className="space-y-2">
          <p><strong>C2PA Content Credentials:</strong></p>
          {c2paCredentials.map((credential, index) => {
            if (typeof credential !== "object" || credential === null) {
              return null;
            }
            const item = credential as Record<string, unknown>;
            return (
              <div key={`${String(item.url ?? item.value ?? index)}-${index}`} className="rounded-xl border border-black/10 bg-white/50 p-3">
                <p className="font-medium text-black/80">
                  {String(item.type ?? "credential")}
                  {item.source ? <span className="font-normal text-black/50"> ({String(item.source)})</span> : null}
                </p>
                {item.url ? (
                  <a className="mt-1 block break-all text-accent underline" href={String(item.url)} target="_blank" rel="noreferrer">
                    {String(item.url)}
                  </a>
                ) : null}
                {item.description ? <p className="mt-1">{String(item.description)}</p> : null}
                {item.value ? <p className="mt-1 break-all"><strong>Value:</strong> {String(item.value)}</p> : null}
              </div>
            );
          })}
        </div>
      ) : null}
      {mediaAssets.length ? (
        <div className="space-y-2">
          <p><strong>Media assets:</strong></p>
          {mediaAssets.map((asset, index) => {
            if (typeof asset !== "object" || asset === null) {
              return null;
            }
            const media = asset as Record<string, unknown>;
            return (
              <div key={`${String(media.url ?? index)}-${index}`} className="rounded-xl border border-black/10 bg-white/50 p-3">
                <p className="font-medium text-black/80">
                  {String(media.type ?? "media")}
                  {media.source ? <span className="font-normal text-black/50"> ({String(media.source)})</span> : null}
                </p>
                {media.url ? (
                  <a className="mt-1 block break-all text-accent underline" href={String(media.url)} target="_blank" rel="noreferrer">
                    {String(media.url)}
                  </a>
                ) : null}
                {media.caption ? <p className="mt-1"><strong>Caption:</strong> {String(media.caption)}</p> : null}
                {media.alt ? <p><strong>Alt:</strong> {String(media.alt)}</p> : null}
                {media.poster ? <p><strong>Poster:</strong> {String(media.poster)}</p> : null}
                {media.mime_type ? <p><strong>MIME:</strong> {String(media.mime_type)}</p> : null}
                {media.width || media.height ? (
                  <p><strong>Dimensions:</strong> {String(media.width ?? "?")} x {String(media.height ?? "?")}</p>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : null}
      {remoteMediaProvenance.length ? (
        <div className="space-y-2">
          <p><strong>Remote media provenance:</strong></p>
          {remoteMediaProvenance.map((item, index) => {
            if (typeof item !== "object" || item === null) {
              return null;
            }
            const provenance = item as Record<string, unknown>;
            const signals = Array.isArray(provenance.signals) ? provenance.signals.map((signal) => String(signal)).join(", ") : "";
            const linkedCredentials = Array.isArray(provenance.linked_credentials) ? provenance.linked_credentials : [];
            return (
              <div key={`${String(provenance.url ?? index)}-${index}`} className="rounded-xl border border-black/10 bg-white/50 p-3">
                <p className="font-medium text-black/80">
                  {String(provenance.type ?? "media")}
                  {provenance.source ? <span className="font-normal text-black/50"> ({String(provenance.source)})</span> : null}
                </p>
                {provenance.url ? (
                  <a className="mt-1 block break-all text-accent underline" href={String(provenance.url)} target="_blank" rel="noreferrer">
                    {String(provenance.url)}
                  </a>
                ) : null}
                {provenance.manifest_url ? (
                  <a className="mt-1 block break-all text-accent underline" href={String(provenance.manifest_url)} target="_blank" rel="noreferrer">
                    Manifest: {String(provenance.manifest_url)}
                  </a>
                ) : null}
                {provenance.description ? <p className="mt-1">{String(provenance.description)}</p> : null}
                {"evidence_found" in provenance ? (
                  <p className="mt-1">
                    <strong>Evidence found:</strong> {provenance.evidence_found ? "yes" : "no"}
                  </p>
                ) : null}
                {"checked" in provenance ? (
                  <p>
                    <strong>Checked:</strong> {provenance.checked ? "yes" : "no"}
                  </p>
                ) : null}
                {provenance.platform ? <p><strong>Platform:</strong> {String(provenance.platform)}</p> : null}
                {provenance.evaluation_state ? <p><strong>Evaluation:</strong> {String(provenance.evaluation_state)}</p> : null}
                {signals ? <p className="mt-1"><strong>Signals:</strong> {signals}</p> : null}
                {provenance.content_type ? <p><strong>Content-Type:</strong> {String(provenance.content_type)}</p> : null}
                {"direct_media_response" in provenance ? (
                  <p><strong>Direct media response:</strong> {provenance.direct_media_response ? "yes" : "no"}</p>
                ) : null}
                {provenance.note ? <p><strong>Note:</strong> {String(provenance.note)}</p> : null}
                {provenance.content_length ? <p><strong>Content-Length:</strong> {String(provenance.content_length)}</p> : null}
                {provenance.etag ? <p><strong>ETag:</strong> {String(provenance.etag)}</p> : null}
                {Array.isArray(provenance.checked_candidates) && provenance.checked_candidates.length > 1 ? (
                  <p><strong>Checked candidates:</strong> {provenance.checked_candidates.map((candidate) => String(candidate)).join(" | ")}</p>
                ) : null}
                {linkedCredentials.length ? (
                  <p className="mt-1"><strong>Linked credentials:</strong> {String(linkedCredentials.length)}</p>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
