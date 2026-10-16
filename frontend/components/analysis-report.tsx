import type { ReactNode } from "react";
import { SimilarityMatch, VerificationResult } from "@/lib/types";
import { SimilarityMatchesPanel } from "@/components/similarity-matches-panel";

type DetailMap = Record<string, unknown>;

function asRecord(value: unknown): DetailMap {
  return value && typeof value === "object" ? (value as DetailMap) : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function yesNo(value: boolean): string {
  return value ? "Yes" : "No";
}

function formatScore100(value: unknown): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "n/a";
  }
  const score = numeric <= 1 ? numeric * 100 : numeric;
  return `${score.toFixed(1)} / 100`;
}

function formatAbsoluteScore100(value: unknown): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "n/a";
  }
  return `${numeric.toFixed(1)} / 100`;
}

function numberOrUndefined(value: unknown): number | undefined {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : undefined;
}

function labelFromScore100(value: unknown): string | undefined {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return undefined;
  }
  const score = numeric <= 1 ? numeric * 100 : numeric;
  if (score >= 70) {
    return "Verified";
  }
  if (score >= 50) {
    return "Likely Verified";
  }
  if (score >= 35) {
    return "Not Verifiable";
  }
  return "Unverified";
}

function shortHash(value: string | null | undefined): string {
  if (!value) {
    return "n/a";
  }
  return `${value.slice(0, 12)}...`;
}

function getProvenanceStatus(result: VerificationResult, details: DetailMap) {
  const verificationStatus = String(details.provenance_verification_status ?? details.verification_status ?? "");
  const layer1 = asRecord(result.layer_results?.layer1);
  const c2paCredentials = asArray(details.c2pa_credentials).map(asRecord);
  const remoteMediaProvenance = asArray(details.remote_media_provenance).map(asRecord);
  const youtubeSignals = asRecord(details.youtube_signals);
  const pageCredentials = c2paCredentials.filter((item) => String(item.source ?? "") !== "remote_media");
  const youtubePageSignal = youtubeSignals.page_level_provenance_signal === true;
  const youtubeOriginalContent = youtubeSignals.original_content_claim === true;
  const rawExternalDetected = remoteMediaProvenance.some((item) =>
    ["credentials_verified", "remote_manifest_verified", "sidecar_manifest_verified"].includes(String(item.verification_state ?? "")),
  );
  const platformExternalDetected =
    youtubePageSignal ||
    youtubeOriginalContent ||
    pageCredentials.length > 0 ||
    remoteMediaProvenance.some((item) => String(item.verification_state ?? "") === "platform_signal_detected_manifest_not_retrievable");
  const externalDetected = rawExternalDetected || platformExternalDetected;
  const internalValid = Boolean(details.internal_manifest_valid || details.platform_manifest_valid);

  if (verificationStatus === "issuer_revoked") {
    return {
      badge: "Issuer revoked",
      summary: "A provenance record exists, but the issuing credential has been revoked.",
    };
  }
  if (verificationStatus === "timestamp_invalid") {
    return {
      badge: "Timestamp invalid",
      summary: "A provenance record exists, but its timestamp could not be validated.",
    };
  }
  if (verificationStatus === "version_chain_broken") {
    return {
      badge: "Version chain broken",
      summary: "A provenance record exists, but the version history link is incomplete or inconsistent.",
    };
  }
  if (verificationStatus === "signature_invalid" || verificationStatus === "hash_mismatch" || layer1.status === "mismatch") {
    return {
      badge: "Verification failed",
      summary: "A signed provenance record was found, but verification failed.",
    };
  }
  if (internalValid) {
    return {
      badge: "Verified platform-issued provenance",
      summary: "A platform-issued provenance record was created and verified for this article snapshot.",
    };
  }
  if (externalDetected) {
    let summary = "External provenance was found for this content.";
    if (rawExternalDetected && platformExternalDetected) {
      summary = "External provenance was found from both raw asset checks and page or platform signals.";
    } else if (rawExternalDetected) {
      summary = "External signed provenance was found from raw asset or manifest checks.";
    } else if (youtubeOriginalContent) {
      summary = "External provenance was found from YouTube page signals, including original-content indicators.";
    } else if (platformExternalDetected) {
      summary = "External provenance was found from page or platform-level signals, even though a raw manifest was not retrieved.";
    }
    return {
      badge: "External signed provenance found",
      summary,
    };
  }
  return {
    badge: "No signed provenance found",
    summary: "Publisher attribution was identified, but no article-specific signed provenance was found.",
  };
}

function summaryRows(result: VerificationResult, details: DetailMap, provenanceStatus: { badge: string; summary: string }) {
  const manifestId = String(details.manifest_id ?? details.internal_manifest_id ?? "n/a");
  const previousVersion = details.previous_version_id ?? details.previous_manifest_id ?? null;
  const issuedBy = String(details.issued_by ?? details.signer_name ?? details.signer_id ?? "NewsProof system");
  const issuerType = String(details.issuer_type ?? (provenanceStatus.badge === "Verified platform-issued provenance" ? "platform" : "unknown"));
  const signedAt = String(details.signed_at ?? details.manifest_created_at ?? result.created_at);
  const timestampSource = String(details.timestamp_source ?? details.timestamp_provider ?? details.timestamp_mode ?? "internal server timestamp");
  const recordType = String(details.record_type ?? "article_snapshot");
  const canonicalHash = String(details.canonical_hash ?? result.proof_summary.content_hash ?? "n/a");
  const signatureAlgorithm = String(details.signature_algorithm ?? "Ed25519");

  return [
    ["Record type", recordType],
    ["Issued by", issuedBy],
    ["Issuer type", issuerType],
    ["Verification result", provenanceStatus.badge],
    ["Signed at", signedAt === "n/a" ? "Not available" : new Date(signedAt).toLocaleString()],
    ["Timestamp source", timestampSource],
    ["Current version", manifestId],
    ["Previous version exists", previousVersion ? "Yes" : "No"],
    ["Canonical hash", canonicalHash],
    ["Signature algorithm", signatureAlgorithm],
  ] as const;
}

function transportEvidenceRows(details: DetailMap) {
  const transport = asRecord(details.transport_evidence);
  if (!Object.keys(transport).length) {
    return [];
  }

  return [
    ["HTTPS enabled", transport.https_enabled === true ? "Yes" : transport.https_enabled === false ? "No" : "Unknown"],
    ["Certificate subject", String(transport.certificate_subject ?? "Not available")],
    ["Certificate issuer", String(transport.certificate_issuer ?? "Not available")],
    ["Certificate valid from", String(transport.certificate_valid_from ?? "Not available")],
    ["Certificate valid to", String(transport.certificate_valid_to ?? "Not available")],
    [
      "Certificate hostname match",
      transport.certificate_hostname_match === true ? "Yes" : transport.certificate_hostname_match === false ? "No" : "Unknown",
    ],
  ] as const;
}

function mediaRows(details: DetailMap, contentPreview: DetailMap) {
  const extractedMedia = asArray(details.extracted_media_assets ?? contentPreview.media_assets).map(asRecord);
  const remoteMedia = asArray(details.remote_media_provenance).map(asRecord);

  if (!extractedMedia.length && !remoteMedia.length) {
    return [];
  }

  return extractedMedia.map((asset, index) => {
    const remote = remoteMedia.find((item) => String(item.url ?? "") === String(asset.url ?? "")) ?? remoteMedia[index] ?? {};
    const contentType = String(remote.content_type ?? asset.mime_type ?? "not returned");
    const verificationState = String(remote.verification_state ?? "");
    const credentialDetected = [
      "credentials_verified",
      "remote_manifest_verified",
      "sidecar_manifest_verified",
      "platform_signal_detected_manifest_not_retrievable",
    ].includes(verificationState);
    let note = String(remote.description ?? "Checked successfully, but no external cryptographic provenance was detected.");
    if (verificationState === "credentials_verified") {
      note = "Signed credentials were fetched and verified from the served asset.";
    } else if (verificationState === "remote_manifest_verified" || verificationState === "sidecar_manifest_verified") {
      note = "A remote or sidecar provenance record was recovered and verified.";
    } else if (verificationState === "platform_signal_detected_manifest_not_retrievable") {
      note =
        asset.type === "video" && String(remote.platform ?? "") === "youtube"
          ? "YouTube indicates provenance for this video, but the served embed or visible asset did not expose a retrievable raw manifest."
          : "The platform appears to signal Content Credentials, but the served media did not expose a retrievable raw manifest.";
    } else if (verificationState === "checked_no_retrievable_credentials") {
      note = "The asset was checked successfully, but no retrievable credentials were found.";
    } else if (verificationState === "could_not_evaluate_asset") {
      note = "The fetched asset could not be evaluated because it did not return a direct media file.";
    } else if (contentType.includes("json")) {
      note = "Endpoint returned JSON rather than a direct media file.";
    } else if (contentType !== "not returned" && asset.type === "image" && !contentType.startsWith("image/")) {
      note = "Endpoint returned non-image content instead of a direct image file.";
    } else if (contentType !== "not returned" && asset.type === "video" && !contentType.startsWith("video/")) {
      note = "Endpoint returned non-video content instead of a direct video file.";
    }
    return {
      url: String(asset.url ?? remote.url ?? "n/a"),
      type: String(asset.type ?? remote.type ?? "media"),
      contentType,
      credentialDetected,
      note,
      checked: remote.checked !== false,
      verificationState,
      platform: String(remote.platform ?? ""),
      forensics: asRecord(remote.forensics),
    };
  });
}

function describeRemoteCheck(remote: DetailMap, assetType?: string): string | null {
  const verificationState = String(remote.verification_state ?? "");
  const platform = String(remote.platform ?? "");
  const contentType = String(remote.content_type ?? "");

  if (verificationState === "credentials_verified") {
    return "Signed credentials were fetched and verified from the served asset.";
  }
  if (verificationState === "remote_manifest_verified" || verificationState === "sidecar_manifest_verified") {
    return "A remote or sidecar provenance record was recovered and verified.";
  }
  if (verificationState === "platform_signal_detected_manifest_not_retrievable") {
    if (platform === "youtube") {
      return "YouTube indicates provenance for this video, but the served embed or visible asset did not expose a retrievable raw manifest.";
    }
    if (platform === "linkedin") {
      return "LinkedIn appears to signal Content Credentials, but the served media did not expose a retrievable raw manifest.";
    }
    return "The platform appears to signal Content Credentials, but the served media did not expose a retrievable raw manifest.";
  }
  if (verificationState === "checked_no_retrievable_credentials") {
    return "The asset was checked successfully, but no retrievable credentials were found.";
  }
  if (verificationState === "could_not_evaluate_asset") {
    return "The fetched asset could not be evaluated because it did not return a direct media file.";
  }
  if (contentType.includes("json")) {
    return "Endpoint returned JSON rather than a direct media file.";
  }
  if (contentType !== "not returned" && assetType === "image" && !contentType.startsWith("image/")) {
    return "Endpoint returned non-image content instead of a direct image file.";
  }
  if (contentType !== "not returned" && assetType === "video" && !contentType.startsWith("video/")) {
    return "Endpoint returned non-video content instead of a direct video file.";
  }
  return remote.description ? String(remote.description) : null;
}

function LayerDisclosure({
  title,
  badge,
  description,
  children,
  defaultOpen = false,
}: {
  title: string;
  badge: string;
  description: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="group rounded-[1.75rem] border-2 border-black/15 bg-white/80 p-6 shadow-[0_0_0_1px_rgba(15,23,42,0.02)]" open={defaultOpen}>
      <summary className="cursor-pointer list-none [&::-webkit-details-marker]:hidden">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <svg
              viewBox="0 0 16 16"
              aria-hidden="true"
              className="mt-1 h-4 w-4 shrink-0 text-black/60 transition-transform duration-200 group-open:rotate-180"
            >
              <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <div>
            <h3 className="font-display text-2xl">{title}</h3>
            <p className="mt-2 text-sm text-black/65">{description}</p>
            </div>
          </div>
          <span className="rounded-full border border-black/10 bg-paper/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-black/70">
            {badge}
          </span>
        </div>
      </summary>
      <div className="mt-5 space-y-8">{children}</div>
    </details>
  );
}

export function AnalysisReport({ result }: { result: VerificationResult }) {
  const layerResults = asRecord(result.layer_results);
  const layer1 = asRecord(layerResults.layer1);
  const layer2 = asRecord(layerResults.layer2);
  const layer3 = asRecord(layerResults.layer3);
  const layer2Details = asRecord(layer2.details);
  const resultInputType = String(result.input_type ?? "").toLowerCase();
  const inputTextMode =
    resultInputType === "text" ||
    resultInputType === "pasted_text";
  const imageMode = resultInputType === "image";
  const audioMode = resultInputType === "audio";
  const videoMode = resultInputType === "video";
  const feedbackMode = asRecord(layer3.details).mode === "feedback";
  const mediaAuthenticity = asRecord(asRecord(result.analysis_report).media_authenticity);
  const systemScoring = asRecord(asRecord(result.analysis_report).system_scoring ?? asRecord(result.proof_summary).system_scoring);
  const systemLayerScores = asRecord(systemScoring.layer_scores);
  const systemLayerBreakdown = asRecord(systemScoring.layer_breakdown);
  const displayLayerScores = asRecord(systemScoring.display_layer_scores);
  const displayLayerBreakdown = asRecord(systemScoring.display_layer_breakdown);
  const urlEvidenceScoring = asRecord(systemScoring.url_evidence_scoring ?? asRecord(systemLayerBreakdown.evidence_weighted));
  const urlEvidenceMode = false;
  const textMode =
    inputTextMode &&
    (
      systemScoring.scoring_method === "text_verification_formula" ||
      systemScoring.formula === "100 * (0.45S + 0.20R + 0.20A + 0.10D + 0.05C) * E * P" ||
      Object.keys(asRecord(systemScoring.text_formula_components)).length > 0 ||
      String(asRecord(systemLayerBreakdown.crypto).weighted_scoring ?? "").toLowerCase().includes("pasted-text formula") ||
      String(asRecord(systemLayerBreakdown.text_formula).adjusted_final_score ?? "").length > 0
    );
  const textFormulaComponents = asRecord(systemScoring.text_formula_components ?? asRecord(systemLayerBreakdown.text_formula) ?? layer2Details.text_component_scores);
  const textS = numberOrUndefined(textFormulaComponents.S ?? textFormulaComponents.claim_evidence_support) ?? 0;
  const textR = numberOrUndefined(textFormulaComponents.R ?? textFormulaComponents.source_reliability) ?? 0;
  const textA = numberOrUndefined(textFormulaComponents.A ?? textFormulaComponents.cross_source_agreement) ?? 0;
  const textD = numberOrUndefined(textFormulaComponents.D ?? textFormulaComponents.evidence_directness) ?? 0;
  const textC = numberOrUndefined(textFormulaComponents.C ?? textFormulaComponents.claim_clarity) ?? 0;
  const textE = numberOrUndefined(textFormulaComponents.E ?? textFormulaComponents.evidence_sufficiency ?? systemScoring.evidence_multiplier ?? layer2Details.evidence_multiplier) ?? 0;
  const textP = numberOrUndefined(textFormulaComponents.P ?? textFormulaComponents.penalty_multiplier ?? systemScoring.penalty_multiplier ?? systemScoring.global_penalty ?? layer2Details.penalty_multiplier) ?? 0;
  const hasTextFormula = Object.keys(textFormulaComponents).length > 0;
  const computedTextRawScore = 100 * (0.45 * textS + 0.20 * textR + 0.20 * textA + 0.10 * textD + 0.05 * textC);
  const computedTextAdjustedScore = computedTextRawScore * textE * textP;
  const textRawScore = hasTextFormula ? computedTextRawScore : numberOrUndefined(systemScoring.raw_score);
  const textAdjustedScore = hasTextFormula ? computedTextAdjustedScore : numberOrUndefined(systemScoring.adjusted_score) ?? numberOrUndefined(systemScoring.adjusted_final_score);
  const textDisplayedScore = textAdjustedScore ?? numberOrUndefined(systemScoring.displayed_score) ?? numberOrUndefined(layer2Details.text_verification_score) ?? numberOrUndefined(systemScoring.final_score);
  const displayedOverallScore = textMode ? textDisplayedScore : systemScoring.final_score;
  const displayedOverallLabel = labelFromScore100(displayedOverallScore) ?? String(systemScoring.final_label ?? result.status);
  const sourceProvenanceCheck = asRecord(systemScoring.source_provenance_check ?? asRecord(systemLayerBreakdown.crypto).source_provenance_check);
  const textNlpLayerScore =
    numberOrUndefined(displayLayerScores.nlp) ??
    numberOrUndefined(asRecord(displayLayerBreakdown.nlp).score) ??
    (textDisplayedScore !== undefined ? Number(textDisplayedScore) / 100 : undefined);
  const activeLayerWeights = asRecord(systemScoring.active_layer_weights);
  const urlFixedFormulaDebug = asRecord(systemScoring.url_fixed_formula_debug);
  const urlMissingLayers = asArray(urlFixedFormulaDebug.missing_layers).map((item) => String(item));
  const urlSupportSources = asArray(urlEvidenceScoring.supporting_sources).map(asRecord);
  const urlRefutingSources = asArray(urlEvidenceScoring.refuting_sources).map(asRecord);
  const urlContextSources = asArray(urlEvidenceScoring.ignored_or_context_sources).map(asRecord);
  const youtubeDebug = asRecord(asRecord(result.analysis_report).youtube_debug);
  const details: DetailMap = {
    ...asRecord(layer1.details),
    extracted_publisher_name: asRecord(layer1.details).extracted_publisher_name ?? result.content_preview.publisher_name,
    extracted_author: asRecord(layer1.details).extracted_author ?? result.content_preview.author,
    extracted_publish_time: asRecord(layer1.details).extracted_publish_time ?? result.content_preview.publish_time,
    extracted_source_domain: asRecord(layer1.details).extracted_source_domain ?? result.content_preview.source_domain,
    extracted_section: asRecord(layer1.details).extracted_section ?? result.content_preview.section,
    extracted_keywords: asRecord(layer1.details).extracted_keywords ?? result.content_preview.keywords,
    extracted_hero_image_url: asRecord(layer1.details).extracted_hero_image_url ?? result.content_preview.hero_image_url,
    extracted_media_assets: asRecord(layer1.details).extracted_media_assets ?? result.content_preview.media_assets,
  };

  const provenanceStatus = getProvenanceStatus(result, details);
  const rows = summaryRows(result, details, provenanceStatus);
  const transportRows = transportEvidenceRows(details);
  const metadataSignals = asArray(details.metadata_signals).map((item) => String(item));
  const mediaAssets = mediaRows(details, asRecord(result.content_preview));
  const pageCredentials = asArray(details.c2pa_credentials)
    .map(asRecord)
    .filter((item) => String(item.source ?? "") !== "remote_media");
  const youtubeSignals = asRecord(details.youtube_signals);
  const youtubePageCredential =
    youtubeSignals.page_level_provenance_signal === true || youtubeSignals.original_content_claim === true
      ? {
          source: "youtube_page",
          type: youtubeSignals.original_content_claim === true ? "original_content_signal" : "platform_signal",
          url: String(result.content_preview.canonical_url ?? details.extracted_source_domain ?? ""),
          description:
            youtubeSignals.original_content_claim === true
              ? "YouTube page signals indicate original content from a verified channel under a non-Creative Commons license."
              : "YouTube indicates content credentials or provenance information in the page experience.",
        }
      : null;
  const effectivePageCredentials = youtubePageCredential
    ? [youtubePageCredential, ...pageCredentials.filter((item) => String(item.source ?? "") !== "youtube_page")]
    : pageCredentials;
  const mediaChecked = mediaAssets.filter((item) => item.checked).length;
  const mediaDetected = mediaAssets.filter((item) => item.credentialDetected).length;
  const mediaCouldNotEvaluate = mediaAssets.filter((item) => item.verificationState === "could_not_evaluate_asset").length;
  const mediaNoCredentials = mediaAssets.filter((item) => item.checked && !item.credentialDetected && item.verificationState !== "could_not_evaluate_asset").length;
  const factMatches = asArray(layer2.details ? asRecord(layer2.details).matches : []);
  const textClaims = asArray(asRecord(layer2.details).extracted_claims);
  const textClaimResults = asArray(asRecord(layer2.details).claim_results).map(asRecord);
  const textDebug = asRecord(asRecord(layer2.details).text_debug);
  const factSignals = asArray(layer2.details ? asRecord(layer2.details).metadata_signals : []).map((item) => String(item));
  const totalMatchCount = Number(asRecord(layer2.details).total_match_count ?? factMatches.length);
  const hasMoreMatches = Boolean(asRecord(layer2.details).has_more_matches);
  const uploadForensics = asRecord(details.forensics);
  const rawExternalDetected = mediaAssets.some((item) =>
    ["credentials_verified", "remote_manifest_verified", "sidecar_manifest_verified"].includes(String(item.verificationState)),
  );
  const platformExternalDetected =
    effectivePageCredentials.length > 0 ||
    mediaAssets.some((item) => String(item.verificationState) === "platform_signal_detected_manifest_not_retrievable");
  const internalProof = {
    snapshotCreated: Boolean(details.internal_manifest_id || result.proof_summary.content_hash || details.record_type === "article_snapshot"),
    canonicalized: Boolean(result.proof_summary.content_hash || details.canonical_hash),
    hashGenerated: Boolean(result.proof_summary.content_hash || details.canonical_hash),
    signatureVerified: Boolean(details.internal_manifest_valid || details.platform_manifest_valid),
    timestampAttached: Boolean(details.timestamp_source || details.timestamp_provider || details.timestamp_token_status),
    versionChainValid: details.version_chain_valid === false ? false : Boolean(details.previous_manifest_id ? details.version_chain_valid !== false : true),
  };

  const timelineRows = result.audit_trail.map((entry, index) => ({
    version: index + 1,
    eventType: entry.event_type,
    issuer: String(details.issued_by ?? details.signer_name ?? "NewsProof system"),
    timestamp: new Date(entry.created_at).toLocaleString(),
    hash: shortHash(entry.entry_hash),
    status: index === result.audit_trail.length - 1 ? provenanceStatus.badge : "Recorded",
  }));

  return (
    <section className="panel mt-8 space-y-8 p-6">
      {Object.keys(systemScoring).length ? (
        <section className="rounded-[1.75rem] border border-black/10 bg-white/80 p-6">
          <div className="flex items-center justify-between gap-4">
            <h3 className="font-display text-2xl">Verification Score</h3>
            <span className="rounded-full border border-black/10 bg-paper/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-black/70">
              {textMode ? "Text Formula" : "Whole System"}
            </span>
          </div>
          <p className="mt-2 text-sm text-black/65">
              {textMode
                ? "For pasted text, this score uses the text-verification thesis formula only: claim support, source reliability, cross-source agreement, evidence directness, claim clarity, evidence sufficiency, and penalty multiplier."
                : feedbackMode
                ? imageMode
                  ? "This score reflects verification confidence from image-analysis evidence and provenance evidence. Review feedback is shown separately and does not count as a verification score."
                  : videoMode
                    ? "This score reflects verification confidence from video-analysis evidence and provenance evidence. Review feedback is shown separately and does not count as a verification score."
                  : audioMode
                    ? "This score reflects verification confidence from audio-analysis evidence and provenance evidence. Review feedback is shown separately and does not count as a verification score."
                  : "This score reflects verification confidence from provenance and NLP evidence. Review feedback is shown separately and does not count as a verification score."
              : "This score reflects verification confidence from provenance, NLP evidence, and review signals. It does not mean absolute truth."}
          </p>
          <div className="mt-5 grid gap-4 lg:grid-cols-[0.85fr_1.15fr]">
            <div className="rounded-[1.5rem] border border-black/10 bg-paper/60 p-5">
              <p className="text-xs uppercase tracking-[0.3em] text-black/50">Overall result</p>
              <p className="mt-3 text-3xl font-semibold text-black/90">{textMode ? formatAbsoluteScore100(displayedOverallScore) : formatScore100(displayedOverallScore)}</p>
              <div className="mt-4 space-y-2 text-sm text-black/80">
                <p><strong>Label:</strong> {displayedOverallLabel}</p>
                {systemScoring.reason ? <p><strong>Reason:</strong> {String(systemScoring.reason)}</p> : null}
                {textMode ? <p><strong>Raw score:</strong> {formatAbsoluteScore100(textRawScore)}</p> : null}
                {textMode ? <p><strong>Adjusted score:</strong> {formatAbsoluteScore100(textAdjustedScore)}</p> : null}
                {textMode ? <p><strong>Displayed score:</strong> {formatAbsoluteScore100(textDisplayedScore)}</p> : null}
                {textMode && systemScoring.override_reason ? <p><strong>Override reason:</strong> {String(systemScoring.override_reason)}</p> : null}
                {textMode ? (
                  <>
                    <p><strong>Evidence sufficiency:</strong> {formatScore100(textMode ? textE : systemScoring.evidence_multiplier)}</p>
                    <p><strong>{textMode ? "Penalty multiplier" : "Global penalty"}:</strong> {formatScore100(textMode ? textP : systemScoring.penalty_multiplier ?? systemScoring.global_penalty)}</p>
                  </>
                ) : asRecord(systemScoring.url_multiplier_policy).principle ? (
                  <>
                    <p><strong>Scoring policy:</strong> {String(asRecord(systemScoring.url_multiplier_policy).principle)}</p>
                    {Object.keys(urlFixedFormulaDebug).length ? (
                      <>
                        <p><strong>Formula mode:</strong> {String(urlFixedFormulaDebug.formula_mode ?? "fixed_url_formula")}</p>
                        <p><strong>Weight re-normalized:</strong> {urlFixedFormulaDebug.weight_renormalized ? "Yes" : "No"}</p>
                        {urlMissingLayers.length ? <p><strong>Missing layers:</strong> {urlMissingLayers.join(", ")}</p> : null}
                      </>
                    ) : null}
                  </>
                ) : (
                  <p><strong>Scoring policy:</strong> URL score is calculated from active layer weights. Evidence and penalty diagnostics are not multiplied into the final score.</p>
                )}
              </div>
              {textMode ? (
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  <div className="rounded-2xl border border-black/10 bg-white/70 p-4 text-sm text-black/80">
                    <p className="text-xs uppercase tracking-[0.2em] text-black/45">Source/Provenance Check</p>
                    <p className="mt-2 text-base font-semibold">{String(sourceProvenanceCheck.label ?? "Not applicable / no provenance available for pasted text")}</p>
                    {sourceProvenanceCheck.source_url ? (
                      <p className="mt-2 break-all text-xs text-black/65">
                        <strong>Recovered source:</strong>{" "}
                        <a className="text-accent underline" href={String(sourceProvenanceCheck.source_url)} target="_blank" rel="noreferrer">
                          {String(sourceProvenanceCheck.source_url)}
                        </a>
                      </p>
                    ) : null}
                    <p className="mt-2 text-xs text-black/60">
                      {String(sourceProvenanceCheck.reason ?? "Pasted text has no source URL, hash, signature, or provenance record.")}
                    </p>
                    <p className="mt-2 text-xs text-black/55">
                      Cryptographic provenance proves origin and integrity. It does not prove factual truth.
                    </p>
                  </div>
                  <div className="rounded-2xl border border-black/10 bg-white/70 p-4 text-sm text-black/80">
                    <p className="text-xs uppercase tracking-[0.2em] text-black/45">Text formula / NLP</p>
                    <p className="mt-2 text-xl font-semibold">{formatScore100(textNlpLayerScore)}</p>
                    <p className="mt-2 text-xs text-black/60">
                      Text formula result after evidence sufficiency and penalty multipliers.
                    </p>
                  </div>
                </div>
              ) : null}
            </div>
            <div className="rounded-[1.5rem] border border-black/10 bg-paper/60 p-5">
              <p className="text-xs uppercase tracking-[0.3em] text-black/50">{textMode ? "Text formula breakdown" : "Layer score breakdown"}</p>
              {textMode ? (
                <div className="mt-4 grid gap-3 md:grid-cols-2">
                  {[
                    ["Claim-evidence support", textS],
                    ["Source reliability", textR],
                    ["Cross-source agreement", textA],
                    ["Evidence directness", textD],
                    ["Claim clarity", textC],
                    ["Evidence sufficiency", textE],
                    ["Penalty multiplier", textP],
                    ["Raw score", textRawScore, "absolute"],
                    ["Final adjusted score", textDisplayedScore, "absolute"],
                  ].map(([label, value, scoreType]) => (
                    <div key={String(label)} className="rounded-2xl border border-black/10 bg-white/70 p-4 text-sm text-black/80">
                      <p className="text-xs uppercase tracking-[0.2em] text-black/45">{String(label)}</p>
                      <p className="mt-2 text-xl font-semibold">{scoreType === "absolute" ? formatAbsoluteScore100(value) : formatScore100(value)}</p>
                    </div>
                  ))}
                </div>
              ) : (
              <div className={feedbackMode ? "mt-4 grid gap-3 md:grid-cols-2" : "mt-4 grid gap-3 md:grid-cols-3"}>
                <div className="rounded-2xl border border-black/10 bg-white/70 p-4 text-sm text-black/80">
                  <p className="text-xs uppercase tracking-[0.2em] text-black/45">Crypto layer</p>
                  <p className="mt-2 text-xl font-semibold">{formatScore100(systemLayerScores.crypto)}</p>
                  <p className="mt-2 text-xs text-black/60">{String(asRecord(systemLayerBreakdown.crypto).weighted_scoring ?? "Active")}</p>
                  {activeLayerWeights.crypto !== undefined ? <p className="mt-1 text-xs text-black/55">Weight: {Number(activeLayerWeights.crypto).toFixed(3)}</p> : null}
                </div>
                <div className="rounded-2xl border border-black/10 bg-white/70 p-4 text-sm text-black/80">
                  <p className="text-xs uppercase tracking-[0.2em] text-black/45">{imageMode ? "Image analysis layer" : audioMode ? "Audio analysis layer" : videoMode ? "Video analysis layer" : "NLP layer"}</p>
                  <p className="mt-2 text-xl font-semibold">{formatScore100(systemLayerScores.nlp)}</p>
                  <p className="mt-2 text-xs text-black/60">{String(asRecord(systemLayerBreakdown.nlp).weighted_scoring ?? "Active")}</p>
                  {activeLayerWeights.nlp !== undefined ? <p className="mt-1 text-xs text-black/55">Weight: {Number(activeLayerWeights.nlp).toFixed(3)}</p> : null}
                </div>
                {!feedbackMode ? (
                  <div className="rounded-2xl border border-black/10 bg-white/70 p-4 text-sm text-black/80">
                    <p className="text-xs uppercase tracking-[0.2em] text-black/45">Consensus layer</p>
                    <p className="mt-2 text-xl font-semibold">{formatScore100(systemLayerScores.community)}</p>
                    <p className="mt-2 text-xs text-black/60">{String(asRecord(systemLayerBreakdown.community).weighted_scoring ?? "Missing: not enough community review data. Treated as neutral / excluded from weighted scoring.")}</p>
                    {activeLayerWeights.community !== undefined ? <p className="mt-1 text-xs text-black/55">Weight: {Number(activeLayerWeights.community).toFixed(3)}</p> : null}
                  </div>
                ) : null}
              </div>
              )}
              {systemScoring.explanation ? (
                <p className="mt-4 text-sm text-black/75">{String(systemScoring.explanation)}</p>
              ) : null}
            </div>
          </div>
        </section>
      ) : null}

      <LayerDisclosure
        title="Cryptographic Provenance"
        badge="Crypto Layer"
        description="Open this layer to inspect provenance details, credential checks, media forensics, and technical proof records."
      >
      <section className="rounded-[1.75rem] border border-black/10 bg-white/80 p-6">
        <h3 className="font-display text-2xl">Provenance Summary</h3>
        <div className="mt-5 grid gap-4 md:grid-cols-2">
          {rows.map(([label, value]) => (
            <div key={label} className="rounded-2xl border border-black/10 bg-paper/60 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-black/45">{label}</p>
              <p className="mt-2 break-all text-sm text-black/80">{value}</p>
            </div>
          ))}
        </div>
        {transportRows.length ? (
          <div className="mt-5">
            <p className="text-xs uppercase tracking-[0.3em] text-black/50">Transport evidence</p>
            <div className="mt-3 grid gap-4 md:grid-cols-2">
              {transportRows.map(([label, value]) => (
                <div key={label} className="rounded-2xl border border-black/10 bg-paper/60 p-4">
                  <p className="text-xs uppercase tracking-[0.2em] text-black/45">{label}</p>
                  <p className="mt-2 break-words text-sm text-black/80">{value}</p>
                </div>
              ))}
            </div>
            <p className="mt-3 text-xs text-black/55">
              Transport certificate details support secure delivery context only. They are not treated as article-level signature proof.
            </p>
          </div>
        ) : null}
      </section>

      <section className="space-y-4 rounded-[1.75rem] border border-black/10 bg-white/80 p-6">
        <h3 className="font-display text-2xl">Proof Types</h3>
        <div className="grid gap-4 xl:grid-cols-3">
          <article className="rounded-[1.5rem] border border-black/10 bg-paper/60 p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-black/50">Publisher attribution</p>
            <p className="mt-3 text-sm text-black/65">This is not cryptographic proof of the article itself.</p>
            <div className="mt-4 space-y-2 text-sm text-black/80">
              <p><strong>Layer confidence score:</strong> {formatScore100(systemLayerScores.crypto)}</p>
              <p><strong>Publisher name:</strong> {String(details.extracted_publisher_name ?? details.publisher ?? "Not found")}</p>
              <p><strong>Source domain:</strong> {String(details.extracted_source_domain ?? details.source_domain ?? "Not found")}</p>
              <p><strong>Author:</strong> {String(details.extracted_author ?? "Not found")}</p>
              <p><strong>Published date:</strong> {details.extracted_publish_time ? new Date(String(details.extracted_publish_time)).toLocaleString() : "Not found"}</p>
              <p><strong>Metadata consistency result:</strong> {metadataSignals.length ? `Consistent fields found: ${metadataSignals.join(", ")}` : "Not enough structured metadata to confirm consistency."}</p>
              <p><strong>Registry match result:</strong> {details.publisher || details.publisher_domain ? "Matched trusted registry publisher." : "No trusted registry match recorded."}</p>
              {Object.keys(youtubeSignals).length || Object.keys(youtubeDebug).length ? (
                <>
                  <p><strong>YouTube video ID:</strong> {String(youtubeDebug.video_id ?? "Not found")}</p>
                  <p><strong>Channel title:</strong> {String(youtubeDebug.channel_title ?? "Not found")}</p>
                  <p><strong>Channel ID:</strong> {String(youtubeDebug.channel_id ?? "Not found")}</p>
                  <p><strong>Channel verification status:</strong> {String(youtubeDebug.channel_verification_status ?? youtubeSignals.channel_verification_status ?? "Not confirmed")}</p>
                  <p><strong>Transcript status:</strong> {String(youtubeDebug.transcript_status ?? "Not available")}</p>
                  <p><strong>License:</strong> {String(youtubeSignals.license_label ?? "Not found")}</p>
                  {youtubeSignals.original_content_claim === true ? (
                    <p><strong>Content classification:</strong> Original content</p>
                  ) : youtubeSignals.creative_commons === true ? (
                    <p><strong>Content classification:</strong> Creative Commons licensed content</p>
                  ) : null}
                </>
              ) : null}
            </div>
          </article>

          <article className="rounded-[1.5rem] border border-black/10 bg-paper/60 p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-black/50">Internal platform provenance</p>
            <div className="mt-4 space-y-2 text-sm text-black/80">
              <p><strong>Layer confidence score:</strong> {formatScore100(systemLayerScores.crypto)}</p>
              <p><strong>Snapshot created:</strong> {yesNo(internalProof.snapshotCreated)}</p>
              <p><strong>Canonicalized:</strong> {yesNo(internalProof.canonicalized)}</p>
              <p><strong>Hash generated:</strong> {yesNo(internalProof.hashGenerated)}</p>
              <p><strong>Signature verified:</strong> {yesNo(internalProof.signatureVerified)}</p>
              <p><strong>Timestamp attached:</strong> {yesNo(internalProof.timestampAttached)}</p>
              <p><strong>Version chain valid:</strong> {yesNo(internalProof.versionChainValid)}</p>
            </div>
          </article>

          <article className="rounded-[1.5rem] border border-black/10 bg-paper/60 p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-black/50">External media/content credentials</p>
            <div className="mt-4 space-y-2 text-sm text-black/80">
              <p>
                <strong>External provenance found:</strong>{" "}
                {rawExternalDetected && platformExternalDetected
                  ? "Yes, from both raw checks and platform/page signals"
                  : rawExternalDetected
                    ? "Yes, from raw asset or manifest checks"
                    : platformExternalDetected
                      ? "Yes, from platform/page signals"
                      : "No"}
              </p>
              <p><strong>Page/source provenance signals:</strong> {effectivePageCredentials.length}</p>
              <p><strong>Media assets checked:</strong> {mediaChecked}</p>
              <p><strong>Assets with detected credentials:</strong> {mediaDetected}</p>
              <p><strong>Assets with no credentials:</strong> {mediaNoCredentials}</p>
              <p><strong>Assets that could not be evaluated:</strong> {mediaCouldNotEvaluate}</p>
            </div>
            {effectivePageCredentials.length ? (
              <div className="mt-4 space-y-3">
                {effectivePageCredentials.map((credential, index) => (
                  <div key={`page-credential-${index}`} className="rounded-2xl border border-black/10 bg-white/70 p-3 text-xs text-black/75">
                    <p><strong>Source signal:</strong> {String(credential.source ?? "page")}</p>
                    <p><strong>Type:</strong> {String(credential.type ?? "manifest")}</p>
                    {credential.url ? (
                      <p>
                        <strong>Link:</strong>{" "}
                        <a className="break-all text-accent underline" href={String(credential.url)} target="_blank" rel="noreferrer">
                          {String(credential.url)}
                        </a>
                      </p>
                    ) : null}
                    <p>{String(credential.description ?? "Page-level provenance signal detected.")}</p>
                  </div>
                ))}
              </div>
            ) : null}
            <div className="mt-4 space-y-3">
              {mediaAssets.length ? mediaAssets.map((asset) => (
                <div key={asset.url} className="rounded-2xl border border-black/10 bg-white/70 p-3 text-xs text-black/75">
                  <p><strong>Asset URL:</strong> <a className="break-all text-accent underline" href={asset.url} target="_blank" rel="noreferrer">{asset.url}</a></p>
                  <p><strong>Type:</strong> {asset.type}</p>
                  {asset.platform ? <p><strong>Platform:</strong> {asset.platform}</p> : null}
                  {asset.verificationState ? <p><strong>Check result:</strong> {asset.verificationState.replaceAll("_", " ")}</p> : null}
                  <p><strong>Returned content-type:</strong> {asset.contentType}</p>
                  <p><strong>Credential or provenance signal detected:</strong> {yesNo(asset.credentialDetected)}</p>
                  <p>{asset.note}</p>
                </div>
              )) : (
                <p className="text-sm text-black/65">No associated media assets were available for external credential checks.</p>
              )}
            </div>
          </article>
        </div>
      </section>

      <section className="rounded-[1.75rem] border border-black/10 bg-white/80 p-6">
        <h3 className="font-display text-2xl">Media Forensics</h3>
        <p className="mt-2 text-sm text-black/65">
          This section reports recoverable technical metadata from image, audio, and video files. It can show device, software, container, codec, or timestamp clues when those details are actually preserved.
        </p>

        {Object.keys(uploadForensics).length ? (
          <div className="mt-5 rounded-[1.5rem] border border-black/10 bg-paper/60 p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-black/50">Submitted file forensics</p>
            <div className="mt-4 space-y-2 text-sm text-black/80">
              <p><strong>Media kind:</strong> {String(uploadForensics.media_kind ?? "unknown")}</p>
              <p><strong>Metadata status:</strong> {String(uploadForensics.metadata_status ?? "unknown")}</p>
              {uploadForensics.format ? <p><strong>Format:</strong> {String(uploadForensics.format)}</p> : null}
              {uploadForensics.format_name ? <p><strong>Container:</strong> {String(uploadForensics.format_name)}</p> : null}
              {uploadForensics.duration_seconds ? <p><strong>Duration:</strong> {String(uploadForensics.duration_seconds)} seconds</p> : null}
              {uploadForensics.size_bytes ? <p><strong>Size:</strong> {String(uploadForensics.size_bytes)} bytes</p> : null}
              {uploadForensics.width || uploadForensics.height ? (
                <p><strong>Dimensions:</strong> {String(uploadForensics.width ?? "?")} x {String(uploadForensics.height ?? "?")}</p>
              ) : null}
              {Object.keys(asRecord(uploadForensics.fields)).length ? (
                <div>
                  <p><strong>Recovered fields:</strong></p>
                  <div className="mt-2 space-y-1 text-xs text-black/70">
                    {Object.entries(asRecord(uploadForensics.fields)).map(([key, value]) => (
                      <p key={key}><strong>{key.replaceAll("_", " ")}:</strong> {String(value)}</p>
                    ))}
                  </div>
                </div>
              ) : null}
              {asArray(uploadForensics.streams).length ? (
                <div>
                  <p><strong>Stream details:</strong></p>
                  <div className="mt-2 space-y-2">
                    {asArray(uploadForensics.streams).map((stream, index) => (
                      <div key={index} className="rounded-xl border border-black/10 bg-white/60 p-3 text-xs text-black/70">
                        {Object.entries(asRecord(stream)).map(([key, value]) => (
                          <p key={key}><strong>{key.replaceAll("_", " ")}:</strong> {String(value)}</p>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
              {asArray(uploadForensics.notes).length ? (
                <div className="space-y-1 text-xs text-black/65">
                  {asArray(uploadForensics.notes).map((note, index) => (
                    <p key={index}>{String(note)}</p>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        ) : null}

        {mediaAssets.length ? (
          <div className="mt-5 space-y-3">
            <p className="text-xs uppercase tracking-[0.3em] text-black/50">Associated remote asset forensics</p>
            {mediaAssets.map((asset) => {
              const forensic = asRecord(asset.forensics);
              const remote = asArray(details.remote_media_provenance).map(asRecord).find((item) => String(item.url ?? "") === asset.url) ?? {};
              return (
                <div key={`forensics-${asset.url}`} className="rounded-[1.5rem] border border-black/10 bg-paper/60 p-4 text-sm text-black/80">
                  <p><strong>Asset URL:</strong> <a className="break-all text-accent underline" href={asset.url} target="_blank" rel="noreferrer">{asset.url}</a></p>
                  <p><strong>Type:</strong> {asset.type}</p>
                  {remote.source ? <p><strong>Found via:</strong> {String(remote.source)}</p> : null}
                  {remote.platform ? <p><strong>Platform:</strong> {String(remote.platform)}</p> : null}
                  {describeRemoteCheck(remote, String(asset.type)) ? (
                    <p><strong>How it was checked:</strong> {describeRemoteCheck(remote, String(asset.type))}</p>
                  ) : null}
                  {asRecord(asset).source ? <p><strong>Asset source:</strong> {String(asRecord(asset).source)}</p> : null}
                  {asRecord(asset).caption ? <p><strong>Caption:</strong> {String(asRecord(asset).caption)}</p> : null}
                  {asRecord(asset).alt ? <p><strong>Alt text:</strong> {String(asRecord(asset).alt)}</p> : null}
                  {asRecord(asset).poster ? <p><strong>Poster:</strong> {String(asRecord(asset).poster)}</p> : null}
                  {asRecord(asset).mime_type ? <p><strong>Declared MIME:</strong> {String(asRecord(asset).mime_type)}</p> : null}
                  <p><strong>Metadata status:</strong> {String(forensic.metadata_status ?? "unknown")}</p>
                  {forensic.format ? <p><strong>Format:</strong> {String(forensic.format)}</p> : null}
                  {forensic.content_type ? <p><strong>Content-Type:</strong> {String(forensic.content_type)}</p> : null}
                  {forensic.content_length ? <p><strong>Content-Length:</strong> {String(forensic.content_length)}</p> : null}
                  {forensic.width || forensic.height ? (
                    <p><strong>Dimensions:</strong> {String(forensic.width ?? "?")} x {String(forensic.height ?? "?")}</p>
                  ) : null}
                  {Object.keys(asRecord(forensic.fields)).length ? (
                    <div className="mt-2 space-y-1 text-xs text-black/70">
                      {Object.entries(asRecord(forensic.fields)).map(([key, value]) => (
                        <p key={key}><strong>{key.replaceAll("_", " ")}:</strong> {String(value)}</p>
                      ))}
                    </div>
                  ) : null}
                  {asArray(forensic.notes).length ? (
                    <div className="mt-2 space-y-1 text-xs text-black/65">
                      {asArray(forensic.notes).map((note, index) => (
                        <p key={index}>{String(note)}</p>
                      ))}
                    </div>
                  ) : null}
                  {Array.isArray(remote.checked_candidates) && remote.checked_candidates.length > 1 ? (
                    <p className="mt-2 text-xs text-black/65"><strong>Checked candidates:</strong> {remote.checked_candidates.map((candidate) => String(candidate)).join(" | ")}</p>
                  ) : null}
                </div>
              );
            })}
          </div>
        ) : null}
      </section>
      </LayerDisclosure>

      {Object.keys(mediaAuthenticity).length && String(mediaAuthenticity.status ?? "") !== "not_applicable" ? (
        <section className="rounded-[1.75rem] border border-black/10 bg-white/80 p-6">
          <div className="flex items-center justify-between gap-4">
            <h3 className="font-display text-2xl">Media Authenticity</h3>
            <span className="rounded-full border border-black/10 bg-paper/70 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-black/70">
              Forensics
            </span>
          </div>
          <p className="mt-2 text-sm text-black/65">
            This section summarizes available media forensic clues and provenance limitations. It does not determine whether the underlying claim is true.
          </p>
          <div className="mt-5 grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
            <div className="rounded-[1.5rem] border border-black/10 bg-paper/60 p-5">
              <p className="text-xs uppercase tracking-[0.3em] text-black/50">Assessment</p>
              <p className="mt-3 text-2xl font-semibold text-black/85">{String(mediaAuthenticity.status ?? "inconclusive").replaceAll("_", " ")}</p>
              <div className="mt-4 space-y-2 text-sm text-black/80">
                <p><strong>Risk level:</strong> {String(mediaAuthenticity.risk_level ?? "unknown")}</p>
                {"score" in mediaAuthenticity ? <p><strong>Authenticity score:</strong> {String(mediaAuthenticity.score)}</p> : null}
                <p>{String(mediaAuthenticity.explanation ?? "No media authenticity assessment was available.")}</p>
              </div>
              <p className="mt-4 text-xs text-black/60">
                This score reflects available media signals, not whether the article claim is true.
              </p>
            </div>

            <div className="rounded-[1.5rem] border border-black/10 bg-paper/60 p-5">
              <p className="text-xs uppercase tracking-[0.3em] text-black/50">Signals and limitations</p>
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div>
                  <p className="text-sm font-semibold text-black/80">Signals found</p>
                  <div className="mt-2 space-y-2 text-sm text-black/75">
                    {asArray(mediaAuthenticity.signals).length ? (
                      asArray(mediaAuthenticity.signals).map((signal, index) => <p key={index}>{String(signal)}</p>)
                    ) : (
                      <p>No strong manipulation signals were recorded.</p>
                    )}
                  </div>
                </div>
                <div>
                  <p className="text-sm font-semibold text-black/80">Limitations</p>
                  <div className="mt-2 space-y-2 text-sm text-black/75">
                    {asArray(mediaAuthenticity.limitations).length ? (
                      asArray(mediaAuthenticity.limitations).map((item, index) => <p key={index}>{String(item)}</p>)
                    ) : (
                      <p>No additional limitations were recorded.</p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      ) : null}

      <LayerDisclosure
        title="Version History and Audit Trail"
        badge="Audit Layer"
        description="Open this layer to inspect version history, audit events, hashes, and technical verification records."
      >
      <section className="rounded-[1.75rem] border border-black/10 bg-white/80 p-6">
        <h3 className="font-display text-2xl">Version History and Audit Trail</h3>
        <div className="mt-5 overflow-x-auto">
          <table className="min-w-full border-separate border-spacing-y-2 text-sm text-black/80">
            <thead>
              <tr className="text-left text-xs uppercase tracking-[0.2em] text-black/45">
                <th className="px-3 py-2">Version</th>
                <th className="px-3 py-2">Event type</th>
                <th className="px-3 py-2">Issuer</th>
                <th className="px-3 py-2">Timestamp</th>
                <th className="px-3 py-2">Short hash</th>
                <th className="px-3 py-2">Verification status</th>
              </tr>
            </thead>
            <tbody>
              {timelineRows.map((row) => (
                <tr key={`${row.version}-${row.hash}`} className="rounded-2xl bg-paper/60">
                  <td className="rounded-l-2xl px-3 py-3">{row.version}</td>
                  <td className="px-3 py-3">{row.eventType}</td>
                  <td className="px-3 py-3">{row.issuer}</td>
                  <td className="px-3 py-3">{row.timestamp}</td>
                  <td className="px-3 py-3">{row.hash}</td>
                  <td className="rounded-r-2xl px-3 py-3">{row.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <details className="mt-6 rounded-2xl border border-black/10 bg-paper/60 p-4">
          <summary className="cursor-pointer text-sm font-semibold text-black/80">Technical details</summary>
          <div className="mt-4 space-y-4 text-xs text-black/70">
            <div>
              <p className="font-medium text-black/80">Canonical JSON preview</p>
              <pre className="mt-2 overflow-auto rounded-2xl bg-ink p-4 text-[11px] text-white">
                {String(details.canonical_json_preview ?? details.canonical_json ?? "Not available")}
              </pre>
            </div>
            <p><strong>Full SHA-256 hash:</strong> {String(details.canonical_hash ?? result.proof_summary.content_hash ?? "Not available")}</p>
            <p><strong>Signature algorithm:</strong> {String(details.signature_algorithm ?? "Ed25519")}</p>
            <p><strong>Issuer key id:</strong> {String(details.issuer_key_id ?? details.key_id ?? "Not available")}</p>
            <p><strong>Timestamp token status:</strong> {String(details.timestamp_token_status ?? details.timestamp_source ?? "Not available")}</p>
            <p><strong>Previous manifest id:</strong> {String(details.previous_manifest_id ?? "None")}</p>
            <div>
              <p className="font-medium text-black/80">Verification logs</p>
              <pre className="mt-2 overflow-auto rounded-2xl bg-ink p-4 text-[11px] text-white">
                {JSON.stringify(
                  {
                    proof_summary: result.proof_summary,
                    layer1,
                    audit_trail: result.audit_trail,
                  },
                  null,
                  2,
                )}
              </pre>
            </div>
          </div>
        </details>
      </section>
      </LayerDisclosure>

      <LayerDisclosure
        title={imageMode ? "Image Analysis" : audioMode ? "Audio Analysis" : videoMode ? "Video Analysis" : textMode ? "Text Verification" : "Fact Checking"}
        badge={imageMode ? "Image Analysis Layer" : audioMode ? "Audio Analysis Layer" : videoMode ? "Video Analysis Layer" : "NLP Layer"}
        description={
          imageMode
            ? "Open this layer to inspect visual integrity checks, manipulation signals, OCR findings, context matching, and related evidence for the uploaded image."
            : videoMode
              ? "Open this layer to inspect audio transcription, sampled-frame OCR, metadata/provenance findings, contextual evidence, and known limitations for the uploaded video."
              : audioMode
                ? "Open this layer to inspect transcript quality, voice authenticity, manipulation checks, context matching, and related evidence for the uploaded audio."
                : textMode
                  ? "Open this layer to inspect the NLP-led claim analysis for pasted text, including extracted claims, supporting evidence, and the final text-verification judgment."
                  : "Open this layer to inspect NLP-based corroboration and matching evidence."
        }
      >
        <div className="grid gap-4 lg:grid-cols-[0.95fr_1.05fr]">
          <div className="rounded-[1.5rem] border border-black/10 bg-paper/60 p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-black/50">{imageMode ? "Image analysis summary" : audioMode ? "Audio analysis summary" : videoMode ? "Video analysis summary" : textMode ? "Text verification summary" : "NLP Summary"}</p>
            <p className="mt-3 text-sm text-black/80">{String(layer2.explanation ?? "No fact-checking summary available.")}</p>
            <div className="mt-4 space-y-2 text-sm text-black/80">
              {audioMode ? (
                <>
                  <p><strong>Transcript confidence:</strong> {formatScore100(Number(asRecord(asRecord(layer2.details).transcription_result).transcript_confidence ?? 0))}</p>
                  <p><strong>Audio authenticity score:</strong> {formatScore100(Number(asRecord(layer2.details).audio_analysis_score ?? 0))}</p>
                  <p><strong>Context fact-check score:</strong> {formatScore100(Number(asRecord(layer2.details).text_verification_score ?? systemLayerScores.nlp ?? 0))}</p>
                  <p><strong>Final NewsProof score:</strong> {formatScore100(Number(asRecord(asRecord(result.proof_summary).system_scoring).final_score ?? 0) / 100)}</p>
                </>
              ) : (
                <p><strong>{textMode ? "Text verification score" : "Layer confidence score"}:</strong> {formatScore100(textMode ? asRecord(layer2.details).text_verification_score : systemLayerScores.nlp)}</p>
              )}
              <p><strong>{imageMode || audioMode || videoMode ? "Finding" : textMode ? "Final label" : "Verdict"}:</strong> {String(asRecord(layer2.details).text_verification_label ?? asRecord(layer2.details).fact_check_verdict ?? (audioMode ? (asRecord(layer2.details).spoof_detected ? "spoofing risk found" : "inconclusive") : videoMode ? (asRecord(layer2.details).technical_warning ? "technical/provenance warning" : "inconclusive") : asRecord(layer2.details).manipulation_detected ? "manipulation risk found" : "inconclusive"))}</p>
              {asRecord(layer2.details).fact_check_assessment ? (
                <p><strong>{textMode ? "Explanation" : "Assessment"}:</strong> {String(asRecord(layer2.details).fact_check_assessment)}</p>
              ) : imageMode ? (
                <p><strong>Assessment:</strong> {String(layer2.explanation ?? "No image-analysis assessment available.")}</p>
              ) : videoMode ? (
                <p><strong>Assessment:</strong> {String(layer2.explanation ?? "No video-analysis assessment available.")}</p>
              ) : audioMode ? (
                <p><strong>Assessment:</strong> {String(layer2.explanation ?? "No audio-analysis assessment available.")}</p>
              ) : null}
              {textMode && asRecord(layer2.details).supporting_evidence_summary ? (
                <p><strong>Evidence summary:</strong> {String(asRecord(layer2.details).supporting_evidence_summary)}</p>
              ) : imageMode && asRecord(layer2.details).ocr_result ? (
                <p><strong>OCR summary:</strong> {String(asRecord(asRecord(layer2.details).ocr_result).ocr_summary ?? "No OCR summary available.")}</p>
              ) : videoMode && asRecord(layer2.details).ocr_result ? (
                <p><strong>OCR summary:</strong> {String(asRecord(asRecord(layer2.details).ocr_result).ocr_summary ?? "No OCR summary available.")}</p>
              ) : audioMode && asRecord(layer2.details).transcription_result ? (
                <p><strong>Transcript summary:</strong> {String(asRecord(asRecord(layer2.details).transcription_result).transcript_summary ?? "No transcript summary available.")}</p>
              ) : asRecord(layer2.details).match_summary ? (
                <p><strong>Match summary:</strong> {String(asRecord(layer2.details).match_summary)}</p>
              ) : null}
              {!textMode && asRecord(layer2.details).trusted_publisher_match ? (
                <p><strong>Publisher context:</strong> {String(asRecord(layer2.details).trusted_publisher_match)}</p>
              ) : null}
              <p><strong>{imageMode || audioMode || videoMode ? "Evidence sources found" : textMode ? "Evidence sources found" : "Similarity sources found"}:</strong> {factMatches.length}</p>
              {textMode && textClaims.length ? (
                <p><strong>Extracted claims:</strong> {textClaims.length}</p>
              ) : null}
              {textMode && asRecord(textDebug.final_source_recovery).status ? (
                <>
                <p><strong>Source recovery status:</strong> {String(asRecord(textDebug.final_source_recovery).status)}</p>
                  {asRecord(textDebug.final_source_recovery).url ? (
                    <p><strong>Recovered source URL:</strong> {String(asRecord(textDebug.final_source_recovery).url)}</p>
                  ) : null}
                  {asRecord(textDebug.final_source_recovery).reason ? (
                    <p><strong>Source recovery reason:</strong> {String(asRecord(textDebug.final_source_recovery).reason)}</p>
                  ) : null}
                </>
              ) : null}
              {textMode && asRecord(layer2.details).simplified_meaning ? (
                <p><strong>Simplified meaning:</strong> {String(asRecord(layer2.details).simplified_meaning)}</p>
              ) : null}
              {textMode && Object.keys(asRecord(asRecord(layer2.details).text_component_scores)).length ? (
                <p>
                  <strong>Formula components:</strong>{" "}
                  {Object.entries(asRecord(asRecord(layer2.details).text_component_scores))
                    .filter(([key]) => ["S", "R", "A", "D", "C", "E", "P"].includes(key))
                    .map(([key, value]) => `${key}=${String(value)}`)
                    .join(" | ")}
                </p>
              ) : null}
              {imageMode ? (
                <>
                  <p><strong>Image fetch status:</strong> {String(asRecord(asRecord(layer2.details).ocr_result).image_fetch_status ?? "not_applicable")}</p>
                  <p><strong>Image decode status:</strong> {String(asRecord(asRecord(layer2.details).ocr_result).image_decode_status ?? "unknown")}</p>
                  <p><strong>Text detected in image:</strong> {yesNo(Boolean(asRecord(asRecord(layer2.details).ocr_result).text_detected))}</p>
                  <p><strong>OCR status:</strong> {String(asRecord(asRecord(layer2.details).ocr_result).ocr_status ?? "not_attempted")}</p>
                  <p><strong>Context retrieval status:</strong> {String(asRecord(layer2.details).context_retrieval_status ?? "not_checked")}</p>
                  <p><strong>Corroboration status:</strong> {String(asRecord(layer2.details).corroboration_status ?? "not_checked")}</p>
                  <p><strong>Manipulation detected:</strong> {yesNo(Boolean(asRecord(layer2.details).manipulation_detected))}</p>
                  <p><strong>Misleading reuse detected:</strong> {yesNo(Boolean(asRecord(layer2.details).misleading_reuse))}</p>
                  {asArray(asRecord(layer2.details).failure_reasons).length ? (
                    <p><strong>Failure reasons:</strong> {asArray(asRecord(layer2.details).failure_reasons).map((item) => String(item)).join(", ")}</p>
                  ) : null}
                </>
              ) : audioMode ? (
                <>
                  <p><strong>Transcript available:</strong> {yesNo(String(asRecord(asRecord(layer2.details).transcription_result).transcript_status ?? "") === "completed")}</p>
                  <p><strong>Spoofing detected:</strong> {yesNo(Boolean(asRecord(layer2.details).spoof_detected))}</p>
                  <p><strong>Manipulation detected:</strong> {yesNo(Boolean(asRecord(layer2.details).manipulation_detected))}</p>
                  {asRecord(layer2.details).claimed_speaker ? <p><strong>Claimed speaker:</strong> {String(asRecord(layer2.details).claimed_speaker)}</p> : null}
                </>
              ) : videoMode ? (
                <>
                  <p><strong>Text detected in frames:</strong> {yesNo(Boolean(asRecord(asRecord(layer2.details).ocr_result).text_detected))}</p>
                  <p><strong>Audio present:</strong> {yesNo(Boolean(asRecord(asRecord(layer2.details).audio_result).audio_present))}</p>
                  <p><strong>Audio transcript extracted:</strong> {yesNo(Boolean(asRecord(layer2.details).transcript_text))}</p>
                  <p><strong>Technical/provenance warning:</strong> {yesNo(Boolean(asRecord(layer2.details).technical_warning))}</p>
                  <p><strong>Full forensic authenticity detection:</strong> Not performed</p>
                </>
              ) : null}
              {!imageMode && !audioMode && !videoMode && asRecord(layer2.details).social_text_input ? (
                <p><strong>Extracted social caption/context:</strong> {String(asRecord(layer2.details).social_text_input)}</p>
              ) : null}
              {!imageMode && !audioMode && !videoMode && asRecord(layer2.details).social_ocr_result ? (
                <p><strong>Social image OCR status:</strong> {String(asRecord(asRecord(layer2.details).social_ocr_result).ocr_status ?? "not_attempted")}</p>
              ) : null}
              {!imageMode && !audioMode && !videoMode && asArray(asRecord(asRecord(layer2.details).social_ocr_result).failure_reasons).length ? (
                <p><strong>Social image OCR failures:</strong> {asArray(asRecord(asRecord(layer2.details).social_ocr_result).failure_reasons).map((item) => String(item)).join(", ")}</p>
              ) : null}
              {factSignals.length ? <p><strong>Signals used:</strong> {factSignals.join(", ")}</p> : null}
            </div>
            {textMode && textClaimResults.length ? (
              <div className="mt-5 space-y-3">
                <p className="text-xs uppercase tracking-[0.3em] text-black/50">Extracted claims</p>
                {textClaimResults.map((claim, index) => (
                  <div key={`claim-${index}`} className="rounded-2xl border border-black/10 bg-white/70 p-3 text-sm text-black/75">
                    <p className="font-medium">{String(claim.claim_text ?? "Claim")}</p>
                    <p className="mt-2"><strong>Claim label:</strong> {String(claim.claim_label ?? "Unknown")}</p>
                    <p><strong>Claim score:</strong> {formatScore100(claim.claim_score)}</p>
                    {claim.evidence_summary ? <p className="mt-2">{String(claim.evidence_summary)}</p> : null}
                  </div>
                ))}
              </div>
            ) : null}
            {textMode && Object.keys(textDebug).length ? (
              <div className="mt-5 space-y-3">
                <p className="text-xs uppercase tracking-[0.3em] text-black/50">Text retrieval debug</p>
                <div className="rounded-2xl border border-black/10 bg-white/70 p-3 text-sm text-black/75 space-y-3">
                  {textDebug.simplified_meaning ? <p><strong>Simplified meaning:</strong> {String(textDebug.simplified_meaning)}</p> : null}
                  {textDebug.cleaned_input ? <p><strong>Cleaned input:</strong> {String(textDebug.cleaned_input)}</p> : null}
                  <p><strong>Exact phrase queries:</strong> {asArray(textDebug.exact_phrase_queries).map((item) => String(item)).join(" | ") || "None"}</p>
                  <p><strong>Lexical/entity queries:</strong> {asArray(textDebug.lexical_queries).map((item) => String(item)).join(" | ") || "None"}</p>
                  {textDebug.simplified_meaning_query ? <p><strong>Simplified meaning query:</strong> {String(textDebug.simplified_meaning_query)}</p> : null}
                  <p><strong>Search queries:</strong> {asArray(textDebug.search_queries).map((item) => String(item)).join(" | ") || "None"}</p>
                  {asArray(textDebug.top_urls).length ? (
                    <p><strong>Top URLs:</strong> {asArray(textDebug.top_urls).map((item) => String(item)).join(" | ")}</p>
                  ) : null}
                  {Object.keys(asRecord(textDebug.component_scores)).length ? (
                    <p>
                      <strong>Component scores:</strong>{" "}
                      {Object.entries(asRecord(textDebug.component_scores))
                        .filter(([key]) => ["S", "R", "A", "D", "C", "E", "P"].includes(key))
                        .map(([key, value]) => `${key}=${String(value)}`)
                        .join(" | ")}
                    </p>
                  ) : null}
                  {asArray(textDebug.support_judgments).length ? (
                    <div>
                      <p><strong>Support judgments:</strong></p>
                      <div className="mt-2 space-y-2">
                        {asArray(textDebug.support_judgments).slice(0, 5).map((item, index) => {
                          const support = asRecord(item);
                          return (
                            <div key={`support-${index}`} className="rounded-xl border border-black/10 bg-paper/40 p-3">
                              <p>{String(support.title ?? support.url ?? "Candidate")}</p>
                              <p className="mt-1 text-xs text-black/65">
                                {String(support.support_judgment ?? "unknown")} | via {String(support.query_type ?? "unknown")} | retrieval {String(support.retrieval_confidence ?? 0)} | verification {String(support.verification_confidence ?? 0)} | overlap {String(support.overlap_score ?? 0)}
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : null}
                  {asArray(textDebug.candidate_fetches).length ? (
                    <div>
                      <p><strong>Candidate fetches:</strong></p>
                      <div className="mt-2 space-y-2">
                        {asArray(textDebug.candidate_fetches).slice(0, 5).map((item, index) => {
                          const fetchItem = asRecord(item);
                          return (
                            <div key={`fetch-${index}`} className="rounded-xl border border-black/10 bg-paper/40 p-3">
                              <p>{String(fetchItem.title ?? fetchItem.url ?? "Candidate")}</p>
                              <p className="mt-1 text-xs text-black/65">
                                {String(fetchItem.fetch_status ?? "unknown")} | {String(fetchItem.content_type ?? "unknown")} | text length {String(fetchItem.extracted_text_length ?? 0)}
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            ) : null}
          </div>

          <div className="rounded-[1.5rem] border border-black/10 bg-paper/60 p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-black/50">{imageMode || audioMode || videoMode ? "Evidence sources" : textMode ? "Evidence sources" : "Matched sources"}</p>
            <SimilarityMatchesPanel
              jobId={result.job_id}
              initialMatches={factMatches as SimilarityMatch[]}
              initialHasMore={hasMoreMatches}
              initialTotal={totalMatchCount}
            />
          </div>
        </div>
      </LayerDisclosure>

      <LayerDisclosure
        title={feedbackMode ? "Feedback and Review" : "Community Consensus"}
        badge={feedbackMode ? "Review Layer" : "Census Layer"}
        description={
          feedbackMode
            ? "Open this layer to inspect structured user feedback, expert review notes, correction requests, and dispute reports linked to the verification history."
            : "Open this layer to inspect user voting signals and supporting community context."
        }
      >
        <div className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="rounded-[1.5rem] border border-black/10 bg-paper/60 p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-black/50">{feedbackMode ? "Feedback Summary" : "Consensus Summary"}</p>
            <div className="mt-4 space-y-2 text-sm text-black/80">
              <p><strong>Layer status:</strong> {String(layer3.status ?? "none")}</p>
              {!feedbackMode ? <p><strong>Layer confidence score:</strong> {formatScore100(systemLayerScores.community)}</p> : null}
              <p>{String(layer3.explanation ?? "No community summary available.")}</p>
              {feedbackMode ? (
                <>
                  <p><strong>Total feedback:</strong> {String(asRecord(layer3.details).feedback_count ?? 0)}</p>
                  <p><strong>Reviewed:</strong> {String(asRecord(layer3.details).reviewed_count ?? 0)}</p>
                  <p><strong>Resolved:</strong> {String(asRecord(layer3.details).resolved_count ?? 0)}</p>
                  <p><strong>Disputes:</strong> {String(asRecord(layer3.details).dispute_count ?? 0)}</p>
                  <p><strong>Corrections:</strong> {String(asRecord(layer3.details).correction_count ?? 0)}</p>
                  <p><strong>Source suggestions:</strong> {String(asRecord(layer3.details).source_suggestion_count ?? 0)}</p>
                </>
              ) : (
                <>
                  <p><strong>Total votes:</strong> {String(asRecord(layer3.details).vote_count ?? 0)}</p>
                  <p><strong>True votes:</strong> {String(asRecord(layer3.details).true_votes ?? 0)}</p>
                  <p><strong>False votes:</strong> {String(asRecord(layer3.details).false_votes ?? 0)}</p>
                  <p><strong>Not sure votes:</strong> {String(asRecord(layer3.details).not_sure_votes ?? 0)}</p>
                  {asRecord(layer3.details).majority_vote ? (
                    <p><strong>Majority vote:</strong> {String(asRecord(layer3.details).majority_vote)}</p>
                  ) : null}
                </>
              )}
            </div>
          </div>
          <div className="rounded-[1.5rem] border border-black/10 bg-paper/60 p-5">
            <p className="text-xs uppercase tracking-[0.3em] text-black/50">How to read this</p>
            <div className="mt-4 space-y-2 text-sm text-black/75">
              {feedbackMode ? (
                <>
                  <p>Feedback entries document review activity and possible revisions, but they do not act as truth-by-voting signals.</p>
                  <p>Use this section to understand disputes, corrections, and source suggestions linked to the signed verification history.</p>
                  <p>It complements the NLP result and cryptographic audit trail rather than replacing either one.</p>
                </>
              ) : (
                <>
                  <p>Community votes can strengthen or weaken the overall assessment when enough users align.</p>
                  <p>They do not prove who issued a record, whether it was signed, or whether a media file preserved original provenance.</p>
                  <p>Use this alongside the provenance and fact-checking sections rather than as a replacement for them.</p>
                </>
              )}
            </div>
          </div>
        </div>
      </LayerDisclosure>
    </section>
  );
}
