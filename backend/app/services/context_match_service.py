from __future__ import annotations

import re

from app.utils import normalize_text


class ContextMatchService:
    """Rule-based claim/evidence context matcher used as verifier fallback.

    This deliberately checks factual context, not semantic similarity. A source
    that discusses the same topic in a different city or country must not count
    as direct support for a claim.
    """

    def compare(self, claim: str, evidence: dict | str) -> dict:
        evidence_text = self._evidence_text(evidence)
        claim_context = self.extract_context_terms(claim)
        evidence_context = self.extract_context_terms(evidence_text)

        claim_locations = set(claim_context["locations"])
        evidence_locations = set(evidence_context["locations"])
        claim_countries = set(claim_context["countries"])
        evidence_countries = set(evidence_context["countries"])
        claim_events = set(claim_context["events"])
        evidence_events = set(evidence_context["events"])
        claim_actors = set(claim_context["actors"])
        evidence_actors = set(evidence_context["actors"])
        claim_times = set(claim_context["times"])
        evidence_times = set(evidence_context["times"])

        mismatch_reason = ""
        location_mismatch = False
        if (claim_locations or claim_countries) and (evidence_locations or evidence_countries):
            claim_location_countries = {self.country_for_location(item) for item in claim_locations}
            claim_location_countries.discard("")
            evidence_location_countries = {self.country_for_location(item) for item in evidence_locations}
            evidence_location_countries.discard("")
            if claim_locations:
                related_geo = bool(claim_locations & evidence_locations) or bool(claim_location_countries & (evidence_countries | evidence_location_countries))
            else:
                related_geo = bool((claim_locations | claim_countries) & (evidence_locations | evidence_countries | evidence_location_countries))
            location_mismatch = not related_geo
            if location_mismatch:
                mismatch_reason = (
                    f"Claim location/country {sorted(claim_locations | claim_countries)} does not match "
                    f"evidence location/country {sorted(evidence_locations | evidence_countries)}."
                )

        event_mismatch = False
        if claim_events and evidence_events and not (claim_events & evidence_events):
            event_mismatch = True
            mismatch_reason = mismatch_reason or f"Claim event {sorted(claim_events)} does not match evidence event {sorted(evidence_events)}."

        actor_mismatch = False
        if claim_actors and evidence_actors:
            actor_mismatch = not (claim_actors & evidence_actors) and len(claim_actors) >= 2
            if actor_mismatch:
                mismatch_reason = mismatch_reason or f"Claim actors {sorted(claim_actors)} do not match evidence actors {sorted(evidence_actors)}."

        time_mismatch = False
        if claim_times and evidence_times and not (claim_times & evidence_times):
            time_mismatch = True
            mismatch_reason = mismatch_reason or f"Claim time {sorted(claim_times)} does not match evidence time {sorted(evidence_times)}."

        location_score = 0.0 if location_mismatch else (1.0 if (claim_locations or claim_countries) else 0.75)
        event_score = 0.0 if event_mismatch else (1.0 if claim_events and evidence_events and bool(claim_events & evidence_events) else 0.65 if claim_events else 0.75)
        entity_score = 0.0 if actor_mismatch else (len(claim_actors & evidence_actors) / max(1, len(claim_actors)) if claim_actors else 0.75)
        time_score = 0.0 if time_mismatch else (1.0 if claim_times and evidence_times and bool(claim_times & evidence_times) else 0.75)
        context_pass = not (location_mismatch or event_mismatch or actor_mismatch or time_mismatch)

        return {
            "claim_locations": sorted(claim_locations),
            "evidence_locations": sorted(evidence_locations),
            "claim_countries": sorted(claim_countries),
            "evidence_countries": sorted(evidence_countries),
            "claim_events": sorted(claim_events),
            "evidence_events": sorted(evidence_events),
            "claim_actors": sorted(claim_actors),
            "evidence_actors": sorted(evidence_actors),
            "claim_times": sorted(claim_times),
            "evidence_times": sorted(evidence_times),
            "entity_match_score": round(max(0.0, min(1.0, entity_score)), 3),
            "location_match_score": round(max(0.0, min(1.0, location_score)), 3),
            "time_match_score": round(max(0.0, min(1.0, time_score)), 3),
            "event_match_score": round(max(0.0, min(1.0, event_score)), 3),
            "location_match_pass": not location_mismatch,
            "event_match_pass": not event_mismatch,
            "entity_match_pass": not actor_mismatch,
            "time_match_pass": not time_mismatch,
            "context_match_pass": context_pass,
            "mismatch_reason": mismatch_reason,
        }

    def extract_context_terms(self, text: str) -> dict[str, list[str]]:
        normalized = normalize_text(text or "")
        location_patterns = {
            "dhaka": r"\bdhaka\b|ঢাকা|ঢাকায়|ঢাকায়",
            "beijing": r"\bbeijing\b|বেইজিং",
            "shanghai": r"\bshanghai\b|সাংহাই",
            "sundarbans": r"\bsundarbans\b|সুন্দরবন",
            "pretoria": r"\bpretoria\b",
            "washington": r"\bwashington\b|ওয়াশিংটন",
        }
        location_patterns["texas"] = r"\btexas\b"
        country_patterns = {
            "bangladesh": r"\bbangladesh\b|বাংলাদেশ",
            "china": r"\bchina\b|\bchinese\b|চীন|চীনা",
            "united states": r"\bunited states\b|\bu\.?s\.?a?\b|\bamerica\b|যুক্তরাষ্ট্র|মার্কিন",
            "south africa": r"\bsouth africa\b|\bsouth african\b",
            "india": r"\bindia\b|ভারত",
        }
        event_patterns = {
            "protest": r"\b(?:protest|movement|demonstration|rally)\b|আন্দোলন|বিক্ষোভ|মিছিল",
            "clash": r"\bclash(?:es|ed)?\b|সংঘর্ষ",
            "fire": r"\b(?:fire|blaze)\b|আগুন|অগ্নিকাণ্ড",
            "sale": r"\b(?:sell|sold|sale|transfer|ownership)\b|বিক্রি|হস্তান্তর",
            "deal": r"\b(?:deal|agreement|signed)\b|চুক্তি",
            "conservation": r"\b(?:conservation|protect|preserve|environment)\b|সংরক্ষণ|পরিবেশ",
            "awareness": r"\b(?:awareness|training|program|workshop)\b|সচেতনতা|প্রশিক্ষণ",
            "presidential_run": r"\b(?:president|presidential|run for president|campaign)\b",
            "citizenship": r"\b(?:citizenship|renounc|natural[ -]born|eligible|eligibility)\b",
            "bitcoin_funding": r"\b(?:bitcoin|crypto|wallet|fund|funding)\b",
        }
        event_patterns["election_result"] = r"\b(?:defeat(?:ed|s|ing)?|beat|beats|won|wins|winner|run[- ]?off|primary|nomination|nominee|senate)\b"
        actor_patterns = {
            "students": r"\bstudent(?:s)?\b|শিক্ষার্থী|ছাত্র|ছাত্রী",
            "police": r"\bpolice\b|পুলিশ",
            "government": r"\bgovernment\b|সরকার",
            "bangladesh government": r"\bbangladesh government\b|বাংলাদেশ সরকার",
            "chinese company": r"\bchinese (?:company|conglomerate|firm|developer)\b|চীনা কোম্পানি",
            "elon musk": r"\belon musk\b|\bmusk\b",
            "ken paxton": r"\bken paxton\b|\bpaxton\b",
            "john cornyn": r"\bjohn cornyn\b|\bcornyn\b",
            "officials": r"\bofficials?\b|কর্মকর্তা",
        }
        times = re.findall(r"\b(?:20\d{2}|19\d{2})\b|আজ|গতকাল|আগামীকাল", normalized)
        return {
            "locations": [key for key, pattern in location_patterns.items() if re.search(pattern, normalized)],
            "countries": [key for key, pattern in country_patterns.items() if re.search(pattern, normalized)],
            "events": [key for key, pattern in event_patterns.items() if re.search(pattern, normalized)],
            "actors": [key for key, pattern in actor_patterns.items() if re.search(pattern, normalized)],
            "times": self._dedupe(times),
        }

    def country_for_location(self, location: str) -> str:
        location_countries = {
            "dhaka": "bangladesh",
            "sundarbans": "bangladesh",
            "beijing": "china",
            "shanghai": "china",
            "pretoria": "south africa",
            "washington": "united states",
            "texas": "united states",
        }
        return location_countries.get(normalize_text(location), "")

    def relation_from_context(self, model_label: str, context_match: dict, relevance: float) -> str:
        if model_label == "contradicts":
            return "contradicts"
        if not bool(context_match.get("location_match_pass")):
            return "unrelated"
        if not bool(context_match.get("event_match_pass")):
            return "background_only" if relevance >= 0.55 else "unrelated"
        if model_label == "supports" and bool(context_match.get("context_match_pass")) and relevance >= 0.8:
            return "supports"
        if model_label in {"partially_related", "partially_supports"} or relevance >= 0.6:
            return "partially_related"
        if relevance >= 0.45:
            return "background_only"
        return "insufficient"

    def relation_explanation(self, relation: str, context_match: dict) -> str:
        mismatch = str(context_match.get("mismatch_reason") or "").strip()
        if relation == "supports":
            return "Accepted because the source directly matches the claim's key entity, location, and event/action context."
        if relation == "contradicts":
            return "Accepted because the source directly contradicts a key part of the claim."
        if relation == "partially_related":
            return "Limited context only: the source is related, but it does not fully verify the exact claim."
        if relation == "background_only":
            return mismatch or "Rejected as direct support because it gives background or broad topic context without verifying the exact claim."
        if relation == "unrelated":
            return mismatch or "Rejected because the evidence context does not match the claim."
        return "Rejected because the source did not provide enough claim-specific evidence."

    def _evidence_text(self, evidence: dict | str) -> str:
        if isinstance(evidence, dict):
            return " ".join(str(evidence.get(key) or "") for key in ("title", "snippet", "passage", "text", "evidence_body"))
        return str(evidence or "")

    def _dedupe(self, items: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in items:
            cleaned = " ".join(str(item or "").split()).strip()
            key = normalize_text(cleaned)
            if key and key not in seen:
                seen.add(key)
                deduped.append(cleaned)
        return deduped
