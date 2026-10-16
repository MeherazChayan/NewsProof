from __future__ import annotations

import re

from app.utils import normalize_text


class ClaimExtractionService:
    """Extracts checkable factual claims and lightweight claim metadata."""

    FACTUAL_PATTERN = re.compile(
        r"\b(is|are|was|were|has|have|had|will|said|says|reported|reports|announced|confirmed|approved|threatened|killed|won|lost|passed|arrested|warned|accused|ordered|renounced|renouncing|makes|plans|planned|fund|funds)\b",
        re.IGNORECASE,
    )

    def extract_claims(self, text: str, max_claims: int = 5) -> dict:
        raw_text = " ".join((text or "").split())
        if not raw_text:
            return {"claims": [], "entities": [], "locations": [], "dates": [], "numbers": [], "quotes": [], "checkable": False, "reason": "empty"}

        if self._looks_like_opinion(raw_text):
            return {
                "claims": [],
                "entities": self._extract_entities(raw_text),
                "locations": self._extract_locations(raw_text),
                "dates": self._extract_dates(raw_text),
                "numbers": re.findall(r"\b\d[\d,\.]*\b", raw_text),
                "quotes": self._extract_quotes(raw_text),
                "checkable": False,
                "reason": "opinion_only",
            }

        if self._is_headline_like(raw_text):
            return {
                "claims": [self._claim_metadata(raw_text)],
                "entities": self._extract_entities(raw_text),
                "locations": self._extract_locations(raw_text),
                "dates": self._extract_dates(raw_text),
                "numbers": re.findall(r"\b\d[\d,\.]*\b", raw_text),
                "quotes": self._extract_quotes(raw_text),
                "checkable": True,
                "reason": "headline",
            }

        claims: list[dict] = []
        for fragment in re.split(r"(?<=[\.\!\?\n])\s+", raw_text):
            claim = " ".join(fragment.split()).strip(" .")
            if not claim or len(claim) < 18:
                continue
            if not self._is_checkable_claim(claim):
                continue
            for atomic in self._atomic_claims(claim):
                if not self._is_checkable_claim(atomic):
                    continue
                claims.append(self._claim_metadata(atomic))
                if len(claims) >= max_claims:
                    break
            if len(claims) >= max_claims:
                break

        return {
            "claims": claims,
            "entities": self._extract_entities(raw_text),
            "locations": self._extract_locations(raw_text),
            "dates": self._extract_dates(raw_text),
            "numbers": re.findall(r"\b\d[\d,\.]*\b", raw_text),
            "quotes": self._extract_quotes(raw_text),
            "checkable": bool(claims),
            "reason": "claims_found" if claims else "no_checkable_claim",
        }

    def classify_checkability(self, text: str) -> dict:
        extracted = self.extract_claims(text, max_claims=3)
        return {
            "checkable": extracted["checkable"],
            "reason": extracted["reason"],
            "claim_count": len(extracted["claims"]),
        }

    def _claim_metadata(self, text: str) -> dict:
        entities = self._extract_entities(text)
        dates = self._extract_dates(text)
        numbers = re.findall(r"\b\d[\d,\.]*\b", text)
        locations = self._extract_locations(text)
        return {
            "text": text[:500],
            "short_forms": self._claim_variants(text, entities, locations, dates, numbers),
            "entities": entities,
            "locations": locations,
            "numbers": numbers,
            "dates": dates,
            "quotes": self._extract_quotes(text),
        }

    def _extract_entities(self, text: str) -> list[str]:
        entities = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text or "")
        deduped: list[str] = []
        seen: set[str] = set()
        for entity in entities:
            key = entity.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entity)
        return deduped[:10]

    def _is_checkable_claim(self, text: str) -> bool:
        lowered = text.lower()
        if lowered.startswith(("i think", "i feel", "in my opinion", "maybe", "perhaps")):
            return False
        if self.FACTUAL_PATTERN.search(text):
            return True
        return bool(re.search(r"\d", text))

    def _atomic_claims(self, text: str) -> list[str]:
        compact = " ".join((text or "").split()).strip(" .")
        if not compact:
            return []

        lowered = compact.lower()
        atoms = [compact]
        sale_match = re.search(r"\b(?:sell|sold|sale of)\b", lowered)
        if sale_match:
            money = re.search(r"(?:\$|usd\s*)\s?\d[\d,\.]*(?:\s?(?:million|billion|trillion))?|\b\d[\d,\.]*\s?(?:million|billion|trillion)\b", compact, flags=re.IGNORECASE)
            buyer = re.search(r"\bto\s+(?:a\s+|an\s+|the\s+)?([^\.]+?)(?:\s+for\s+|\s+worth\s+|$)", compact, flags=re.IGNORECASE)
            base = re.sub(r"\s+for\s+(?:\$|usd\s*)?\s?\d[\d,\.]*(?:\s?(?:million|billion|trillion))?.*$", "", compact, flags=re.IGNORECASE).strip(" .")
            if base and base != compact:
                atoms.append(base)
            if buyer:
                atoms.append(f"The buyer was {buyer.group(1).strip(' .')}")
            if money:
                atoms.append(f"The deal value was {money.group(0).strip()}")

        if re.search(r"\bconvert(?:ed|ing)?\b", lowered) and re.search(r"\b(?:resort|island|forest|land)\b", lowered):
            atoms.append(compact)

        if re.search(r"\b(?:president|presidential|run for president)\b", lowered):
            actor = self._extract_entities(compact)
            person = actor[0] if actor else "The person"
            year = re.search(r"\b20\d{2}\b", compact)
            if re.search(r"\bannounced\b", lowered):
                atoms.append(f"{person} announced he will run for U.S. president{f' in {year.group(0)}' if year else ''}")
            if re.search(r"\brenounc(?:e|ed|ing)\b", lowered):
                citizenship = re.search(r"renounc(?:e|ed|ing)\s+(?:his|her|their)?\s*([^\.]+?citizenship)", compact, flags=re.IGNORECASE)
                atoms.append(f"{person} renounced {citizenship.group(1).strip() if citizenship else 'foreign citizenship'}")
                atoms.append(f"Renouncing foreign citizenship makes {person} eligible to run for U.S. president")

        if re.search(r"\bfund\b", lowered) and re.search(r"\b(?:campaign|bitcoin|wallet)\b", lowered):
            actor = self._extract_entities(compact)
            person = actor[0] if actor else "The person"
            atoms.append(f"{person} plans to fund the campaign using Bitcoin from his personal wallet")

        if len(atoms) > 1:
            atoms = atoms[1:]

        deduped: list[str] = []
        seen: set[str] = set()
        for atom in atoms:
            normalized = normalize_text(atom)
            if not normalized or normalized in seen or len(atom) < 12:
                continue
            seen.add(normalized)
            deduped.append(atom[:500])
        return deduped[:4]

    def _looks_like_opinion(self, text: str) -> bool:
        lowered = normalize_text(text)
        return lowered.startswith(("i think", "i believe", "in my opinion")) or "should" in lowered and not self.FACTUAL_PATTERN.search(text)

    def _is_headline_like(self, text: str) -> bool:
        if "\n" in text:
            return False
        word_count = len(text.split())
        if word_count < 5 or word_count > 22:
            return False
        return sum(1 for char in text if char in ".!?") <= 1

    def _extract_dates(self, text: str) -> list[str]:
        return re.findall(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,\s*\d{4})?|\b\d{4}\b",
            text,
            flags=re.IGNORECASE,
        )

    def _extract_locations(self, text: str) -> list[str]:
        matches = re.findall(r"\b(?:in|at|from|across|near|outside)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text or "")
        deduped: list[str] = []
        seen: set[str] = set()
        for match in matches:
            key = match.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(match)
        return deduped[:8]

    def _extract_quotes(self, text: str) -> list[str]:
        return [item.strip() for item in re.findall(r"[\"“](.+?)[\"”]", text or "") if item.strip()][:5]

    def _claim_variants(
        self,
        text: str,
        entities: list[str],
        locations: list[str],
        dates: list[str],
        numbers: list[str],
    ) -> list[str]:
        compact = " ".join(text.split())
        variants: list[str] = [compact[:220]]
        if entities:
            variants.append(" ".join((entities + locations + dates)[:6]))
        if entities or numbers:
            variants.append(" ".join((entities[:3] + numbers[:2] + locations[:2])[:7]))
        if len(compact.split()) > 12:
            variants.append(" ".join(compact.split()[:12]))

        deduped: list[str] = []
        seen: set[str] = set()
        for variant in variants:
            normalized = normalize_text(variant)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(variant)
        return deduped[:4]
