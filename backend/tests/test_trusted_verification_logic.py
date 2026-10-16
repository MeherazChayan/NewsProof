from app.services.context_detector import extract_claim_context
from app.services.evidence_alignment import classify_evidence, compare_claim_with_evidence
from app.services.evidence_search import build_country_aware_queries
from app.services.fact_check_parser import extract_fact_check_verdict
from app.services.trusted_sources import get_source_type
from app.services.verification import VerificationService
from app.services.verdict_engine import decide_final_verdict, score_evidence
from types import SimpleNamespace


def _decision(items):
    return decide_final_verdict(score_evidence(items))["verdict"]


def test_fact_check_page_says_false_returns_unverified():
    verdict = extract_fact_check_verdict("Rating: False. This claim is fabricated.")
    assert verdict == "refutes"
    assert _decision([{"source_type": "fact_check_page", "alignment": "refutes", "verdict_extracted": verdict}]) == "Unverified"


def test_fact_check_page_says_true_returns_verified():
    verdict = extract_fact_check_verdict("Verdict: True. The evidence confirms the claim.")
    assert verdict == "supports"
    assert _decision([{"source_type": "fact_check_page", "alignment": "supports", "verdict_extracted": verdict}]) == "Verified"


def test_trusted_news_keyword_only_mention_is_not_verifiable():
    claim = "Bangladesh sold the Sundarbans to China for $50 billion."
    evidence = "China is investing in Bangladesh infrastructure and energy projects."
    alignment = compare_claim_with_evidence(claim, evidence)
    assert alignment["alignment"] in {"mentions_only", "irrelevant"}
    assert _decision([{"source_type": "trusted_news_page", "alignment": "mentions_only"}]) == "Not Verifiable"


def test_two_trusted_news_pages_confirming_exact_claim_can_verify():
    items = [
        {"source_type": "trusted_news_page", "alignment": "supports", "domain": "reuters.com"},
        {"source_type": "trusted_news_page", "alignment": "supports", "domain": "apnews.com"},
    ]
    assert _decision(items) == "Verified"


def test_untrusted_source_support_does_not_verify():
    assert _decision([{"source_type": "unknown_or_untrusted_page", "alignment": "supports"}]) == "Not Verifiable"


def test_bangladesh_claim_selects_bangladesh_sources_and_ignores_unrelated_china_context():
    claim = "The government of Bangladesh secretly signed a deal to sell the Sundarbans to a Chinese real estate conglomerate for $50 billion."
    context = extract_claim_context(claim)
    queries = build_country_aware_queries(context)
    assert context["country"] == "Bangladesh"
    assert any("site:thedailystar.net" in query for query in queries)

    alignment = compare_claim_with_evidence(claim, "A Chinese real estate company announced a project in Shanghai.")
    assert alignment["alignment"] == "irrelevant"


def test_audio_transcription_incomplete_claim_stays_not_verifiable():
    assert _decision([]) == "Not Verifiable"
    partial = compare_claim_with_evidence("Elon Musk announced", "Elon Musk appears in recent technology news.")
    assert partial["alignment"] != "supports"


def test_image_ocr_partial_text_stays_not_verifiable_without_exact_support():
    partial_ocr = "Sundarbans China 50 billion"
    context = extract_claim_context(partial_ocr)
    assert context["main_claim"] == partial_ocr
    assert _decision([{"source_type": "trusted_news_page", "alignment": "mentions_only"}]) == "Not Verifiable"


def test_source_type_registry_classifies_fact_check_and_social_pages():
    assert get_source_type("https://www.snopes.com/fact-check/example") == "fact_check_page"
    assert get_source_type("https://www.facebook.com/some-post") == "social_media_page"


def test_fact_check_exact_claim_false_splits_aboutness_from_stance():
    claim = "Stephen Colbert and his wife, Evelyn McGee-Colbert, opened a free hospital for the homeless in early 2026."
    evidence = (
        "Claim: Stephen Colbert and his wife opened a free hospital for the homeless in early 2026. "
        "Rating: False. The rumor is not true."
    )
    result = classify_evidence(claim, evidence, is_fact_check=True)
    assert result["aboutness"] == "exact_claim"
    assert result["stance"] == "refutes"
    assert result["relation"] == "refutes"
    assert result["verdict_extracted"] == "false"
    assert result["support_score"] == 0.0
    assert result["refute_score"] == 1.0
    assert _decision([{"source_type": "fact_check_page", **result}]) == "Unverified"


def test_snopes_false_verdict_overrides_near_duplicate_overlap_for_ruffalo_claim():
    claim = "Mark Ruffalo invested $369K in 'Where Truth Leads, Justice Follows'."
    evidence = (
        "Claim: Mark Ruffalo invested $369K in Where Truth Leads, Justice Follows. "
        "Rating: False. There is no evidence this investment happened."
    )
    result = classify_evidence(claim, evidence, is_fact_check=True)
    assert result["aboutness"] == "exact_claim"
    assert result["stance"] == "refutes"
    assert result["relation"] == "refutes"
    assert _decision([{"source_type": "fact_check_page", **result}]) == "Unverified"


def test_shared_entity_without_same_action_is_mentions_only():
    claim = "Mark Ruffalo invested $369K in a project."
    evidence = "Mark Ruffalo spoke about responsibility to justice during a public event."
    alignment = compare_claim_with_evidence(claim, evidence)
    assert alignment["alignment"] == "mentions_only"
    assert alignment["action_match"] is False


def test_partial_matches_without_strong_support_stay_not_verifiable():
    service = VerificationService.__new__(VerificationService)
    result = service._finalize_text_verification(
        [
            {
                "claim_text": "Mark Ruffalo invested $369K in a project.",
                "support_score": 0.0,
                "source_reliability": 0.9,
                "agreement_score": 0.7,
                "directness_score": 0.7,
                "clarity_score": 1.0,
                "claim_score": 0.6,
                "claim_label": "Likely Verified",
                "claim_verdict": "partially_supported",
                "evidence_sources": [
                    {
                        "title": "Mark Ruffalo speaks about justice",
                        "url": "https://www.reuters.com/world/example",
                        "domain": "reuters.com",
                        "source_type": "trusted_news_page",
                        "evidence_relation": "partially_related",
                        "alignment": "mentions_only",
                    }
                ],
            }
        ],
        [{"title": "Mark Ruffalo speaks about justice", "url": "https://www.reuters.com/world/example"}],
        {"simplified_meaning": "Mark Ruffalo invested $369K in a project.", "exact_phrase_queries": []},
    )
    assert result["final_label"] == "Not Verifiable"
    assert result["final_score"] < 0.35


def test_text_system_scoring_uses_adjusted_text_formula_not_stale_raw_score():
    service = VerificationService.__new__(VerificationService)
    service._text_crypto_score_inputs = lambda layer1, layer2, extracted: {}
    service._generate_text_explanation = lambda claim_results, final_score, final_label: "text explanation"
    layer = SimpleNamespace(status="none", confidence=0.0, explanation="", details={})
    layer2 = SimpleNamespace(
        status="supporting",
        confidence=0.6,
        explanation="",
        details={
            "text_component_scores": {"S": 0.6, "R": 0.9, "A": 0.7, "D": 0.7, "C": 1.0, "E": 0.4, "P": 0.4},
            "evidence_multiplier": 0.4,
            "penalty_multiplier": 0.4,
            "text_verification_label": "Unverified",
            "claim_results": [
                {
                    "claim_verdict": "supported",
                    "support_score": 0.6,
                    "source_reliability": 0.9,
                    "agreement_score": 0.7,
                    "directness_score": 0.7,
                    "clarity_score": 1.0,
                    "evidence_sources": [{"evidence_relation": "supports"}],
                }
            ],
        },
    )
    extracted = SimpleNamespace(main_text="Claim text")

    result = service._build_text_system_scoring(layer, layer2, layer, extracted)

    assert round(result["raw_score"], 1) == 71.0
    assert round(result["adjusted_final_score"], 1) == 11.4
    assert round(result["final_score"], 1) == 11.4
    assert result["final_label"] == "Unverified"
    assert result["scoring_method"] == "text_verification_formula"
    assert set(result["display_layer_scores"]) == {"nlp", "community"}
    assert result["source_provenance_check"]["scored_as_truth"] is False
    assert result["source_provenance_check"]["status"] == "no_provenance_available"


def test_not_verifiable_label_does_not_floor_displayed_score_to_35():
    service = VerificationService.__new__(VerificationService)
    service._text_crypto_score_inputs = lambda layer1, layer2, extracted: {}
    service._generate_text_explanation = lambda claim_results, final_score, final_label: "text explanation"
    layer = SimpleNamespace(status="none", confidence=0.0, explanation="", details={})
    layer2 = SimpleNamespace(
        status="none",
        confidence=0.0,
        explanation="",
        details={
            "text_component_scores": {"S": 0.0, "R": 0.2, "A": 0.0, "D": 0.25, "C": 1.0, "E": 0.4, "P": 0.6},
            "text_verification_label": "Not Verifiable",
            "claim_results": [
                {
                    "claim_verdict": "partially_supported",
                    "evidence_sources": [{"evidence_relation": "partially_related"}],
                }
            ],
        },
    )
    result = service._build_text_system_scoring(layer, layer2, layer, SimpleNamespace(main_text="Claim text"))

    assert result["final_label"] == "Unverified"
    assert result["adjusted_final_score"] < 35.0
    assert result["displayed_score"] == result["adjusted_final_score"]
    assert "without a floor" in result["override_reason"]


def test_talarico_same_race_article_is_neutral_context_not_support():
    claim = "Republicans claim James Talarico, Texas candidate for US Senate, is vegan. He's not."
    evidence = "Republicans target Talarico's social media clips in Texas race."
    result = classify_evidence(claim, evidence, is_fact_check=False)

    assert result["aboutness"] == "same_entity_related_event"
    assert result["stance"] == "neutral"
    assert result["relation"] == "mentions_only"
    assert result["support_score"] == 0.0
    assert "does not directly verify the exact claim" in result["reason"]
