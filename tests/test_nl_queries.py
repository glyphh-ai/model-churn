"""Test that NL queries match the correct exemplar risk levels.

These tests encode NL text queries (via the semantic layer) and compare
against training exemplars to verify broad domain queries like "churn"
and "healthy" match the expected risk categories.

Customer-to-exemplar matching uses the metrics layer.
NL query-to-exemplar matching uses the semantic layer.
This file tests the semantic path.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG, encode_query, entry_to_record

glyphh = pytest.importorskip("glyphh")

from glyphh import Encoder, Concept
from glyphh.core.ops import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def encoder():
    return Encoder(ENCODER_CONFIG)


@pytest.fixture(scope="module")
def pattern_glyphs(encoder):
    """Encode all training exemplars into glyphs with their metadata."""
    exemplars_path = DATA_DIR / "exemplars.jsonl"
    glyphs = []
    with open(exemplars_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            record = entry_to_record(entry)
            concept = Concept(
                name=record["concept_text"],
                attributes=record["attributes"],
            )
            glyph = encoder.encode(concept)
            glyphs.append((glyph, record["metadata"]))
    return glyphs


def _semantic_similarity(glyph1, glyph2):
    """Semantic layer cortex similarity only."""
    v1 = glyph1.layers["semantic"].cortex.data
    v2 = glyph2.layers["semantic"].cortex.data
    return float(cosine_similarity(v1, v2))


def _encode_nl_query(query_text, encoder):
    """Encode an NL query using the model's encode_query function."""
    concept_dict = encode_query(query_text)
    concept = Concept(
        name=concept_dict["name"],
        attributes=concept_dict["attributes"],
    )
    return encoder.encode(concept)


def _find_best_match(query_glyph, pattern_glyphs, sim_fn=_semantic_similarity):
    """Find the training exemplar most similar to a query glyph."""
    best_score = -1
    best_meta = None
    for pattern_glyph, meta in pattern_glyphs:
        score = sim_fn(query_glyph, pattern_glyph)
        if score > best_score:
            best_score = score
            best_meta = meta
    return best_meta, best_score


def _best_score_for_level(query_glyph, pattern_glyphs, level):
    """Best semantic similarity score against exemplars of a given risk level."""
    return max(
        _semantic_similarity(query_glyph, pg)
        for pg, m in pattern_glyphs
        if m["risk_level"] == level
    )


def _best_score_for_driver(query_glyph, pattern_glyphs, driver):
    """Best semantic similarity score against exemplars of a given churn driver."""
    scores = [
        _semantic_similarity(query_glyph, pg)
        for pg, m in pattern_glyphs
        if m.get("churn_driver") == driver
    ]
    return max(scores) if scores else -1


# ---------------------------------------------------------------------------
# High-risk: Churn / exit queries (multiple phrasings)
# ---------------------------------------------------------------------------

class TestChurnExitQueries:
    """Churn, attrition, and exit-related queries → high risk."""

    def test_what_customers_likely_to_churn(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("what customers are likely to churn", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_which_accounts_at_risk_of_leaving(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("which accounts are at risk of leaving", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_show_me_customers_who_might_cancel(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("show me customers who might cancel", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_whos_about_to_churn(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("who's about to churn", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_customers_at_risk_of_attrition(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("customers at risk of attrition", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"


# ---------------------------------------------------------------------------
# High-risk: Inactive / ghost / dormant queries
# ---------------------------------------------------------------------------

class TestInactiveQueries:
    """Inactive, dormant, ghost customer queries → high risk."""

    def test_dormant_accounts(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("dormant accounts", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_ghost_customers(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("ghost customers", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_customers_who_stopped_using_the_product(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("customers who stopped using the product", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_inactive_accounts_this_month(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("inactive accounts this month", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_show_me_inactive_customers(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("show me inactive customers", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"


# ---------------------------------------------------------------------------
# High-risk: Support burden queries
# ---------------------------------------------------------------------------

class TestSupportBurdenQueries:
    """Support ticket and frustration queries → high risk."""

    def test_customers_filing_too_many_support_tickets(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("customers filing too many support tickets", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_accounts_with_escalated_cases(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("accounts with escalated cases", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_who_is_struggling_with_our_product(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("who is struggling with our product", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_frustrated_customers_with_lots_of_issues(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("frustrated customers with lots of issues", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"


# ---------------------------------------------------------------------------
# High-risk: Defect / quality queries
# ---------------------------------------------------------------------------

class TestDefectQueries:
    """Bug, defect, and quality queries → high risk."""

    def test_customers_affected_by_bugs(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("customers affected by bugs", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_accounts_hitting_recurring_issues(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("accounts hitting recurring issues", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_who_is_dealing_with_quality_problems(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("who is dealing with quality problems", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"


# ---------------------------------------------------------------------------
# High-risk: Billing / cancel queries
# ---------------------------------------------------------------------------

class TestBillingQueries:
    """Billing, pricing, and cancellation queries → high risk."""

    def test_customers_who_disputed_their_invoice(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("customers who disputed their invoice", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_accounts_concerned_about_pricing(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("accounts concerned about pricing", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] != "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_who_asked_about_cancellation_or_refund(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("who asked about cancellation or refund", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_customers_about_to_cancel_subscription(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("customers about to cancel their subscription", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"


# ---------------------------------------------------------------------------
# High-risk: Low adoption queries
# ---------------------------------------------------------------------------

class TestAdoptionQueries:
    """Low adoption and underutilization queries → high risk."""

    def test_customers_not_using_key_features(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("customers not using key features", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] != "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_underutilized_accounts(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("underutilized accounts", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] != "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_who_hasnt_completed_onboarding(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("who hasn't completed onboarding", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] != "low", f"matched {meta['risk_level']} ({score:.4f})"


# ---------------------------------------------------------------------------
# Low-risk: Healthy / retained queries
# ---------------------------------------------------------------------------

class TestHealthyQueries:
    """Healthy, retained, and satisfied customer queries → low risk."""

    def test_our_best_customers(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("our best customers", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_power_users(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("power users", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_happy_customers(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("happy customers", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_champion_accounts(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("champion accounts", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_most_engaged_users(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("most engaged users", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_loyal_customers(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("loyal customers", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_healthy_growing_accounts(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("show me healthy growing accounts", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_retained_customers(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("retained customers with strong adoption", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"


# ---------------------------------------------------------------------------
# Low-risk: Growth / expansion queries
# ---------------------------------------------------------------------------

class TestGrowthQueries:
    """Growth, expansion, and renewal queries → low risk."""

    def test_accounts_ready_for_upsell(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("accounts ready for upsell", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_growing_customers(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("growing customers", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_who_renewed_early(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("who renewed early", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"

    def test_expanding_accounts(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("expanding accounts", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"


# ---------------------------------------------------------------------------
# Score ordering — directional correctness per driver
# ---------------------------------------------------------------------------

class TestScoreOrdering:
    """Verify high-risk queries score higher against high-risk exemplars
    and low-risk queries score higher against low-risk exemplars."""

    def test_churn_high_over_low(self, encoder, pattern_glyphs):
        """'churn' query: best high-risk score > best low-risk score."""
        glyph = _encode_nl_query("which customers are at risk of churning", encoder)
        best_high = _best_score_for_level(glyph, pattern_glyphs, "high")
        best_low = _best_score_for_level(glyph, pattern_glyphs, "low")
        assert best_high > best_low, f"high={best_high:.4f} low={best_low:.4f}"

    def test_healthy_low_over_high(self, encoder, pattern_glyphs):
        """'healthy' query: best low-risk score > best high-risk score."""
        glyph = _encode_nl_query("show me healthy satisfied customers", encoder)
        best_high = _best_score_for_level(glyph, pattern_glyphs, "high")
        best_low = _best_score_for_level(glyph, pattern_glyphs, "low")
        assert best_low > best_high, f"low={best_low:.4f} high={best_high:.4f}"

    def test_support_tickets_high_over_low(self, encoder, pattern_glyphs):
        """'support tickets' query should prefer high-risk."""
        glyph = _encode_nl_query("customers with too many support tickets", encoder)
        best_high = _best_score_for_level(glyph, pattern_glyphs, "high")
        best_low = _best_score_for_level(glyph, pattern_glyphs, "low")
        assert best_high > best_low, f"high={best_high:.4f} low={best_low:.4f}"

    def test_billing_dispute_high_over_low(self, encoder, pattern_glyphs):
        """'billing dispute' query should prefer high-risk."""
        glyph = _encode_nl_query("customers who disputed their billing", encoder)
        best_high = _best_score_for_level(glyph, pattern_glyphs, "high")
        best_low = _best_score_for_level(glyph, pattern_glyphs, "low")
        assert best_high > best_low, f"high={best_high:.4f} low={best_low:.4f}"

    def test_bugs_and_defects_high_over_low(self, encoder, pattern_glyphs):
        """'bugs and defects' query should prefer high-risk."""
        glyph = _encode_nl_query("customers dealing with bugs and defects", encoder)
        best_high = _best_score_for_level(glyph, pattern_glyphs, "high")
        best_low = _best_score_for_level(glyph, pattern_glyphs, "low")
        assert best_high > best_low, f"high={best_high:.4f} low={best_low:.4f}"

    def test_inactive_high_over_low(self, encoder, pattern_glyphs):
        """'inactive' query should prefer high-risk."""
        glyph = _encode_nl_query("inactive dormant customers", encoder)
        best_high = _best_score_for_level(glyph, pattern_glyphs, "high")
        best_low = _best_score_for_level(glyph, pattern_glyphs, "low")
        assert best_high > best_low, f"high={best_high:.4f} low={best_low:.4f}"

    def test_engaged_low_over_high(self, encoder, pattern_glyphs):
        """'engaged' query should prefer low-risk."""
        glyph = _encode_nl_query("our most engaged active customers", encoder)
        best_high = _best_score_for_level(glyph, pattern_glyphs, "high")
        best_low = _best_score_for_level(glyph, pattern_glyphs, "low")
        assert best_low > best_high, f"low={best_low:.4f} high={best_high:.4f}"


# ---------------------------------------------------------------------------
# Stemming — inflected forms should match the same tier
# ---------------------------------------------------------------------------

class TestStemming:
    """Stemmed variants of key terms should match the same risk level."""

    def test_churning_matches_high(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("churning customers", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_cancelling_matches_high(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("customers cancelling their plans", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_declining_matches_high(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("accounts with declining usage", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "high", f"matched {meta['risk_level']} ({score:.4f})"

    def test_engaged_active_matches_low(self, encoder, pattern_glyphs):
        glyph = _encode_nl_query("customers who are engaged and active", encoder)
        meta, score = _find_best_match(glyph, pattern_glyphs)
        assert meta["risk_level"] == "low", f"matched {meta['risk_level']} ({score:.4f})"
