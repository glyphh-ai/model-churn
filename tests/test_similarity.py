"""Test that raw customer data matches the correct risk patterns.

These tests encode raw customer metrics (no risk labels) and compare
against the training patterns to verify the model correctly identifies
risk levels from metrics alone.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG, entry_to_record

# These tests require the glyphh SDK
glyphh = pytest.importorskip("glyphh")

from glyphh import Encoder, Concept, SimilarityCalculator

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def encoder():
    return Encoder(ENCODER_CONFIG)


@pytest.fixture(scope="module")
def similarity():
    return SimilarityCalculator()


@pytest.fixture(scope="module")
def pattern_glyphs(encoder):
    """Encode all training patterns into glyphs with their metadata."""
    patterns_path = DATA_DIR / "patterns.jsonl"
    glyphs = []
    with open(patterns_path) as f:
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


def _encode_customer(customer, encoder):
    """Encode a raw customer record (metrics only) into a glyph."""
    return encoder.encode(Concept(
        name=customer["customer_id"],
        attributes={
            "customer_id": customer["customer_id"],
            "risk_level": "",
            "churn_driver": "",
            "usage_band": "",
            "keywords": "",
            "logins": customer["logins"],
            "support_cases": customer["support_cases"],
            "defects": customer["defects"],
            "feature_adoption": customer["feature_adoption"],
        },
    ))


def _find_best_match(customer_glyph, pattern_glyphs, similarity):
    """Find the training pattern most similar to a customer glyph."""
    best_score = -1
    best_meta = None
    for pattern_glyph, meta in pattern_glyphs:
        result = similarity.compute(customer_glyph, pattern_glyph)
        if result.score > best_score:
            best_score = result.score
            best_meta = meta
    return best_meta, best_score


def test_inactive_customer_matches_high_risk(
    encoder, similarity, pattern_glyphs, expected_high_risk
):
    """acme-corp (0 logins, 0 everything) should match high-risk patterns."""
    acme = next(c for c in expected_high_risk if c["customer_id"] == "acme-corp")
    glyph = _encode_customer(acme, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs, similarity)

    assert meta["risk_level"] == "high", (
        f"acme-corp matched {meta['risk_level']} (score={score:.4f}), expected high"
    )


def test_support_heavy_customer_matches_high_risk(
    encoder, similarity, pattern_glyphs, expected_high_risk
):
    """beta-inc (15 support cases) should match high-risk patterns."""
    beta = next(c for c in expected_high_risk if c["customer_id"] == "beta-inc")
    glyph = _encode_customer(beta, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs, similarity)

    assert meta["risk_level"] == "high", (
        f"beta-inc matched {meta['risk_level']} (score={score:.4f}), expected high"
    )


def test_defect_heavy_customer_matches_high_risk(
    encoder, similarity, pattern_glyphs, expected_high_risk
):
    """gamma-llc (9 defects, declining) should match high-risk patterns."""
    gamma = next(c for c in expected_high_risk if c["customer_id"] == "gamma-llc")
    glyph = _encode_customer(gamma, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs, similarity)

    assert meta["risk_level"] == "high", (
        f"gamma-llc matched {meta['risk_level']} (score={score:.4f}), expected high"
    )


def test_power_user_matches_low_risk(
    encoder, similarity, pattern_glyphs, expected_low_risk
):
    """omega-ai (150 logins, 95% adoption) should match low-risk patterns."""
    omega = next(c for c in expected_low_risk if c["customer_id"] == "omega-ai")
    glyph = _encode_customer(omega, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs, similarity)

    assert meta["risk_level"] == "low", (
        f"omega-ai matched {meta['risk_level']} (score={score:.4f}), expected low"
    )


def test_expanding_customer_matches_low_risk(
    encoder, similarity, pattern_glyphs, expected_low_risk
):
    """sigma-dev (110 logins, 80% adoption) should match low-risk patterns."""
    sigma = next(c for c in expected_low_risk if c["customer_id"] == "sigma-dev")
    glyph = _encode_customer(sigma, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs, similarity)

    assert meta["risk_level"] == "low", (
        f"sigma-dev matched {meta['risk_level']} (score={score:.4f}), expected low"
    )


def test_high_risk_scores_higher_than_low_risk(
    encoder, similarity, pattern_glyphs, expected_high_risk, expected_low_risk
):
    """High-risk customers should have stronger matches to high-risk patterns
    than low-risk customers do."""
    # Pick one high-risk and one low-risk customer
    acme = next(c for c in expected_high_risk if c["customer_id"] == "acme-corp")
    omega = next(c for c in expected_low_risk if c["customer_id"] == "omega-ai")

    acme_glyph = _encode_customer(acme, encoder)
    omega_glyph = _encode_customer(omega, encoder)

    # Get their best match scores against high-risk patterns only
    high_patterns = [(g, m) for g, m in pattern_glyphs if m["risk_level"] == "high"]

    acme_best = max(
        similarity.compute(acme_glyph, pg).score for pg, _ in high_patterns
    )
    omega_best = max(
        similarity.compute(omega_glyph, pg).score for pg, _ in high_patterns
    )

    assert acme_best > omega_best, (
        f"acme-corp ({acme_best:.4f}) should score higher against high-risk "
        f"patterns than omega-ai ({omega_best:.4f})"
    )
