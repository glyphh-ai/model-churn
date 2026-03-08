"""Test that raw customer data matches the correct risk patterns.

These tests encode raw customer metrics (no risk labels) and compare
against the training exemplars to verify the model correctly identifies
risk levels from metrics alone.

Customer-to-exemplar matching uses the metrics layer cortex since
customers have metrics but no text. The semantic layer drives NL query
matching; the metrics layer drives customer data matching.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG, entry_to_record

# These tests require the glyphh SDK
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
            # Skip semantic-only exemplars (no meaningful metrics)
            if entry.get("risk_level") == "all":
                continue
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
            "description": "",
            "keywords": "",
            "logins": customer["logins"],
            "support_cases": customer["support_cases"],
            "defects": customer["defects"],
            "feature_adoption": customer["feature_adoption"],
        },
    ))


def _metrics_score(glyph1, glyph2):
    """Cosine similarity on metrics layer cortex vectors.

    Customer matching is metrics-driven (customers have data, no text).
    """
    v1 = glyph1.layers["metrics"].cortex.data
    v2 = glyph2.layers["metrics"].cortex.data
    return float(cosine_similarity(v1, v2))


def _find_best_match(customer_glyph, pattern_glyphs):
    """Find the training exemplar most similar to a customer glyph."""
    best_score = -1
    best_meta = None
    for pattern_glyph, meta in pattern_glyphs:
        score = _metrics_score(customer_glyph, pattern_glyph)
        if score > best_score:
            best_score = score
            best_meta = meta
    return best_meta, best_score


def test_inactive_customer_matches_high_risk(
    encoder, pattern_glyphs, expected_high_risk
):
    """acme-corp (0 logins, 0 everything) should match high-risk exemplars."""
    acme = next(c for c in expected_high_risk if c["customer_id"] == "acme-corp")
    glyph = _encode_customer(acme, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)

    assert meta["risk_level"] == "high", (
        f"acme-corp matched {meta['risk_level']} (score={score:.4f}), expected high"
    )


def test_support_heavy_customer_matches_high_risk(
    encoder, pattern_glyphs, expected_high_risk
):
    """beta-inc (18 support cases) should match high-risk exemplars."""
    beta = next(c for c in expected_high_risk if c["customer_id"] == "beta-inc")
    glyph = _encode_customer(beta, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)

    assert meta["risk_level"] == "high", (
        f"beta-inc matched {meta['risk_level']} (score={score:.4f}), expected high"
    )


def test_defect_heavy_customer_matches_high_risk(
    encoder, pattern_glyphs, expected_high_risk
):
    """gamma-llc (9 defects, declining) should match high-risk exemplars."""
    gamma = next(c for c in expected_high_risk if c["customer_id"] == "gamma-llc")
    glyph = _encode_customer(gamma, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)

    assert meta["risk_level"] == "high", (
        f"gamma-llc matched {meta['risk_level']} (score={score:.4f}), expected high"
    )


def test_power_user_matches_low_risk(
    encoder, pattern_glyphs, expected_low_risk
):
    """omega-ai (150 logins, 95% adoption) should match low-risk exemplars."""
    omega = next(c for c in expected_low_risk if c["customer_id"] == "omega-ai")
    glyph = _encode_customer(omega, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)

    assert meta["risk_level"] == "low", (
        f"omega-ai matched {meta['risk_level']} (score={score:.4f}), expected low"
    )


def test_expanding_customer_matches_low_risk(
    encoder, pattern_glyphs, expected_low_risk
):
    """sigma-dev (110 logins, 80% adoption) should match low-risk exemplars."""
    sigma = next(c for c in expected_low_risk if c["customer_id"] == "sigma-dev")
    glyph = _encode_customer(sigma, encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)

    assert meta["risk_level"] == "low", (
        f"sigma-dev matched {meta['risk_level']} (score={score:.4f}), expected low"
    )


def test_high_risk_scores_higher_than_low_risk(
    encoder, pattern_glyphs, expected_high_risk, expected_low_risk
):
    """High-risk customers should have stronger matches to high-risk exemplars
    than low-risk customers do."""
    acme = next(c for c in expected_high_risk if c["customer_id"] == "acme-corp")
    omega = next(c for c in expected_low_risk if c["customer_id"] == "omega-ai")

    acme_glyph = _encode_customer(acme, encoder)
    omega_glyph = _encode_customer(omega, encoder)

    # Get their best match scores against high-risk exemplars only
    high_exemplars = [(g, m) for g, m in pattern_glyphs if m["risk_level"] == "high"]

    acme_best = max(_metrics_score(acme_glyph, pg) for pg, _ in high_exemplars)
    omega_best = max(_metrics_score(omega_glyph, pg) for pg, _ in high_exemplars)

    assert acme_best > omega_best, (
        f"acme-corp ({acme_best:.4f}) should score higher against high-risk "
        f"exemplars than omega-ai ({omega_best:.4f})"
    )
