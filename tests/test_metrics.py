"""Test numeric binning edge cases for metric roles."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG

glyphh = pytest.importorskip("glyphh")

from glyphh import Encoder, Concept, SimilarityCalculator


@pytest.fixture(scope="module")
def encoder():
    return Encoder(ENCODER_CONFIG)


@pytest.fixture(scope="module")
def similarity():
    return SimilarityCalculator()


def _make_customer(encoder, cid, logins=50, support=2, defects=1, adoption=50):
    """Encode a raw customer with specific metric values."""
    return encoder.encode(Concept(
        name=cid,
        attributes={
            "customer_id": cid,
            "risk_level": "",
            "churn_driver": "",
            "usage_band": "",
            "keywords": "",
            "logins": logins,
            "support_cases": support,
            "defects": defects,
            "feature_adoption": adoption,
        },
    ))


def test_zero_logins_encodes(encoder):
    """Zero logins (min boundary) should encode without error."""
    glyph = _make_customer(encoder, "zero-logins", logins=0)
    assert glyph is not None
    assert glyph.global_cortex.dimension == ENCODER_CONFIG.dimension


def test_max_logins_encodes(encoder):
    """Max logins (200, upper boundary) should encode without error."""
    glyph = _make_customer(encoder, "max-logins", logins=200)
    assert glyph is not None


def test_over_max_logins_encodes(encoder):
    """Logins above max_value should clamp and still encode."""
    glyph = _make_customer(encoder, "over-max", logins=500)
    assert glyph is not None


def test_adjacent_logins_more_similar_than_distant(encoder, similarity):
    """Customers with 50 vs 60 logins should be more similar than 50 vs 200."""
    base = _make_customer(encoder, "base", logins=50)
    near = _make_customer(encoder, "near", logins=60)
    far = _make_customer(encoder, "far", logins=200)

    near_score = similarity.compute(base, near).score
    far_score = similarity.compute(base, far).score

    assert near_score > far_score, (
        f"Adjacent logins ({near_score:.4f}) should be more similar than distant ({far_score:.4f})"
    )


def test_zero_support_vs_high_support(encoder, similarity):
    """Zero support cases should be very different from 15 support cases."""
    low = _make_customer(encoder, "low-support", support=0)
    high = _make_customer(encoder, "high-support", support=15)
    mid = _make_customer(encoder, "mid-support", support=3)

    mid_low = similarity.compute(mid, low).score
    high_low = similarity.compute(high, low).score

    assert mid_low > high_low


def test_feature_adoption_boundaries(encoder):
    """Feature adoption at 0% and 100% should both encode."""
    low = _make_customer(encoder, "no-adoption", adoption=0)
    full = _make_customer(encoder, "full-adoption", adoption=100)
    assert low is not None
    assert full is not None


def test_all_zeros_encodes(encoder):
    """A customer with all zero metrics should still encode."""
    glyph = _make_customer(encoder, "all-zeros", logins=0, support=0, defects=0, adoption=0)
    assert glyph is not None
    assert glyph.global_cortex.dimension == ENCODER_CONFIG.dimension


def test_all_max_encodes(encoder):
    """A customer with all max metrics should still encode."""
    glyph = _make_customer(encoder, "all-max", logins=200, support=20, defects=15, adoption=100)
    assert glyph is not None
