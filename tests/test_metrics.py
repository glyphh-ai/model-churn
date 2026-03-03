"""Test numeric binning edge cases for metric roles."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG

glyphh = pytest.importorskip("glyphh")

from glyphh import Encoder, Concept
from glyphh.core.ops import cosine_similarity


@pytest.fixture(scope="module")
def encoder():
    return Encoder(ENCODER_CONFIG)


def _make_customer(encoder, cid, logins=50, support=2, defects=1, adoption=50):
    """Encode a raw customer with specific metric values."""
    return encoder.encode(Concept(
        name=cid,
        attributes={
            "customer_id": cid,
            "description": "",
            "keywords": "",
            "logins": logins,
            "support_cases": support,
            "defects": defects,
            "feature_adoption": adoption,
        },
    ))


def _metrics_score(glyph1, glyph2):
    """Cosine similarity on metrics layer cortex vectors."""
    v1 = glyph1.layers["metrics"].cortex.data
    v2 = glyph2.layers["metrics"].cortex.data
    return float(cosine_similarity(v1, v2))


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


def test_nearby_logins_more_similar_than_distant(encoder):
    """Customers with nearby logins should be more similar than distant ones."""
    base = _make_customer(encoder, "base", logins=10)
    near = _make_customer(encoder, "near", logins=30)
    far = _make_customer(encoder, "far", logins=180)

    near_score = _metrics_score(base, near)
    far_score = _metrics_score(base, far)

    assert near_score > far_score, (
        f"Nearby logins ({near_score:.4f}) should be more similar than distant ({far_score:.4f})"
    )


def test_low_support_vs_extreme_support(encoder):
    """Low support should be much more similar to zero than extreme support is."""
    zero = _make_customer(encoder, "no-support", support=0, logins=0, defects=0, adoption=0)
    low = _make_customer(encoder, "low-support", support=2, logins=0, defects=0, adoption=0)
    extreme = _make_customer(encoder, "extreme-support", support=20, logins=0, defects=0, adoption=0)

    low_zero = _metrics_score(low, zero)
    extreme_zero = _metrics_score(extreme, zero)

    assert low_zero > extreme_zero, (
        f"Low support ({low_zero:.4f}) should be more similar to zero than extreme ({extreme_zero:.4f})"
    )


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
