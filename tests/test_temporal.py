"""Test temporal behavior — same customer on different days = distinct glyphs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG

glyphh = pytest.importorskip("glyphh")

from glyphh import Encoder, Concept


@pytest.fixture(scope="module")
def encoder():
    return Encoder(ENCODER_CONFIG)


def test_same_customer_different_metrics_different_glyphs(encoder, test_customers):
    """Same customer_id with changed metrics should produce different vectors."""
    acme = next(c for c in test_customers if c["customer_id"] == "acme-corp")

    # Day 1: original metrics (0 logins)
    glyph_day1 = encoder.encode(Concept(
        name=acme["customer_id"],
        attributes={
            "customer_id": acme["customer_id"],
            "risk_level": "",
            "churn_driver": "",
            "usage_band": "",
            "keywords": "",
            "logins": acme["logins"],
            "support_cases": acme["support_cases"],
            "defects": acme["defects"],
            "feature_adoption": acme["feature_adoption"],
        },
    ))

    # Day 2: logins recovered to 40
    glyph_day2 = encoder.encode(Concept(
        name=acme["customer_id"],
        attributes={
            "customer_id": acme["customer_id"],
            "risk_level": "",
            "churn_driver": "",
            "usage_band": "",
            "keywords": "",
            "logins": 40,
            "support_cases": acme["support_cases"],
            "defects": acme["defects"],
            "feature_adoption": acme["feature_adoption"],
        },
    ))

    # Vectors should differ because metrics changed
    assert glyph_day1.global_cortex.data.tolist() != glyph_day2.global_cortex.data.tolist()


def test_customer_id_is_key_part():
    """Verify customer_id is marked as key_part in the config."""
    semantic = next(l for l in ENCODER_CONFIG.layers if l.name == "semantic")
    identity = next(s for s in semantic.segments if s.name == "identity")
    cid = next(r for r in identity.roles if r.name == "customer_id")
    assert cid.key_part is True


def test_temporal_source_is_auto():
    """Temporal source must be auto for daily snapshot use case."""
    assert ENCODER_CONFIG.temporal_source == "auto"
    assert ENCODER_CONFIG.temporal_config.signal_type == "auto"
