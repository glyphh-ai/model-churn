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


def _full_similarity(glyph1, glyph2):
    """Full bundled cortex similarity (both layers combined)."""
    return float(glyph1.similarity(glyph2))


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


def _find_top_k_matches(query_glyph, pattern_glyphs, k=3, sim_fn=_semantic_similarity):
    """Return top-k matches sorted by similarity."""
    scored = []
    for pattern_glyph, meta in pattern_glyphs:
        score = sim_fn(query_glyph, pattern_glyph)
        scored.append((meta, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


# ---------------------------------------------------------------------------
# High-risk NL queries should match high-risk exemplars
# ---------------------------------------------------------------------------

def test_churn_query_matches_high_risk(encoder, pattern_glyphs):
    """'likely to churn' should match high-risk exemplars on semantic layer."""
    glyph = _encode_nl_query("what customers are likely to churn", encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)
    assert meta["risk_level"] == "high", (
        f"'churn' query matched {meta['risk_level']} (score={score:.4f}), expected high"
    )


def test_cancel_query_matches_high_risk(encoder, pattern_glyphs):
    """'about to cancel' should match high-risk exemplars."""
    glyph = _encode_nl_query("customers about to cancel their subscription", encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)
    assert meta["risk_level"] == "high", (
        f"'cancel' query matched {meta['risk_level']} (score={score:.4f}), expected high"
    )


def test_inactive_query_matches_high_risk(encoder, pattern_glyphs):
    """'inactive customers' should match high-risk exemplars."""
    glyph = _encode_nl_query("show me inactive customers", encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)
    assert meta["risk_level"] == "high", (
        f"'inactive' query matched {meta['risk_level']} (score={score:.4f}), expected high"
    )


def test_frustrated_query_matches_high_risk(encoder, pattern_glyphs):
    """'frustrated customers' should match high-risk exemplars."""
    glyph = _encode_nl_query("frustrated customers filing lots of tickets", encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)
    assert meta["risk_level"] == "high", (
        f"'frustrated' query matched {meta['risk_level']} (score={score:.4f}), expected high"
    )


# ---------------------------------------------------------------------------
# Low-risk NL queries should match low-risk exemplars
# ---------------------------------------------------------------------------

def test_healthy_query_matches_low_risk(encoder, pattern_glyphs):
    """'healthy accounts' should match low-risk exemplars."""
    glyph = _encode_nl_query("show me healthy growing accounts", encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)
    assert meta["risk_level"] == "low", (
        f"'healthy' query matched {meta['risk_level']} (score={score:.4f}), expected low"
    )


def test_engaged_query_matches_low_risk(encoder, pattern_glyphs):
    """'engaged customers' should match low-risk exemplars."""
    glyph = _encode_nl_query("customers who are engaged and active", encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)
    assert meta["risk_level"] == "low", (
        f"'engaged' query matched {meta['risk_level']} (score={score:.4f}), expected low"
    )


def test_retained_query_matches_low_risk(encoder, pattern_glyphs):
    """'retained customers' should match low-risk exemplars."""
    glyph = _encode_nl_query("retained customers with strong adoption", encoder)
    meta, score = _find_best_match(glyph, pattern_glyphs)
    assert meta["risk_level"] == "low", (
        f"'retained' query matched {meta['risk_level']} (score={score:.4f}), expected low"
    )


# ---------------------------------------------------------------------------
# Score ordering — high-risk queries should score higher against high-risk
# ---------------------------------------------------------------------------

def test_churn_query_high_risk_score_greater_than_low_risk(encoder, pattern_glyphs):
    """'churn' query should score higher against high-risk exemplars than low-risk."""
    glyph = _encode_nl_query("which customers are at risk of churning", encoder)

    high_exemplars = [(g, m) for g, m in pattern_glyphs if m["risk_level"] == "high"]
    low_exemplars = [(g, m) for g, m in pattern_glyphs if m["risk_level"] == "low"]

    best_high = max(_semantic_similarity(glyph, pg) for pg, _ in high_exemplars)
    best_low = max(_semantic_similarity(glyph, pg) for pg, _ in low_exemplars)

    assert best_high > best_low, (
        f"'churn' query: best high-risk score ({best_high:.4f}) should exceed "
        f"best low-risk score ({best_low:.4f})"
    )


def test_healthy_query_low_risk_score_greater_than_high_risk(encoder, pattern_glyphs):
    """'healthy' query should score higher against low-risk exemplars."""
    glyph = _encode_nl_query("show me healthy satisfied customers", encoder)

    high_exemplars = [(g, m) for g, m in pattern_glyphs if m["risk_level"] == "high"]
    low_exemplars = [(g, m) for g, m in pattern_glyphs if m["risk_level"] == "low"]

    best_high = max(_semantic_similarity(glyph, pg) for pg, _ in high_exemplars)
    best_low = max(_semantic_similarity(glyph, pg) for pg, _ in low_exemplars)

    assert best_low > best_high, (
        f"'healthy' query: best low-risk score ({best_low:.4f}) should exceed "
        f"best high-risk score ({best_high:.4f})"
    )
