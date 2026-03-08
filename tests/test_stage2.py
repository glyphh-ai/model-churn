"""Test Stage 2: exemplar → find similar customers via metrics layer.

Validates that FIND SIMILAR TO glyph(exemplar) AT LAYER metrics returns
customers with similar usage patterns, not just any customer with
overlapping support_cases/defects values.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from encoder import ENCODER_CONFIG, entry_to_record

glyphh = pytest.importorskip("glyphh")

from glyphh import Encoder, Concept
from glyphh.core.ops import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"


@pytest.fixture(scope="module")
def encoder():
    return Encoder(ENCODER_CONFIG)


@pytest.fixture(scope="module")
def exemplar_glyphs(encoder):
    """Encode exemplars, keyed by question text."""
    result = {}
    with open(DATA_DIR / "exemplars.jsonl") as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("risk_level") == "all":
                continue
            record = entry_to_record(entry)
            concept = Concept(name=record["concept_text"], attributes=record["attributes"])
            glyph = encoder.encode(concept)
            result[entry["question"]] = (entry, glyph)
    return result


@pytest.fixture(scope="module")
def customer_glyphs(encoder):
    """Encode all demo customers."""
    result = {}
    with open(DEMO_DIR / "customers.jsonl") as f:
        for line in f:
            c = json.loads(line.strip())
            glyph = encoder.encode(Concept(
                name=c["customer_id"],
                attributes={
                    "customer_id": c["customer_id"],
                    "description": "", "keywords": "",
                    "logins": c["logins"], "support_cases": c["support_cases"],
                    "defects": c["defects"], "feature_adoption": c["feature_adoption"],
                },
            ))
            result[c["customer_id"]] = (c, glyph)
    return result


def _stage2_top(exemplar_glyph, customer_glyphs, n=5):
    """Find top-n customers by metrics layer similarity to an exemplar."""
    scores = []
    for cid, (cdata, cglyph) in customer_glyphs.items():
        v1 = exemplar_glyph.layers["metrics"].cortex.data
        v2 = cglyph.layers["metrics"].cortex.data
        score = float(cosine_similarity(v1, v2))
        scores.append((cid, cdata, score))
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:n]


class TestDowngradeExemplar:
    """'customer downgrading their plan' (L=30, S=2, D=0, A=35)
    should match mid-range customers, not power users."""

    def test_top_match_has_similar_logins(self, exemplar_glyphs, customer_glyphs):
        _, glyph = exemplar_glyphs["customer downgrading their plan"]
        top = _stage2_top(glyph, customer_glyphs, n=3)
        # Top match should have logins within 25 of exemplar (30)
        for cid, cdata, score in top:
            assert cdata["logins"] <= 60, (
                f"Top match {cid} has {cdata['logins']} logins, too far from exemplar (30)"
            )

    def test_power_users_not_top(self, exemplar_glyphs, customer_glyphs):
        _, glyph = exemplar_glyphs["customer downgrading their plan"]
        top = _stage2_top(glyph, customer_glyphs, n=5)
        top_ids = {cid for cid, _, _ in top}
        power_users = {"taskpilot", "saaspro", "rocketship-ai", "alphadev", "greenfield-io"}
        overlap = top_ids & power_users
        assert not overlap, f"Power users in top 5: {overlap}"


class TestHighLoginExemplar:
    """'customer with high logins and growing feature adoption' (L=120, A=80)
    should match power users."""

    def test_top_matches_are_power_users(self, exemplar_glyphs, customer_glyphs):
        _, glyph = exemplar_glyphs["customer with high logins and growing feature adoption"]
        top = _stage2_top(glyph, customer_glyphs, n=5)
        for cid, cdata, score in top:
            assert cdata["logins"] >= 70, (
                f"Top match {cid} has {cdata['logins']} logins, expected power user"
            )


class TestInactiveExemplar:
    """Inactive exemplar should match ghost/inactive customers."""

    def test_top_matches_are_inactive(self, exemplar_glyphs, customer_glyphs):
        _, glyph = exemplar_glyphs["customers who stopped logging in completely"]
        top = _stage2_top(glyph, customer_glyphs, n=5)
        for cid, cdata, score in top:
            assert cdata["logins"] <= 20, (
                f"Top match {cid} has {cdata['logins']} logins, expected inactive"
            )


class TestSupportHeavyExemplar:
    """Support-heavy exemplar should match frustrated customers."""

    def test_top_matches_have_high_support(self, exemplar_glyphs, customer_glyphs):
        _, glyph = exemplar_glyphs["customer hitting same bug repeatedly"]
        top = _stage2_top(glyph, customer_glyphs, n=5)
        for cid, cdata, score in top:
            assert cdata["support_cases"] >= 3 or cdata["defects"] >= 2, (
                f"Top match {cid} (S={cdata['support_cases']}, D={cdata['defects']}) "
                f"expected high support/defects"
            )
